"""공통 파이프라인: 데이터 로드, item-item 점수 → top-100 추천, 결과 저장.

이 모듈은 학습용 데이터(val_train.csv 또는 원본 train.csv)만 읽는다.
정답(val_truth.csv)은 evaluate.py에서만 읽는다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "teamkit"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = DATA / "cache"
SPLIT_ID = "f0dfad2f4431a876"
N_USERS = 36043
N_ITEMS = 19547

MODES = {
    "val": {"fit": KIT / "validation/val_train.csv", "users": KIT / "validation/val_users.csv"},
    "final": {"fit": DATA / "train.csv", "users": DATA / "test.csv"},
    # reranker 학습용 내부 split: teamkit split --train validation/val_train.csv 로 만든 것
    "inner": {"fit": DATA / "inner/val_train.csv", "users": DATA / "inner/val_users.csv"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fit(mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    df = pd.read_csv(MODES[mode]["fit"])
    users = pd.read_csv(MODES[mode]["users"])["user_id"].to_numpy()
    return df, users


def interaction_matrix(df: pd.DataFrame, weight: str = "binary") -> sp.csr_matrix:
    if weight == "binary":
        vals = np.ones(len(df), np.float32)
    elif weight == "rating":
        vals = (df["rating"].to_numpy(np.float32) / 5.0)
    else:
        raise ValueError(weight)
    return sp.csr_matrix(
        (vals, (df["user_id"].to_numpy(), df["item_id"].to_numpy())), shape=(N_USERS, N_ITEMS)
    )


def popularity(df: pd.DataFrame) -> np.ndarray:
    return np.bincount(df["item_id"].to_numpy(), minlength=N_ITEMS).astype(np.float32)


def recommend_item_item(
    X: sp.csr_matrix,
    S: torch.Tensor,
    users: np.ndarray,
    pop: np.ndarray,
    topn: int = 100,
    batch: int = 2048,
    pop_beta: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """점수 = X[u] @ S + pop_beta * 인기도(0~1). 이미 본 아이템은 제외.
    pop_beta가 작으면(기본 1e-6) 인기도는 동점·0점 채우기에만 쓰인다."""
    seen = (X > 0).astype(np.float32).tocsr()
    pop_t = torch.from_numpy(pop / pop.max()).to(DEVICE) * pop_beta
    out_items = np.empty((len(users), topn), np.int64)
    out_scores = np.empty((len(users), topn), np.float32)
    for start in range(0, len(users), batch):
        u = users[start : start + batch]
        xb = torch.from_numpy(X[u].toarray()).to(DEVICE)
        mask = torch.from_numpy(seen[u].toarray()).to(DEVICE) > 0
        scores = xb @ S + pop_t
        scores[mask] = -torch.inf
        val, idx = torch.topk(scores, topn, dim=1)
        out_items[start : start + len(u)] = idx.cpu().numpy()
        out_scores[start : start + len(u)] = val.cpu().numpy()
    return out_items, out_scores


def recommend_scores_fn(score_fn, X: sp.csr_matrix, users: np.ndarray, pop: np.ndarray,
                        topn: int = 100, batch: int = 2048):
    """score_fn(user_idx_array) -> torch [b, N_ITEMS] 점수. 마스킹·top-k는 여기서."""
    seen = (X > 0).astype(np.float32).tocsr()
    pop_t = torch.from_numpy(pop / pop.max()).to(DEVICE) * 1e-6
    out_items = np.empty((len(users), topn), np.int64)
    out_scores = np.empty((len(users), topn), np.float32)
    for start in range(0, len(users), batch):
        u = users[start : start + batch]
        scores = score_fn(u).float() + pop_t
        mask = torch.from_numpy(seen[u].toarray()).to(DEVICE) > 0
        scores[mask] = -torch.inf
        val, idx = torch.topk(scores, topn, dim=1)
        out_items[start : start + len(u)] = idx.cpu().numpy()
        out_scores[start : start + len(u)] = val.cpu().numpy()
    return out_items, out_scores


def topk_sparsify(sim: torch.Tensor, k: int | None) -> torch.Tensor:
    """행마다 상위 k개만 남긴다. 대각선(자기 자신)은 0."""
    sim.fill_diagonal_(0)
    if k is None or k >= sim.shape[1]:
        return sim
    val, idx = torch.topk(sim, k, dim=1)
    out = torch.zeros_like(sim)
    out.scatter_(1, idx, val)
    return out


def cosine_item_sim(E, k: int | None, chunk: int = 2048) -> torch.Tensor:
    """E: 행 L2 정규화된 아이템 특성 (scipy sparse 또는 numpy/torch dense)."""
    if sp.issparse(E):
        E = E.tocsr().astype(np.float32)
        Et = E.T.tocsc()
        S = torch.zeros((N_ITEMS, N_ITEMS), dtype=torch.float32, device=DEVICE)
        for s in range(0, N_ITEMS, chunk):
            block = torch.from_numpy((E[s : s + chunk] @ Et).toarray()).to(DEVICE)
            S[s : s + chunk] = block
    else:
        Et = torch.as_tensor(E, dtype=torch.float32, device=DEVICE)
        Et = torch.nn.functional.normalize(Et, dim=1)
        S = Et @ Et.T
    S.clamp_(min=0)
    return topk_sparsify(S, k)


def save_outputs(name: str, mode: str, users: np.ndarray, items: np.ndarray, scores: np.ndarray) -> Path:
    """results/{name}_{mode}.csv (제출 형식 top-20) + results/{name}_{mode}_top100.csv.gz."""
    RESULTS.mkdir(exist_ok=True)
    pred = RESULTS / f"{name}_{mode}.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id"])
        for u, row in zip(users, items):
            w.writerow([int(u), " ".join(map(str, row[:20]))])
    n = items.shape[1]
    top = pd.DataFrame({
        "user_id": np.repeat(users, n),
        "item_id": items.ravel(),
        "score": scores.ravel(),
        "rank": np.tile(np.arange(1, n + 1), len(users)),
    })
    top.to_csv(RESULTS / f"{name}_{mode}_top100.csv.gz", index=False)
    return pred
