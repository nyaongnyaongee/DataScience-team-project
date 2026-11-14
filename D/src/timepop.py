"""시간 인기도 (timepop): 유저가 활동하던 시기에 인기 있던 아이템을 추천한다.

- 타임스탬프를 bin(기본 30일)으로 나눠 아이템별 bin 상호작용 수 P[bin, item]을 만든다.
- 유저 시간 가중치 w_u[bin] = Σ_(유저의 학습 상호작용 j) K(bin - bin_j),  K = 가우시안(σ bin)
  (--kernel window 이면 [첫 구매 - pad, 마지막 구매 + pad] 구간 균일 가중치)
- 점수 = w_u @ P_norm + pop_beta · 전체 인기도.  P_norm: --item-norm 에 따라 원본 또는 아이템 총량^γ 로 나눔.

학습 데이터만 읽는다. val 모드에서는 evaluate.py(키트 score)로 채점한다.

예) python src/timepop.py --sigma 3
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import common as C

DAY_MS = 86_400_000


def time_bins(ts: np.ndarray, bin_days: float, t0: int) -> np.ndarray:
    return ((ts - t0) // int(bin_days * DAY_MS)).astype(np.int64)


def build_scores(df, bin_days: float, sigma: float, kernel: str, pad: float, gamma: float):
    """(user_weights[nU, nB] torch, item_bin[nB, nI] torch) 반환."""
    t0 = int(df["timestamp"].min())
    b = time_bins(df["timestamp"].to_numpy(), bin_days, t0)
    n_b = int(b.max()) + 1 + 24  # 최종 모드에서 범위가 조금 넓어져도 되도록 여유
    u = df["user_id"].to_numpy()
    i = df["item_id"].to_numpy()
    dev = C.DEVICE
    P = torch.zeros((n_b, C.N_ITEMS), device=dev)
    P.index_put_((torch.from_numpy(b).to(dev), torch.from_numpy(i).to(dev)),
                 torch.ones(len(b), device=dev), accumulate=True)
    if gamma > 0:
        tot = P.sum(0).clamp(min=1)
        P = P / tot.pow(gamma)
    H = torch.zeros((C.N_USERS, n_b), device=dev)
    H.index_put_((torch.from_numpy(u).to(dev), torch.from_numpy(b).to(dev)),
                 torch.ones(len(b), device=dev), accumulate=True)
    if kernel == "gauss":
        r = int(np.ceil(4 * sigma))
        x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        W = torch.nn.functional.conv1d(H[:, None, :], k[None, None, :], padding=r)[:, 0, :]
    elif kernel == "window":
        idx = torch.arange(n_b, device=dev)[None, :]
        has = H > 0
        first = torch.where(has, idx, n_b).min(1).values[:, None]
        last = torch.where(has, idx, -1).max(1).values[:, None]
        W = ((idx >= first - pad) & (idx <= last + pad)).float()
    else:
        raise ValueError(kernel)
    return W, P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-days", type=float, default=30)
    ap.add_argument("--kernel", default="gauss", choices=["gauss", "window"])
    ap.add_argument("--sigma", type=float, default=3.0, help="가우시안 폭 (bin 단위)")
    ap.add_argument("--pad", type=float, default=3.0, help="window 커널 여유 (bin 단위)")
    ap.add_argument("--gamma", type=float, default=0.0, help="아이템 총 인기도^γ 로 나눔")
    ap.add_argument("--pop-beta", type=float, default=0.0, help="전체 인기도 혼합 (0~1 정규화 후)")
    ap.add_argument("--mode", default="val", choices=list(C.MODES))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    df, users = C.load_fit(a.mode)
    X = C.interaction_matrix(df)
    pop = C.popularity(df)
    W, P = build_scores(df, a.bin_days, a.sigma, a.kernel, a.pad, a.gamma)
    pop_t = torch.from_numpy(pop / pop.max()).to(C.DEVICE)

    @torch.no_grad()
    def score_fn(us):
        w = W[torch.as_tensor(us, device=C.DEVICE)]
        s = w @ P
        s = s / s.max(1, keepdim=True).values.clamp(min=1e-12)
        return s + a.pop_beta * pop_t

    items, scores = C.recommend_scores_fn(score_fn, X, users, pop)
    kpart = f"g{a.sigma:g}" if a.kernel == "gauss" else f"w{a.pad:g}"
    name = f"timepop_b{a.bin_days:g}_{kpart}_gm{a.gamma:g}_pb{a.pop_beta:g}"
    pred = C.save_outputs(name, a.mode, users, items, scores)
    if a.mode == "val":
        from evaluate import evaluate
        evaluate(pred, name, extra_inputs="timestamp", log=not a.no_log)
    else:
        print(f"wrote {pred}")


if __name__ == "__main__":
    main()
