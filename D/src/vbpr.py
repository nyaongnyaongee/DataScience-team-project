"""VBPR (He & McAuley, 2016): BPR-MF + 고정 아이템 feature의 선형 투영.

score(u, i) = b_i + γ_u·γ_i + θ_u·(E f_i) + β'·f_i

f_i: 콘텐츠 임베딩(기본 MiniLM + 0.5·CLIP 이미지 + CLIP 텍스트, 1920차원).
--features none 이면 θ/E/β' 없이 순수 BPR-MF (feature 기여도 비교용).

학습 코드는 정답 파일을 읽지 않는다. --eval-every epoch마다 예측을 저장하고,
val 모드에서는 evaluate.py(키트 score)로 채점만 한다. early stopping 없음.

예) python src/vbpr.py --epochs 60 --eval-every 10 --seed 0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import common as C
import features as F


def load_item_features(name: str) -> torch.Tensor | None:
    if name == "none":
        return None
    meta = F.load_meta()
    mini = F.sbert_matrix(meta, "sentence-transformers/all-MiniLM-L6-v2", fields=("title", "store", "categories"))
    img, txt = F.clip_matrix(meta, "image"), F.clip_matrix(meta, "text")
    parts = {"fuse": [(mini, 1), (img, 0.5), (txt, 1)], "text": [(mini, 1), (txt, 1)], "image": [(img, 1)]}[name]
    return torch.from_numpy(F.fuse(*parts)).float()


class VBPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, feats: torch.Tensor | None, d: int, d_vis: int):
        super().__init__()
        self.gu = nn.Embedding(n_users, d)
        self.gi = nn.Embedding(n_items, d)
        self.bi = nn.Embedding(n_items, 1)
        for e in (self.gu, self.gi):
            nn.init.normal_(e.weight, std=0.01)
        nn.init.zeros_(self.bi.weight)
        self.use_feat = feats is not None
        if self.use_feat:
            self.register_buffer("f", feats)
            self.tu = nn.Embedding(n_users, d_vis)
            nn.init.normal_(self.tu.weight, std=0.01)
            self.E = nn.Linear(feats.shape[1], d_vis, bias=False)
            self.bp = nn.Linear(feats.shape[1], 1, bias=False)
            nn.init.normal_(self.E.weight, std=0.01)
            nn.init.zeros_(self.bp.weight)

    def score(self, u, i):
        s = (self.gu(u) * self.gi(i)).sum(-1) + self.bi(i).squeeze(-1)
        if self.use_feat:
            fi = self.f[i]
            s = s + (self.tu(u) * self.E(fi)).sum(-1) + self.bp(fi).squeeze(-1)
        return s

    def reg(self, u, i, j):
        r = self.gu(u).pow(2).sum(-1) + self.gi(i).pow(2).sum(-1) + self.gi(j).pow(2).sum(-1)
        r = r + self.bi(i).squeeze(-1).pow(2) + self.bi(j).squeeze(-1).pow(2)
        if self.use_feat:
            r = r + self.tu(u).pow(2).sum(-1)
        return r.mean()

    @torch.no_grad()
    def full_scores_fn(self):
        item_bias = self.bi.weight.squeeze(-1)
        Gi = self.gi.weight
        if self.use_feat:
            Ef = self.E(self.f)
            item_bias = item_bias + self.bp(self.f).squeeze(-1)

        @torch.no_grad()
        def fn(users: np.ndarray) -> torch.Tensor:
            u = torch.as_tensor(users, device=Gi.device)
            s = self.gu(u) @ Gi.T + item_bias
            if self.use_feat:
                s = s + self.tu(u) @ Ef.T
            return s
        return fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="fuse", choices=["fuse", "text", "image", "none"])
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--d-vis", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-4, help="임베딩 L2 (배치 평균)")
    p.add_argument("--wd-E", type=float, default=1e-5, help="E, β' weight decay")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="val", choices=list(C.MODES))
    p.add_argument("--tag", default="")
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = C.DEVICE

    df, users = C.load_fit(a.mode)
    X = C.interaction_matrix(df)
    pop = C.popularity(df)
    U = torch.from_numpy(df["user_id"].to_numpy().copy()).to(dev)
    I = torch.from_numpy(df["item_id"].to_numpy().copy()).to(dev)
    seen_keys = torch.sort(U * C.N_ITEMS + I).values

    feats = load_item_features(a.features)
    model = VBPR(C.N_USERS, C.N_ITEMS, feats, a.d, a.d_vis).to(dev)
    emb_params = [p_ for n, p_ in model.named_parameters() if not n.startswith(("E.", "bp."))]
    groups = [{"params": emb_params, "weight_decay": 0.0}]
    if model.use_feat:
        groups.append({"params": list(model.E.parameters()) + list(model.bp.parameters()), "weight_decay": a.wd_E})
    opt = torch.optim.Adam(groups, lr=a.lr)

    name_base = (f"vbpr_{a.features}_d{a.d}" + (f"v{a.d_vis}" if model.use_feat else "")
                 + f"_lr{a.lr:g}_reg{a.reg:g}" + (f"_{a.tag}" if a.tag else ""))
    n = len(U)
    for ep in range(1, a.epochs + 1):
        model.train()
        t, total = time.time(), 0.0
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for s in range(0, n, a.batch):
            b = perm[s : s + a.batch]
            u, i = U[b], I[b]
            j = torch.randint(0, C.N_ITEMS, (len(b),), device=dev)
            # 이미 본 아이템이 뽑히면 한 번 다시 뽑는다
            k = u * C.N_ITEMS + j
            pos = torch.searchsorted(seen_keys, k).clamp(max=len(seen_keys) - 1)
            hit = seen_keys[pos] == k
            j = torch.where(hit, torch.randint(0, C.N_ITEMS, (len(b),), device=dev), j)
            loss = -Fn.logsigmoid(model.score(u, i) - model.score(u, j)).mean() + a.reg * model.reg(u, i, j)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"epoch {ep} loss {total / n:.4f} ({time.time() - t:.1f}s)", flush=True)

        if ep % a.eval_every == 0 or ep == a.epochs:
            model.eval()
            items, scores = C.recommend_scores_fn(model.full_scores_fn(), X, users, pop)
            name = f"{name_base}_e{ep}_s{a.seed}"
            pred = C.save_outputs(name, a.mode, users, items, scores)
            if a.mode == "val":
                from evaluate import evaluate
                evaluate(pred, name, seed=str(a.seed),
                         extra_inputs="item_meta.jsonl+images" if model.use_feat else "none",
                         log=not a.no_log)
            else:
                print(f"wrote {pred}")


if __name__ == "__main__":
    main()
