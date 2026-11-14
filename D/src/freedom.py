"""FREEDOM (Zhou & Shen, 2023): 고정 modality item-item kNN 그래프 + degree-sensitive edge pruning LightGCN.

- item-item 그래프: 이미지/텍스트 feature cosine top-k (이진) → 대칭 정규화 → w_img·A_img + (1-w_img)·A_txt, 고정
- user-item 그래프: 매 epoch 엣지를 1/sqrt(d_u d_i) 비례 확률로 (1-dropout)만 샘플 → 대칭 정규화 → LightGCN
- 아이템 표현 = LightGCN 아이템 출력 + (item-item 그래프 L_mm층 전파한 아이템 ID 임베딩)
- loss = BPR + reg_weight·(BPR(u, W_v f_v) + BPR(u, W_t f_t))

학습 코드는 정답 파일을 읽지 않는다. --eval-every epoch마다 예측을 저장하고 val 모드에서는
evaluate.py(키트 score)로 채점만 한다.

예) python src/freedom.py --epochs 200 --eval-every 25 --seed 0
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


def load_modalities() -> tuple[torch.Tensor, torch.Tensor]:
    meta = F.load_meta()
    mini = F.sbert_matrix(meta, "sentence-transformers/all-MiniLM-L6-v2", fields=("title", "store", "categories"))
    txt = F.fuse((mini, 1.0), (F.clip_matrix(meta, "text"), 1.0))
    img = F.clip_matrix(meta, "image")
    return torch.from_numpy(img).float(), torch.from_numpy(txt).float()


def sym_norm(idx: torch.Tensor, val: torch.Tensor, n: int) -> torch.Tensor:
    deg = torch.zeros(n, device=val.device).index_add_(0, idx[0], val)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0
    v = dinv[idx[0]] * val * dinv[idx[1]]
    return torch.sparse_coo_tensor(idx, v, (n, n)).coalesce()


def knn_graph(feat: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    f = Fn.normalize(feat, dim=1)
    n = f.shape[0]
    rows, cols = [], []
    for s in range(0, n, chunk):
        sim = f[s : s + chunk] @ f.T
        _, idx = torch.topk(sim, k, dim=1)  # 자기 자신 포함 (원 논문 구현과 동일)
        rows.append(torch.arange(s, s + len(idx), device=f.device).repeat_interleave(k))
        cols.append(idx.reshape(-1))
    idx = torch.stack([torch.cat(rows), torch.cat(cols)])
    return sym_norm(idx, torch.ones(idx.shape[1], device=f.device), n)


class FREEDOM(nn.Module):
    def __init__(self, n_users, n_items, img, txt, d, n_layers, n_mm_layers, knn_k, w_img):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers, self.n_mm_layers = n_layers, n_mm_layers
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.register_buffer("img", img)
        self.register_buffer("txt", txt)
        self.img_trs = nn.Linear(img.shape[1], d)
        self.txt_trs = nn.Linear(txt.shape[1], d)
        nn.init.xavier_normal_(self.img_trs.weight)
        nn.init.xavier_normal_(self.txt_trs.weight)
        with torch.no_grad():
            a_img = knn_graph(img.to(C.DEVICE), knn_k)
            a_txt = knn_graph(txt.to(C.DEVICE), knn_k)
            self.mm_adj = (w_img * a_img + (1 - w_img) * a_txt).coalesce()

    def forward(self, adj: torch.Tensor):
        h = self.item_emb.weight
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight])
        outs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego)
            outs.append(ego)
        allm = torch.stack(outs, 1).mean(1)
        u, i = allm[: self.n_users], allm[self.n_users :]
        return u, i + h


def bpr(u, p, n):
    return -Fn.logsigmoid((u * p).sum(-1) - (u * n).sum(-1)).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-mm-layers", type=int, default=1)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--w-img", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.8, help="학습 중 잘라낼 user-item 엣지 비율")
    ap.add_argument("--reg-weight", type=float, default=1e-4, help="modality BPR loss 가중치")
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="val", choices=list(C.MODES))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = C.DEVICE
    nU, nI = C.N_USERS, C.N_ITEMS

    df, users = C.load_fit(a.mode)
    X = C.interaction_matrix(df)
    pop = C.popularity(df)
    U = torch.from_numpy(df["user_id"].to_numpy().copy()).to(dev)
    I = torch.from_numpy(df["item_id"].to_numpy().copy()).to(dev)
    seen_keys = torch.sort(U * nI + I).values
    deg_u = torch.bincount(U, minlength=nU).float()
    deg_i = torch.bincount(I, minlength=nI).float()
    edge_p = 1.0 / torch.sqrt(deg_u[U] * deg_i[I])

    def build_adj(keep_idx=None):
        u, i = (U, I) if keep_idx is None else (U[keep_idx], I[keep_idx])
        idx = torch.cat([torch.stack([u, i + nU]), torch.stack([i + nU, u])], 1)
        return sym_norm(idx, torch.ones(idx.shape[1], device=dev), nU + nI)

    full_adj = build_adj()
    img, txt = load_modalities()
    model = FREEDOM(nU, nI, img, txt, a.d, a.n_layers, a.n_mm_layers, a.knn_k, a.w_img).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)

    name_base = (f"freedom_d{a.d}_L{a.n_layers}_k{a.knn_k}_wi{a.w_img:g}_do{a.dropout:g}"
                 f"_rw{a.reg_weight:g}_wd{a.wd:g}_lr{a.lr:g}")
    n = len(U)
    n_keep = int(n * (1 - a.dropout))
    for ep in range(1, a.epochs + 1):
        model.train()
        t, total = time.time(), 0.0
        keep = torch.multinomial(edge_p, n_keep, replacement=False) if a.dropout > 0 else None
        adj = build_adj(keep)
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for s in range(0, n, a.batch):
            b = perm[s : s + a.batch]
            u, i = U[b], I[b]
            j = torch.randint(0, nI, (len(b),), device=dev)
            k = u * nI + j
            pos = torch.searchsorted(seen_keys, k).clamp(max=len(seen_keys) - 1)
            j = torch.where(seen_keys[pos] == k, torch.randint(0, nI, (len(b),), device=dev), j)
            ua, ia = model(adj)
            loss = bpr(ua[u], ia[i], ia[j])
            if a.reg_weight > 0:
                fv, ft = model.img_trs(model.img), model.txt_trs(model.txt)
                loss = loss + a.reg_weight * (bpr(ua[u], fv[i], fv[j]) + bpr(ua[u], ft[i], ft[j]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep} loss {total / n:.4f} ({time.time() - t:.1f}s)", flush=True)

        if ep % a.eval_every == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                ua, ia = model(full_adj)

            @torch.no_grad()
            def score_fn(us):
                return ua[torch.as_tensor(us, device=dev)] @ ia.T

            items, scores = C.recommend_scores_fn(score_fn, X, users, pop)
            name = f"{name_base}_e{ep}_s{a.seed}"
            pred = C.save_outputs(name, a.mode, users, items, scores)
            if a.mode == "val":
                from evaluate import evaluate
                evaluate(pred, name, seed=str(a.seed), extra_inputs="item_meta.jsonl+images", log=not a.no_log)
            else:
                print(f"wrote {pred}")


if __name__ == "__main__":
    main()
