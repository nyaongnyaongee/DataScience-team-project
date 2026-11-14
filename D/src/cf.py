"""순수 CF 모델: EASE, RP3β, iALS (블렌드 재료용).

- EASE (Steck 2019): B = -P/diag(P), P = (XᵀX + λI)⁻¹, diag(B)=0. 점수 X·B.
- RP3β (Paudel 2017): W = (P_iu^α · P_ui^α) / pop_j^β, 행마다 top-k. 점수 X·W.
- iALS (Hu 2008): 신뢰도 c = 1 + α·x, 정확 해 ALS (유저·아이템 교대로 batched solve).

학습 데이터만 읽는다. val 모드에서는 evaluate.py(키트 score)로 채점한다.

예) python src/cf.py ease --lam 500
    python src/cf.py rp3b --alpha 1 --beta 0.4 --k 200
    python src/cf.py ials --d 128 --alpha 10 --reg 1 --iters 15 --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import torch

import common as C


def dense_X(X: sp.csr_matrix) -> torch.Tensor:
    return torch.from_numpy(X.toarray()).to(C.DEVICE)


def ease(X: sp.csr_matrix, lam: float) -> torch.Tensor:
    Xd = dense_X(X)
    G = Xd.T @ Xd
    del Xd
    G.diagonal().add_(lam)
    P = torch.linalg.inv(G)
    del G
    B = -P / torch.diagonal(P)[None, :]
    B.fill_diagonal_(0)
    return B


def rp3b(X: sp.csr_matrix, alpha: float, beta: float, k: int) -> torch.Tensor:
    Xd = dense_X(X)
    pui = Xd / Xd.sum(1, keepdim=True).clamp(min=1)          # user → item
    piu = (Xd / Xd.sum(0, keepdim=True).clamp(min=1)).T      # item → user
    del Xd
    W = piu.pow(alpha) @ pui.pow(alpha)
    del pui, piu
    pop = torch.from_numpy(np.asarray(X.sum(0)).ravel()).float().to(C.DEVICE).clamp(min=1)
    W = W / pop.pow(beta)[None, :]
    return C.topk_sparsify(W, k)


def ials(X: sp.csr_matrix, d: int, alpha: float, reg: float, iters: int, seed: int,
         chunk: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    dev = C.DEVICE
    coo = X.tocoo()
    u_idx = torch.from_numpy(coo.row.astype(np.int64)).to(dev)
    i_idx = torch.from_numpy(coo.col.astype(np.int64)).to(dev)
    nU, nI = X.shape
    U = torch.randn(nU, d, device=dev) * 0.01
    V = torch.randn(nI, d, device=dev) * 0.01
    eye = torch.eye(d, device=dev)

    def solve(rows, cols, n_rows, Y):
        # A_r = YᵀY + α Σ_{c∈r} y_c y_cᵀ + reg·I,   b_r = (1+α) Σ_{c∈r} y_c
        # 행 블록 단위로 A를 만들어 메모리를 nU×d×d 전체보다 작게 유지
        base = Y.T @ Y + reg * eye
        out = torch.empty(n_rows, d, device=dev)
        order = torch.argsort(rows)
        rows_s, cols_s = rows[order], cols[order]
        bounds = torch.searchsorted(rows_s, torch.arange(0, n_rows + 1, device=dev))
        row_block = max(1, int(2e9 // (4 * d * d)))  # 블록당 약 2GB
        for r0 in range(0, n_rows, row_block):
            r1 = min(n_rows, r0 + row_block)
            A = base.expand(r1 - r0, d, d).clone()
            b = torch.zeros(r1 - r0, d, device=dev)
            e0, e1 = int(bounds[r0]), int(bounds[r1])
            step = max(1, min(chunk, int(5e8 // (d * d))))  # 외적 버퍼 약 2GB 이하
            for s in range(e0, e1, step):
                r, c = rows_s[s : min(s + step, e1)] - r0, cols_s[s : min(s + step, e1)]
                y = Y[c]
                A.index_add_(0, r, alpha * y[:, :, None] * y[:, None, :])
                b.index_add_(0, r, (1 + alpha) * y)
            out[r0:r1] = torch.linalg.solve(A, b)
        return out

    for it in range(iters):
        U = solve(u_idx, i_idx, nU, V)
        V = solve(i_idx, u_idx, nI, U)
    return U, V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["ease", "rp3b", "ials"])
    ap.add_argument("--lam", type=float, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="val", choices=list(C.MODES))
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    df, users = C.load_fit(a.mode)
    X = C.interaction_matrix(df)
    pop = C.popularity(df)
    seed = "NA"
    with torch.no_grad():
        if a.model == "ease":
            name = f"ease_l{a.lam:g}"
            items, scores = C.recommend_item_item(X, ease(X, a.lam), users, pop)
        elif a.model == "rp3b":
            name = f"rp3b_a{a.alpha:g}_b{a.beta:g}_k{a.k}"
            items, scores = C.recommend_item_item(X, rp3b(X, a.alpha, a.beta, a.k), users, pop)
        else:
            name = f"ials_d{a.d}_a{a.alpha:g}_r{a.reg:g}_it{a.iters}_s{a.seed}"
            seed = str(a.seed)
            U, V = ials(X, a.d, a.alpha, a.reg, a.iters, a.seed)
            items, scores = C.recommend_scores_fn(
                lambda us: U[torch.as_tensor(us, device=C.DEVICE)] @ V.T, X, users, pop)
    pred = C.save_outputs(name, a.mode, users, items, scores)
    if a.mode == "val":
        from evaluate import evaluate
        evaluate(pred, name, seed=seed, extra_inputs="none", log=not a.no_log)
    else:
        print(f"wrote {pred}")


if __name__ == "__main__":
    main()
