"""팀원 C 코드(src/ext/, 원문 그대로)로 top-100 파일을 만든다.

- lightgcn: C001 설정 (3층, d128, lr .003, reg 1e-4, 100,000 표본/epoch, seed 42, 575 epoch)
- sgl     : C005 설정 (3층, d128, lr .003, reg 1e-4, SSL .0005, drop .1, tau .2, 525 epoch)
- ncl     : C007 설정 (3층, d128, lr .003, reg 1e-4, 구조 SSL .001, prototype 0, 425 epoch)
            sgl/ncl은 C 원문 sgl_ncl_runner.py를 그대로 실행(--checkpoint)하고, 저장된 가중치로 top-100을 뽑는다.
- simplex : C011 설정 (X01: d128, lr .002, gamma .5, neg 20, margin .2, neg_weight 4,
            reg 1e-5, 35 step/epoch, batch 2048, seed 42, 650 epoch)

C 원본은 val에서 val_truth로 best epoch을 골라 그 시점 가중치를 썼다.
여기서는 세 모드 모두 val_truth 없이 그 epoch 수로 고정 학습한다 (epoch 수 자체는 C가 val에서 고른 값).
점수(score)는 C 코드가 쓰는 내적/코사인 그대로이다.

예) python src/team_models.py lightgcn --mode inner
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common as C

sys.path.insert(0, str(Path(__file__).resolve().parent / "ext"))
import graph_models as gm  # noqa: E402
import lightgcn_top100_export as lx  # noqa: E402
import ultra_simplex_runner as ux  # noqa: E402


def write_top100(data, user_vecs, item_vecs, name, mode, device, batch=512, topk=100):
    """C의 predict_and_write와 같은 방식(내적, 본 아이템 제외)으로 top-100 저장."""
    users = data.target_users
    items_all = np.empty((len(users), topk), np.int64)
    scores_all = np.empty((len(users), topk), np.float32)
    item_ids = np.array([int(x) for x in data.item_ids])
    iv = item_vecs.to(device)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        idx = torch.tensor([data.user_index[u] for u in ub], device=device)
        sc = user_vecs.to(device)[idx] @ iv.T
        for r, u in enumerate(ub):
            seen = list(data.seen_items[data.user_index[u]])
            sc[r, seen] = -torch.inf
        v, ix = torch.topk(sc, topk, dim=1)
        items_all[s:s + len(ub)] = item_ids[ix.cpu().numpy()]
        scores_all[s:s + len(ub)] = v.cpu().numpy()
    uids = np.array([int(u) for u in users])
    return C.save_outputs(name, mode, uids, items_all, scores_all)


def train_simplex(data, device, epochs, seed=42, dim=128, lr=.002, gamma=.5, negatives=20,
                  margin=.2, neg_weight=4., reg=1e-5, steps=35, batch=2048):
    """ultra_simplex_runner.run 의 simplex 분기와 같은 학습 루프 (평가·조기종료 제거)."""
    ux.seed_all(seed)
    mat = ux.get_interaction_matrix(data)
    model = ux.SimpleXStyle(data.num_users, data.num_items, dim, gamma).to(device)
    rowptr = torch.tensor(mat.indptr.astype(np.int64), device=device)
    flat_history = torch.tensor(mat.indices.astype(np.int64), device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    u_edges = torch.tensor(data.positive_users, dtype=torch.long, device=device)
    i_edges = torch.tensor(data.positive_items, dtype=torch.long, device=device)
    observed = (u_edges * data.num_items + i_edges).sort().values.unique(sorted=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        acc = 0.0
        for _ in range(steps):
            u, i, j = ux.sample_batch(u_edges, i_edges, data.num_items, observed, batch, negatives)
            optim.zero_grad(set_to_none=True)
            history_sum, lengths = ux.history_sum_for_batch(model.item_emb, flat_history, rowptr, u)
            agg_loo = (history_sum - model.item_emb(i)) / (lengths - 1).clamp_min(1)[:, None]
            user_r = torch.nn.functional.normalize(model.user_emb(u) + model.gamma * agg_loo, dim=-1)
            p = torch.nn.functional.normalize(model.item_emb(i), dim=-1)
            n = torch.nn.functional.normalize(model.item_emb(j), dim=-1)
            pos_s = (user_r * p).sum(-1)
            neg_s = (user_r[:, None, :] * n).sum(-1)
            loss = ((1.0 - pos_s) + neg_weight * torch.nn.functional.relu(neg_s - margin).square().mean(-1)).mean()
            r = .5 * (model.user_emb(u).square().sum(-1).mean() + model.item_emb(i).square().sum(-1).mean()
                      + model.item_emb(j).square().sum(-1).mean())
            loss = loss + reg * r
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            optim.step()
            acc += loss.item()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"simplex seed{seed} epoch {epoch}/{epochs} loss {acc / steps:.5f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        uv, iv = model.vectors(mat)
    return uv.detach(), iv.detach()


def run_ssl(model, data, mode, epochs, seed, device):
    import json
    import subprocess
    import tempfile
    cfg = {"sgl": dict(ssl=.0005, proto=0.), "ncl": dict(ssl=.001, proto=0.)}[model]
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "ck.pt"
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "ext/sgl_ncl_runner.py"), "--model", model,
               "--fit", str(C.MODES[mode]["fit"]), "--users", str(C.MODES[mode]["users"]),
               "--out", str(Path(td) / "top20.csv"), "--epochs", str(epochs), "--lr", "0.003", "--reg", "0.0001",
               "--seed", str(seed), "--device", "cuda:0", "--dim", "128", "--layers", "3",
               "--samples-per-epoch", "100000", "--eval-every", "25", "--patience", "0",
               "--ssl-weight", str(cfg["ssl"]), "--proto-weight", str(cfg["proto"]), "--drop", "0.1", "--tau", "0.2",
               "--ssl-nodes", "192", "--warmup", "100", "--checkpoint", str(ck)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        state = torch.load(ck, map_location=device)
    m = gm.LightGCN(data.num_users, data.num_items, 128, 3).to(device)
    m.load_state_dict(state["weights"])
    m.eval()
    with torch.no_grad():
        uv, iv = m(gm.normalized_graph(data, device))
    return uv, iv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["lightgcn", "simplex", "sgl", "ncl"])
    ap.add_argument("--mode", default="val", choices=list(C.MODES))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name")
    a = ap.parse_args()
    dev = C.DEVICE
    data = gm.load_data(C.MODES[a.mode]["fit"], C.MODES[a.mode]["users"])
    t0 = time.time()
    if a.model == "lightgcn":
        ep = a.epochs or 575
        name = a.name or f"lgcnC_e{ep}_s{a.seed}"
        graph = gm.normalized_graph(data, dev)
        uv, iv, _ = lx.train_one(data, graph, dev, seed=a.seed, epochs=ep, dim=128, layers=3, lr=.003,
                                 reg=1e-4, samples_per_epoch=100000, truth_file=None,
                                 eval_every=25, eval_batch=128)
    elif a.model in ("sgl", "ncl"):
        ep = a.epochs or {"sgl": 525, "ncl": 425}[a.model]
        name = a.name or f"{a.model}C_e{ep}_s{a.seed}"
        uv, iv = run_ssl(a.model, data, a.mode, ep, a.seed, dev)
    else:
        ep = a.epochs or 650
        name = a.name or f"simplexC_e{ep}_s{a.seed}"
        uv, iv = train_simplex(data, dev, ep, seed=a.seed)
    pred = write_top100(data, uv, iv, name, a.mode, dev)
    print(f"[{name} {a.mode}] train+predict {time.time() - t0:.0f}s -> {pred}", flush=True)
    if a.mode == "val":
        from evaluate import evaluate
        evaluate(pred, name, extra_inputs="none", seed=str(a.seed),
                 notes=f"팀원 C 코드 원문(src/ext) 고정 {ep} epoch, val_truth 미사용")


if __name__ == "__main__":
    main()
