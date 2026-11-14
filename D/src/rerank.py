"""2단계 reranker: 기본 모델들의 top-100 후보 → 피처 → LightGBM LambdaRank → top-20.

데이터 규칙:
- 학습: inner 모드 (data/inner, teamkit split --train validation/val_train.csv 로 만든 내부 split).
  기본 모델도 data/inner/val_train.csv 로 학습한 결과(results/*_inner_top100.csv.gz)를 쓰고,
  라벨은 data/inner/val_truth.csv. validation/val_truth.csv 는 학습에 쓰지 않는다.
- 평가: val 모드 후보·피처에 학습된 모델을 적용 → evaluate.py(키트 score)로 채점만.
- 최종: final 모드 후보·피처에 같은 모델 적용 (재학습 없음).

예) python src/rerank.py --models content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 \
        --apply val --name rerank_v1
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common as C
import features as F
import timepop as TP

DAY_MS = 86_400_000
MISSING_RANK = 101


def keys(u: np.ndarray, i: np.ndarray) -> np.ndarray:
    return u.astype(np.int64) * C.N_ITEMS + i.astype(np.int64)


def lookup(sorted_keys: np.ndarray, values: np.ndarray, q: np.ndarray, fill: float) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, q).clip(max=len(sorted_keys) - 1)
    hit = sorted_keys[pos] == q
    out = np.full(len(q), fill, np.float32)
    out[hit] = values[pos[hit]]
    return out


class Static:
    """모드와 무관한 아이템 메타 피처."""

    def __init__(self):
        meta = F.load_meta()
        self.price = pd.to_numeric(meta["price"], errors="coerce").to_numpy(np.float32)
        store = meta["store"].fillna("").astype(str).str.strip().str.lower()
        self.store = pd.factorize(store)[0]
        cats = meta["categories"].apply(lambda c: [x for x in (c or []) if x != "Clothing, Shoes & Jewelry"])
        self.cat1 = pd.factorize(cats.apply(lambda c: ">".join(c[:1])))[0]
        self.cat2 = pd.factorize(cats.apply(lambda c: ">".join(c[:2])))[0]
        self.leaf = pd.factorize(cats.apply(lambda c: ">".join(c)))[0]
        dev = C.DEVICE
        mini = F.sbert_matrix(meta, "sentence-transformers/all-MiniLM-L6-v2", fields=("title", "store", "categories"))
        self.emb = {
            "mini": torch.nn.functional.normalize(torch.as_tensor(mini, dtype=torch.float32, device=dev), dim=1),
            "clipimg": torch.nn.functional.normalize(
                torch.as_tensor(F.clip_matrix(meta, "image"), dtype=torch.float32, device=dev), dim=1),
            "cliptxt": torch.nn.functional.normalize(
                torch.as_tensor(F.clip_matrix(meta, "text"), dtype=torch.float32, device=dev), dim=1),
        }


def group_match(X: sp.csr_matrix, codes: np.ndarray, u: np.ndarray, i: np.ndarray) -> np.ndarray:
    """유저 이력 중 후보와 같은 그룹(store/카테고리)인 비율."""
    n_g = codes.max() + 1
    G = sp.csr_matrix((np.ones(C.N_ITEMS, np.float32), (np.arange(C.N_ITEMS), codes)), shape=(C.N_ITEMS, n_g))
    UG = (X @ G).tocoo()
    k = UG.row.astype(np.int64) * n_g + UG.col
    order = np.argsort(k)
    cnt = lookup(k[order], UG.data[order].astype(np.float32), u.astype(np.int64) * n_g + codes[i], 0.0)
    n_u = np.asarray(X.sum(1)).ravel().astype(np.float32)
    return cnt / np.maximum(n_u[u], 1)


def build(mode: str, models: list[str], topn: int, st: Static, robust: bool = False, extra: bool = False) -> pd.DataFrame:
    t = time.time()
    df, users = C.load_fit(mode)
    X = C.interaction_matrix(df)
    # 1) 후보와 모델별 rank/정규화 score
    tops = {}
    cand = []
    for m in models:
        top = pd.read_csv(C.RESULTS / f"{m}_{mode}_top100.csv.gz")
        top["ns"] = top["score"] / top.groupby("user_id")["score"].transform("max").abs().clip(lower=1e-12)
        k = keys(top["user_id"].to_numpy(), top["item_id"].to_numpy())
        o = np.argsort(k)
        tops[m] = (k[o], top["rank"].to_numpy(np.float32)[o], top["ns"].to_numpy(np.float32)[o])
        cand.append(k[top["rank"].to_numpy() <= topn])
    ck = np.unique(np.concatenate(cand))
    u = (ck // C.N_ITEMS).astype(np.int64)
    i = (ck % C.N_ITEMS).astype(np.int64)
    out = {"user_id": u, "item_id": i}
    ranks = []
    for m, (k, r, s) in tops.items():
        out[f"r_{m}"] = lookup(k, r, ck, MISSING_RANK)
        out[f"s_{m}"] = lookup(k, s, ck, 0.0)
        ranks.append(out[f"r_{m}"])
    R = np.stack(ranks, 1)
    out["r_min"] = R.min(1)
    out["r_mean"] = R.mean(1)
    out["n_in20"] = (R <= 20).sum(1).astype(np.float32)
    out["rrf"] = (1.0 / (60 + R)).sum(1)

    # 2) 아이템 인기·시간 (학습 데이터 기준)
    n_fit = len(df)
    pop = np.bincount(df["item_id"], minlength=C.N_ITEMS).astype(np.float32)
    out["pop_share"] = pop[i] / n_fit * 1e4
    out["pop_pct"] = (pd.Series(pop).rank(pct=True).to_numpy(np.float32))[i]
    rat = df.groupby("item_id")["rating"].agg(["mean", "count"]).reindex(range(C.N_ITEMS))
    out["item_rating"] = rat["mean"].to_numpy(np.float32)[i]
    tday = df["timestamp"].to_numpy() / DAY_MS
    it_t = pd.Series(tday).groupby(df["item_id"].to_numpy()).agg(["min", "median", "max"]).reindex(range(C.N_ITEMS))
    us_t = pd.Series(tday).groupby(df["user_id"].to_numpy()).agg(["min", "median", "max"]).reindex(range(C.N_USERS))
    out["dt_median"] = np.abs(us_t["median"].to_numpy()[u] - it_t["median"].to_numpy()[i]).astype(np.float32)
    out["item_first_minus_user_last"] = (it_t["min"].to_numpy()[i] - us_t["max"].to_numpy()[u]).astype(np.float32)
    out["user_last_minus_item_last"] = (us_t["max"].to_numpy()[u] - it_t["max"].to_numpy()[i]).astype(np.float32)
    span = (us_t["max"] - us_t["min"]).to_numpy(np.float32)
    n_u = np.asarray(X.sum(1)).ravel().astype(np.float32)
    if robust:
        # 학습 데이터 크기에 따라 커지는 개수형 피처를 fit 안의 상대값으로 바꾼다
        active = n_u > 0
        pct = np.zeros(C.N_USERS, np.float32)
        pct[active] = pd.Series(n_u[active]).rank(pct=True).to_numpy(np.float32)
        out["user_n_pct"] = pct[u]
        # 활동 기간도 fit 안 백분위로 (전체 기간으로 나누면 세 모드의 분모가 같아 효과가 없음)
        sp_pct = np.zeros(C.N_USERS, np.float32)
        sp_pct[active] = pd.Series(span[active]).rank(pct=True).to_numpy(np.float32)
        out["user_span_pct"] = sp_pct[u]
    else:
        out["user_span"] = span[u]
        out["user_n"] = n_u[u]

    # 3) 시간 인기도 (timepop σ=8)
    W, P = TP.build_scores(df, 30, 8.0, "gauss", 0, 0.0)
    W = W / W.sum(1, keepdim=True).clamp(min=1e-12)
    P = P / P.sum(1, keepdim=True).clamp(min=1)  # bin 안에서의 점유율
    tp = np.empty(len(ck), np.float32)
    dev = C.DEVICE
    for s in range(0, len(ck), 200_000):
        uu = torch.as_tensor(u[s : s + 200_000], device=dev)
        ii = torch.as_tensor(i[s : s + 200_000], device=dev)
        tp[s : s + 200_000] = (W[uu] * P[:, ii].T).sum(1).cpu().numpy() * 1e4
    out["timepop"] = tp
    del W, P

    # 4) 콘텐츠: 유저 프로필(이력 평균 임베딩)과의 cosine
    Xt = torch.sparse_csr_tensor(torch.as_tensor(X.indptr, dtype=torch.int64), torch.as_tensor(X.indices, dtype=torch.int64),
                                 torch.as_tensor(X.data), size=X.shape).to(dev)
    for name, E in st.emb.items():
        prof = torch.nn.functional.normalize(Xt @ E, dim=1)
        v = np.empty(len(ck), np.float32)
        for s in range(0, len(ck), 500_000):
            uu = torch.as_tensor(u[s : s + 500_000], device=dev)
            ii = torch.as_tensor(i[s : s + 500_000], device=dev)
            v[s : s + 500_000] = (prof[uu] * E[ii]).sum(1).cpu().numpy()
        out[f"cos_{name}"] = v

    if extra:
        for name, v in hist_sims(df, u, i, st, ("mini", "clipimg")).items():
            out[name] = v

    # 5) 메타: store/카테고리 일치 비율, 가격
    out["m_store"] = group_match(X, st.store, u, i)
    out["m_cat1"] = group_match(X, st.cat1, u, i)
    out["m_cat2"] = group_match(X, st.cat2, u, i)
    out["m_leaf"] = group_match(X, st.leaf, u, i)
    out["price"] = st.price[i]
    feats = pd.DataFrame(out)
    print(f"[{mode}] candidates {len(feats):,} ({len(feats) / len(users):.1f}/user) in {time.time() - t:.0f}s", flush=True)
    return feats


def hist_sims(df, u, i, st: Static, names, L: int = 64, recent: int = 5, B: int = 512) -> dict:
    """후보와 유저 이력(최근 L개) 사이 cosine의 최대값, 최근 recent개 평균. u는 정렬돼 있어야 함."""
    dev = C.DEVICE
    d = df.sort_values(["user_id", "timestamp"], ascending=[True, False])
    d = d[d.groupby("user_id").cumcount() < L]
    pos = d.groupby("user_id").cumcount().to_numpy()
    H = torch.full((C.N_USERS, L), -1, dtype=torch.int64)
    H[torch.as_tensor(d["user_id"].to_numpy()), torch.as_tensor(pos)] = torch.as_tensor(d["item_id"].to_numpy())
    H = H.to(dev)
    out = {f"{k}_{n}": np.zeros(len(u), np.float32) for n in names for k in ("maxcos", "recentcos")}
    starts = np.searchsorted(u, np.arange(C.N_USERS + 1))
    users = np.unique(u)
    for b0 in range(0, len(users), B):
        ub = users[b0 : b0 + B]
        lo, hi = starts[ub], starts[ub + 1]
        cnt = hi - lo
        cmax = int(cnt.max())
        rows = np.concatenate([np.arange(a, z) for a, z in zip(lo, hi)])
        col = np.concatenate([np.arange(c) for c in cnt])
        brow = np.repeat(np.arange(len(ub)), cnt)
        ci = torch.zeros((len(ub), cmax), dtype=torch.int64, device=dev)
        ci[torch.as_tensor(brow, device=dev), torch.as_tensor(col, device=dev)] = torch.as_tensor(i[rows], device=dev)
        h = H[torch.as_tensor(ub, device=dev)]
        valid = h >= 0
        hc = h.clamp(min=0)
        rmask = valid & (torch.arange(L, device=dev)[None, :] < recent)
        for n in names:
            E = st.emb[n]
            S = torch.bmm(E[ci], E[hc].transpose(1, 2))  # [b, cmax, L]
            mx = S.masked_fill(~valid[:, None, :], -2).max(2).values
            rc = (S * rmask[:, None, :]).sum(2) / rmask.sum(1, keepdim=True).clamp(min=1)
            sel = (torch.as_tensor(brow, device=dev), torch.as_tensor(col, device=dev))
            out[f"maxcos_{n}"][rows] = mx[sel].cpu().numpy()
            out[f"recentcos_{n}"][rows] = rc[sel].cpu().numpy()
    return out


def load_truth_inner() -> np.ndarray:
    tr = pd.read_csv(C.DATA / "inner/val_truth.csv")
    return np.sort(keys(tr["user_id"].to_numpy(), tr["item_id"].to_numpy()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--topn", type=int, default=50, help="모델별 후보 수")
    ap.add_argument("--apply", nargs="+", default=["val"], choices=["val", "final"], help="같은 학습 모델을 적용할 모드들")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--min-leaf", type=int, default=100)
    ap.add_argument("--objective", default="lambdarank", choices=["lambdarank", "binary"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="여러 개면 예측을 평균")
    ap.add_argument("--robust", action="store_true", help="개수형 피처를 fit 내 상대값으로")
    ap.add_argument("--extra", action="store_true", help="이력 최대·최근 cosine 피처 추가")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    st = Static()
    tr = build("inner", a.models, a.topn, st, a.robust, a.extra)
    truth = load_truth_inner()
    ck = keys(tr["user_id"].to_numpy(), tr["item_id"].to_numpy())
    pos = np.searchsorted(truth, ck).clip(max=len(truth) - 1)
    tr["label"] = (truth[pos] == ck).astype(np.int8)
    print(f"[inner] candidate recall (truth covered) {tr['label'].sum() / len(truth):.4f}", flush=True)
    # 양성 후보가 있는 유저만 학습 (없으면 lambdarank gradient 0)
    has = tr.groupby("user_id")["label"].transform("max") > 0
    tr = tr[has]
    tr_users = tr["user_id"].unique()
    fcols = [c for c in tr.columns if c not in ("user_id", "item_id", "label")]

    def ds(d):
        return lgb.Dataset(d[fcols], d["label"], group=d.groupby("user_id", sort=False).size().to_numpy())

    models = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        va_users = set(rng.choice(tr_users, size=len(tr_users) // 10, replace=False).tolist())
        is_va = tr["user_id"].isin(va_users).to_numpy()
        params = dict(objective=a.objective, metric="ndcg", eval_at=[20], learning_rate=a.lr, num_leaves=a.leaves,
                      min_data_in_leaf=a.min_leaf, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambdarank_truncation_level=40, seed=seed, verbose=-1)
        model = lgb.train(params, ds(tr[~is_va]), a.rounds, valid_sets=[ds(tr[is_va])],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
        imp = pd.Series(model.feature_importance("gain"), index=fcols).sort_values(ascending=False)
        print(f"[seed {seed}] best_iter {model.best_iteration} top features:",
              ", ".join(f"{k}={v:.0f}" for k, v in imp.head(15).items()), flush=True)
        model.save_model(str(C.RESULTS / (f"{a.name}_lgb.txt" if len(a.seeds) == 1 else f"{a.name}_s{seed}_lgb.txt")))
        models.append(model)
    del tr

    desc = (f"{a.objective} lr{a.lr} leaves{a.leaves} minleaf{a.min_leaf} topn{a.topn}"
            f"{' robust' if a.robust else ''}{' extra' if a.extra else ''} seeds {'/'.join(map(str, a.seeds))}"
            f" iters {'/'.join(str(m.best_iteration) for m in models)} models " + "+".join(a.models))

    def write(te, name, mode, users, seed_tag):
        te = te.sort_values(["user_id", "pred", "item_id"], ascending=[True, False, True])
        te = te.set_index("user_id").loc[users].reset_index()
        te = te.groupby("user_id", sort=False).head(100)
        n = int(te.groupby("user_id").size().min())
        te = te.groupby("user_id", sort=False).head(n)
        items = te["item_id"].to_numpy().reshape(len(users), n)
        scores = te["pred"].to_numpy(np.float32).reshape(len(users), n)
        pred = C.save_outputs(name, mode, users, items, scores)
        if mode == "val":
            from evaluate import evaluate
            evaluate(pred, name, seed=seed_tag, extra_inputs="rerank(inner split)", notes=desc, log=not a.no_log)
        else:
            print(f"wrote {pred}")

    for mode in a.apply:
        te = build(mode, a.models, a.topn, st, a.robust, a.extra)
        _, users = C.load_fit(mode)
        preds = [m.predict(te[fcols], num_iteration=m.best_iteration) for m in models]
        if len(models) > 1 and mode == "val":
            for seed, p in zip(a.seeds, preds):
                write(te.assign(pred=p), f"{a.name}_s{seed}", mode, users, str(seed))
        te["pred"] = np.mean(preds, 0)
        write(te, a.name, mode, users, "/".join(map(str, a.seeds)))

if __name__ == "__main__":
    main()
