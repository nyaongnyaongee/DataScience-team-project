"""여러 모델의 top-100 결과를 합쳐 top-20을 만든다 (같은 split의 예측끼리만).

입력: results/<name>_<mode>_top100.csv.gz (user_id,item_id,score,rank)
방식:
  rrf    : Σ w_m / (c + rank_m)
  minmax : Σ w_m · (유저별 min-max 정규화 점수)
리스트에 없는 아이템은 그 모델에서 0점. top-100은 이미 본 아이템이 빠져 있다.

예) python src/blend.py --inputs content_fuse_1_0.5_1_k10_pop7 vbpr_fuse_..._e100_s0 \
        --weights 1 0.5 --method rrf --name blend_knn_vbpr
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C


def load_top(name: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(C.RESULTS / f"{name}_{mode}_top100.csv.gz")


def blend(tops: list[pd.DataFrame], weights: list[float], method: str, c: float = 60.0) -> pd.DataFrame:
    parts = []
    for df, w in zip(tops, weights):
        df = df[["user_id", "item_id", "score", "rank"]].copy()
        if method == "rrf":
            df["s"] = w / (c + df["rank"])
        elif method == "minmax":
            g = df.groupby("user_id")["score"]
            lo, hi = g.transform("min"), g.transform("max")
            df["s"] = w * (df["score"] - lo) / (hi - lo).replace(0, 1)
        else:
            raise ValueError(method)
        parts.append(df[["user_id", "item_id", "s"]])
    out = pd.concat(parts).groupby(["user_id", "item_id"], as_index=False)["s"].sum()
    out = out.sort_values(["user_id", "s", "item_id"], ascending=[True, False, True])
    out["rank"] = out.groupby("user_id").cumcount() + 1
    return out[out["rank"] <= 100]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--method", default="rrf", choices=["rrf", "minmax"])
    p.add_argument("--rrf-c", type=float, default=60.0)
    p.add_argument("--mode", default="val", choices=list(C.MODES))
    p.add_argument("--name", required=True)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    w = a.weights or [1.0] * len(a.inputs)
    assert len(w) == len(a.inputs)

    _, users = C.load_fit(a.mode)
    out = blend([load_top(n, a.mode) for n in a.inputs], w, a.method, a.rrf_c)
    counts = out.groupby("user_id").size()
    assert (counts.reindex(users).fillna(0) >= 20).all(), "어떤 유저는 후보가 20개 미만"
    out = out.set_index("user_id").loc[users].reset_index()
    n = 100 if (counts >= 100).all() else int(counts.min())
    out = out.groupby("user_id", sort=False).head(n)
    items = out["item_id"].to_numpy().reshape(len(users), n)
    scores = out["s"].to_numpy(np.float32).reshape(len(users), n)
    pred = C.save_outputs(a.name, a.mode, users, items, scores)
    if a.mode == "val":
        from evaluate import evaluate
        desc = f"{a.method} " + " + ".join(f"{x}*{n_}" for x, n_ in zip(w, a.inputs))
        evaluate(pred, a.name, extra_inputs="blend", notes=desc, log=not a.no_log)
    else:
        print(f"wrote {pred}")


if __name__ == "__main__":
    main()
