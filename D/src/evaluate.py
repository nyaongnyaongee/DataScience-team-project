"""검증 채점: 키트의 score 명령 + 활동량 그룹별 Recall@20 + experiments.csv 기록.

val_truth.csv를 읽는 곳은 이 파일뿐이다. 점수는 키트의 score 명령과
키트의 recall_at_20 함수로만 계산한다.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from common import DATA, KIT, ROOT, SPLIT_ID

sys.path.insert(0, str(KIT))
import teamkit  # noqa: E402

VAL_TRAIN = KIT / "validation/val_train.csv"
VAL_TRUTH = KIT / "validation/val_truth.csv"
EXPERIMENTS = ROOT / "experiments.csv"
GROUPS = [("<=13", 0, 13), ("14-18", 14, 18), (">=19", 19, 10**9)]


def kit_score(pred: Path) -> dict:
    out = subprocess.run(
        [sys.executable, str(KIT / "teamkit.py"), "score", "--pred", str(pred),
         "--fit", str(VAL_TRAIN), "--truth", str(VAL_TRUTH)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"teamkit score failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout)


def group_recall(pred: Path) -> dict:
    counts = pd.read_csv(DATA / "train.csv", usecols=["user_id"])["user_id"].value_counts()
    preds = teamkit.read_predictions(pred)
    truth = teamkit.read_truth(VAL_TRUTH)
    res = {}
    for name, lo, hi in GROUPS:
        us = {str(u) for u, c in counts.items() if lo <= c <= hi}
        r = teamkit.recall_at_20({u: p for u, p in preds.items() if u in us},
                                 {u: t for u, t in truth.items() if u in us})
        res[name] = round(r["recall_at_20"], 5)
    return res


def evaluate(pred: Path, model: str, extra_inputs: str = "none", seed: str = "NA",
             notes: str = "", log: bool = True) -> dict:
    s = kit_score(pred)
    g = group_recall(pred)
    print(f"[{model}] recall@20={s['recall_at_20']:.5f} hits={s['hits']} groups={g}", flush=True)
    if log:
        with open(EXPERIMENTS, "a", newline="") as f:
            csv.writer(f).writerow([
                SPLIT_ID, model, seed, "validation/val_train.csv", extra_inputs,
                s["recall_at_20"], s["hits"], s["denominator"],
                (notes + " " if notes else "") + "groups " + " ".join(f"{k}:{v}" for k, v in g.items()),
            ])
    return {**s, "groups": g}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    evaluate(Path(a.pred), a.model, log=not a.no_log)
