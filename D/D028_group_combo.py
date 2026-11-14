"""D028 그룹별 가중치 조합. D/ 에서 실행: python D028_group_combo.py (gw_k*_f* val top-100 필요)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import pandas as pd, numpy as np, common as C
from evaluate import evaluate
df, users = C.load_fit("val")
n = df["user_id"].value_counts().reindex(users).fillna(0).to_numpy()
pick = {0: "gw_k1_f0.8", 1: "gw_k1.5_f1.3", 2: "gw_k1.5_f1.3"}   # 그룹별 최고 (원본 그룹 기준 recall로 고름)
g = np.where(n <= 10, 0, np.where(n <= 14, 1, 2))
print({k: int((g == k).sum()) for k in range(3)})
tops = {k: pd.read_csv(C.RESULTS / f"{v}_val_top100.csv.gz") for k, v in pick.items()}
gm = pd.Series(g, index=users)
parts = [t[t["user_id"].map(gm) == k] for k, t in tops.items()]
out = pd.concat(parts).set_index("user_id").loc[users].reset_index()
items = out["item_id"].to_numpy().reshape(len(users), 100); scores = out["score"].to_numpy(np.float32).reshape(len(users), 100)
pred = C.save_outputs("gw_combo", "val", users, items, scores)
evaluate(pred, "gw_combo", extra_inputs="blend", notes="group-wise weights by val_train count <=10:k1_f0.8 11-14:k1.5_f1.3 >=15:k1.5_f1.3")
