"""콘텐츠 기반 item-kNN: 아이템 특성 cosine 유사도 → 유저가 본 아이템의 이웃 점수 합.

예) python src/run_content.py --features meta --k 20 50 100 200 500
    python src/run_content.py --features tfidf --k 50 100 200
    python src/run_content.py --features pop          # 파이프라인 확인용 (인기순)
"""

from __future__ import annotations

import argparse
import time

import torch

import common as C
import features as F


def build_features(name: str, meta):
    if name == "meta":
        return F.metadata_matrix(meta, {"cat": 1.0, "store": 1.0, "price": 0.5, "dept": 0.5})
    if name == "meta_cat":
        return F.metadata_matrix(meta, {"cat": 1.0})
    if name == "meta_store":
        return F.metadata_matrix(meta, {"store": 1.0})
    if name == "tfidf":
        return F.tfidf_matrix(meta)
    if name == "tfidf_title":
        return F.tfidf_matrix(meta, fields=("title",))
    if name == "minilm":
        return F.sbert_matrix(meta, "sentence-transformers/all-MiniLM-L6-v2")
    if name == "minilm_title":
        return F.sbert_matrix(meta, "sentence-transformers/all-MiniLM-L6-v2", fields=("title", "store", "categories"))
    if name == "mpnet":
        return F.sbert_matrix(meta, "sentence-transformers/all-mpnet-base-v2", batch_size=128)
    if name == "bge":
        return F.sbert_matrix(meta, "BAAI/bge-base-en-v1.5", max_seq_length=512, batch_size=64)
    if name == "clip_img":
        return F.clip_matrix(meta, "image")
    if name == "clip_txt":
        return F.clip_matrix(meta, "text")
    if name.startswith("fuse"):  # fuse_<w_minilm>_<w_clipimg>_<w_cliptxt>
        w = [float(x) for x in name.split("_")[1:]]
        mini = F.sbert_matrix(meta, "sentence-transformers/all-MiniLM-L6-v2", fields=("title", "store", "categories"))
        return F.fuse((mini, w[0]), (F.clip_matrix(meta, "image"), w[1]), (F.clip_matrix(meta, "text"), w[2]))
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[100])
    p.add_argument("--weight", default="binary", choices=["binary", "rating"])
    p.add_argument("--mode", default="val", choices=list(C.MODES))
    p.add_argument("--pop-beta", type=float, nargs="+", default=[1e-6])
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()

    df, users = C.load_fit(a.mode)
    X = C.interaction_matrix(df, a.weight)
    pop = C.popularity(df)

    if a.features == "pop":
        S_list = [(None, torch.zeros((C.N_ITEMS, C.N_ITEMS), device=C.DEVICE))]
    else:
        t = time.time()
        E = build_features(a.features, F.load_meta())
        print(f"features {a.features}: shape {E.shape}, {time.time() - t:.1f}s", flush=True)
        full = C.cosine_item_sim(E, None)
        S_list = ((k, C.topk_sparsify(full.clone(), k)) for k in a.k)

    for k, S in S_list:
        for beta in a.pop_beta:
            name = (f"content_{a.features}" + (f"_k{k}" if k else "")
                    + (f"_pop{beta:g}" if beta > 1e-6 else "")
                    + (f"_{a.weight}" if a.weight != "binary" else ""))
            items, scores = C.recommend_item_item(X, S, users, pop, pop_beta=beta)
            pred = C.save_outputs(name, a.mode, users, items, scores)
            if a.mode == "val":
                from evaluate import evaluate
                evaluate(pred, name, extra_inputs="item_meta.jsonl" if a.features != "pop" else "none",
                         log=not a.no_log)
            else:
                print(f"wrote {pred}")
        del S
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
