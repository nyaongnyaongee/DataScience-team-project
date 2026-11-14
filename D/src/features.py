"""item_meta.jsonl → 아이템 특성 행렬 (행 순서 = item_id 0..N_ITEMS-1)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from common import DATA, N_ITEMS


def load_meta() -> pd.DataFrame:
    rows = [json.loads(l) for l in open(DATA / "item_meta.jsonl", encoding="utf-8")]
    meta = pd.DataFrame(rows).set_index("item_id").sort_index()
    assert len(meta) == N_ITEMS and (meta.index == np.arange(N_ITEMS)).all()
    return meta


def _tfidf_onehot(tokens: list[list[str]]) -> sp.csr_matrix:
    mlb = MultiLabelBinarizer(sparse_output=True)
    M = mlb.fit_transform(tokens).astype(np.float32)
    return TfidfTransformer(sublinear_tf=True).fit_transform(M).tocsr()


def metadata_blocks(meta: pd.DataFrame) -> dict[str, sp.csr_matrix]:
    """카테고리 경로 prefix / 브랜드 / 가격 구간 / Department 블록, 각각 L2 정규화."""
    def cat_prefixes(cats):
        cats = [c for c in (cats or []) if c != "Clothing, Shoes & Jewelry"]
        return [">".join(cats[: i + 1]) for i in range(len(cats))] or ["<none>"]

    cat = _tfidf_onehot([cat_prefixes(c) for c in meta["categories"]])
    store = _tfidf_onehot([[str(s).strip().lower() or "<none>"] for s in meta["store"]])

    price = pd.to_numeric(meta["price"], errors="coerce")
    bins = pd.qcut(np.log1p(price), 10, labels=False, duplicates="drop")
    # 이웃 구간도 약하게 겹치도록 bin과 bin±1을 함께 켠다
    price_tok = [["p_na"] if pd.isna(b) else [f"p{int(b)}", f"p{int(b)}", f"p{int(b) - 1}", f"p{int(b) + 1}"]
                 for b in bins]
    price_m = normalize(sp.csr_matrix(MultiLabelBinarizer(sparse_output=True).fit_transform(price_tok).astype(np.float32)))

    dept = _tfidf_onehot([[str((d or {}).get("Department", "<none>")).lower()] for d in meta["details"]])
    return {"cat": normalize(cat), "store": normalize(store), "price": price_m, "dept": normalize(dept)}


def metadata_matrix(meta: pd.DataFrame, w: dict[str, float]) -> sp.csr_matrix:
    blocks = metadata_blocks(meta)
    M = sp.hstack([blocks[k] * np.sqrt(v) for k, v in w.items() if v > 0]).tocsr()
    return normalize(M)


def _join(x) -> str:
    if isinstance(x, list):
        return " ".join(map(str, x))
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def item_text(meta: pd.DataFrame, fields=("title", "store", "categories", "features", "description")) -> list[str]:
    parts = []
    for _, r in meta.iterrows():
        t = " . ".join(_join(r[f]) for f in fields if _join(r[f]))
        parts.append(re.sub(r"\s+", " ", t))
    return parts


def tfidf_matrix(meta: pd.DataFrame, fields=("title", "store", "categories", "features", "description"),
                 ngram=(1, 2), min_df=2, max_features=300_000) -> sp.csr_matrix:
    vec = TfidfVectorizer(sublinear_tf=True, min_df=min_df, max_df=0.5, ngram_range=ngram,
                          max_features=max_features, stop_words="english", dtype=np.float32)
    return vec.fit_transform(item_text(meta, fields)).tocsr()


def sbert_matrix(meta: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 fields=("title", "store", "categories", "features", "description"),
                 max_seq_length: int = 256, batch_size: int = 256) -> np.ndarray:
    """Sentence embedding (L2 정규화). data/cache/에 캐시."""
    from common import CACHE
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = model_name.split("/")[-1] + "_" + "-".join(f[:3] for f in fields) + f"_{max_seq_length}"
    path = CACHE / f"sbert_{tag}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cuda")
    model.max_seq_length = max_seq_length
    emb = model.encode(item_text(meta, fields), batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(path, emb)
    return emb


def clip_matrix(meta: pd.DataFrame, kind: str = "image", model_name: str = "openai/clip-vit-large-patch14",
                batch_size: int = 128) -> np.ndarray:
    """CLIP 이미지/텍스트 임베딩 (L2 정규화). 이미지가 없는 아이템은 0 벡터. data/cache/에 캐시."""
    from common import CACHE
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"clip_{kind}_{model_name.split('/')[-1]}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    dim = model.config.projection_dim
    emb = np.zeros((len(meta), dim), np.float32)

    if kind == "text":
        texts = item_text(meta, ("title", "store", "categories"))
        for s in range(0, len(texts), batch_size):
            inp = proc.tokenizer(texts[s : s + batch_size], padding=True, truncation=True,
                                 max_length=77, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.get_text_features(**inp)
                out = getattr(out, "pooler_output", out)
            emb[s : s + batch_size] = torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy()
    elif kind == "image":
        from concurrent.futures import ThreadPoolExecutor
        img_dir = DATA / "images"

        def load(i):
            p = img_dir / f"{i}.jpg"
            try:
                return i, Image.open(p).convert("RGB")
            except Exception:
                return i, None

        ids = list(range(len(meta)))
        with ThreadPoolExecutor(16) as ex:
            for s in range(0, len(ids), batch_size):
                pairs = [(i, im) for i, im in ex.map(load, ids[s : s + batch_size]) if im is not None]
                if not pairs:
                    continue
                idx, ims = zip(*pairs)
                px = proc.image_processor(list(ims), return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
                with torch.no_grad():
                    out = model.get_image_features(pixel_values=px)
                    out = getattr(out, "pooler_output", out)
                emb[list(idx)] = torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy()
    else:
        raise ValueError(kind)
    np.save(path, emb)
    return emb


def fuse(*parts_and_weights) -> np.ndarray:
    """(emb, w) 쌍들을 sqrt(w)로 가중해 이어 붙인 뒤 L2 정규화 → cosine = 가중 평균 cosine."""
    M = np.hstack([e * np.sqrt(w) for e, w in parts_and_weights if w > 0])
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)
