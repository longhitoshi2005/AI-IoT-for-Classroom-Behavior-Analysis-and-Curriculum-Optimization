import os, pickle, json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_PATH  = os.getenv("FAISS_INDEX_PATH", "data/education.index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/education_chunks.pkl")


_model = None
_index = None
_chunks: List[str] = []


def load_assets():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if not _chunks:
        with open(CHUNKS_PATH, "rb") as f:
            _chunks[:] = pickle.load(f)


def _encode(texts: List[str]) -> np.ndarray:
    emb = _model.encode(texts)
    return np.asarray(emb, dtype="float32")


def retrieve(query: str, k: int = 3) -> str:
    load_assets()
    q = query.lower()
    vec = _encode([q])
    D, I = _index.search(vec, k)
    ctx = "\n---\n".join(_chunks[i] for i in I[0] if 0 <= i < len(_chunks))
    return ctx


def summarize_csv(paths: List[str]) -> str:
    out = []
    for p in paths or []:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
            cols = list(df.columns)
            rows = len(df)
            # ví dụ đặc thù: đếm nhãn hành vi phổ biến
            behavior_cols = [c for c in cols if c.lower() in {"handraise", "studying", "phone_usage"}]
            freq = {}
            for c in behavior_cols:
                freq[c] = int(df[c].sum()) if str(df[c].dtype).startswith("int") or str(df[c].dtype).startswith("float") else df[c].value_counts().to_dict()
            out.append(f"CSV:{os.path.basename(p)} | rows={rows} | tóm tắt={freq}")
        except Exception as e:
            out.append(f"CSV:{os.path.basename(p)} lỗi đọc: {e}")
    return "\n".join(out)


def summarize_json(paths: List[str]) -> str:
    out = []
    for p in paths or []:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # Bốc tách vài số liệu điển hình nếu có
            beh = obj.get("behavior_analysis", {})
            occ = beh.get("occurrence_counts", {})
            avg = beh.get("average_session_durations", {})
            out.append(f"JSON:{os.path.basename(p)} | occurrences={occ} | avg_durations={avg}")
        except Exception as e:
            out.append(f"JSON:{os.path.basename(p)} lỗi đọc: {e}")
    return "\n".join(out)


def build_context(query: str, k: int, csv_paths: List[str], json_paths: List[str]) -> str:
    base = retrieve(query, k=k)
    csv_part = summarize_csv(csv_paths)
    json_part = summarize_json(json_paths)
    parts = [p for p in [base, csv_part, json_part] if p]
    return "\n\n".join(parts)
