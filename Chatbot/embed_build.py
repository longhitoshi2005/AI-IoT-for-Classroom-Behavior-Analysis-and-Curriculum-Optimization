import os, pickle
import faiss, numpy as np
from sentence_transformers import SentenceTransformer


SRC = os.getenv("SRC_TEXT", "data/education_content.txt")
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/education.index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/education_chunks.pkl")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


text = open(SRC, "r", encoding="utf-8").read()
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
model = SentenceTransformer(EMBED_MODEL)
emb = model.encode(chunks)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(np.asarray(emb, dtype="float32"))
faiss.write_index(index, INDEX_PATH)
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)
print("Saved:", INDEX_PATH, CHUNKS_PATH)
