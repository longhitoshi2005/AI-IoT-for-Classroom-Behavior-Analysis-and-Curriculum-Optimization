# embed_and_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load scraped text
with open("education_content.txt", "r", encoding='utf-8') as f:
    full_text = f.read()

# Chunking
chunk_size = 500
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and chunks
faiss.write_index(index, "education.index")
with open("education_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Saved education.index and education_chunks.pkl")
