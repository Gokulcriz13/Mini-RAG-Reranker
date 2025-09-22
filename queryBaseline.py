# query_baseline.py
import faiss
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import json

DB_PATH = "data/chunks.db"
FAISS_PATH = "data/faiss_index/index.faiss"
DB_IDS_PATH = "data/faiss_index/db_ids.npy"
SOURCES_PATH = "data/sources.json"

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS...")
index = faiss.read_index(FAISS_PATH)
print("Index vectors:", index.ntotal)
db_ids = np.load(DB_IDS_PATH)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

def baseline_search(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    distances, indices = index.search(q_emb.astype('float32'), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        db_id = int(db_ids[idx])
        cursor.execute("SELECT file_name, title, url, chunk FROM chunks WHERE id = ?", (db_id,))
        row = cursor.fetchone()
        if not row:
            continue
        score = float(distances[0][i])  # inner product (cosine) between 0 and 1
        results.append({"db_id": db_id, "file": row[0], "title": row[1], "url": row[2], "chunk": row[3], "score": score})
    return results

if __name__ == "__main__":
    while True:
        q = input("\nQuery (or 'exit'): ")
        if q.strip().lower() == "exit":
            break
        res = baseline_search(q, top_k=5)
        print("\nBaseline (FAISS) top results:")
        for i,r in enumerate(res,1):
            print("="*80)
            print(f"[{i}] {r['title']} ({r['file']}) score={r['score']:.4f}")
            print(r['chunk'])
            print("="*80 + "\n")
