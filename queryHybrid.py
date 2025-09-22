# queryHybrid.py
import faiss
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import re

DB_PATH = "data/chunks.db"
FAISS_PATH = "data/faiss_index/index.faiss"
DB_IDS_PATH = "data/faiss_index/db_ids.npy"

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS...")
index = faiss.read_index(FAISS_PATH)
db_ids = np.load(DB_IDS_PATH)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

def clean_query(query: str) -> str:
    safe = query.replace('"', '""')
    safe = re.sub(r'[^\w\s]', ' ', safe)  # keep only words and spaces
    return safe.strip()

def faiss_scores(query, fetch_k=10):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    distances, indices = index.search(q_emb.astype('float32'), fetch_k)
    out = {}
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        db_id = int(db_ids[idx])
        score = float(distances[0][i])  # cosine similarity
        out[db_id] = score
    return out

def fts_scores(query, fetch_k=50):
    keyword_scores = {}
    safe_query = clean_query(query)

    try:
        sql = f'SELECT rowid, bm25(chunks_fts) as s FROM chunks_fts WHERE chunks_fts MATCH "{safe_query}" LIMIT {fetch_k}'
        cursor.execute(sql)
        rows = cursor.fetchall()

        if rows:
            s_vals = [r[1] for r in rows]
            max_s, min_s = max(s_vals), min(s_vals)
            for rowid, s in rows:
                keyword_scores[rowid] = (max_s - s) / (max_s - min_s + 1e-9)

    except Exception:
        sql = f'SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH "{safe_query}" LIMIT {fetch_k}'
        cursor.execute(sql)
        rows = cursor.fetchall()
        for i, (rowid,) in enumerate(rows):
            keyword_scores[rowid] = 1.0 / (i + 1)

    return keyword_scores

def hybrid_search(query, top_k=5, alpha=0.7, faiss_k=20):
    f_scores = faiss_scores(query, fetch_k=faiss_k)
    k_scores = fts_scores(query, fetch_k=faiss_k)

    # normalize faiss scores
    if f_scores:
        vals = np.array(list(f_scores.values()))
        minv, maxv = vals.min(), vals.max()
        for k in f_scores:
            if maxv - minv > 1e-9:
                f_scores[k] = (f_scores[k] - minv) / (maxv - minv)
            else:
                f_scores[k] = 1.0

    combined = {}
    keys = set(f_scores.keys()) | set(k_scores.keys())
    for db_id in keys:
        fs = f_scores.get(db_id, 0.0)
        ks = k_scores.get(db_id, 0.0)
        final = alpha * fs + (1.0 - alpha) * ks
        cursor.execute("SELECT file_name, title, url, chunk FROM chunks WHERE id = ?", (db_id,))
        row = cursor.fetchone()
        if row:
            combined[db_id] = {
                "db_id": db_id,
                "file": row[0],
                "title": row[1],
                "url": row[2],
                "chunk": row[3],
                "faiss": float(fs),
                "keyword": float(ks),
                "score": float(final)
            }

    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

if __name__ == "__main__":
    while True:
        q = input("\nQuery (or exit): ")
        if q.strip().lower() == "exit":
            break
        out = hybrid_search(q, top_k=5, alpha=0.7)
        for i, r in enumerate(out, 1):
            print("="*80)
            print(f"[{i}] {r['title']} ({r['file']}) score={r['score']:.4f}  faiss={r['faiss']:.4f} keyword={r['keyword']:.4f}")
            print(r['chunk'])
            print("="*80 + "\n")