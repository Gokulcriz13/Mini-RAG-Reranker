# ingest.py
import os
import fitz  # PyMuPDF
import sqlite3
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

# Paths
PDF_DIR = "data/pdfs"
DB_PATH = "data/chunks.db"
FAISS_DIR = "data/faiss_index"
FAISS_PATH = os.path.join(FAISS_DIR, "index.faiss")
EMB_PATH = "data/embeddings.npy"
SOURCES_PATH = "data/sources.json"
DB_IDS_PATH = os.path.join(FAISS_DIR, "db_ids.npy")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Reset DB
if os.path.exists(DB_PATH):
    print("Remove old DB")
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Main table
cursor.execute("""
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    title TEXT,
    url TEXT,
    chunk TEXT
)
""")

# FTS5 table for keyword retrieval
cursor.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk, content='chunks', content_rowid='id')")

all_chunks = []
sources = []
file_list = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

if not file_list:
    print("No PDFs found in", PDF_DIR)
else:
    for file in file_list:
        path = os.path.join(PDF_DIR, file)
        print("Processing", file)
        text = extract_text(path)
        chunks = chunk_text(text)
        title = os.path.splitext(file)[0]
        url = f"file://{os.path.abspath(path)}"
        sources.append({"title": title, "url": url, "file": file})
        for chunk in chunks:
            cursor.execute("INSERT INTO chunks (file_name, title, url, chunk) VALUES (?, ?, ?, ?)",
                           (file, title, url, chunk))
            all_chunks.append(chunk)

    conn.commit()

    # populate FTS5 from chunks table
    cursor.execute("INSERT INTO chunks_fts(rowid, chunk) SELECT id, chunk FROM chunks")
    conn.commit()

    # save db_ids (FAISS index -> DB id mapping)
    cursor.execute("SELECT id FROM chunks ORDER BY id ASC")
    db_ids = np.array([row[0] for row in cursor.fetchall()], dtype=np.int64)
    np.save(DB_IDS_PATH, db_ids)
    print("Saved DB ID mapping to", DB_IDS_PATH)

    # save sources
    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        json.dump(sources, fh, ensure_ascii=False, indent=2)
    print("Saved sources.json")

# embeddings + FAISS
if all_chunks:
    print("Generating embeddings for", len(all_chunks), "chunks...")
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    np.save(EMB_PATH, embeddings)
    print("Saved embeddings to", EMB_PATH)

    # Normalize for cosine similarity (recommended)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product for cosine (vectors are normalized)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, FAISS_PATH)
    print("Saved FAISS index to", FAISS_PATH)
else:
    print("No chunks -> skipped embeddings + FAISS")

conn.close()
print("Done")
