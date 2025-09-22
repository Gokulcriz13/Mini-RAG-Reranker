# Mini RAG + Reranker Sprint

## Overview
This project implements a small Question Answering (QA) service over a set of **20 public PDFs on industrial & machine safety**.  
It uses embeddings (all-MiniLM-L6-v2) + FAISS for baseline similarity search, then improves ranking with a **hybrid reranker** (vector + BM25 keyword).

## Features
- PDF ingestion + chunking → stored in SQLite
- Embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Baseline FAISS search (cosine similarity)
- Hybrid reranker (FAISS + SQLite FTS5 BM25)
- Flask API `/ask` for answering questions
- Abstain mechanism when results are too weak
- Answers are short, extractive, and cite their source

## Setup
```bash
# clone repo
git clone <your_repo_url>
cd <repo_name>

# create venv
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt

# run API
python api.py

# Easy example
curl -X POST http://127.0.0.1:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "What are the main industrial robot safety standards?", "k": 3, "mode": "hybrid"}'

# Tricky example
curl -X POST http://127.0.0.1:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "How does CE marking affect machine safety requirements?", "k": 5, "mode": "hybrid"}'
