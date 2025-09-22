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

## Results: Baseline vs Hybrid

| Question                                                             | Baseline Top Doc                                                                 | Baseline Score | Hybrid Top Doc                                                                 | Hybrid Score |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------|--------------|
| What are the main industrial robot safety standards?                 | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper          | 0.7783         | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper         | 0.7000       |
| Explain the difference between collaborative and traditional robots. | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper          | 0.5262         | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper         | 0.7000       |
| What sensors are typically used for robot safety?                    | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper          | 0.6700         | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper         | 0.7000       |
| How does ISO 10218 regulate robot safety?                            | 06 - Todd Dickey - IRSC 2022 (Intro to Industrial Robot Safety ISO 10218 Parts)  | 0.7011         | 06 - Todd Dickey - IRSC 2022 (Intro to Industrial Robot Safety ISO 10218 Parts) | 0.7000       |
| What are the main hazards in industrial robotics?                    | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper          | 0.7160         | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper         | 0.7000       |
| How do safety-rated monitored stops work?                            | special_information_guide_for_safe_machinery_en_im0014678                        | 0.6701         | special_information_guide_for_safe_machinery_en_im0014678                       | 0.7000       |
| Explain the concept of power and force limiting in cobots.           | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper          | 0.3980         | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper         | 0.7000       |
| What is the role of light curtains in robot safety?                  | Machine-Safety-Brochure-Guide                                                    | 0.6761         | Machine-Safety-Brochure-Guide                                                   | 0.7000       |

