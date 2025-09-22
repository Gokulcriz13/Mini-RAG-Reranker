from flask import Flask, request, jsonify
from queryBaseline import baseline_search
from queryHybrid import hybrid_search
import os

app = Flask(__name__)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "queries.log")

def log_query(query, results, mode):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"MODE: {mode}\nQUERY: {query}\n")
        for r in results:
            f.write(f"  DB_ID: {r['db_id']}, SCORE: {r['score']}\n")
        f.write("\n")

def highlight_terms(chunk, query):
    """Highlight each query word in the chunk"""
    for term in query.split():
        if term:
            chunk = chunk.replace(term, f"<mark>{term}</mark>")
    return chunk

@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json() or {}
    q = payload.get("q", "").strip()
    k = int(payload.get("k", 5))
    mode = payload.get("mode", "hybrid")  # baseline|hybrid
    alpha = float(payload.get("alpha", 0.7))  # hybrid weight
    page = int(payload.get("page", 1))  # pagination

    if not q:
        return jsonify({"error": "empty query"}), 400

    # Perform search
    if mode == "baseline":
        results = baseline_search(q, top_k=k*page)
        reranker_used = "baseline"
        contexts = [
            {
                "db_id": r["db_id"],
                "file": r["file"],
                "title": r["title"],
                "url": r["url"],
                "chunk": highlight_terms(r["chunk"], q),
                "score": r["score"]
            }
            for r in results
        ]
    else:
        results = hybrid_search(q, top_k=k*page, alpha=alpha)
        reranker_used = "hybrid"
        contexts = [
            {
                "db_id": r["db_id"],
                "file": r["file"],
                "title": r["title"],
                "url": r["url"],
                "chunk": highlight_terms(r["chunk"], q),
                "score": r["score"],
                "faiss": r["faiss"],
                "keyword": r["keyword"]
            }
            for r in results
        ]

    # Pagination slicing
    start_idx = (page - 1) * k
    end_idx = start_idx + k
    contexts_page = contexts[start_idx:end_idx]

    # Short grounded answer with abstain + citation
    answer = None
    if contexts_page and contexts_page[0]["score"] > 0.3:  # ✅ threshold
        top = contexts_page[0]
        clean_chunk = top["chunk"].replace("\n", " ").strip()
        answer = f"{clean_chunk[:250]} (Source: {top['title']})"

    # Log the query
    log_query(q, contexts_page, mode)

    return jsonify({
        "answer": answer,
        "contexts": contexts_page,
        "reranker_used": reranker_used,
        "page": page,
        "k": k,
        "alpha": alpha
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
