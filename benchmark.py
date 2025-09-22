# benchmark_compare.py
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8080/ask"  # Change if deployed
QUESTIONS = [
    "What are the main industrial robot safety standards?",
    "Explain the difference between collaborative and traditional robots.",
    "What sensors are typically used for robot safety?",
    "How does ISO 10218 regulate robot safety?",
    "What are the main hazards in industrial robotics?",
    "How do safety-rated monitored stops work?",
    "Explain the concept of power and force limiting in cobots.",
    "What is the role of light curtains in robot safety?"
]

def query_api(q, mode="baseline", k=3):
    """Send a POST request to our API and return contexts"""
    payload = {"q": q, "k": k, "mode": mode}
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        return resp.json().get("contexts", [])
    else:
        print(f"Error {resp.status_code}: {resp.text}")
        return []

def benchmark():
    rows = []
    for q in QUESTIONS:
        baseline_results = query_api(q, mode="baseline")
        hybrid_results = query_api(q, mode="hybrid")

        # Record top-1 results for compact README table
        baseline_top = baseline_results[0] if baseline_results else {}
        hybrid_top = hybrid_results[0] if hybrid_results else {}

        rows.append({
            "Question": q,
            "Baseline Top": f"{baseline_top.get('title','-')} ({baseline_top.get('score','-')})",
            "Hybrid Top": f"{hybrid_top.get('title','-')} ({hybrid_top.get('score','-')})"
        })

    df = pd.DataFrame(rows)
    # Save as markdown for README
    with open("results_table.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print("✅ Results written to results_table.md")

if __name__ == "__main__":
    benchmark()
