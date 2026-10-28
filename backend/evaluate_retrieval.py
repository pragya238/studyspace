"""Small deterministic retrieval benchmark for the bundled Studyspace sample corpus.

Run after loading the sample course and starting the API:
    python evaluate_retrieval.py
The benchmark reports hit-rate@k and reciprocal rank on a tiny labeled query set.
"""
import json
import urllib.error
import urllib.request

API = "http://localhost:8000"
CASES = [
    {"query": "Where does glycolysis happen?", "expected": "cytoplasm"},
    {"query": "What does oxygen do in the electron transport chain?", "expected": "final electron acceptor"},
    {"query": "How does fermentation let glycolysis continue without oxygen?", "expected": "regenerating NAD"},
    {"query": "Where does the Calvin cycle take place?", "expected": "stroma"},
    {"query": "What does ATP synthase use to make ATP?", "expected": "proton"},
]


def request(path, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(API + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    try:
        demo = request("/api/demo", "POST", {})
        course_id = demo["course_id"]
        reciprocal_ranks, hits = [], 0
        for case in CASES:
            result = request("/api/chat", "POST", {"course_id": course_id, "question": case["query"]})
            sources = result.get("sources", [])
            rank = next((i + 1 for i, source in enumerate(sources) if case["expected"].lower() in source["text"].lower()), None)
            hits += rank is not None
            reciprocal_ranks.append(1 / rank if rank else 0)
            print(f"{case['query']:<58} {'hit@5' if rank else 'miss':<7} rank={rank or '-'}")
        n = len(CASES)
        print(f"\nHit-rate@5: {hits / n:.3f}")
        print(f"MRR@5:      {sum(reciprocal_ranks) / n:.3f}")
        print(f"Queries:    {n}")
    except (urllib.error.URLError, KeyError) as exc:
        raise SystemExit(f"Could not run benchmark. Start the API first. Details: {exc}")


if __name__ == "__main__":
    main()
