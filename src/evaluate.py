"""Phase 9-10: measure how often retrieval finds the right chunk, for each search method.

A question is a HIT when a retrieved chunk comes from the expected PDF and contains the
expected keyword. MRR (mean reciprocal rank) rewards finding it HIGH: rank 1 = 1, rank 4 = 0.25.
No Gemini calls, so it is free and fast.
"""
import json
from pathlib import Path

from retriever import hybrid_search, vector_search

QUESTIONS_FILE = Path("eval/questions.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
K_VALUES = [5, 10]
METHODS = {"vector": vector_search, "hybrid": hybrid_search}


def is_hit(text, source, item):
    return source == item["source"] and item["keyword"].lower() in text.lower()


def find_rank(results, item):
    for rank, (text, meta) in enumerate(results, start=1):
        if is_hit(text, meta["source"], item):
            return rank
    return None


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    ranks = {name: [] for name in METHODS}
    print(f"{'vector':>7} {'hybrid':>7}   question")
    for item in questions:
        if not any(is_hit(c["text"], c["source"], item) for c in chunks):
            print(f"BROKEN TEST   {item['question']}  ('{item['keyword']}' not in {item['source']})")
            continue
        row = []
        for name, search in METHODS.items():
            rank = find_rank(search(item["question"], max(K_VALUES)), item)
            ranks[name].append(rank)
            row.append(f"{rank if rank else 'MISS':>7}")
        print(" ".join(row) + f"   {item['question']}")

    valid = len(ranks["vector"])
    print(f"\nValid test questions: {valid}/{len(questions)}\n")
    print(f"{'method':<8}" + "".join(f"{'Hit@' + str(k):>9}" for k in K_VALUES) + f"{'MRR':>7}")
    for name, rs in ranks.items():
        hit_cells = "".join(f"{sum(1 for r in rs if r and r <= k) / valid:>9.0%}" for k in K_VALUES)
        mrr = sum(1 / r for r in rs if r) / valid
        print(f"{name:<8}{hit_cells}{mrr:>7.2f}")


if __name__ == "__main__":
    main()