"""Phase 9: measure how often retrieval finds the right chunk for each test question.

A question is a HIT when one of the retrieved chunks comes from the expected PDF
and contains the expected keyword. No Gemini calls, so it is free and fast.
"""
import json
from pathlib import Path

import chromadb

QUESTIONS_FILE = Path("eval/questions.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
K_VALUES = [5, 10]   # we score "found in the top 5" and "found in the top 10"


def is_hit(text, source, item):
    return source == item["source"] and item["keyword"].lower() in text.lower()


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    collection = chromadb.PersistentClient(path="chroma_db").get_collection("apple_diseases")

    hits = {k: 0 for k in K_VALUES}
    valid = 0
    for item in questions:
        # 1. Sanity check: does the answer exist in our chunks at all? If not, the TEST is broken.
        if not any(is_hit(c["text"], c["source"], item) for c in chunks):
            print(f"BROKEN TEST   {item['question']}  ('{item['keyword']}' not in {item['source']})")
            continue
        valid += 1

        # 2. Retrieve and find the rank of the first correct chunk (1 = best).
        results = collection.query(query_texts=[item["question"]], n_results=max(K_VALUES))
        rank = None
        for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
            if is_hit(text, meta["source"], item):
                rank = i
                break
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        status = f"HIT rank {rank:<2}" if rank else "MISS       "
        print(f"{status}   {item['question']}")

    print(f"\nValid test questions: {valid}/{len(questions)}")
    for k in K_VALUES:
        print(f"Hit@{k}: {hits[k]}/{valid} = {hits[k] / max(valid, 1):.0%}")


if __name__ == "__main__":
    main()