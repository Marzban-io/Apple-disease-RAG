"""Phase 5: turn every chunk into an embedding and store it in a Chroma vector database.

Reads  data/processed/chunks.jsonl
Writes chroma_db/   (the vector database, a folder on your disk)
"""
import json
from pathlib import Path

import chromadb

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DB_DIR = "chroma_db"
COLLECTION = "apple_diseases"
BATCH_SIZE = 100   # embed 100 chunks at a time


def main():
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)   # start fresh, so re-running never duplicates chunks
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )
        print(f"Embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    print(f"\nDone. The collection '{COLLECTION}' holds {collection.count()} chunks in {DB_DIR}/")


if __name__ == "__main__":
    main()