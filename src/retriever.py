"""Phase 10: hybrid retrieval = meaning search (vectors) + keyword search (BM25), merged with RRF."""
import json
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
CANDIDATES = 30   # how many results each method proposes before merging
RRF_K = 60        # standard constant of Reciprocal Rank Fusion


def tokenize(text):
    """Lowercase words: 'Petal fall, bloom!' -> ['petal', 'fall', 'bloom']"""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def load():
    """Load chunks, build the keyword index and open the vector database (only once)."""
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    collection = chromadb.PersistentClient(path="chroma_db").get_collection("apple_diseases")
    return chunks, bm25, collection


def as_results(ids):
    """Turn a list of chunk ids into (text, metadata) pairs."""
    chunks, _, _ = load()
    by_id = {c["id"]: c for c in chunks}
    return [(by_id[i]["text"], {"source": by_id[i]["source"], "page": by_id[i]["page"]}) for i in ids]


def vector_ids(question, n):
    _, _, collection = load()
    return collection.query(query_texts=[question], n_results=n)["ids"][0]


def keyword_ids(question, n):
    chunks, bm25, _ = load()
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:n]
    return [chunks[i]["id"] for i in best]


def vector_search(question, k):
    return as_results(vector_ids(question, k))


def hybrid_search(question, k):
    """Reciprocal Rank Fusion: a chunk scores 1/(60 + rank) in each list; the scores are added."""
    scores = {}
    for ranking in (vector_ids(question, CANDIDATES), keyword_ids(question, CANDIDATES)):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return as_results(best)