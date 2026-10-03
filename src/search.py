"""Phase 6: search the vector database.  Example:
   python src/search.py what diseases attack apple during bloom
"""
import sys

import chromadb

TOP_K = 5   # how many chunks to return

question = " ".join(sys.argv[1:])
collection = chromadb.PersistentClient(path="chroma_db").get_collection("apple_diseases")
results = collection.query(query_texts=[question], n_results=TOP_K)

print(f"Question: {question}")
for rank, (text, meta, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
    print(f"\n#{rank}  similarity {1 - distance:.2f}  |  {meta['source']}, page {meta['page']}")
    print(text[:400] + ("..." if len(text) > 400 else ""))