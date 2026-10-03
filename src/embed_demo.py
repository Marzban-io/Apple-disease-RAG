"""Phase 5 demo: see what an embedding is, on 4 example sentences."""
import numpy as np
from chromadb.utils import embedding_functions

embed = embedding_functions.DefaultEmbeddingFunction()  # model all-MiniLM-L6-v2, runs on your laptop

sentences = [
    "Apple scab spreads in rainy weather.",
    "Wet conditions in spring favour scab infections on apple leaves.",
    "Fire blight is a bacterial disease of apple and pear.",
    "The price of tractors increased last year.",
]
vectors = embed(sentences)
print(f"Each sentence becomes a list of {len(vectors[0])} numbers.")
print(f"First 5 numbers of sentence 1: {np.round(vectors[0][:5], 3)}")


def similarity(a, b):
    """Cosine similarity: 1 = same meaning, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"\nSimilarity to: '{sentences[0]}'")
for sentence, vector in zip(sentences[1:], vectors[1:]):
    print(f"  {similarity(vectors[0], vector):.2f}  {sentence}")