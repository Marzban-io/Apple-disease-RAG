"""Phase 4: cut the cleaned pages into small overlapping chunks.

Reads  data/processed/pages_clean.jsonl
Writes data/processed/chunks.jsonl   (one record per chunk: id, source, page, text)
"""
import json
import random
import re
from pathlib import Path

IN_FILE = Path("data/processed/pages_clean.jsonl")
OUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 1000   # target maximum characters per chunk (~200 words)
OVERLAP = 200       # characters repeated from the end of the previous chunk
MIN_CHUNK = 50      # drop tiny leftover chunks (usually table fragments)


def split_sentences(text):
    """Split text after . ! or ? followed by a space. Cut over-long 'sentences' (e.g. tables) by words."""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > CHUNK_SIZE:
            cut = sentence.rfind(" ", 0, CHUNK_SIZE)
            cut = cut if cut > 0 else CHUNK_SIZE
            sentences.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def chunk_text(text):
    """Group whole sentences into chunks of up to CHUNK_SIZE characters, with OVERLAP between chunks."""
    chunks, current = [], []
    for sentence in split_sentences(text):
        if current and len(" ".join(current + [sentence])) > CHUNK_SIZE:
            chunks.append(" ".join(current))
            # keep the last sentences (up to OVERLAP characters) to start the next chunk
            overlap = []
            for previous in reversed(current):
                if len(" ".join([previous] + overlap)) > OVERLAP:
                    break
                overlap.insert(0, previous)
            # only keep the overlap if the next sentence still fits next to it
            if len(" ".join(overlap + [sentence])) > CHUNK_SIZE:
                overlap = []
            current = overlap
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) >= MIN_CHUNK]


def main():
    records = []
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            for i, text in enumerate(chunk_text(page["text"])):
                records.append({
                    "id": f"{page['source']}-p{page['page']}-c{i}",
                    "source": page["source"],
                    "page": page["page"],
                    "text": text,
                })

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    lengths = [len(r["text"]) for r in records]
    print(f"Saved {len(records)} chunks to {OUT_FILE}")
    print(f"Length in characters: min {min(lengths)}, average {sum(lengths) // len(lengths)}, max {max(lengths)}")

    random.seed(42)  # same "random" examples every run
    for r in random.sample(records, 2):
        print(f"\n--- example: {r['id']} ({len(r['text'])} chars) ---\n{r['text']}")


if __name__ == "__main__":
    main()