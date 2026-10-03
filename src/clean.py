"""Phase 3b: clean the extracted text so only real content is left.

Reads  data/processed/pages.jsonl        (raw text, one record per page)
Writes data/processed/pages_clean.jsonl  (cleaned text, near-empty pages dropped)
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

IN_FILE = Path("data/processed/pages.jsonl")
OUT_FILE = Path("data/processed/pages_clean.jsonl")

REPEAT_SHARE = 0.3   # a line on 30%+ of a document's pages counts as header/footer
MIN_LINE_CHARS = 4   # shorter lines ("5", "4a") are page numbers or figure labels
MIN_PAGE_CHARS = 50  # pages with less text after cleaning are dropped


def line_key(line):
    """Compare lines without digits, so 'Page 12' and 'Page 13' count as the same line."""
    return re.sub(r"\d+", "", line).strip().lower()


def find_repeated_lines(pages):
    """Return the set of line keys that repeat on many pages of one document."""
    counts = Counter()
    for page in pages:
        lines = [l for l in page["text"].splitlines() if len(l.strip()) >= MIN_LINE_CHARS]
        edges = lines[:3] + lines[-3:]   # headers sit at the top, footers at the bottom
        counts.update({line_key(l) for l in edges})
    if len(pages) < 5:
        return set()
    return {key for key, n in counts.items() if n / len(pages) >= REPEAT_SHARE}


def clean_text(text, repeated):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < MIN_LINE_CHARS:   # page numbers, figure labels
            continue
        if line_key(stripped) in repeated:   # headers and footers
            continue
        lines.append(stripped)
    text = "\n".join(lines)
        text = re.sub(r"(\w)[-\u00ad\u2010\u2011]\n(\w)", r"\1\2", text)   # "bacte-\nrial" -> "bacterial"
    text = text.replace("\n", " ")                 # join broken lines into flowing text
    text = re.sub(r"\s{2,}", " ", text)            # collapse extra spaces
    return text.strip()


def main():
    by_source = defaultdict(list)
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            by_source[record["source"]].append(record)

    kept_total = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for source, pages in by_source.items():
            repeated = find_repeated_lines(pages)
            before = sum(len(p["text"]) for p in pages)
            after, kept = 0, 0
            for page in pages:
                text = clean_text(page["text"], repeated)
                if len(text) < MIN_PAGE_CHARS:
                    continue
                out.write(json.dumps({**page, "text": text}, ensure_ascii=False) + "\n")
                after += len(text)
                kept += 1
            kept_total += kept
            print(f"{source}: kept {kept}/{len(pages)} pages, {before:,} -> {after:,} characters, "
                  f"{len(repeated)} repeated lines removed")
            for key in sorted(repeated)[:5]:
                print(f"    removed repeated line: '{key[:60]}'")
    print(f"\nSaved {kept_total} clean pages to {OUT_FILE}")


if __name__ == "__main__":
    main()