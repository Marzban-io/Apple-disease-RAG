"""Phase 3: extract text from every PDF, page by page, keeping file name and page number."""
import json
from pathlib import Path

import pymupdf  # the PyMuPDF library

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/pages.jsonl")


def extract_pages(pdf_path):
    """Return a list of {source, page, text} records, one per PDF page."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": page.get_text(),
            })
    return pages


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
            pages = extract_pages(pdf_path)
            chars = sum(len(p["text"]) for p in pages)
            empty = sum(1 for p in pages if len(p["text"].strip()) < 50)
            print(f"{pdf_path.name}: {len(pages)} pages, {chars:,} characters, {empty} near-empty pages")
            for p in pages:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            total_pages += len(pages)
    print(f"\nSaved {total_pages} pages to {OUT_FILE}")


if __name__ == "__main__":
    main()