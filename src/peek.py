"""Print the text of one page.
   python src/peek.py kentucky_apple_scouting.pdf 5          -> raw text
   python src/peek.py kentucky_apple_scouting.pdf 5 clean    -> cleaned text
"""
import json
import sys

source, page = sys.argv[1], int(sys.argv[2])
file = "data/processed/pages_clean.jsonl" if len(sys.argv) > 3 else "data/processed/pages.jsonl"

with open(file, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        if record["source"] == source and record["page"] == page:
            print(record["text"])
            break
    else:
        print("Not found. Check the file name and page number (empty pages are dropped in the clean file).")