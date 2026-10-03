"""Phase 7: retrieve the best chunks, then ask Gemini to answer ONLY from them, with citations.
   python src/answer.py which apple diseases spread with rain
"""
import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors

MODEL = "gemini-3.8-flash"   # pick a name printed by: python src\list_models.py
TOP_K = 5                    # how many chunks the LLM gets to read
RETRIES = 4                  # how many times to try when Gemini is busy

PROMPT_TEMPLATE = """You are an assistant that helps apple farmers with plant diseases.
Answer the question using ONLY the numbered sources below.

Rules:
- After every fact, cite the source number in brackets, like [1] or [2][4].
- If the sources do not contain the answer, or only part of it, say so clearly. Never invent facts or numbers.
- Write in simple, practical language a farmer understands. Use short bullet points when listing things.

Sources:
{sources}

Question: {question}

Answer:"""


def retrieve(question):
    collection = chromadb.PersistentClient(path="chroma_db").get_collection("apple_diseases")
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return list(zip(results["documents"][0], results["metadatas"][0]))


def build_prompt(question, chunks):
    sources = "\n\n".join(
        f"[{i}] ({meta['source']}, page {meta['page']})\n{text}"
        for i, (text, meta) in enumerate(chunks, start=1)
    )
    return PROMPT_TEMPLATE.format(sources=sources, question=question)


def generate(client, prompt):
    """Ask Gemini; if its servers are busy (error 5xx), wait and try again: 5 s, 10 s, 20 s."""
    for attempt in range(1, RETRIES + 1):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except errors.ServerError:
            if attempt == RETRIES:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"Gemini is busy. Retrying in {wait} seconds (attempt {attempt}/{RETRIES})...")
            time.sleep(wait)


def main():
    question = " ".join(sys.argv[1:])
    chunks = retrieve(question)                      # R: retrieval
    prompt = build_prompt(question, chunks)          # A: augmentation

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    answer = generate(client, prompt)                # G: generation

    print(f"Question: {question}\n")
    print(answer)
    print("\nSources:")
    for i, (_, meta) in enumerate(chunks, start=1):
        print(f"  [{i}] {meta['source']}, page {meta['page']}")


if __name__ == "__main__":
    main()