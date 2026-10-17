import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from sage.ingestion import extract_pdf_pages, chunk_pages
from sage.retrieval import HybridRetriever
from sage.answering import answer_question

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not model:
    raise RuntimeError("GEMINI_MODEL is missing")

root = Path(__file__).resolve().parents[2]
pdf_path = root / "src/sage/data/sources/jesc110.pdf"

pages = extract_pdf_pages(str(pdf_path))
chunks = chunk_pages(pages)
retriever = HybridRetriever(chunks)

client = genai.Client(api_key=api_key)

result = answer_question(
    question="How does the human eye focus light on the retina?",
    retriever=retriever,
    client=client,
    model=model,
)

print("\nANSWER\n")
print(result["answer"])

print("\nRETRIEVED SOURCES\n")
for source in result["sources"]:
    print(
        f"[{source['evidence_id']}] " f"{source['source']} — PDF page {source['page']}"
    )
