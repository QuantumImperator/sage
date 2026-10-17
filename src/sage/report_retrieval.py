import json
from pathlib import Path

from sage.ingestion import extract_pdf_pages, chunk_pages
from sage.retrieval import (
    BM25Retriever,
    SemanticRetriever,
    HybridRetriever,
)

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "src/sage/data/sources/jesc110.pdf"
QUESTIONS_PATH = ROOT / "src/sage/data/evaluation/retrieval_questions.json"
REPORT_PATH = ROOT / "src/sage/data/evaluation/retrieval_report.json"


def main():
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)

    pages = extract_pdf_pages(str(PDF_PATH))
    chunks = chunk_pages(pages)

    retrievers = {
        "bm25": BM25Retriever(chunks),
        "semantic": SemanticRetriever(chunks),
        "hybrid": HybridRetriever(chunks),
    }

    report = []

    for question in questions:
        entry = {
            "id": question["id"],
            "question": question["question"],
            "expected_concept": question["expected_concept"],
            "retrievals": {},
        }

        for name, retriever in retrievers.items():
            results = retriever.search(question["question"], top_k=3)

            entry["retrievals"][name] = [
                {
                    "chunk_id": result["chunk_id"],
                    "page": result["page"],
                    "score": result.get("rrf_score", result.get("score")),
                    "content": result["content"],
                }
                for result in results
            ]

        report.append(entry)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Report saved to: {REPORT_PATH}")
    print(f"Questions processed: {len(report)}")


if __name__ == "__main__":
    main()
