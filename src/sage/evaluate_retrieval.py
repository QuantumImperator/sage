import json
from pathlib import Path

from sage.ingestion import extract_pdf_pages, chunk_pages
from sage.retrieval import (
    BM25Retriever,
    SemanticRetriever,
    HybridRetriever,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = PROJECT_ROOT / "src/sage/data/sources/jesc110.pdf"
QUESTIONS_PATH = PROJECT_ROOT / "src/sage/data/evaluation/retrieval_questions.json"


def recall_at_k(results: list[dict], relevant_pages: set[int], k: int) -> float:
    if not relevant_pages:
        return 0.0

    retrieved_pages = {item["page"] for item in results[:k]}
    return len(retrieved_pages & relevant_pages) / len(relevant_pages)


def reciprocal_rank(results: list[dict], relevant_pages: set[int]) -> float:
    for rank, item in enumerate(results, start=1):
        if item["page"] in relevant_pages:
            return 1.0 / rank

    return 0.0


def main():
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)

    pages = extract_pdf_pages(str(PDF_PATH))
    chunks = chunk_pages(pages)

    retrievers = {
        "BM25": BM25Retriever(chunks),
        "Semantic": SemanticRetriever(chunks),
        "Hybrid": HybridRetriever(chunks),
    }

    evaluated = [question for question in questions if question["relevant_pages"]]

    if not evaluated:
        print("No verified relevant pages yet.")
        print("Add ground-truth labels before interpreting metrics.")
        return

    for name, retriever in retrievers.items():
        recalls = []
        reciprocal_ranks = []

        for question in evaluated:
            relevant_pages = set(question["relevant_pages"])
            results = retriever.search(question["question"], top_k=3)

            recalls.append(recall_at_k(results, relevant_pages, k=3))
            reciprocal_ranks.append(reciprocal_rank(results, relevant_pages))

        print(
            f"{name:10} | "
            f"Recall@3: {sum(recalls) / len(recalls):.3f} | "
            f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        )


if __name__ == "__main__":
    main()
