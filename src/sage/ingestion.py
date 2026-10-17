from pathlib import Path

import pymupdf


def extract_pdf_pages(path: str) -> list[dict]:
    """Extract non-empty text from a PDF, preserving page numbers."""
    file_path = Path(path)
    pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                pages.append(
                    {
                        "source": str(file_path),
                        "page": page_number,
                        "content": text,
                    }
                )

    return pages


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict]:
    """Split extracted pages into overlapping, traceable chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []

    for page in pages:
        content = page["content"]
        start = 0
        chunk_number = 1

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": (
                            f"{Path(page['source']).stem}"
                            f"-p{page['page']}-c{chunk_number}"
                        ),
                        "source": page["source"],
                        "page": page["page"],
                        "content": chunk_text,
                    }
                )

            if end == len(content):
                break

            start = end - overlap
            chunk_number += 1

    return chunks
