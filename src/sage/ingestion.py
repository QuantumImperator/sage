from pathlib import Path
import re

import pymupdf


def extract_pdf_pages(path: str) -> list[dict]:
    """Extract non-empty text from a PDF, preserving PDF page numbers."""
    file_path = Path(path)
    pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({
                    "source": str(file_path),
                    "page": page_number,
                    "content": text,
                })

    return pages


def _choose_chunk_end(
    content: str,
    start: int,
    target_end: int,
) -> int:
    """Prefer a sentence, paragraph, or word boundary near the target."""
    if target_end >= len(content):
        return len(content)

    minimum = start + int((target_end - start) * 0.65)

    boundaries = [
        content.rfind("\n\n", minimum, target_end),
        content.rfind(". ", minimum, target_end),
        content.rfind("? ", minimum, target_end),
        content.rfind("! ", minimum, target_end),
        content.rfind("; ", minimum, target_end),
        content.rfind(" ", minimum, target_end),
    ]

    boundary = max(boundaries)

    if boundary > start:
        # Keep sentence punctuation, but exclude a whitespace separator.
        if content[boundary:boundary + 2] in {". ", "? ", "! ", "; "}:
            return boundary + 1
        return boundary

    return target_end


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict]:
    """Create overlapping chunks while preferring readable boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []

    for page in pages:
        content = page["content"].strip()
        start = 0
        chunk_number = 1

        while start < len(content):
            target_end = min(start + chunk_size, len(content))
            end = _choose_chunk_end(content, start, target_end)

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": (
                        f"{Path(page['source']).stem}"
                        f"-p{page['page']}-c{chunk_number}"
                    ),
                    "source": page["source"],
                    "page": page["page"],
                    "content": chunk_text,
                })
                chunk_number += 1

            if end >= len(content):
                break

            # Aim for the requested overlap, then align the next start
            # to a nearby sentence or word boundary when possible.
            next_start = max(start + 1, end - overlap)
            minimum = next_start
            search_end = end

            boundaries = [
                content.rfind("\n\n", minimum, search_end),
                content.rfind(". ", minimum, search_end),
                content.rfind("? ", minimum, search_end),
                content.rfind("! ", minimum, search_end),
                content.rfind(" ", minimum, search_end),
            ]

            boundary = max(boundaries)

            if boundary >= next_start:
                if content[boundary:boundary + 2] in {
                    ". ", "? ", "! "
                }:
                    next_start = boundary + 2
                else:
                    next_start = boundary + 1

            start = next_start

    return chunks
