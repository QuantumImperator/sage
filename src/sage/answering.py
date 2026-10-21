from pathlib import Path

from google import genai


def answer_question(
    question: str,
    retriever,
    client: genai.Client,
    model: str,
    top_k: int = 3,
) -> dict:
    """Answer a question using retrieved textbook evidence."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    chunks = retriever.search(question, top_k=top_k)

    if not chunks:
        return {
            "answer": (
                "I could not find sufficient supporting evidence "
                "in the available textbook."
            ),
            "sources": [],
            "status": "insufficient_evidence",
        }

    evidence_blocks = []
    sources = []

    for index, chunk in enumerate(chunks, start=1):
        evidence_id = f"E{index}"
        page = chunk["page"]
        source_name = Path(chunk["source"]).name

        evidence_blocks.append(
            f"[{evidence_id}]\n"
            f"Source: {source_name}, PDF page {page}\n"
            f"Passage:\n{chunk['content']}"
        )

        sources.append(
            {
                "evidence_id": evidence_id,
                "source": source_name,
                "page": page,
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
            }
        )

    prompt = f"""
You are S.A.G.E., a source-grounded NCERT learning assistant.

Answer the learner's question using ONLY the evidence below.

Rules:
1. Do not introduce unsupported factual claims.
2. Cite supporting passages using their exact IDs, such as [E1].
3. Do not invent evidence IDs or page numbers.
4. Explain clearly at a school student's level.
5. If the evidence is insufficient, explicitly say so.
6. Distinguish related concepts instead of confusing them.

LEARNER QUESTION:
{question}

TEXTBOOK EVIDENCE:
{chr(10).join(evidence_blocks)}

Write the answer with inline evidence IDs.
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        answer = response.text
    except Exception as exc:
        # Keep diagnostic details limited; never expose them to the learner.
        print(
            f"S.A.G.E. generation failed: {type(exc).__name__}"
        )
        return {
            "answer": (
                "Answer generation is temporarily unavailable. "
                "Your textbook evidence is still available below. "
                "Please try again later."
            ),
            "sources": sources,
            "status": "generation_unavailable",
        }

    if not answer:
        return {
            "answer": (
                "The model returned no answer. "
                "Your retrieved textbook evidence is available below."
            ),
            "sources": sources,
            "status": "empty_generation",
        }

    return {
        "answer": answer,
        "sources": sources,
        "status": "success",
    }
