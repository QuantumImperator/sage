import re
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable word tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("Cannot build retrieval over empty chunks")

        self.chunks = chunks
        self.tokenized_chunks = [tokenize(chunk["content"]) for chunk in chunks]
        self.index = BM25Okapi(self.tokenized_chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for i in ranked_indices[:top_k]:
            if scores[i] <= 0:
                continue

            result = self.chunks[i].copy()
            result["score"] = float(scores[i])
            results.append(result)

        return results

class SemanticRetriever:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("Cannot build retrieval over empty chunks")

        self.chunks = chunks

        # Load the embedding model.
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(Path.home() / ".cache" / "fastembed"),
        )

        # Embed every passage once and store the vectors.
        contents = [chunk["content"] for chunk in chunks]
        self.embeddings = np.asarray(
            list(self.model.passage_embed(contents)),
            dtype=np.float32,
        )

        # Precompute passage vector magnitudes.
        self.embedding_norms = np.linalg.norm(
            self.embeddings, axis=1
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        # Embed the query.
        query_vector = np.asarray(
            list(self.model.query_embed([query]))[0],
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(query_vector)

        # Calculate cosine similarity against every passage.
        denominators = self.embedding_norms * query_norm

        scores = np.divide(
            self.embeddings @ query_vector,
            denominators,
            out=np.zeros_like(self.embedding_norms),
            where=denominators != 0,
        )

        # Rank passage indices by similarity, highest first.
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Preserve source, page, and other chunk metadata.
        results = []

        for i in ranked_indices:
            result = self.chunks[int(i)].copy()
            result["score"] = float(scores[i])
            results.append(result)
        
        return results

class HybridRetriever:
    def __init__(self, chunks: list[dict]):
        self.bm25 = BM25Retriever(chunks)
        self.semantic = SemanticRetriever(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not query.strip():
            return []

        # Retrieve candidate passages from both systems.
        candidate_k = max(top_k * 3, top_k)

        bm25_results = self.bm25.search(query, candidate_k)
        semantic_results = self.semantic.search(query, candidate_k)

        fused = {}

        # Add each passage's contribution from its BM25 rank.
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = result.copy()
                fused[chunk_id]["rrf_score"] = 0.0

            fused[chunk_id]["bm25_score"] = result["score"]
            fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

        # Add each passage's contribution from its semantic rank.
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = result.copy()
                fused[chunk_id]["rrf_score"] = 0.0

            fused[chunk_id]["semantic_score"] = result["score"]
            fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

        # Return passages with the highest combined ranking scores.
        ranked_results = sorted(
            fused.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        return ranked_results[:top_k]
