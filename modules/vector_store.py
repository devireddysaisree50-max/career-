from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass

import numpy as np

from modules.document_chunker import DocumentChunk
from modules.embeddings import Embedder, EmbeddingError


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", text.casefold())


class HybridVectorStore:
    """Session-owned Chroma index with reciprocal-rank semantic/BM25 fusion."""

    def __init__(self, embedder: Embedder) -> None:
        try:
            import chromadb

            self.client = chromadb.EphemeralClient()
            self.collection = self.client.create_collection(
                name=f"resume_{uuid.uuid4().hex}",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise RuntimeError(
                "Could not initialize the in-memory Chroma vector index."
            ) from error
        self.embedder = embedder
        self._chunks: list[DocumentChunk] = []

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty document.")
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        if embeddings.shape[0] != len(chunks):
            raise EmbeddingError("Embedding count does not match the chunk count.")

        ids = [uuid.uuid4().hex for _ in chunks]
        try:
            self.collection.add(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {"source": chunk.source, "chunk_index": chunk.index}
                    for chunk in chunks
                ],
                embeddings=embeddings.tolist(),
            )
        except Exception as error:
            raise RuntimeError("Could not add document chunks to Chroma.") from error
        self._chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        if not query.strip() or not self._chunks:
            return []

        query_vector = self.embedder.embed([query])
        candidate_count = min(max(top_k * 3, top_k), len(self._chunks))
        try:
            semantic = self.collection.query(
                query_embeddings=query_vector.tolist(),
                n_results=candidate_count,
                include=["documents", "metadatas"],
            )
        except Exception as error:
            raise RuntimeError("Semantic retrieval from Chroma failed.") from error

        semantic_docs = semantic.get("documents", [[]])[0]
        semantic_meta = semantic.get("metadatas", [[]])[0]
        by_key: dict[tuple[str, int], DocumentChunk] = {
            (chunk.source, chunk.index): chunk for chunk in self._chunks
        }
        ranked: dict[tuple[str, int], float] = {}
        rank_constant = 60
        for rank, (text, metadata) in enumerate(
            zip(semantic_docs, semantic_meta), start=1
        ):
            key = (str(metadata["source"]), int(metadata["chunk_index"]))
            ranked[key] = ranked.get(key, 0.0) + 1.0 / (rank_constant + rank)

        query_terms = set(_terms(query))
        if query_terms:
            corpus_terms = [_terms(chunk.text) for chunk in self._chunks]
            average_length = sum(map(len, corpus_terms)) / len(corpus_terms) or 1.0
            document_frequency = Counter(
                term for document in corpus_terms for term in set(document)
            )
            keyword_scores: list[tuple[int, float]] = []
            for index, document in enumerate(corpus_terms):
                frequencies = Counter(document)
                score = 0.0
                for term in query_terms.intersection(frequencies):
                    frequency = frequencies[term]
                    inverse = np.log(
                        1
                        + (len(corpus_terms) - document_frequency[term] + 0.5)
                        / (document_frequency[term] + 0.5)
                    )
                    denominator = frequency + 1.2 * (
                        1 - 0.75 + 0.75 * len(document) / average_length
                    )
                    score += inverse * frequency * 2.2 / denominator
                keyword_scores.append((index, score))
            keyword_scores.sort(key=lambda item: item[1], reverse=True)
            positive = [item for item in keyword_scores if item[1] > 0][:candidate_count]
            for rank, (index, _) in enumerate(positive, start=1):
                chunk = self._chunks[index]
                key = (chunk.source, chunk.index)
                ranked[key] = ranked.get(key, 0.0) + 1.0 / (rank_constant + rank)

        results = [
            RetrievedChunk(
                text=by_key[key].text,
                source=by_key[key].source,
                chunk_index=by_key[key].index,
                score=score,
            )
            for key, score in ranked.items()
            if key in by_key
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
