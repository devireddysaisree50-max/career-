from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    index: int


def chunk_document(
    text: str,
    source: str,
    chunk_size_words: int = 450,
    overlap_words: int = 80,
) -> list[DocumentChunk]:
    if chunk_size_words < 1:
        raise ValueError("Chunk size must be at least one word.")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    words = re.findall(r"\S+", text)
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks: list[DocumentChunk] = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_size_words]).strip()
        if piece:
            chunks.append(DocumentChunk(piece, source, len(chunks)))
        if start + chunk_size_words >= len(words):
            break
    return chunks
