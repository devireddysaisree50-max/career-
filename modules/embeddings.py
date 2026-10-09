from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingError(RuntimeError):
    """Local embedding model is missing or cannot generate embeddings."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalSentenceTransformer:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise EmbeddingError(
                f"Local embedding model not found at '{self.model_path}'. "
                "Download a Sentence Transformers model explicitly and set "
                "EMBEDDING_MODEL_PATH; the app does not download models automatically."
            )
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                str(self.model_path), local_files_only=True
            )
        except Exception as error:
            raise EmbeddingError(
                "The local embedding model could not be loaded. Confirm that the "
                "model files are complete and compatible with the installed "
                "Sentence Transformers and PyTorch versions."
            ) from error

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            vectors = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        except Exception as error:
            raise EmbeddingError("The local embedding model failed to encode text.") from error
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise EmbeddingError("The embedding model returned malformed vectors.")
        return array
