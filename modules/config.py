from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    embedding_model_path: str = os.getenv(
        "EMBEDDING_MODEL_PATH", ".models/all-MiniLM-L6-v2"
    )
    upload_limit_bytes: int = 10 * 1024 * 1024
    chunk_size_words: int = 450
    chunk_overlap_words: int = 80
    retrieval_top_k: int = 5
    request_timeout_seconds: int = 120
    maximum_generation_tokens: int = 1200


DEFAULT_CONFIG = AppConfig()
