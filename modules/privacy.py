from __future__ import annotations

from typing import Any


SESSION_DATA_KEYS = (
    "resume_bytes",
    "resume_text",
    "resume_file_name",
    "resume_profile",
    "job_text",
    "job_match",
    "vector_store",
    "indexed_sources",
    "chat_history",
    "recent_analyses",
)


def clear_candidate_data(session_state: Any) -> None:
    for key in SESSION_DATA_KEYS:
        session_state.pop(key, None)
