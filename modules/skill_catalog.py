from __future__ import annotations

import re


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "Java": ("java",),
    "JavaScript": ("javascript", "js"),
    "TypeScript": ("typescript",),
    "C++": ("c++", "cpp"),
    "C#": ("c#", "c sharp"),
    "Go": ("golang",),
    "Rust": ("rust",),
    "SQL": ("sql",),
    "R": ("r language",),
    "HTML/CSS": ("html", "css"),
    "React": ("react", "react.js", "reactjs"),
    "Node.js": ("node.js", "nodejs"),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi",),
    "Streamlit": ("streamlit",),
    "REST APIs": ("rest api", "restful api", "rest apis"),
    "Git": ("git", "github", "gitlab"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure",),
    "Google Cloud": ("google cloud", "gcp"),
    "Linux": ("linux",),
    "Machine Learning": ("machine learning", "ml"),
    "Deep Learning": ("deep learning",),
    "NLP": ("natural language processing", "nlp"),
    "LLMs": ("large language models", "llm", "llms"),
    "RAG": ("retrieval augmented generation", "rag"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "PyTorch": ("pytorch", "torch"),
    "TensorFlow": ("tensorflow",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "FAISS": ("faiss",),
    "ChromaDB": ("chromadb", "chroma db"),
    "MongoDB": ("mongodb", "mongo db"),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "Redis": ("redis",),
    "CI/CD": ("ci/cd", "continuous integration", "continuous delivery"),
    "Agile": ("agile", "scrum"),
    "Data Structures": ("data structures", "data structure"),
    "Algorithms": ("algorithms", "algorithm design"),
    "Microsoft Excel": ("microsoft excel", "excel"),
}

RELATED_SKILLS: dict[str, set[str]] = {
    "PyTorch": {"TensorFlow", "Machine Learning"},
    "TensorFlow": {"PyTorch", "Machine Learning"},
    "PostgreSQL": {"MySQL", "SQL"},
    "MySQL": {"PostgreSQL", "SQL"},
    "React": {"JavaScript", "TypeScript"},
    "Django": {"Flask", "FastAPI"},
    "Flask": {"Django", "FastAPI"},
    "FastAPI": {"Django", "Flask"},
    "AWS": {"Azure", "Google Cloud"},
    "Azure": {"AWS", "Google Cloud"},
    "Google Cloud": {"AWS", "Azure"},
}


def _contains_alias(text: str, alias: str) -> bool:
    escaped = re.escape(alias)
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def extract_skills(text: str) -> list[str]:
    return [
        skill
        for skill, aliases in SKILL_ALIASES.items()
        if any(_contains_alias(text, alias) for alias in aliases)
    ]


def skill_evidence(text: str, skill: str) -> str | None:
    normalized = text.casefold()
    aliases = SKILL_ALIASES.get(skill, (skill.casefold(),))
    for alias in aliases:
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.IGNORECASE
        )
        match = pattern.search(normalized)
        if match:
            start = max(0, match.start() - 65)
            end = min(len(text), match.end() + 100)
            return re.sub(r"\s+", " ", text[start:end]).strip()
    return None
