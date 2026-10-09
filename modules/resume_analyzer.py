from __future__ import annotations

import re
from dataclasses import dataclass

from modules.skill_catalog import extract_skills, skill_evidence


SECTION_ALIASES = {
    "education": ("education", "academic background", "qualifications"),
    "experience": ("experience", "employment", "work history", "internship"),
    "projects": ("projects", "selected projects", "project experience"),
    "certifications": ("certifications", "certificates", "licenses"),
    "achievements": ("achievements", "awards", "honors"),
}
KNOWN_HEADINGS = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


@dataclass(frozen=True)
class ResumeProfile:
    candidate_name: str | None
    email: str | None
    phone: str | None
    education: tuple[str, ...]
    experience: tuple[str, ...]
    projects: tuple[str, ...]
    certifications: tuple[str, ...]
    achievements: tuple[str, ...]
    skills: tuple[str, ...]
    strengths: tuple[str, ...]
    missing_information: tuple[str, ...]


def _sections(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {name: [] for name in SECTION_ALIASES}
    active: str | None = None
    for line in text.splitlines():
        cleaned = line.strip(" \t|:-")
        heading = re.sub(r"[^a-z ]", "", cleaned.casefold()).strip()
        section = KNOWN_HEADINGS.get(heading)
        if section:
            active = section
        elif cleaned and active:
            found[active].append(cleaned)
    return found


def analyze_resume(text: str) -> ResumeProfile:
    if not text.strip():
        raise ValueError("Resume text is empty.")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_name = None
    for line in lines[:8]:
        if "@" in line or re.search(r"\d", line) or len(line) > 80:
            continue
        if re.search(r"\b(resume|curriculum vitae|cv)\b", line, re.I):
            continue
        if len(line.split()) <= 5 and re.fullmatch(
            r"[A-Za-z][A-Za-z .'-]*", line
        ):
            candidate_name = line
            break

    email_match = re.search(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE
    )
    phone_match = re.search(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)", text)
    sections = _sections(text)
    skills = extract_skills(text)
    experience_and_projects = "\n".join(
        sections["experience"] + sections["projects"]
    )
    strengths = tuple(
        f"Documented {skill} in experience or project material."
        for skill in skills
        if skill_evidence(experience_and_projects, skill)
    )
    missing = []
    if not candidate_name:
        missing.append("Candidate name was not confidently identified.")
    if not email_match:
        missing.append("No email address was found.")
    if not sections["education"]:
        missing.append("No clearly labeled education section was found.")
    if not sections["experience"]:
        missing.append("No clearly labeled experience section was found.")
    if not skills:
        missing.append("No recognized technical skills were found.")

    return ResumeProfile(
        candidate_name=candidate_name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0).strip() if phone_match else None,
        education=tuple(sections["education"]),
        experience=tuple(sections["experience"]),
        projects=tuple(sections["projects"]),
        certifications=tuple(sections["certifications"]),
        achievements=tuple(sections["achievements"]),
        skills=tuple(skills),
        strengths=strengths,
        missing_information=tuple(missing),
    )
