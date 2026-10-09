from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from modules.skill_catalog import RELATED_SKILLS, SKILL_ALIASES, skill_evidence


MatchStatus = Literal["Matched", "Partial", "Missing"]
RequirementType = Literal["Required", "Preferred", "Unspecified"]


@dataclass(frozen=True)
class SkillMatch:
    skill: str
    requirement_type: RequirementType
    status: MatchStatus
    weight: float
    evidence: str | None
    explanation: str


@dataclass(frozen=True)
class JobMatchReport:
    score: int
    skills: tuple[SkillMatch, ...]
    relevant_projects: tuple[str, ...]
    recommendations: tuple[str, ...]
    interview_topics: tuple[str, ...]


def _occurrence_context(text: str, skill: str) -> str:
    aliases = SKILL_ALIASES.get(skill, (skill.casefold(),))
    for alias in aliases:
        match = re.search(re.escape(alias), text, re.IGNORECASE)
        if match:
            return text[max(0, match.start() - 100) : match.end() + 100].casefold()
    return ""


def _requirement_type(text: str, skill: str) -> RequirementType:
    context = _occurrence_context(text, skill)
    if re.search(r"\b(preferred|nice to have|desirable|bonus)\b", context):
        return "Preferred"
    if re.search(r"\b(required|must have|must|required to|essential)\b", context):
        return "Required"
    return "Unspecified"


def _resume_mention(resume_text: str, skill: str) -> str | None:
    evidence = skill_evidence(resume_text, skill)
    if evidence:
        return evidence
    related = sorted(RELATED_SKILLS.get(skill, set()))
    for neighbor in related:
        evidence = skill_evidence(resume_text, neighbor)
        if evidence:
            return f"{neighbor}: {evidence}"
    return None


def match_resume_to_job(resume_text: str, job_text: str) -> JobMatchReport:
    if not resume_text.strip():
        raise ValueError("Resume text is empty.")
    if not job_text.strip():
        raise ValueError("Job description is empty.")

    requirements = [
        skill for skill in SKILL_ALIASES if skill_evidence(job_text, skill)
    ]
    if not requirements:
        raise ValueError(
            "No supported skills were identified in the job description. Add clear "
            "technical skill requirements to calculate a transparent match score."
        )

    skill_results: list[SkillMatch] = []
    for skill in requirements:
        requirement_type = _requirement_type(job_text, skill)
        evidence = skill_evidence(resume_text, skill)
        if evidence:
            status: MatchStatus = "Matched"
            explanation = "The resume explicitly mentions this skill."
        else:
            evidence = _resume_mention(resume_text, skill)
            if evidence:
                status = "Partial"
                explanation = (
                    "The resume mentions a related skill, not this requirement itself."
                )
            else:
                status = "Missing"
                explanation = "No supporting mention was found in the resume text."
        weight = {
            "Required": 2.0,
            "Unspecified": 1.5,
            "Preferred": 1.0,
        }[requirement_type]
        skill_results.append(
            SkillMatch(
                skill, requirement_type, status, weight, evidence, explanation
            )
        )

    denominator = sum(item.weight for item in skill_results)
    numerator = sum(
        item.weight
        * {"Matched": 1.0, "Partial": 0.5, "Missing": 0.0}[item.status]
        for item in skill_results
    )
    score = round(100 * numerator / denominator)

    project_text = "\n".join(
        line
        for line in resume_text.splitlines()
        if re.search(r"\b(project|built|developed|implemented|created)\b", line, re.I)
    )
    relevant = tuple(
        line.strip()
        for line in project_text.splitlines()
        if any(skill_evidence(line, item.skill) for item in skill_results)
    )
    recommendations = tuple(
        f"Add a truthful example of {item.skill} if you have used it."
        if item.status == "Missing"
        else f"Make your {item.skill} contribution and outcome more explicit."
        for item in skill_results
        if item.status != "Matched"
    )
    topics = tuple(item.skill for item in skill_results)
    return JobMatchReport(score, tuple(skill_results), relevant, recommendations, topics)
