from __future__ import annotations

from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from modules.job_matcher import JobMatchReport
from modules.resume_analyzer import ResumeProfile


def generate_pdf_report(
    profile: ResumeProfile,
    match: JobMatchReport | None,
    model_name: str,
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        output = BytesIO()
        document = SimpleDocTemplate(
            output,
            pagesize=letter,
            rightMargin=0.65 * inch,
            leftMargin=0.65 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.65 * inch,
            title="AI Resume Analyzer - Career Report",
        )
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=styles["Heading2"],
                textColor=colors.HexColor("#6557d2"),
                spaceBefore=12,
            )
        )
        styles.add(
            ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5)
        )
        story = [
            Paragraph("AI Resume Analyzer &amp; JobMatch AI", styles["Title"]),
            Paragraph(f"Report date: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}", styles["Small"]),
            Paragraph(f"Selected model: {escape(model_name)}", styles["Small"]),
            Spacer(1, 12),
            Paragraph("Resume summary", styles["SectionHeading"]),
        ]

        def add_list(items: tuple[str, ...], empty: str = "Not identified.") -> None:
            for item in items or (empty,):
                story.append(Paragraph(f"&bull; {escape(item)}", styles["BodyText"]))

        story.append(
            Paragraph(
                f"Candidate: {escape(profile.candidate_name or 'Not identified')}",
                styles["BodyText"],
            )
        )
        if profile.email:
            story.append(Paragraph(f"Email: {escape(profile.email)}", styles["BodyText"]))
        story.append(Paragraph("Technical skills", styles["SectionHeading"]))
        add_list(profile.skills)
        story.append(Paragraph("Education", styles["SectionHeading"]))
        add_list(profile.education)
        story.append(Paragraph("Experience", styles["SectionHeading"]))
        add_list(profile.experience)
        story.append(Paragraph("Projects", styles["SectionHeading"]))
        add_list(profile.projects)
        story.append(Paragraph("Certifications", styles["SectionHeading"]))
        add_list(profile.certifications)

        if match:
            story.extend(
                [
                    Paragraph("Job match", styles["SectionHeading"]),
                    Paragraph(
                        f"Approximate evidence-coverage score: {match.score}%",
                        styles["BodyText"],
                    ),
                    Paragraph(
                        "Method: required skills weigh 2, unspecified requirements "
                        "weigh 1.5, and preferred skills weigh 1. A supported match "
                        "earns full weight, a related skill half weight, and a missing "
                        "skill zero. This is not a hiring probability.",
                        styles["Small"],
                    ),
                    Spacer(1, 6),
                ]
            )
            rows = [["Skill", "Priority", "Result", "Resume evidence"]]
            rows.extend(
                [
                    item.skill,
                    item.requirement_type,
                    item.status,
                    Paragraph(
                        escape(item.evidence or "No supporting mention found."),
                        styles["Small"],
                    ),
                ]
                for item in match.skills
            )
            table = Table(rows, colWidths=[1.15 * inch, 0.9 * inch, 0.75 * inch, 4.0 * inch], repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeaff")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d6d3e5")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
            story.append(Paragraph("Recommendations", styles["SectionHeading"]))
            add_list(match.recommendations)

        story.extend(
            [
                Spacer(1, 12),
                Paragraph(
                    "Limitations: this report is an automated decision-support aid. "
                    "Extracted text and skill evidence can be incomplete. Review all "
                    "results and do not use this score as an automatic hiring decision.",
                    styles["Small"],
                ),
            ]
        )
        document.build(story)
        return output.getvalue()
    except Exception as error:
        raise RuntimeError(f"Could not generate the PDF report: {error}") from error
