"""Export the generated report to a PDF (Phase 5)."""
from __future__ import annotations

import io

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def report_to_pdf(report_text: str, title: str = "AI Chest X-ray Report") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for para in report_text.split("\n"):
        story.append(Paragraph(para or "&nbsp;", styles["Normal"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "<i>AI-generated educational draft — not for clinical use.</i>",
        styles["Italic"],
    ))
    doc.build(story)
    return buf.getvalue()
