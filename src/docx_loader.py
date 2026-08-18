"""Load DOCX files and extract text."""

from __future__ import annotations

from typing import List
import os


def _style_to_heading(text: str, style_name: str) -> str:
    """Mark headings with # prefix if style contains 'heading'."""
    if not text:
        return text
    if style_name and "heading" in style_name.lower():
        return f"\n\n# {text.strip()}\n"
    return text.strip()


def load_docx_text(path: str) -> str:
    """Load a DOCX file and return concatenated extracted text.

    Preserves paragraph boundaries and heading structure where possible.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DOCX not found: {path}")

    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX support. Install it via: pip install python-docx"
        ) from exc

    document = Document(path)
    blocks: List[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style_name = paragraph.style.name if paragraph.style is not None else ""
        blocks.append(_style_to_heading(text, style_name))

    return "\n\n".join(blocks)
