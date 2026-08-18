"""Load PDF files and extract text with page metadata."""

from typing import List, Dict
from PyPDF2 import PdfReader
import os


def load_pdf_text(path: str) -> str:
    """Load PDF and return concatenated text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def load_pdf_with_pages(path: str) -> List[Dict[str, object]]:
    """Load PDF with page metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    
    reader = PdfReader(path)
    pages = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        
        pages.append({
            "page": page_num,
            "text": text
        })
    
    return pages

