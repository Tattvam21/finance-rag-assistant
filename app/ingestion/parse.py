import os
import re
import pymupdf
from typing import List, Tuple

def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from PDF page by page so page numbers are preserved.
    Returns list of (page_number, raw_text).
    """
    doc = pymupdf.open(pdf_path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]

def clean_text(text: str) -> str:
    """
    Cleans extracted text:
    - fixes hyphenation across line breaks (e.g., 'disburse-\nment' -> 'disbursement')
    - removes standard header / footer lines
    - collapses redundant horizontal whitespace while preserving paragraph breaks
    """
    # 1. Fix line-break hyphenation
    cleaned = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 2. Normalize corrupted rupee glyphs (■ or I followed by digits)
    cleaned = re.sub(r'■(?=\d)', '₹', cleaned)
    cleaned = re.sub(r'(?<=\s)I(?=\d{1,3}(?:,\d{2,3})+|\d+)', '₹', cleaned)

    lines = cleaned.splitlines()
    filtered_lines = []

    # Common repetitive headers or footers to filter out
    boilerplate_patterns = [
        re.compile(r"Corporate Finance & Operations Policy Manual", re.IGNORECASE),
        re.compile(r"^Page \d+( of \d+)?$", re.IGNORECASE)
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue
        
        # Check against boilerplate
        if any(pat.search(stripped) for pat in boilerplate_patterns):
            continue

        # Collapse excessive whitespace within line
        collapsed = re.sub(r'[ \t]+', ' ', stripped)
        filtered_lines.append(collapsed)

    # Rejoin and collapse multiple consecutive blank lines to double newlines (paragraphs)
    result = "\n".join(filtered_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()
