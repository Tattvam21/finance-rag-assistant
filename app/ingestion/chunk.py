import re
import tiktoken
from typing import List, Dict, Any

# Standard OpenAI cl100k_base tokenizer
ENCODING = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def infer_doc_type(source_file: str) -> str:
    name = source_file.lower().replace(".pdf", "")
    if "payment" in name:
        return "payment_policy"
    elif "vendor" in name:
        return "vendor_policy"
    elif "expense" in name:
        return "expense_policy"
    elif "invoice" in name:
        return "invoice_terms"
    elif "compliance" in name:
        return "compliance"
    return "general"

def extract_doc_title(cleaned_text: str) -> str:
    """Extracts human-readable document title from the text header before numbered sections."""
    heading_pattern = re.compile(r'(?=^(?:§\s*)?\d+\.\d+\s+[^\n]+)', re.MULTILINE)
    raw_blocks = heading_pattern.split(cleaned_text)
    if raw_blocks:
        lines = [l.strip() for l in raw_blocks[0].splitlines() if l.strip()]
        title_lines = [
            l for l in lines
            if not re.search(r'Corporate Finance|Version|Page \d+', l, re.IGNORECASE)
            and not re.match(r'^\d+\.\d+', l)
        ]
        if title_lines:
            return " ".join(title_lines).strip()
    return "Corporate Finance Policy"

def chunk_page_text(
    source_file: str,
    page: int,
    cleaned_text: str,
    target_chunk_tokens: int = 500,
    overlap_tokens: int = 50,
    starting_chunk_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Chunks text using token counts with ~500-token chunks and ~50-token overlap,
    splitting primarily on numbered section headings (e.g. '1.1 Standard Payment Terms',
    '4.2 Overdue Invoices and Late Fee Structure').
    Eliminates title-only overview chunks and stores the document title as chunk metadata.
    """
    doc_type = infer_doc_type(source_file)
    doc_title = extract_doc_title(cleaned_text)
    default_overview_title = f"Overview: {doc_title}"

    # Regex matching numbered section headings: e.g. "1.1 Standard Payment Terms" or "4.2 Overdue..."
    heading_pattern = re.compile(r'(?=^(?:§\s*)?\d+\.\d+\s+[^\n]+)', re.MULTILINE)
    raw_blocks = heading_pattern.split(cleaned_text)

    blocks = []
    for b in raw_blocks:
        b_clean = b.strip()
        if not b_clean:
            continue
        # Split on double-newlines if multiple paragraphs exist within block
        sub_paras = b_clean.split("\n\n")
        blocks.extend([sp.strip() for sp in sub_paras if sp.strip()])

    section_regex = re.compile(r'^(?:§\s*)?(\d+\.\d+)\s+([^\n]+)')
    current_section = default_overview_title

    chunks = []
    current_paragraphs = []
    chunk_idx = starting_chunk_index

    for block in blocks:
        # Check if block starts with a section heading
        sec_match = section_regex.match(block)
        if sec_match:
            # If we already have accumulated text from a valid previous section, flush it
            if current_paragraphs and current_section != default_overview_title:
                chunk_text = "\n\n".join(current_paragraphs).strip()
                if chunk_text:
                    chunks.append({
                        "source_file": source_file,
                        "page": page,
                        "section": current_section,
                        "doc_type": doc_type,
                        "doc_title": doc_title,
                        "chunk_index": chunk_idx,
                        "content": chunk_text,
                        "tokens": count_tokens(chunk_text)
                    })
                    chunk_idx += 1
            current_paragraphs = []
            current_section = f"{sec_match.group(1)} {sec_match.group(2).strip()}"

        # Skip accumulating text that belongs to the doc title/header block before first section
        if current_section == default_overview_title:
            continue

        candidate_text = "\n\n".join(current_paragraphs + [block])
        candidate_tokens = count_tokens(candidate_text)

        if candidate_tokens <= target_chunk_tokens:
            current_paragraphs.append(block)
        else:
            if current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs).strip()
                chunks.append({
                    "source_file": source_file,
                    "page": page,
                    "section": current_section,
                    "doc_type": doc_type,
                    "doc_title": doc_title,
                    "chunk_index": chunk_idx,
                    "content": chunk_text,
                    "tokens": count_tokens(chunk_text)
                })
                chunk_idx += 1

                # Calculate overlap: take trailing tokens from current paragraphs
                encoded = ENCODING.encode(chunk_text)
                overlap_encoded = encoded[-overlap_tokens:] if len(encoded) > overlap_tokens else []
                overlap_text = ENCODING.decode(overlap_encoded) if overlap_encoded else ""

                current_paragraphs = [overlap_text, block] if overlap_text else [block]
            else:
                # Single block exceeds target size: chunk it directly
                chunks.append({
                    "source_file": source_file,
                    "page": page,
                    "section": current_section,
                    "doc_type": doc_type,
                    "doc_title": doc_title,
                    "chunk_index": chunk_idx,
                    "content": block,
                    "tokens": candidate_tokens
                })
                chunk_idx += 1
                current_paragraphs = []

    # Emit final remaining chunk
    if current_paragraphs and current_section != default_overview_title:
        chunk_text = "\n\n".join(current_paragraphs).strip()
        if chunk_text:
            chunks.append({
                "source_file": source_file,
                "page": page,
                "section": current_section,
                "doc_type": doc_type,
                "doc_title": doc_title,
                "chunk_index": chunk_idx,
                "content": chunk_text,
                "tokens": count_tokens(chunk_text)
            })

    return chunks
