"""PDF ingestion module for extracting structured content from PDF files."""
import pdfplumber
from collections import defaultdict
from typing import List, Dict

from .common import (
    clean_text,
    merge_semantic_lines,
    build_fixed_levels,
    assign_levels,
    merge_same_level,
    nest_by_levels,
    extract_titles_and_chunks,
)


def extract_lines(pdf_path, bucket_size=4):
    """Extract text lines from PDF with font size and position information."""
    lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            chars = page.chars
            if not chars:
                continue

            buckets = defaultdict(list)
            for ch in chars:
                key = round(float(ch["top"]) / bucket_size)
                buckets[key].append(ch)

            for _, chs in sorted(buckets.items(), key=lambda kv: kv[0]):
                chs = sorted(chs, key=lambda c: float(c["x0"]))

                raw = "".join(c["text"] for c in chs)

                text = clean_text(raw)
                if not text:
                    continue

                sizes = [float(c["size"]) for c in chs]
                tops = [float(c["top"]) for c in chs]

                lines.append(
                    {
                        "text": text,
                        "size": max(sizes),
                        "top": min(tops),
                    }
                )

    lines.sort(key=lambda l: l["top"])
    return lines


def pdf_to_json(pdf_path, max_levels=5):
    """Extract structured content from PDF and return titles and markdown.
    
    Args:
        pdf_path: Path to the PDF file
        max_levels: Maximum number of hierarchy levels to detect
        
    Returns:
        dict: Dictionary with 'titles' and 'markdown' keys
    """
    lines = extract_lines(pdf_path)
    lines = merge_semantic_lines(lines)
    level_map = build_fixed_levels(lines, max_levels=max_levels)
    assigned = assign_levels(lines, level_map)
    merged = merge_same_level(assigned)
    nested = nest_by_levels(merged)
    titles_bullets, markdown_content = extract_titles_and_chunks(nested)
    return {
        "titles": titles_bullets,
        "markdown": markdown_content,
    }
