"""Image ingestion module for extracting structured content from images using OCR."""
import pytesseract
from PIL import Image

from .common import (
    clean_text,
    merge_semantic_lines,
    build_fixed_levels,
    assign_levels,
    merge_same_level,
    nest_by_levels,
    extract_titles_and_chunks,
)


def ocr_extract_words(image_path):
    """Run OCR and extract each word with bounding boxes.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        list: List of dictionaries with word information:
            {text, x0, y0, x1, y1, height}
    """
    img = Image.open(image_path)

    # Use TSV output from Tesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=["text"])

    words = []
    for _, row in data.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        x0 = int(row["left"])
        y0 = int(row["top"])
        w = int(row["width"])
        h = int(row["height"])

        words.append({
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x0 + w,
            "y1": y0 + h,
            "height": h
        })

    return words


def group_words_into_lines(words, bucket_size=10):
    """Groups OCR words into lines using vertical tolerance.
    
    Args:
        words: List of word dictionaries from ocr_extract_words
        bucket_size: Vertical tolerance for grouping words into lines
        
    Returns:
        list: Lines with text, size (height), and top position
    """
    lines = {}
    for w in words:
        key = round(w["y0"] / bucket_size)
        lines.setdefault(key, []).append(w)

    results = []
    for key, ws in sorted(lines.items()):
        ws_sorted = sorted(ws, key=lambda x: x["x0"])
        text = " ".join(w["text"] for w in ws_sorted)
        max_h = max(w["height"] for w in ws_sorted)
        top = min(w["y0"] for w in ws_sorted)

        results.append({
            "text": clean_text(text),
            "size": max_h,  # visual font size = height
            "top": top
        })

    # sort lines by top
    results.sort(key=lambda l: l["top"])
    return results


def image_to_json(image_path, max_levels=5):
    """Extract content from image and return titles and markdown.
    
    Uses similar approach to PDF extraction:
    1. Extract words using OCR
    2. Group words into lines
    3. Merge semantic lines
    4. Detect body font size and build level mapping
    5. Assign levels, merge, nest
    6. Extract titles and markdown
    
    Args:
        image_path: Path to the image file
        max_levels: Maximum number of hierarchy levels to detect
        
    Returns:
        dict: Dictionary with 'titles' and 'markdown' keys
    """
    words = ocr_extract_words(image_path)
    lines = group_words_into_lines(words)
    
    if not lines:
        return {
            "titles": "",
            "markdown": "",
        }
    
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
