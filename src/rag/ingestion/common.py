"""Common text processing and hierarchy building functions shared across ingestion modules."""
import re
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple


def clean_text(text: str):
    """Clean text by removing special characters and normalizing whitespace."""
    text = re.sub(r"[\u200b\u200c\u200d\u202a\u202b\u202c\u202d\u202e]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{3,}", "", text)  
    return text.strip()


def merge_semantic_lines(lines):
    """Merge lines that semantically belong together (e.g., sentence continuation)."""
    if not lines:
        return lines

    merged = []
    buf = None

    def starts_with_lower_alpha(text: str) -> bool:
        for ch in text.lstrip():
            if ch.isalpha():
                return ch.islower()
        return False

    for ln in lines:
        if buf is None:
            buf = dict(ln)
            continue

        prev_text = buf["text"].rstrip()
        curr_text = ln["text"].lstrip()

        prev_ends_with_full_stop = prev_text.endswith(".")
        curr_starts_lower = starts_with_lower_alpha(curr_text)

        should_merge = curr_starts_lower and not prev_ends_with_full_stop

        if should_merge:
            buf["text"] = prev_text + " " + curr_text
            buf["size"] = max(buf["size"], ln["size"])
        else:
            merged.append(buf)
            buf = dict(ln)

    if buf is not None:
        merged.append(buf)

    return merged


def detect_body_font_size(lines):
    """Detect the most common font size (body text size) from lines."""
    sizes = [round(l["size"], 1) for l in lines]
    freq = Counter(sizes)
    body_size, _ = max(freq.items(), key=lambda kv: kv[1])
    return body_size


def filter_small_fonts(lines, body_size):
    """Filter out lines with font sizes smaller than body size."""
    return [ln for ln in lines if ln["size"] >= body_size]


def build_fixed_levels(lines, max_levels=5):
    """Build a mapping from font sizes to hierarchy levels using clustering."""
    body_size = detect_body_font_size(lines)
    filtered_lines = filter_small_fonts(lines, body_size)

    if not filtered_lines:
        filtered_lines = lines

    sizes = [round(l["size"], 1) for l in filtered_lines]
    unique_sizes = sorted(set(sizes), reverse=True)

    if len(unique_sizes) <= max_levels:
        return {sz: idx for idx, sz in enumerate(unique_sizes)}

    X = np.array([[s] for s in sizes], dtype=float)
    kmeans = KMeans(n_clusters=max_levels, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)

    size_to_cluster = {}
    for size_val, label in zip(sizes, labels):
        size_to_cluster[size_val] = label

    cluster_max = {}
    for cl in range(max_levels):
        members = [sz for sz in size_to_cluster if size_to_cluster[sz] == cl]
        cluster_max[cl] = max(members)

    ordered = sorted(cluster_max.items(), key=lambda x: -x[1])
    cluster_to_level = {cluster: idx for idx, (cluster, _) in enumerate(ordered)}

    size_to_level = {sz: cluster_to_level[size_to_cluster[sz]] for sz in size_to_cluster}

    return size_to_level


def assign_levels(lines, level_map):
    """Assign hierarchy levels to lines based on font size mapping."""
    max_level = max(level_map.values()) if level_map else 0

    assigned = []
    for ln in lines:
        sz = round(ln["size"], 1)
        lvl = level_map.get(sz, max_level)

        assigned.append({
            "text": ln["text"],
            "level": lvl,
        })

    return assigned


def merge_same_level(items):
    """Merge consecutive items that have the same hierarchy level."""
    merged = []
    buf = None

    for item in items:
        if buf is None:
            buf = dict(item)
            continue

        if buf["level"] == item["level"]:
            buf["text"] += " " + item["text"]
        else:
            merged.append(buf)
            buf = dict(item)

    if buf:
        merged.append(buf)

    return merged


def nest_by_levels(items):
    """Build a nested hierarchy structure from items with levels."""
    root = []
    stack = []

    for item in items:
        node = {"text": item["text"], "level": item["level"], "children": []}

        if not stack:
            root.append(node)
            stack.append(node)
            continue

        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()

        if not stack:
            root.append(node)
        else:
            stack[-1]["children"].append(node)

        stack.append(node)

    return root


def extract_titles_and_chunks(nested: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Extract numbered bullets for titles and generate markdown content.
    
    Returns:
        tuple: (numbered_bullets_string, markdown_string)
    """
    if not nested:
        return "", ""

    # Collect all levels in one pass
    all_levels = set()
    for root in nested:
        stack = [root]
        while stack:
            node = stack.pop()
            all_levels.add(node["level"])
            stack.extend(node.get("children", []))

    if not all_levels:
        return "", ""

    sorted_levels = sorted(all_levels)
    body_level = sorted_levels[-1]
    title_levels = set(sorted_levels[:min(len(sorted_levels) - 1, 3)])
    level_to_heading = {level: idx + 1 if level != body_level else 0 
                       for idx, level in enumerate(sorted_levels)}

    # Build titles tree and generate markdown in single pass
    titles_root = {"title": "<root>", "children": []}
    title_stack: List[Dict[str, Any]] = []
    markdown_lines = []

    def dfs(node: Dict[str, Any]):
        """Single DFS to build titles tree and generate markdown."""
        level = node["level"]
        text = node["text"].strip()
        is_title = level in title_levels

        # Build titles tree
        if is_title and text:
            while title_stack and title_stack[-1]["level"] >= level:
                title_stack.pop()
            
            title_node = {"title": text, "children": [], "level": level}
            if not title_stack:
                titles_root["children"].append(title_node)
            else:
                title_stack[-1].setdefault("children", []).append(title_node)
            title_stack.append(title_node)

        # Generate markdown
        if text:
            heading_level = level_to_heading[level]
            if heading_level == 0:
                markdown_lines.append(text)
            else:
                markdown_lines.append(f"{'#' * heading_level} {text}")

        # Recurse
        for child in node.get("children", []):
            dfs(child)

        # Cleanup title stack
        if is_title and title_stack and title_stack[-1]["level"] == level:
            title_stack.pop()

    # Process all roots
    for root in nested:
        dfs(root)

    # Generate numbered bullets from titles tree
    bullets = []
    def walk_titles(children, numbers):
        for idx, child in enumerate(children, start=1):
            num_str = ".".join(str(n) for n in numbers + [idx]) + "."
            bullets.append(f"{num_str} {child['title']}")
            if child.get("children"):
                walk_titles(child["children"], numbers + [idx])

    walk_titles(titles_root.get("children", []), [])

    return "\n".join(bullets), "\n\n".join(markdown_lines)

