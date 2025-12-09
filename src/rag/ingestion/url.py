"""URL/HTML ingestion module for extracting structured content from web pages."""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

from .common import (
    clean_text,
    merge_semantic_lines,
    build_fixed_levels,
    assign_levels,
    merge_same_level,
    nest_by_levels,
    extract_titles_and_chunks,
)


def fetch_html(url):
    """Fetch HTML content from URL.
    
    Args:
        url: URL to fetch
        
    Returns:
        tuple: (BeautifulSoup object, base_url)
    """
    res = requests.get(url, timeout=15, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser"), url


def parse_css_rules(css_text: str) -> list:
    """Parse CSS text into list of (selector, properties) tuples.
    
    Args:
        css_text: Raw CSS text
        
    Returns:
        list: List of (selector, properties) tuples
    """
    rules = []
    # Remove comments
    css_text = re.sub(r'/\*.*?\*/', '', css_text, flags=re.DOTALL)
    
    # Split by } to get individual rules
    for rule_block in css_text.split('}'):
        if not rule_block.strip():
            continue
        
        # Extract selector and properties
        parts = rule_block.split('{', 1)
        if len(parts) != 2:
            continue
        
        selector = parts[0].strip()
        properties = parts[1].strip()
        
        if selector and properties:
            rules.append((selector, properties))
    
    return rules


def fetch_stylesheets(soup, base_url: str) -> str:
    """Fetch all external stylesheets and combine with inline styles.
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL for resolving relative stylesheet URLs
        
    Returns:
        str: Combined CSS text
    """
    css_text = ""
    
    # Get inline styles
    for style_tag in soup.find_all('style'):
        css_text += style_tag.string or ""
    
    # Fetch linked stylesheets
    for link in soup.find_all('link', rel='stylesheet'):
        href = link.get('href')
        if not href:
            continue
        
        try:
            # Resolve relative URLs
            stylesheet_url = urljoin(base_url, href)
            res = requests.get(stylesheet_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if res.status_code == 200:
                css_text += "\n" + res.text
        except Exception:
            # Skip if stylesheet can't be fetched
            continue
    
    return css_text


def build_css_cache(soup, base_url: str) -> dict:
    """Build a cache of CSS rules indexed by selector.
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL for resolving relative stylesheet URLs
        
    Returns:
        dict: CSS cache mapping selectors to font-size and font-weight
    """
    css_text = fetch_stylesheets(soup, base_url)
    rules = parse_css_rules(css_text)
    
    css_cache = {}
    for selector, properties in rules:
        # Normalize selector (remove whitespace)
        selector = re.sub(r'\s+', ' ', selector.strip())
        
        # Extract font-size and font-weight
        font_size = parse_font_size(properties)
        font_weight = get_font_weight(properties)
        
        if font_size is not None or font_weight != 400:
            if selector not in css_cache:
                css_cache[selector] = {}
            if font_size is not None:
                css_cache[selector]['font-size'] = font_size
            css_cache[selector]['font-weight'] = font_weight
    
    return css_cache


def match_css_selector(selector: str, tag) -> bool:
    """Check if a CSS selector matches the given tag.
    
    Args:
        selector: CSS selector string
        tag: BeautifulSoup tag object
        
    Returns:
        bool: True if selector matches tag
    """
    selector = selector.strip()
    
    # Simple tag selector (e.g., "h1", "p")
    if selector == tag.name:
        return True
    
    # Class selector (e.g., ".class-name")
    if selector.startswith('.'):
        class_name = selector[1:].split()[0]  # Take first class, ignore pseudo-classes
        classes = tag.get('class', [])
        if isinstance(classes, list):
            return class_name in classes
        return class_name == classes
    
    # ID selector (e.g., "#id-name")
    if selector.startswith('#'):
        id_name = selector[1:].split()[0]
        return tag.get('id') == id_name
    
    # Tag.class selector (e.g., "div.content")
    if '.' in selector and not selector.startswith('.'):
        tag_part, class_part = selector.split('.', 1)
        class_name = class_part.split()[0]
        if tag.name == tag_part:
            classes = tag.get('class', [])
            if isinstance(classes, list):
                return class_name in classes
            return class_name == classes
    
    # Tag#id selector (e.g., "div#main")
    if '#' in selector and not selector.startswith('#'):
        tag_part, id_part = selector.split('#', 1)
        id_name = id_part.split()[0]
        return tag.name == tag_part and tag.get('id') == id_name
    
    return False


def get_computed_font_size(tag, css_cache: dict) -> tuple:
    """Get computed font-size and font-weight from CSS cache.
    
    Args:
        tag: BeautifulSoup tag object
        css_cache: CSS cache dictionary
        
    Returns:
        tuple: (font_size, font_weight) or (None, 400) if not found
    """
    best_match = None
    best_specificity = -1
    
    # Calculate specificity: inline > id > class > tag
    def calculate_specificity(selector: str) -> int:
        specificity = 0
        if '#' in selector:
            specificity += 1000  # ID
        if '.' in selector or selector.startswith('.'):
            specificity += 100   # Class
        if any(c.isalpha() for c in selector if c not in '.#'):
            specificity += 10    # Tag
        return specificity
    
    # Check all CSS rules for matches
    for selector, styles in css_cache.items():
        if match_css_selector(selector, tag):
            specificity = calculate_specificity(selector)
            if specificity > best_specificity:
                best_specificity = specificity
                best_match = styles
    
    if best_match:
        return best_match.get('font-size'), best_match.get('font-weight', 400)
    
    return None, 400


def parse_font_size(style_str: str) -> float:
    """Parse font-size from CSS style string. Returns size in pixels.
    
    Args:
        style_str: CSS style string
        
    Returns:
        float: Font size in pixels, or None if not found
    """
    if not style_str:
        return None
    
    # Look for font-size in style string
    match = re.search(r'font-size:\s*([\d.]+)(px|em|rem|pt)?', style_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or 'px').lower()
        
        # Convert to pixels (approximate)
        if unit == 'px':
            return value
        elif unit == 'pt':
            return value * 1.33  # 1pt ≈ 1.33px
        elif unit == 'em' or unit == 'rem':
            return value * 16  # Assume base font size of 16px
        else:
            return value
    
    return None


def get_font_weight(style_str: str) -> int:
    """Extract font-weight from style string. Returns numeric weight.
    
    Args:
        style_str: CSS style string
        
    Returns:
        int: Font weight (400 for normal, 700 for bold, etc.)
    """
    if not style_str:
        return 400
    
    match = re.search(r'font-weight:\s*(\d+|bold|normal|bolder|lighter)', style_str, re.IGNORECASE)
    if match:
        weight = match.group(1).lower()
        if weight == 'bold' or weight == 'bolder':
            return 700
        elif weight == 'normal' or weight == 'lighter':
            return 400
        else:
            return int(weight)
    
    return 400


def extract_lines_from_html(soup, base_url: str):
    """Extract text lines with font size information from HTML.
    
    Similar to PDF extraction, but extracts from HTML elements.
    Uses CSS stylesheets, inline styles, semantic tags, and element hierarchy.
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL for resolving relative stylesheet URLs
        
    Returns:
        list: Lines with text, size, and top position
    """
    # Build CSS cache from stylesheets and style tags (before removing them)
    css_cache = build_css_cache(soup, base_url)
    
    # Remove non-content elements
    for element in soup.find_all(['nav', 'footer', 'header', 'aside', 'script', 'style', 'noscript']):
        element.decompose()
    
    lines = []
    
    # Get all text-containing elements (block and inline-block)
    # Focus on common content elements
    content_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 
                   'article', 'section', 'main', 'li', 'td', 'th']
    
    # Track position for ordering
    position = 0
    
    for tag in soup.find_all(content_tags):
        # Skip if this element contains block-level children (to avoid duplicate extraction)
        has_block_children = any(child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section'] 
                                 for child in tag.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section'], recursive=False))
        if has_block_children:
            continue
        
        text = tag.get_text(strip=True)
        # Skip very short text or common non-content patterns
        if not text or len(text) < 3:
            continue
        if text.lower() in ['skip to content', 'menu', 'search', 'login', 'sign up', 'subscribe']:
            continue
        
        # Priority order: inline style > CSS cache > semantic tags
        style_str = tag.get('style', '')
        font_size = parse_font_size(style_str)
        font_weight = get_font_weight(style_str)
        
        # If no inline style, check CSS cache
        if font_size is None:
            css_font_size, css_font_weight = get_computed_font_size(tag, css_cache)
            if css_font_size is not None:
                font_size = css_font_size
            if font_weight == 400:  # Only override if not set by inline
                font_weight = css_font_weight
        
        # If still no font-size, use semantic hints
        if font_size is None:
            tag_name = tag.name.lower()
            if tag_name.startswith('h'):
                # Headings: h1=24px, h2=20px, h3=18px, h4=16px, h5=14px, h6=12px
                level = int(tag_name[1])
                font_size = 28 - (level * 2)
                if font_weight == 400:  # Only set if not already set
                    font_weight = 700 if level <= 2 else 600
            elif tag_name in ['p', 'div', 'span', 'li', 'td', 'th']:
                # Default body text size
                font_size = 16
                font_weight = 400
            else:
                font_size = 16
                font_weight = 400
        
        # Use font-weight as size modifier (heavier = larger perceived size)
        # Add small adjustment based on weight
        size_adjustment = (font_weight - 400) / 100  # -4 to +3 adjustment
        adjusted_size = font_size + size_adjustment
        
        lines.append({
            "text": clean_text(text),
            "size": adjusted_size,
            "top": position,  # Use position for ordering
        })
        position += 1
    
    # Sort by position (DOM order)
    lines.sort(key=lambda l: l["top"])
    return lines


def webpage_to_json(url, max_levels=5):
    """Extract content from webpage and return titles and markdown.
    
    Uses similar approach to PDF extraction:
    1. Extract text lines with font size information (from CSS, inline styles, semantic tags)
    2. Merge semantic lines
    3. Detect body font size and build level mapping
    4. Assign levels, merge, nest
    5. Extract titles and markdown
    
    Args:
        url: URL to extract content from
        max_levels: Maximum number of hierarchy levels to detect
        
    Returns:
        dict: Dictionary with 'titles' and 'markdown' keys
    """
    soup, base_url = fetch_html(url)
    lines = extract_lines_from_html(soup, base_url)
    
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
