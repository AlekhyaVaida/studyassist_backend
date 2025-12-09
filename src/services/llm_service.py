"""LLM service for generating pages from documents."""

import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor

from src.config import config

# Maximum words per page
MAX_WORDS_PER_PAGE = 8000


def generate_notebook_metadata(
    documents_content: List[Dict[str, str]],
    toc_items: List[str],
    user_prompt: Optional[str] = None
) -> Dict[str, str]:
    """Generate notebook title and description from documents and TOC using LLM.
    
    Args:
        documents_content: List of dicts with 'titles' and 'markdown' keys
        toc_items: List of table of contents items
        user_prompt: Optional user prompt/query
        
    Returns:
        dict: Dictionary with 'title' and 'description' keys
    """
    if not config.validate_api_key():
        raise ValueError("OpenAI API key is not configured")
    
    client = OpenAI(api_key=config.openai_api_key)
    
    # Combine all markdown content (limit to avoid token limits)
    combined_markdown = "\n\n---\n\n".join([
        f"# {doc.get('titles', 'Document')}\n\n{doc.get('markdown', '')[:5000]}"
        for doc in documents_content
    ])
    
    # Combine TOC items
    toc_text = "\n".join([f"- {item}" for item in toc_items])
    
    system_prompt = """You are an assistant that generates appropriate titles and descriptions for notebooks based on document content.

Your task is to:
1. Analyze the provided document content and table of contents
2. Generate a concise, descriptive title (3-8 words)
3. Generate a SHORT one-sentence description that summarizes what the notebook contains

Return a JSON object with this structure:
{
  "title": "Concise Notebook Title",
  "description": "One short sentence describing what this notebook contains."
}

The title should be:
- Concise and descriptive (3-8 words)
- Reflect the main topic or theme
- Professional and clear

The description should be:
- Exactly ONE sentence
- Short and concise (under 20 words)
- Describe what topics/content the notebook covers
- Focus on the actual content, not the user's query or purpose"""

    user_prompt_text = f"""Analyze the following documents and table of contents to generate an appropriate notebook title and description.

Table of Contents:
{toc_text}

Document Content Summary:
{combined_markdown[:8000]}

Generate:
1. A concise title (3-8 words)
2. A SHORT one-sentence description (under 20 words) that describes what topics and content this notebook contains

The description should ONLY describe what the documents actually contain - what topics, subjects, or content are covered in the notebook."""

    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_text}
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
            timeout=60.0  # 1 minute timeout for metadata generation
        )
        
        result = response.choices[0].message.content
        metadata = json.loads(result)
        
        if "title" in metadata and "description" in metadata:
            return {
                "title": metadata["title"],
                "description": metadata["description"]
            }
        else:
            # Fallback: use first document title
            first_title = documents_content[0].get("titles", "").split("\n")[0] if documents_content else "Untitled Notebook"
            return {
                "title": first_title[:50] if first_title else "Untitled Notebook",
                "description": f"Notebook containing {len(toc_items)} sections about {first_title}"
            }
            
    except Exception as e:
        print(f"Error generating notebook metadata: {e}")
        # Fallback
        first_title = documents_content[0].get("titles", "").split("\n")[0] if documents_content else "Untitled Notebook"
        return {
            "title": first_title[:50] if first_title else "Untitled Notebook",
            "description": f"Notebook with {len(toc_items)} sections"
        }


# Common non-content sections to exclude from TOC
EXCLUDED_SECTIONS = [
    "references",
    "reference",
    "bibliography",
    "glossary",
    "appendix",
    "appendices",
    "index",
    "table of contents",
    "contents",
    "toc",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "abstract",
    "preface",
    "foreword",
    "introduction to",
    "about the author",
    "author information",
    "copyright",
    "legal notice",
    "disclaimer"
]


def count_words_in_tiptap(content: Dict[str, Any]) -> int:
    """Count words in TipTap JSON content.
    
    Args:
        content: TipTap JSON content dictionary
        
    Returns:
        int: Word count
    """
    word_count = 0
    
    def extract_text(node: Any) -> None:
        nonlocal word_count
        if isinstance(node, dict):
            # Extract text from text nodes
            if node.get("type") == "text" and "text" in node:
                text = node["text"]
                word_count += len(text.split())
            # Recursively process content arrays
            if "content" in node and isinstance(node["content"], list):
                for child in node["content"]:
                    extract_text(child)
        elif isinstance(node, list):
            for item in node:
                extract_text(item)
    
    extract_text(content)
    return word_count


def truncate_tiptap_content(content: Dict[str, Any], max_words: int) -> Dict[str, Any]:
    """Truncate TipTap JSON content to fit within word limit.
    
    Args:
        content: TipTap JSON content dictionary
        max_words: Maximum word count
        
    Returns:
        dict: Truncated TipTap JSON content
    """
    if not isinstance(content, dict) or content.get("type") != "doc":
        return content
    
    word_count = count_words_in_tiptap(content)
    
    if word_count <= max_words:
        return content
    
    # Truncate by removing content nodes until we're under the limit
    if "content" not in content or not isinstance(content["content"], list):
        return content
    
    truncated_content = {
        "type": "doc",
        "content": []
    }
    
    current_word_count = 0
    
    for node in content["content"]:
        node_words = count_words_in_tiptap(node)
        
        if current_word_count + node_words <= max_words:
            truncated_content["content"].append(node)
            current_word_count += node_words
        else:
            # Try to truncate this node if it's a paragraph
            if isinstance(node, dict) and node.get("type") == "paragraph":
                if "content" in node and isinstance(node["content"], list):
                    truncated_para = {
                        "type": "paragraph",
                        "content": []
                    }
                    para_word_count = 0
                    
                    for text_node in node["content"]:
                        if isinstance(text_node, dict) and text_node.get("type") == "text":
                            text = text_node.get("text", "")
                            words = text.split()
                            remaining_words = max_words - current_word_count - para_word_count
                            
                            if len(words) <= remaining_words:
                                truncated_para["content"].append(text_node)
                                para_word_count += len(words)
                            else:
                                # Truncate text
                                truncated_words = words[:remaining_words]
                                truncated_text = " ".join(truncated_words)
                                truncated_para["content"].append({
                                    "type": "text",
                                    "text": truncated_text + "..."
                                })
                                truncated_content["content"].append(truncated_para)
                                break
                    
                    if truncated_para["content"]:
                        truncated_content["content"].append(truncated_para)
                        current_word_count += para_word_count
            
            # Stop if we've reached the limit
            if current_word_count >= max_words:
                break
    
    return truncated_content


def clean_section_title(title: str) -> str:
    """Remove chapter numbers, topic numbers, and numbering prefixes from section titles.
    
    Args:
        title: Section title to clean
        
    Returns:
        str: Cleaned title without numbers
    """
    import re
    
    # Remove common numbering patterns:
    # "Chapter 1: Title" -> "Title"
    # "1. Title" -> "Title"
    # "1.1 Title" -> "Title"
    # "Chapter 1 Title" -> "Title"
    # "Section 1.1 Title" -> "Title"
    
    cleaned = title.strip()
    
    # Remove "Chapter X:" or "Chapter X " patterns
    cleaned = re.sub(r'^Chapter\s+\d+[:\s\.-]+', '', cleaned, flags=re.IGNORECASE)
    
    # Remove "Section X.X:" or "Section X.X " patterns
    cleaned = re.sub(r'^Section\s+\d+\.?\d*\.?\d*[:\s\.-]+', '', cleaned, flags=re.IGNORECASE)
    
    # Remove "Topic X.X.X:" patterns
    cleaned = re.sub(r'^Topic\s+\d+\.?\d*\.?\d*\.?\d*[:\s\.-]+', '', cleaned, flags=re.IGNORECASE)
    
    # Remove "ID: X.X" or "ID X.X" patterns
    cleaned = re.sub(r'^ID\s*[:\s]+\d+\.?\d*\.?\d*[:\s\.-]+', '', cleaned, flags=re.IGNORECASE)
    
    # Remove leading numbers with dots: "1. ", "1.1 ", "1.1.1 ", etc.
    cleaned = re.sub(r'^\d+\.\d*\.?\d*\.?\s*', '', cleaned)
    
    # Remove leading numbers with colon: "1: ", "1.1: ", etc.
    cleaned = re.sub(r'^\d+\.?\d*\.?\d*:\s*', '', cleaned)
    
    # Remove leading numbers with dash: "1 - ", "1.1 - ", etc.
    cleaned = re.sub(r'^\d+\.?\d*\.?\d*\s*-\s*', '', cleaned)
    
    # Remove leading numbers with parentheses: "(1)", "(1.1)", etc.
    cleaned = re.sub(r'^\(\d+\.?\d*\.?\d*\)\s*', '', cleaned)
    
    # Remove leading numbers with space: "1 Title" -> "Title" (but keep if it's part of the title)
    # Only if it's at the very start
    cleaned = re.sub(r'^\d+\.?\d*\.?\d*\s+', '', cleaned)
    
    # Remove any remaining leading/trailing punctuation from cleaning
    cleaned = cleaned.strip(' :.-')
    
    return cleaned.strip()


def should_exclude_section(title: str) -> bool:
    """Check if a section title should be excluded from page generation.
    
    Args:
        title: Section title to check
        
    Returns:
        bool: True if section should be excluded
    """
    title_lower = title.lower().strip()
    
    # Check if title contains any excluded keywords
    for excluded in EXCLUDED_SECTIONS:
        if excluded in title_lower:
            return True
    
    # Check for common patterns
    if title_lower.startswith("list of"):
        return True
    
    return False


def extract_table_of_contents(
    documents_content: List[Dict[str, str]]
) -> List[str]:
    """Extract table of contents from documents using LLM.
    
    Args:
        documents_content: List of dicts with 'titles' and 'markdown' keys
        notebook_title: Title of the notebook
        notebook_description: Optional description of the notebook
        
    Returns:
        list: List of TOC item titles
    """
    if not config.validate_api_key():
        raise ValueError("OpenAI API key is not configured")
    
    client = OpenAI(api_key=config.openai_api_key)
    
    # Combine all markdown content
    combined_markdown = "\n\n---\n\n".join([
        f"# {doc.get('titles', 'Document')}\n\n{doc.get('markdown', '')}"
        for doc in documents_content
    ])
    
    system_prompt = """You are an assistant that extracts table of contents from documents.

Your task is to:
1. Analyze the provided markdown content
2. Extract ONLY the most important major sections and chapters
3. Group related subsections together into main sections
4. EXCLUDE non-content sections like: References, Bibliography, Glossary, Appendix, Index, Table of Contents, Acknowledgments, Abstract, Preface, etc.
5. Return a concise list of high-level topics (aim for 5-15 main sections maximum)
6. IMPORTANT: Remove ALL numbers, chapter identifiers, section IDs, and numbering from titles

Return a JSON object with this structure:
{
  "toc": [
    "Main Section Title",
    "Another Section Title",
    "Third Section Title",
    ...
  ]
}

The TOC should include:
- Only major chapters/sections (H1, H2 level headings)
- Group related subsections together
- Avoid listing every small subsection
- EXCLUDE: References, Bibliography, Glossary, Appendix, Index, Table of Contents, Acknowledgments, Abstract, Preface, Foreword, etc.
- DO NOT include: Chapter numbers (e.g., "Chapter 1", "1.1"), section IDs, topic numbers, or any numbering
- Use clean, descriptive titles without any numbering or identifiers
- Aim for 5-15 comprehensive sections that cover the main content
- Each section should be substantial enough to warrant its own page"""

    user_prompt = f"""Extract the table of contents from the following document content.

IMPORTANT: 
- Extract ONLY the major sections/chapters. Group related subsections together.
- Aim for 5-15 comprehensive sections maximum. Do NOT list every small subsection.
- EXCLUDE non-content sections: References, Bibliography, Glossary, Appendix, Index, Table of Contents, Acknowledgments, Abstract, Preface, Foreword, "List of Figures", "List of Tables", etc.
- Only include sections with actual educational/content value.
- CRITICAL: Remove ALL numbers, chapter identifiers, section IDs, and numbering from titles
- Do NOT include: "Chapter 1", "1.1", "Section 2.3", "Topic 1.2.3", or any similar numbering
- Use clean, descriptive titles like "Introduction" instead of "Chapter 1: Introduction"
- The page order provides sequence, so numbers are unnecessary

Document Content:
{combined_markdown[:10000]}  # Limit to avoid token limits

Return a JSON object with a "toc" array containing ONLY the major content sections (5-15 items maximum).
Each section should be substantial enough to contain comprehensive content.
EXCLUDE all reference, appendix, glossary, and metadata sections.
ALL titles must be clean and free of numbers, chapter identifiers, or section IDs."""

    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent TOC extraction
            response_format={"type": "json_object"},
            timeout=120.0  # 2 minutes timeout for TOC extraction
        )
        
        result = response.choices[0].message.content
        toc_data = json.loads(result)
        
        if "toc" in toc_data and isinstance(toc_data["toc"], list):
            toc_items = toc_data["toc"]
            # Filter out empty items and excluded sections, then clean numbers
            toc_items = [
                clean_section_title(item.strip())
                for item in toc_items 
                if item.strip() and not should_exclude_section(item)
            ]
            # Remove duplicates after cleaning
            seen = set()
            unique_toc_items = []
            for item in toc_items:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    unique_toc_items.append(item)
            return unique_toc_items if unique_toc_items else extract_toc_from_markdown(combined_markdown)
        else:
            # Fallback: extract headings from markdown
            return extract_toc_from_markdown(combined_markdown)
            
    except Exception as e:
        print(f"Error extracting TOC with LLM: {e}")
        # Fallback: extract headings from markdown
        return extract_toc_from_markdown(combined_markdown)


def extract_toc_from_markdown(markdown: str) -> List[str]:
    """Extract table of contents from markdown by parsing headings.
    Only extracts major headings (H1, H2) and groups related sections.
    
    Args:
        markdown: Markdown string
        
    Returns:
        list: List of major heading titles (limited to 15)
    """
    toc = []
    lines = markdown.split("\n")
    seen_headings = set()
    
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # Only extract H1 and H2 headings (not deeper levels)
            level = len(line) - len(line.lstrip("#"))
            if level <= 2:  # Only H1 and H2
                heading_text = line.lstrip("#").strip()
                # Skip excluded sections
                if heading_text and not should_exclude_section(heading_text):
                    # Clean numbers from heading
                    cleaned_heading = clean_section_title(heading_text)
                    # Normalize and deduplicate
                    normalized = cleaned_heading.lower().strip()
                    if normalized not in seen_headings and cleaned_heading:
                        seen_headings.add(normalized)
                        toc.append(cleaned_heading)
                        
                        # Limit to 15 major sections
                        if len(toc) >= 15:
                            break
    
    # If we have too many, try to combine related ones
    if len(toc) > 15:
        # Group by first word or common prefix
        grouped = []
        current_group = []
        for heading in toc:
            if not current_group or len(current_group) < 3:
                current_group.append(heading)
            else:
                # Combine group into one section
                grouped.append(" / ".join(current_group[:2]))  # Take first 2 as combined title
                current_group = [heading]
        if current_group:
            grouped.append(" / ".join(current_group[:2]))
        toc = grouped[:15]  # Limit to 15
    
    return toc if toc else ["Introduction", "Main Content", "Conclusion"]


def generate_page_for_toc_item(
    toc_item: str,
    documents_content: List[Dict[str, str]],
    notebook_title: str,
    notebook_description: Optional[str] = None,
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    notebook_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a comprehensive page for a single TOC item.
    
    Args:
        toc_item: Title of the TOC item/section
        documents_content: List of dicts with 'titles' and 'markdown' keys
        notebook_title: Title of the notebook
        notebook_description: Optional description of the notebook
        query: Optional query/prompt to guide content generation
        
    Returns:
        dict: Page dictionary with 'title' and 'content' (TipTap JSON)
    """
    if not config.validate_api_key():
        raise ValueError("OpenAI API key is not configured")
    
    client = OpenAI(api_key=config.openai_api_key)
    
    # Combine all markdown content
    combined_markdown = "\n\n---\n\n".join([
        f"# {doc.get('titles', 'Document')}\n\n{doc.get('markdown', '')}"
        for doc in documents_content
    ])
    
    system_prompt = f"""You are an assistant that generates comprehensive page content using RAG-retrieved context and document content.

Your task:
1. Use the RAG-retrieved context (from user's documents) as the primary source
2. Supplement with the original document section
3. Generate comprehensive, detailed content for the section
4. Output the content in Markdown format
5. IMPORTANT: Generate as much content as possible, up to approximately {MAX_WORDS_PER_PAGE} words
6. DO NOT include chapter numbers, section numbers, or topic numbers in headings or content

Return a JSON object with this structure:
{{
  "title": "Section Title (without numbers)",
  "markdown": "# Section Title\\n\\nYour markdown content here..."
}}

Markdown format - use these elements:
- Headings: # H1, ## H2, ### H3 (only these three levels)
- Bold: **bold text**
- Italics: *italic text*
- Links: [link text](url)
- Unordered lists: - item or * item
- Numbered lists: 1. item, 2. item, etc.
- Code snippets: `inline code` or ```language\ncode\n```
- Checkboxes: - [ ] unchecked or - [x] checked
- Paragraphs: Regular text (separate paragraphs with blank lines)

DO NOT use:
- Chapter numbers (e.g., "Chapter 1", "1.1", "1.2.3")
- Section numbers or topic numbers in headings
- Blockquotes
- Tables
- Images
- Any other markdown elements

IMPORTANT: Remove all numbering from headings and content. The page order already provides sequence.
Generate comprehensive content using the RAG context. Include details, examples, and explanations.
Generate as much content as possible while staying under {MAX_WORDS_PER_PAGE} words."""

    # Extract relevant section from markdown (faster than RAG query)
    # Find the section in the markdown that matches this TOC item
    relevant_content = combined_markdown
    toc_lower = toc_item.lower()
    
    # Try to find the section in markdown
    lines = combined_markdown.split("\n")
    section_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("#") and toc_lower in line.lower():
            section_start = i
            break
    
    # Extract relevant section (next 200 lines or until next major heading)
    if section_start >= 0:
        section_lines = []
        for i in range(section_start, min(section_start + 200, len(lines))):
            line = lines[i]
            # Stop at next H1 or H2 heading (unless it's the first one)
            if i > section_start and line.strip().startswith("#") and len(line.strip()) - len(line.strip().lstrip("#")) <= 2:
                break
            section_lines.append(line)
        relevant_content = "\n".join(section_lines)
    else:
        # If section not found, use first 5000 chars
        relevant_content = combined_markdown[:5000]
    
    user_prompt = f"""Generate comprehensive markdown content for this section: "{toc_item}"

Content:
{relevant_content[:4000]}

Extract and expand ALL content related to "{toc_item}" using the provided context.
Generate comprehensive markdown content with:
- Headings (# H1, ## H2, ### H3 only) - WITHOUT chapter numbers or section numbers
- Bold (**text**)
- Italics (*text*)
- Links ([text](url))
- Unordered lists (- item or * item)
- Numbered lists (1. item, 2. item, etc.)
- Code snippets (`inline code` or ```language\ncode\n```)
- Checkboxes (- [ ] unchecked or - [x] checked)
- Paragraphs

IMPORTANT: 
- Remove all chapter numbers, section numbers, and topic numbers from headings and content
- Do NOT use "Chapter 1", "1.1", "Section 2.3", etc. anywhere
- The page order already provides sequence, so numbers are not needed
- Use clean titles like "Introduction" instead of "Chapter 1: Introduction"

Generate as much detailed content as possible, up to {MAX_WORDS_PER_PAGE} words.
Include examples, explanations, code snippets, and comprehensive details."""

    try:
        # Add timeout to prevent hanging (4 minutes should be enough for most responses)
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for faster, more deterministic responses
            response_format={"type": "json_object"},
            timeout=240.0  # 4 minutes timeout to prevent hanging
        )
        
        result = response.choices[0].message.content
        page_data = json.loads(result)
        
        # Validate response - expect markdown format
        if isinstance(page_data, dict) and "title" in page_data and "markdown" in page_data:
            title = page_data["title"]
            # Clean title from numbers
            title = clean_section_title(title)
            markdown_content = page_data["markdown"]
            
            # Remove numbers from markdown headings before conversion
            import re
            lines = markdown_content.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith("#"):
                    # Extract heading level and text
                    heading_prefix = line[:len(line) - len(line.lstrip('#'))]
                    heading_text = line.lstrip("#").strip()
                    # Clean numbers from heading text
                    cleaned_text = clean_section_title(heading_text)
                    # Reconstruct heading line
                    cleaned_line = heading_prefix + " " + cleaned_text
                    cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append(line)
            markdown_content = "\n".join(cleaned_lines)
            
            # Convert markdown to TipTap JSON
            tiptap_content = markdown_to_tiptap(markdown_content)
            
            # Check word count and truncate if needed
            word_count = count_words_in_tiptap(tiptap_content)
            if word_count > MAX_WORDS_PER_PAGE:
                print(f"Content exceeds {MAX_WORDS_PER_PAGE} words ({word_count} words), truncating...")
                tiptap_content = truncate_tiptap_content(tiptap_content, MAX_WORDS_PER_PAGE)
                final_word_count = count_words_in_tiptap(tiptap_content)
                print(f"Truncated to {final_word_count} words")
            
            return {
                "title": title,
                "content": tiptap_content
            }
        
        # Invalid format, use fallback
        raise ValueError(f"Invalid page format: {page_data}")
            
    except Exception as e:
        print(f"Error generating page for '{toc_item}': {e}")
        # Fallback: create simple page from markdown
        return create_fallback_page_for_toc(toc_item, documents_content)


def create_fallback_page_for_toc(
    toc_item: str,
    documents_content: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Create a simple page for TOC item as fallback.
    
    Args:
        toc_item: Title of the TOC item
        documents_content: List of dicts with 'titles' and 'markdown' keys
        
    Returns:
        dict: Page dictionary
    """
    # Find relevant markdown content
    combined_markdown = "\n\n".join([
        doc.get("markdown", "")
        for doc in documents_content
    ])
    
    # Simple markdown to TipTap conversion
    content = markdown_to_tiptap(combined_markdown)
    
    return {
        "title": toc_item[:100] if toc_item else "Untitled Page",
        "content": content
    }


async def generate_pages_from_documents(
    documents_content: List[Dict[str, str]],
    notebook_title: str,
    notebook_description: Optional[str] = None,
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """Generate pages from processed documents using LLM (parallelized).
    
    Process:
    1. Extract table of contents from documents
    2. For each TOC item, generate a comprehensive page in parallel
    
    Args:
        documents_content: List of dicts with 'titles' and 'markdown' keys
        notebook_title: Title of the notebook (LLM-generated)
        notebook_description: Description of the notebook (LLM-generated)
        query: Optional query/prompt to guide page generation
        user_id: User ID for RAG queries
        notebook_id: Notebook ID for RAG queries
        max_workers: Maximum number of concurrent page generations
        
    Returns:
        list: List of page dictionaries with 'title' and 'content' (TipTap JSON)
    """
    if not documents_content:
        return []
    
    # Step 1: Extract table of contents
    print(f"Extracting table of contents...")
    toc_items = extract_table_of_contents(documents_content)
    
    if not toc_items:
        # Fallback: create a single page
        return create_fallback_pages(documents_content)
    
    print(f"Found {len(toc_items)} TOC items. Generating pages in parallel (max {max_workers} concurrent)...")
    
    # Step 2: Generate pages in parallel using asyncio with timeout and retry
    async def generate_single_page(toc_item: str, index: int, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """Generate a single page asynchronously with timeout and retry logic."""
        max_retries = 2
        print(f"Generating page {index + 1}/{len(toc_items)}: {toc_item}" + (f" (retry {retry_count + 1}/{max_retries})" if retry_count > 0 else ""))
        
        try:
            # Run sync function in thread pool with timeout
            loop = asyncio.get_event_loop()
            # Set timeout to 5 minutes per page (should be more than enough)
            page = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate_page_for_toc_item(
                        toc_item,
                        documents_content,
                        notebook_title,
                        notebook_description,
                        query,
                        user_id=user_id,
                        notebook_id=notebook_id
                    )
                ),
                timeout=300.0  # 5 minutes timeout per page
            )
            if retry_count > 0:
                print(f"Successfully completed page {index + 1}/{len(toc_items)}: {toc_item} after {retry_count} retries")
            else:
                print(f"Completed page {index + 1}/{len(toc_items)}: {toc_item}")
            return page
        except asyncio.TimeoutError:
            if retry_count < max_retries:
                print(f"Timeout generating page {index + 1}/{len(toc_items)}: {toc_item} (attempt {retry_count + 1}/{max_retries + 1}). Retrying...")
                # Wait a bit before retrying (exponential backoff)
                await asyncio.sleep(2 ** retry_count)  # 1s, 2s delay
                # Retry the generation
                return await generate_single_page(toc_item, index, retry_count + 1)
            else:
                print(f"Timeout generating page {index + 1}/{len(toc_items)}: {toc_item} (exceeded {max_retries + 1} attempts, giving up)")
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:
            if retry_count < max_retries:
                print(f"Error generating page for '{toc_item}' (attempt {retry_count + 1}/{max_retries + 1}): {e}. Retrying...")
                # Wait a bit before retrying (exponential backoff)
                await asyncio.sleep(2 ** retry_count)  # 1s, 2s delay
                # Retry the generation
                return await generate_single_page(toc_item, index, retry_count + 1)
            else:
                print(f"Error generating page for '{toc_item}' (exceeded {max_retries + 1} attempts): {e}")
                import traceback
                traceback.print_exc()
                return None
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_workers)
    
    async def generate_with_semaphore(toc_item: str, index: int):
        """Generate page with semaphore to limit concurrency."""
        async with semaphore:
            return await generate_single_page(toc_item, index)
    
    # Generate all pages in parallel with overall timeout
    tasks = [
        generate_with_semaphore(toc_item, i)
        for i, toc_item in enumerate(toc_items)
    ]
    
    try:
        # Set overall timeout: 10 minutes per page * number of pages, but cap at 30 minutes
        overall_timeout = min(30 * 60, 10 * 60 * len(toc_items))
        print(f"Starting parallel page generation with {len(tasks)} tasks (overall timeout: {overall_timeout/60:.1f} minutes)")
        
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=overall_timeout
        )
    except asyncio.TimeoutError:
        print(f"WARNING: Overall timeout ({overall_timeout/60:.1f} minutes) exceeded for page generation")
        print("Collecting completed pages...")
        # Try to get any completed results
        results = []
        for task in tasks:
            if task.done():
                try:
                    result = await task
                    results.append(result)
                except asyncio.CancelledError:
                    # Task was cancelled due to timeout
                    results.append(asyncio.TimeoutError("Task timed out"))
                except Exception as e:
                    results.append(e)
                except BaseException as e:
                    # Catch any other base exceptions (including CancelledError)
                    results.append(e)
            else:
                # Task is still running, cancel it
                task.cancel()
                results.append(asyncio.TimeoutError("Task timed out"))
    
    # Filter out None results and exceptions
    pages = []
    for i, result in enumerate(results):
        # Check for TimeoutError first (before generic Exception)
        if isinstance(result, asyncio.TimeoutError):
            print(f"Timeout generating page for '{toc_items[i]}'")
            continue
        if isinstance(result, Exception):
            print(f"Exception generating page for '{toc_items[i]}': {result}")
            continue
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            # Handle BaseException subclasses (like CancelledError) that aren't Exception
            print(f"BaseException generating page for '{toc_items[i]}': {result}")
            continue
        if result is not None:
            pages.append(result)
    
    print(f"Successfully generated {len(pages)}/{len(toc_items)} pages")
    
    # If we got at least some pages, return them. Otherwise use fallback.
    if pages:
        return pages
    else:
        print("No pages generated, using fallback pages")
        return create_fallback_pages(documents_content)


def create_fallback_pages(documents_content: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Create simple pages from markdown as fallback when LLM fails.
    
    Args:
        documents_content: List of dicts with 'titles' and 'markdown' keys
        
    Returns:
        list: List of page dictionaries
    """
    pages = []
    
    for doc in documents_content:
        title = doc.get("titles", "").split("\n")[0] if doc.get("titles") else "Untitled Page"
        markdown = doc.get("markdown", "")
        
        # Simple markdown to TipTap conversion
        content = markdown_to_tiptap(markdown)
        
        pages.append({
            "title": title[:100] if title else "Untitled Page",  # Limit title length
            "content": content
        })
    
    return pages


def parse_inline_formatting(text: str) -> List[Dict[str, Any]]:
    """Parse inline markdown formatting (bold, italics, links) into TipTap nodes.
    
    Args:
        text: Text with markdown formatting
        
    Returns:
        list: List of TipTap text nodes with marks
    """
    import re
    
    nodes = []
    i = 0
    
    while i < len(text):
        # Links: [text](url)
        link_match = re.match(r'\[([^\]]+)\]\(([^\)]+)\)', text[i:])
        if link_match:
            if i > 0:
                nodes.append({"type": "text", "text": text[:i]})
            nodes.append({
                "type": "text",
                "text": link_match.group(1),
                "marks": [{"type": "link", "attrs": {"href": link_match.group(2)}}]
            })
            text = text[i + link_match.end():]
            i = 0
            continue
        
        # Bold: **text** or __text__
        bold_match = re.match(r'\*\*([^\*]+)\*\*|__([^_]+)__', text[i:])
        if bold_match:
            if i > 0:
                nodes.append({"type": "text", "text": text[:i]})
            bold_text = bold_match.group(1) or bold_match.group(2)
            # Recursively parse formatting inside bold
            inner_nodes = parse_inline_formatting(bold_text)
            for node in inner_nodes:
                if node.get("type") == "text":
                    if "marks" not in node:
                        node["marks"] = []
                    node["marks"].append({"type": "bold"})
            nodes.extend(inner_nodes)
            text = text[i + bold_match.end():]
            i = 0
            continue
        
        # Italics: *text* or _text_ (but not ** or __)
        italic_match = re.match(r'(?<!\*)\*(?!\*)([^\*]+)\*(?!\*)|(?<!_)_(?!_)([^_]+)_(?!_)', text[i:])
        if italic_match:
            if i > 0:
                nodes.append({"type": "text", "text": text[:i]})
            italic_text = italic_match.group(1) or italic_match.group(2)
            # Recursively parse formatting inside italics
            inner_nodes = parse_inline_formatting(italic_text)
            for node in inner_nodes:
                if node.get("type") == "text":
                    if "marks" not in node:
                        node["marks"] = []
                    node["marks"].append({"type": "italic"})
            nodes.extend(inner_nodes)
            text = text[i + italic_match.end():]
            i = 0
            continue
        
        i += 1
    
    # Add remaining text
    if text:
        nodes.append({"type": "text", "text": text})
    
    return nodes if nodes else [{"type": "text", "text": ""}]


def markdown_to_tiptap(markdown: str) -> Dict[str, Any]:
    """Convert markdown to TipTap JSON format.
    Supports: H1, H2, H3, bold, italics, links, paragraphs, lists, code blocks, checkboxes.
    
    Args:
        markdown: Markdown string
        
    Returns:
        dict: TipTap JSON document
    """
    import re
    
    if not markdown.strip():
        return {"type": "doc", "content": []}
    
    lines = markdown.split("\n")
    content = []
    current_paragraph_lines = []
    in_code_block = False
    code_block_lines = []
    code_block_language = None
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()  # Keep leading spaces, remove trailing
        
        # Code blocks: ```language\ncode\n```
        if line.strip().startswith("```"):
            if in_code_block:
                # End code block
                code_text = "\n".join(code_block_lines)
                content.append({
                    "type": "codeBlock",
                    "attrs": {"language": code_block_language or ""},
                    "content": [{"type": "text", "text": code_text}]
                })
                code_block_lines = []
                code_block_language = None
                in_code_block = False
            else:
                # Start code block
                # Flush current paragraph
                if current_paragraph_lines:
                    paragraph_text = " ".join(current_paragraph_lines)
                    paragraph_nodes = parse_inline_formatting(paragraph_text)
                    content.append({
                        "type": "paragraph",
                        "content": paragraph_nodes
                    })
                    current_paragraph_lines = []
                
                # Extract language if present
                lang_match = line.strip()[3:].strip()
                code_block_language = lang_match if lang_match else None
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue
        
        # Empty line - end current paragraph
        if not line.strip():
            if current_paragraph_lines:
                paragraph_text = " ".join(current_paragraph_lines)
                paragraph_nodes = parse_inline_formatting(paragraph_text)
                content.append({
                    "type": "paragraph",
                    "content": paragraph_nodes
                })
                current_paragraph_lines = []
            i += 1
            continue
        
        # Headings (only H1, H2, H3)
        if line.startswith("#"):
            # Flush current paragraph
            if current_paragraph_lines:
                paragraph_text = " ".join(current_paragraph_lines)
                paragraph_nodes = parse_inline_formatting(paragraph_text)
                content.append({
                    "type": "paragraph",
                    "content": paragraph_nodes
                })
                current_paragraph_lines = []
            
            level = len(line) - len(line.lstrip("#"))
            if level > 3:
                level = 3  # Limit to H3
            
            heading_text = line.lstrip("#").strip()
            heading_nodes = parse_inline_formatting(heading_text)
            content.append({
                "type": "heading",
                "attrs": {"level": level},
                "content": heading_nodes
            })
            i += 1
            continue
        
        # Checkboxes: - [ ] or - [x] or * [ ] or * [x]
        checkbox_match = None
        if line.strip().startswith("- [") or line.strip().startswith("* ["):
            checkbox_match = re.match(r'^[\s]*[-*]\s*\[([ xX])\]\s*(.*)$', line)
        
        if checkbox_match:
            # Flush current paragraph
            if current_paragraph_lines:
                paragraph_text = " ".join(current_paragraph_lines)
                paragraph_nodes = parse_inline_formatting(paragraph_text)
                content.append({
                    "type": "paragraph",
                    "content": paragraph_nodes
                })
                current_paragraph_lines = []
            
            checked = checkbox_match.group(1).lower() == 'x'
            checkbox_text = checkbox_match.group(2).strip()
            checkbox_nodes = parse_inline_formatting(checkbox_text)
            
            # TipTap uses taskList and taskItem for checkboxes
            content.append({
                "type": "taskList",
                "content": [{
                    "type": "taskItem",
                    "attrs": {"checked": checked},
                    "content": [{
                        "type": "paragraph",
                        "content": checkbox_nodes
                    }]
                }]
            })
            i += 1
            continue
        
        # Unordered lists: - item or * item
        if line.strip().startswith("- ") or (line.strip().startswith("* ") and not line.strip().startswith("* [")):
            # Flush current paragraph
            if current_paragraph_lines:
                paragraph_text = " ".join(current_paragraph_lines)
                paragraph_nodes = parse_inline_formatting(paragraph_text)
                content.append({
                    "type": "paragraph",
                    "content": paragraph_nodes
                })
                current_paragraph_lines = []
            
            # Collect consecutive list items
            list_items = []
            while i < len(lines) and (lines[i].strip().startswith("- ") or 
                                     (lines[i].strip().startswith("* ") and not lines[i].strip().startswith("* ["))):
                list_text = lines[i].strip()[2:].strip()
                list_nodes = parse_inline_formatting(list_text)
                list_items.append({
                    "type": "listItem",
                    "content": [{
                        "type": "paragraph",
                        "content": list_nodes
                    }]
                })
                i += 1
            
            if list_items:
                content.append({
                    "type": "bulletList",
                    "content": list_items
                })
            continue
        
        # Numbered lists: 1. item, 2. item, etc.
        numbered_match = re.match(r'^\s*(\d+)\.\s+(.*)$', line)
        if numbered_match:
            # Flush current paragraph
            if current_paragraph_lines:
                paragraph_text = " ".join(current_paragraph_lines)
                paragraph_nodes = parse_inline_formatting(paragraph_text)
                content.append({
                    "type": "paragraph",
                    "content": paragraph_nodes
                })
                current_paragraph_lines = []
            
            # Collect consecutive numbered list items
            list_items = []
            while i < len(lines):
                numbered_match = re.match(r'^\s*(\d+)\.\s+(.*)$', lines[i])
                if numbered_match:
                    list_text = numbered_match.group(2).strip()
                    list_nodes = parse_inline_formatting(list_text)
                    list_items.append({
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": list_nodes
                        }]
                    })
                    i += 1
                else:
                    break
            
            if list_items:
                content.append({
                    "type": "orderedList",
                    "content": list_items
                })
            continue
        
        # Regular paragraph line
        current_paragraph_lines.append(line)
        i += 1
    
    # Add remaining paragraph
    if current_paragraph_lines:
        paragraph_text = " ".join(current_paragraph_lines)
        paragraph_nodes = parse_inline_formatting(paragraph_text)
        content.append({
            "type": "paragraph",
            "content": paragraph_nodes
        })
    
    return {
        "type": "doc",
        "content": content if content else [{
            "type": "paragraph",
            "content": [{"type": "text", "text": ""}]
        }]
    }
