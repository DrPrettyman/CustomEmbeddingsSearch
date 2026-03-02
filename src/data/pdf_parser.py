"""PDF text extraction and passage chunking for ESG documents.

Extracts text from ESG report PDFs and chunks into retrieval-appropriate
passages. Handles the specific challenges of sustainability reports:
- Mixed narrative and tabular content
- Multi-column layouts
- Section headers following GRI/TCFD/CSRD structure
- Page headers/footers that should be stripped
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum passage length in characters — shorter passages are noise
MIN_CHUNK_CHARS = 100
# Maximum passage length — longer passages should be split
MAX_CHUNK_CHARS = 2000
# Target passage length for splitting
TARGET_CHUNK_CHARS = 500
# Overlap between consecutive chunks (characters)
CHUNK_OVERLAP_CHARS = 50

# Patterns for content we want to strip
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),  # Page numbers only
    re.compile(r"^\s*©.*\d{4}.*$", re.IGNORECASE),  # Copyright lines
    re.compile(r"^\s*table of contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*\.\.\.\.\.*\s*\d+\s*$"),  # TOC dot leaders
]

# ESG section header patterns (GRI, TCFD, SASB, etc.)
SECTION_HEADER_PATTERNS = [
    re.compile(r"^(?:GRI\s+\d{3})", re.IGNORECASE),
    re.compile(r"^(?:TCFD|SASB|CSRD|CDP)\b", re.IGNORECASE),
    re.compile(r"^(?:Scope\s+[123])\b", re.IGNORECASE),
    re.compile(r"^(?:C\d+\.?\s)", re.IGNORECASE),  # CDP question sections
    re.compile(
        r"^(?:Governance|Strategy|Risk Management|Metrics and Targets)\s*$",
        re.IGNORECASE,
    ),
]


@dataclass
class ExtractedPage:
    """Text extracted from a single PDF page."""

    page_num: int
    text: str


@dataclass
class Passage:
    """A chunked passage ready for embedding training."""

    text: str
    source_file: str
    page_num: int
    chunk_index: int
    passage_id: str  # Deterministic ID based on content hash


def extract_text_from_pdf(pdf_path: Path) -> list[ExtractedPage]:
    """Extract text from each page of a PDF using PyMuPDF."""
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            if text.strip():
                pages.append(ExtractedPage(page_num=page_num, text=text))
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
    logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def clean_page_text(text: str) -> str:
    """Clean extracted page text: strip boilerplate, normalize whitespace."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        # Skip boilerplate
        if any(pat.match(stripped) for pat in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(stripped)

    text = "\n".join(cleaned)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_section_header(line: str) -> bool:
    """Check if a line looks like an ESG section header."""
    stripped = line.strip()
    if not stripped:
        return False
    # Short lines in title case or all caps are likely headers
    if len(stripped) < 80 and (stripped.istitle() or stripped.isupper()):
        return True
    # Known ESG framework patterns
    if any(pat.match(stripped) for pat in SECTION_HEADER_PATTERNS):
        return True
    return False


def _make_passage_id(source_file: str, text: str) -> str:
    """Create a deterministic passage ID from source and content."""
    content = f"{source_file}:{text}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def chunk_text(
    text: str,
    source_file: str,
    page_num: int,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[Passage]:
    """Split text into retrieval-appropriate passages.

    Strategy:
    1. Split on paragraph boundaries (double newlines)
    2. Merge short paragraphs up to target_chars
    3. Split long paragraphs at sentence boundaries
    4. Filter out chunks below min_chars
    """
    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If this paragraph alone exceeds max_chars, split at sentence boundaries
        if len(para) > max_chars:
            # Flush current chunk first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Split long paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_chunk = ""
            for sent in sentences:
                if len(sent_chunk) + len(sent) + 1 > target_chars and sent_chunk:
                    chunks.append(sent_chunk)
                    # Keep overlap from end of previous chunk
                    if overlap_chars > 0 and len(sent_chunk) > overlap_chars:
                        sent_chunk = sent_chunk[-overlap_chars:] + " " + sent
                    else:
                        sent_chunk = sent
                else:
                    sent_chunk = (sent_chunk + " " + sent).strip() if sent_chunk else sent
            if sent_chunk:
                chunks.append(sent_chunk)
            continue

        # Try to merge with current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > target_chars:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    # Filter and create Passage objects
    passages = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < min_chars:
            continue
        passage_id = _make_passage_id(source_file, chunk)
        passages.append(
            Passage(
                text=chunk,
                source_file=source_file,
                page_num=page_num,
                chunk_index=i,
                passage_id=passage_id,
            )
        )

    return passages


def extract_passages_from_pdf(
    pdf_path: Path,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    target_chars: int = TARGET_CHUNK_CHARS,
) -> list[Passage]:
    """Full pipeline: PDF → extracted text → cleaned → chunked passages."""
    pages = extract_text_from_pdf(pdf_path)
    all_passages = []

    for page in pages:
        cleaned = clean_page_text(page.text)
        if len(cleaned) < min_chars:
            continue
        passages = chunk_text(
            cleaned,
            source_file=pdf_path.name,
            page_num=page.page_num,
            min_chars=min_chars,
            max_chars=max_chars,
            target_chars=target_chars,
        )
        all_passages.extend(passages)

    logger.info(
        f"Extracted {len(all_passages)} passages from {pdf_path.name} "
        f"({len(pages)} pages)"
    )
    return all_passages


def download_pdf(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a PDF from a URL. Returns True on success."""
    import requests

    if output_path.exists():
        logger.info(f"Already downloaded: {output_path}")
        return True
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info(f"Downloaded {output_path.name} ({len(resp.content) / 1024:.0f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
