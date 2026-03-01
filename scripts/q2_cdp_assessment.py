"""Pre-commitment Q2: Assess CDP data accessibility and parseability.

Downloads a sample CDP climate change questionnaire response (Apple 2024)
and checks whether structured Q&A pairs can be extracted programmatically.

Go/no-go threshold (from plan): Structured Q&A pairs extractable from CDP
without manual effort per company. If CDP requires manual download per company,
scope data augmentation more heavily toward synthetic generation.
"""

import logging
import re
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Sample CDP responses (publicly available on company websites)
SAMPLE_CDP_URLS = {
    "apple_2024": "https://www.apple.com/environment/pdf/Apple_CDP-Climate-Change-Questionnaire_2024.pdf",
    "bp_2024": "https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/sustainability/group-reports/bp-cdp-climate-change-questionnaire-2024.pdf",
}

# CDP question pattern: C followed by digits and sub-question markers
CDP_QUESTION_PATTERN = re.compile(
    r"^\(?C(\d+)\.(\d+[a-z]?)\)?\s+(.+?)$",
    re.MULTILINE,
)


def download_sample_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF. Returns True on success."""
    if output_path.exists():
        logger.info(f"Already downloaded: {output_path}")
        return True
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info(f"Downloaded {output_path} ({len(resp.content)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF if available, else basic extraction."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed. Trying pdfplumber...")

    try:
        import pdfplumber

        text = ""
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        logger.error(
            "Neither PyMuPDF nor pdfplumber installed. "
            "Install one: pip install pymupdf  or  pip install pdfplumber"
        )
        return ""


def find_cdp_qa_pairs(text: str) -> list[dict]:
    """Extract CDP question-answer pairs from extracted PDF text.

    CDP questionnaires use numbering like (1.1), (4.1.1), (7.55.2), etc.
    Questions are followed by company responses in various formats.
    """
    # Split text by CDP question markers: (X.Y) or (X.Y.Z) format
    sections = re.split(r"\n(?=\(\d+\.\d+(?:\.\d+)?\))", text)

    pairs = []
    for section in sections[1:]:  # Skip everything before first question
        lines = section.strip().split("\n")
        if not lines:
            continue

        # First line should contain the question identifier and question text
        first_line = lines[0].strip()
        match = re.match(r"\((\d+\.\d+(?:\.\d+)?)\)\s*(.*)", first_line)
        if not match:
            continue

        q_id = match.group(1)
        q_text_first_line = match.group(2).strip()

        # Question text may span multiple lines until we hit the answer
        # Heuristic: question text is the continuous prose after the ID,
        # answer starts after blank lines or specific markers
        q_text_lines = [q_text_first_line]
        answer_start = 1
        for i, line in enumerate(lines[1:], 1):
            stripped = line.strip()
            # If we hit an empty line or a "Select from:" marker, answer starts
            if not stripped or stripped.startswith("Select") or stripped.startswith("☑"):
                answer_start = i
                break
            # If the line looks like continuation of the question (no special markers)
            if len(stripped) > 20 and not stripped[0].isdigit():
                q_text_lines.append(stripped)
                answer_start = i + 1
            else:
                answer_start = i
                break

        q_text = " ".join(q_text_lines).strip()

        # Rest of the section is the answer
        answer_lines = lines[answer_start:]
        answer = "\n".join(line.strip() for line in answer_lines).strip()

        # Clean up: remove trailing dots from table-of-contents entries
        if re.match(r".*\.{5,}\s*\d+\s*$", q_text):
            continue  # Skip table-of-contents entries

        # Skip empty answers or very short ones
        if len(answer) < 50:
            continue

        # Skip if question text is too short (likely a sub-header)
        if len(q_text) < 10:
            continue

        pairs.append({
            "question_id": q_id,
            "question": q_text,
            "answer": answer[:500],  # Truncate for display
            "answer_full_length": len(answer),
        })

    return pairs


def main():
    project_root = Path(__file__).resolve().parent.parent
    cdp_dir = project_root / "data" / "raw" / "cdp"
    cdp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pre-commitment Q2: CDP Data Assessment")
    print("=" * 60)

    # Try to download a sample CDP response
    sample_name = "apple_2024"
    sample_url = SAMPLE_CDP_URLS[sample_name]
    pdf_path = cdp_dir / f"{sample_name}_cdp.pdf"

    print(f"\n1. Downloading sample CDP response: {sample_name}")
    if not download_sample_pdf(sample_url, pdf_path):
        print("FAIL: Could not download sample CDP PDF")
        return

    print(f"   PDF size: {pdf_path.stat().st_size / 1024:.0f} KB")

    # Try to extract text
    print("\n2. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("FAIL: Could not extract text (install pymupdf or pdfplumber)")
        print("   pip install pymupdf")
        return

    print(f"   Extracted {len(text)} characters from PDF")
    print(f"   First 200 chars: {text[:200]}")

    # Try to find CDP Q&A structure
    print("\n3. Looking for CDP Q&A structure...")
    pairs = find_cdp_qa_pairs(text)
    print(f"   Found {len(pairs)} Q&A sections with answers >= 50 chars")

    if pairs:
        print("\n   Sample pairs:")
        for pair in pairs[:5]:
            print(f"\n   [{pair['question_id']}] {pair['question'][:80]}")
            print(f"   Answer ({pair['answer_full_length']} chars): {pair['answer'][:120]}...")

    # Assessment
    print("\n" + "=" * 60)
    print("CDP Assessment Summary:")
    print(f"  - PDF download: Manual per company (not bulk downloadable)")
    print(f"  - Text extraction: {'SUCCESS' if text else 'FAILED'}")
    print(f"  - Q&A structure found: {len(pairs)} extractable pairs")
    print(f"  - Estimated pairs per company: {len(pairs)}")

    if len(pairs) >= 30:
        print(f"\n  VIABLE: ~{len(pairs)} pairs/company × 20 companies = ~{len(pairs) * 20} pairs")
        print("  Requires: manual PDF collection + automated parsing")
    else:
        print(f"\n  LIMITED: Only {len(pairs)} extractable pairs per company")
        print("  Shift strategy toward synthetic generation from ESG report passages")
    print("=" * 60)


if __name__ == "__main__":
    main()
