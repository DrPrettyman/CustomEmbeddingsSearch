"""Parse CDP climate change questionnaire responses into (query, passage) pairs.

CDP questionnaires have a structured format:
- Questions numbered (X.Y) or (X.Y.Z) — e.g., (1.3.3), (4.1.1), (7.55.2)
- Company responses follow each question
- Responses range from checkbox selections to multi-paragraph narratives

We extract substantive text responses as passages and pair them with the
CDP questions as queries. Checkbox/numeric-only answers are filtered out.
"""

import logging
import re
from pathlib import Path

from src.data.pdf_parser import download_pdf, extract_text_from_pdf, clean_page_text
from src.data.schemas import QueryPassagePair

logger = logging.getLogger(__name__)

# CDP question pattern: (X.Y) or (X.Y.Z) at start of line
CDP_QUESTION_RE = re.compile(r"\((\d+\.\d+(?:\.\d+)?)\)\s*(.*)")

# Minimum substantive answer length — filters checkbox/numeric-only responses
MIN_ANSWER_CHARS = 150

# Known CDP response URLs (publicly published by companies)
CDP_REPORT_URLS = {
    "apple_2024": "https://www.apple.com/environment/pdf/Apple_CDP-Climate-Change-Questionnaire_2024.pdf",
    "bp_2024": "https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/sustainability/group-reports/bp-cdp-climate-change-questionnaire-2024.pdf",
    "oracle_2024": "https://www.oracle.com/a/ocom/docs/cdp-climate-change-questionnaire-2024.pdf",
    "cisco_2024": "https://www.cisco.com/c/dam/m/en_us/about/csr/esg-hub/_pdf/2024-Cisco-CDP-Climate-Change-Response.pdf",
    "bnp_paribas_2024": "https://cdn-group.bnpparibas.com/uploads/file/bnpparibas_cdp_climate_change_questionnaire_2024.pdf",
}

# Map CDP question number ranges to ESG topics
CDP_TOPIC_MAP = {
    range(1, 4): "governance",  # C1-C3: Introduction, governance
    range(4, 6): "governance",  # C4-C5: Governance, supply chain engagement
    range(6, 8): "environmental",  # C6-C7: Environmental performance
    range(8, 10): "environmental",  # C8-C9: Energy, additional metrics
    range(10, 13): "environmental",  # C10-C12: Verification, carbon pricing, engagement
    range(13, 20): "environmental",  # C13+: Additional
}


def _get_topic_for_question(q_id: str) -> str:
    """Map a CDP question ID to an ESG topic."""
    try:
        module_num = int(q_id.split(".")[0])
        for num_range, topic in CDP_TOPIC_MAP.items():
            if module_num in num_range:
                return topic
    except (ValueError, IndexError):
        pass
    return "unknown"


def _is_substantive_answer(text: str) -> bool:
    """Check if an answer contains substantive text (not just checkboxes/numbers)."""
    # Strip checkbox markers and "Select from:" text
    cleaned = re.sub(r"☑\s*\w+", "", text)
    cleaned = re.sub(r"Select (?:from|all that apply):?", "", cleaned)
    cleaned = re.sub(r"\[.*?\]", "", cleaned)  # Remove [Fixed row], [Add row], etc.
    cleaned = cleaned.strip()

    # Check if remaining text is substantive
    # Count words that are actual prose (not just numbers/symbols)
    words = [w for w in cleaned.split() if len(w) > 2 and w.isalpha()]
    return len(words) >= 20 and len(cleaned) >= MIN_ANSWER_CHARS


def extract_cdp_qa_pairs(text: str, company: str, doc_name: str) -> list[QueryPassagePair]:
    """Extract (question, answer) pairs from CDP questionnaire text.

    Returns only pairs where the answer is substantive text (>= MIN_ANSWER_CHARS
    of prose content, not checkbox/numeric answers).
    """
    # Split text at question markers
    sections = re.split(r"\n(?=\(\d+\.\d+(?:\.\d+)?\))", text)

    pairs = []
    for section in sections[1:]:  # Skip content before first question
        lines = section.strip().split("\n")
        if not lines:
            continue

        match = CDP_QUESTION_RE.match(lines[0].strip())
        if not match:
            continue

        q_id = match.group(1)
        q_text_start = match.group(2).strip()

        # Build question text — in CDP format, the question is usually on the
        # first line (possibly wrapping onto a second). The answer follows after
        # a blank line or starts on the next non-empty line.
        # Be conservative: only merge the next line into the question if the
        # first line doesn't end with punctuation (incomplete question text).
        q_lines = [q_text_start] if q_text_start else []
        answer_start = 1

        for i, line in enumerate(lines[1:], 1):
            stripped = line.strip()
            if not stripped:
                answer_start = i + 1
                break
            if stripped.startswith(("Select", "☑", "☐", "[Fixed", "[Add")):
                answer_start = i
                break
            # Only continue question if previous line didn't end with sentence-ending
            # punctuation and this line is short (likely a wrapped question, not an answer)
            prev_text = " ".join(q_lines)
            if (
                not prev_text.endswith(("?", ".", ":", ")"))
                and len(stripped) < 120
                and not any(c.isdigit() for c in stripped[:3])
            ):
                q_lines.append(stripped)
                answer_start = i + 1
            else:
                answer_start = i
                break

        question = " ".join(q_lines).strip()
        if not question or len(question) < 15:
            continue

        # Clean up question: remove trailing dots from TOC-style entries
        if re.search(r"\.{5,}\s*\d+\s*$", question):
            continue

        # Build answer text
        answer_lines = lines[answer_start:]
        answer = "\n".join(line.strip() for line in answer_lines).strip()

        if not _is_substantive_answer(answer):
            continue

        topic = _get_topic_for_question(q_id)

        pair = QueryPassagePair(
            query=question,
            passage=answer,
            source="cdp",
            topic=topic,
            doc_id=doc_name,
            metadata={
                "company": company,
                "question_id": q_id,
                "cdp_module": q_id.split(".")[0],
            },
        )
        pairs.append(pair)

    logger.info(
        f"Extracted {len(pairs)} substantive Q&A pairs from {doc_name} "
        f"(from {len(sections) - 1} total sections)"
    )
    return pairs


def parse_cdp_pdf(
    pdf_path: Path, company: str, doc_name: str | None = None
) -> list[QueryPassagePair]:
    """Parse a CDP response PDF into (query, passage) pairs."""
    if doc_name is None:
        doc_name = pdf_path.stem

    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join(clean_page_text(p.text) for p in pages)

    return extract_cdp_qa_pairs(full_text, company=company, doc_name=doc_name)


def download_and_parse_cdp_reports(
    output_dir: Path,
) -> list[QueryPassagePair]:
    """Download and parse all known CDP report PDFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    for name, url in CDP_REPORT_URLS.items():
        company = name.rsplit("_", 1)[0].replace("_", " ").title()
        pdf_path = output_dir / f"{name}_cdp.pdf"

        if not download_pdf(url, pdf_path):
            continue

        pairs = parse_cdp_pdf(pdf_path, company=company, doc_name=name)
        all_pairs.extend(pairs)
        logger.info(f"  {name}: {len(pairs)} pairs")

    logger.info(f"Total CDP pairs: {len(all_pairs)}")
    return all_pairs
