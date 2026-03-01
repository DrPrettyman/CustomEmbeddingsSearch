"""Parse ESGBench QA data into (query, passage) pairs for training.

ESGBench (https://github.com/sherinegeorge21/ESGBench) provides QA pairs from
real ESG documents with evidence text tracing back to source passages.

We convert each record into (query, passage) pairs where:
- query = the ESGBench question
- passage = the evidence_text (source passage, not the extracted answer)

This preserves the retrieval framing: "given this query, find the right passage."
"""

import json
import logging
from pathlib import Path

import requests

from src.data.schemas import QueryPassagePair

logger = logging.getLogger(__name__)

ESGBENCH_URLS = {
    "open_source": "https://raw.githubusercontent.com/sherinegeorge21/ESGBench/main/data/esgbench_open_source.jsonl",
    "numeric": "https://raw.githubusercontent.com/sherinegeorge21/ESGBench/main/data/esgbench_open_source_num.jsonl",
}

# Map ESGBench categories to our E/S/G topic labels
CATEGORY_MAP = {
    "environmental": "environmental",
    "social": "social",
    "governance": "governance",
    "strategy": "governance",  # strategy is governance-adjacent
    "risk": "governance",  # risk management falls under governance pillar
}

MIN_PASSAGE_LENGTH = 50  # characters — filter out very short evidence texts


def download_esgbench(output_dir: Path) -> dict[str, Path]:
    """Download ESGBench JSONL files to output_dir. Returns dict of name->path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in ESGBENCH_URLS.items():
        dest = output_dir / f"esgbench_{name}.jsonl"
        if dest.exists():
            logger.info(f"Already downloaded: {dest}")
            paths[name] = dest
            continue
        logger.info(f"Downloading {name} from {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text)
        paths[name] = dest
        logger.info(f"Saved {dest} ({len(resp.text)} bytes)")
    return paths


def parse_esgbench_record(record: dict) -> list[QueryPassagePair]:
    """Convert a single ESGBench record into (query, passage) pairs.

    Each record may have multiple evidence items — each becomes a separate pair.
    """
    pairs = []
    question = record.get("question", "").strip()
    if not question:
        return pairs

    category = record.get("category", "unknown").lower().strip()
    topic = CATEGORY_MAP.get(category, "unknown")
    company = record.get("company", "")
    doc_name = record.get("doc_name", "")

    evidence_list = record.get("evidence", [])
    if not evidence_list:
        return pairs

    for evidence in evidence_list:
        evidence_text = evidence.get("evidence_text", "").strip()
        if len(evidence_text) < MIN_PASSAGE_LENGTH:
            continue

        pair = QueryPassagePair(
            query=question,
            passage=evidence_text,
            source="esgbench",
            topic=topic,
            doc_id=doc_name,
            metadata={
                "company": company,
                "category": category,
                "page_num": evidence.get("evidence_page_num"),
                "evidence_doc": evidence.get("evidence_doc_name", ""),
            },
        )
        pairs.append(pair)

    return pairs


def parse_esgbench_file(filepath: Path) -> list[QueryPassagePair]:
    """Parse all records from a single ESGBench JSONL file."""
    pairs = []
    skipped = 0
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON on line {line_num} of {filepath}")
                skipped += 1
                continue
            pairs.extend(parse_esgbench_record(record))

    logger.info(
        f"Parsed {filepath.name}: {len(pairs)} pairs from {line_num} records "
        f"({skipped} skipped)"
    )
    return pairs


def load_esgbench(raw_dir: Path) -> list[QueryPassagePair]:
    """Download and parse all ESGBench data. Returns combined list of pairs."""
    paths = download_esgbench(raw_dir)
    all_pairs = []
    for name, path in paths.items():
        pairs = parse_esgbench_file(path)
        all_pairs.extend(pairs)
        logger.info(f"  {name}: {len(pairs)} pairs")

    logger.info(f"Total ESGBench pairs: {len(all_pairs)}")
    return all_pairs


def get_esgbench_stats(pairs: list[QueryPassagePair]) -> dict:
    """Compute summary statistics for parsed ESGBench pairs."""
    if not pairs:
        return {"total": 0}

    topics = {}
    companies = set()
    docs = set()
    query_lengths = []
    passage_lengths = []

    for p in pairs:
        topics[p.topic] = topics.get(p.topic, 0) + 1
        companies.add(p.metadata.get("company", "unknown"))
        docs.add(p.doc_id)
        query_lengths.append(len(p.query))
        passage_lengths.append(len(p.passage))

    return {
        "total": len(pairs),
        "topic_distribution": topics,
        "unique_companies": len(companies),
        "unique_documents": len(docs),
        "query_length": {
            "min": min(query_lengths),
            "max": max(query_lengths),
            "mean": sum(query_lengths) / len(query_lengths),
        },
        "passage_length": {
            "min": min(passage_lengths),
            "max": max(passage_lengths),
            "mean": sum(passage_lengths) / len(passage_lengths),
        },
    }
