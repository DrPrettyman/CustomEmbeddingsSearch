"""Phase 1 pipeline: Build the full ESG training dataset.

Steps:
1. Load ESGBench seed pairs
2. Download and parse CDP questionnaire responses
3. Download ESG reports and extract passages
4. Generate synthetic queries from passages
5. Merge all sources, deduplicate, quality filter
6. Split into train/eval sets
7. Build eval corpus with distractors
8. Save to data/processed/

Usage:
    python scripts/build_dataset.py --skip-synthetic   # Skip LLM query generation (dev mode)
    python scripts/build_dataset.py --backend anthropic # Use Anthropic API for synthetic
    python scripts/build_dataset.py --max-passages 100  # Limit synthetic generation (cost control)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cdp_parser import download_and_parse_cdp_reports, parse_cdp_pdf
from src.data.dataset_builder import (
    build_eval_corpus,
    deduplicate_pairs,
    quality_filter,
    save_dataset,
    split_train_eval,
)
from src.data.esgbench_parser import load_esgbench
from src.data.pdf_parser import download_pdf, extract_passages_from_pdf
from src.data.report_sources import ALL_REPORTS
from src.data.schemas import QueryPassagePair
from src.data.synthetic_generator import generate_synthetic_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def step1_esgbench(raw_dir: Path) -> list[QueryPassagePair]:
    """Load ESGBench seed pairs."""
    logger.info("=" * 60)
    logger.info("Step 1: Loading ESGBench seed pairs")
    pairs = load_esgbench(raw_dir / "esgbench")
    logger.info(f"  ESGBench: {len(pairs)} pairs")
    return pairs


def step2_cdp(raw_dir: Path) -> list[QueryPassagePair]:
    """Download and parse CDP questionnaire responses."""
    logger.info("=" * 60)
    logger.info("Step 2: Parsing CDP questionnaire responses")
    pairs = download_and_parse_cdp_reports(raw_dir / "cdp")
    logger.info(f"  CDP: {len(pairs)} pairs")
    return pairs


def step3_report_passages(raw_dir: Path) -> list[dict]:
    """Download ESG reports and extract passages for synthetic query generation."""
    logger.info("=" * 60)
    logger.info("Step 3: Downloading reports and extracting passages")

    reports_dir = raw_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    all_passages = []

    for name, info in ALL_REPORTS.items():
        # Skip CDP responses here (handled separately in step 2)
        if info["doc_type"] == "cdp_response":
            continue

        pdf_path = reports_dir / f"{name}.pdf"
        if not download_pdf(info["url"], pdf_path):
            continue

        passages = extract_passages_from_pdf(pdf_path)

        # Determine ESG topic heuristically from content
        for p in passages:
            text_lower = p.text.lower()
            if any(kw in text_lower for kw in [
                "emission", "carbon", "climate", "energy", "water", "waste",
                "scope 1", "scope 2", "scope 3", "renewable", "biodiversity",
            ]):
                topic = "environmental"
            elif any(kw in text_lower for kw in [
                "diversity", "employee", "safety", "health", "human rights",
                "community", "labor", "workforce", "inclusion",
            ]):
                topic = "social"
            elif any(kw in text_lower for kw in [
                "governance", "board", "compliance", "ethics", "audit",
                "risk management", "shareholder", "executive compensation",
            ]):
                topic = "governance"
            else:
                topic = "unknown"

            all_passages.append({
                "text": p.text,
                "source_file": p.source_file,
                "company": info["company"],
                "doc_type": info["doc_type"],
                "topic": topic,
                "original_source": name,
                "page_num": p.page_num,
            })

    logger.info(f"  Extracted {len(all_passages)} passages from reports")
    return all_passages


def step4_synthetic(
    passages: list[dict],
    backend: str = "anthropic",
    max_passages: int | None = None,
    n_queries: int = 3,
) -> list[QueryPassagePair]:
    """Generate synthetic queries from passages."""
    logger.info("=" * 60)
    logger.info(f"Step 4: Generating synthetic queries (backend={backend})")

    if max_passages:
        logger.info(f"  Limiting to {max_passages} passages (cost control)")

    pairs = generate_synthetic_pairs(
        passages,
        backend=backend,
        n_queries_per_passage=n_queries,
        max_passages=max_passages,
    )
    logger.info(f"  Synthetic: {len(pairs)} pairs")
    return pairs


def step5_merge_and_build(
    all_pairs: list[QueryPassagePair],
    output_dir: Path,
) -> None:
    """Merge, deduplicate, filter, split, and save."""
    logger.info("=" * 60)
    logger.info("Step 5: Building final dataset")

    logger.info(f"  Total raw pairs: {len(all_pairs)}")

    # Source breakdown
    sources = {}
    for p in all_pairs:
        sources[p.source] = sources.get(p.source, 0) + 1
    logger.info(f"  By source: {sources}")

    # Quality filter
    filtered = quality_filter(all_pairs)

    # Deduplicate
    unique = deduplicate_pairs(filtered)

    # Split
    train, eval_pairs = split_train_eval(unique)

    # Build eval corpus
    eval_queries, eval_corpus = build_eval_corpus(eval_pairs, train)

    # Save
    save_dataset(train, eval_queries, eval_corpus, output_dir)

    logger.info("=" * 60)
    logger.info("Dataset construction complete!")
    logger.info(f"  Training pairs: {len(train)}")
    logger.info(f"  Eval queries: {len(eval_queries)}")
    logger.info(f"  Eval corpus: {len(eval_corpus)}")
    logger.info(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build ESG training dataset")
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip synthetic query generation (dev/test mode)",
    )
    parser.add_argument(
        "--backend", default="claude-code",
        choices=["claude-code", "anthropic", "openai"],
        help="LLM backend for synthetic query generation. "
             "'claude-code' uses the claude CLI (no API key needed). "
             "'anthropic' requires ANTHROPIC_API_KEY. "
             "'openai' requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--max-passages", type=int, default=None,
        help="Max passages for synthetic generation (cost control)",
    )
    parser.add_argument(
        "--n-queries", type=int, default=3,
        help="Number of synthetic queries per passage",
    )
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "processed"

    # Step 1: ESGBench
    esgbench_pairs = step1_esgbench(raw_dir)

    # Step 2: CDP
    cdp_pairs = step2_cdp(raw_dir)

    # Step 3: Extract passages from ESG reports
    passages = step3_report_passages(raw_dir)

    # Step 4: Synthetic query generation
    synthetic_pairs = []
    if not args.skip_synthetic:
        synthetic_pairs = step4_synthetic(
            passages,
            backend=args.backend,
            max_passages=args.max_passages,
            n_queries=args.n_queries,
        )
    else:
        logger.info("Step 4: SKIPPED (--skip-synthetic)")

    # Step 5: Merge and build
    all_pairs = esgbench_pairs + cdp_pairs + synthetic_pairs
    step5_merge_and_build(all_pairs, output_dir)


if __name__ == "__main__":
    main()
