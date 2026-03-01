"""Pre-commitment Q1: Download ESGBench and assess usability for training.

Go/no-go threshold (from plan): ≥800 usable QA pairs convertible to (query, passage) format.

Outputs:
- Downloads raw ESGBench data to data/raw/esgbench/
- Prints summary statistics
- Reports go/no-go result
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.esgbench_parser import get_esgbench_stats, load_esgbench

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GO_THRESHOLD = 800  # Plan's original threshold


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "esgbench"

    print("=" * 60)
    print("Pre-commitment Q1: ESGBench Assessment")
    print("=" * 60)

    # Download and parse
    pairs = load_esgbench(raw_dir)
    stats = get_esgbench_stats(pairs)

    # Report
    print(f"\nTotal usable (query, passage) pairs: {stats['total']}")
    print(f"Unique companies: {stats['unique_companies']}")
    print(f"Unique documents: {stats['unique_documents']}")

    print("\nTopic distribution:")
    for topic, count in sorted(stats.get("topic_distribution", {}).items()):
        print(f"  {topic}: {count}")

    if stats["total"] > 0:
        ql = stats["query_length"]
        pl = stats["passage_length"]
        print(f"\nQuery length (chars): min={ql['min']}, max={ql['max']}, mean={ql['mean']:.0f}")
        print(f"Passage length (chars): min={pl['min']}, max={pl['max']}, mean={pl['mean']:.0f}")

    # Show a few example pairs
    print("\n--- Example pairs ---")
    for pair in pairs[:3]:
        print(f"\n  Q: {pair.query[:100]}...")
        print(f"  P: {pair.passage[:100]}...")
        print(f"  Topic: {pair.topic} | Source: {pair.source} | Doc: {pair.doc_id}")

    # Go/no-go
    print("\n" + "=" * 60)
    if stats["total"] >= GO_THRESHOLD:
        print(f"GO: {stats['total']} pairs >= {GO_THRESHOLD} threshold")
    else:
        print(f"BELOW THRESHOLD: {stats['total']} pairs < {GO_THRESHOLD}")
        print(
            "ESGBench alone is insufficient. Project viability depends on"
            " CDP, GRI, TCFD, and synthetic augmentation."
        )
        print(
            "This is a KNOWN RISK from the plan — ESGBench was always the"
            " seed, not the full dataset."
        )
    print("=" * 60)

    # Save stats for reference
    stats_path = project_root / "data" / "raw" / "esgbench" / "assessment_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
