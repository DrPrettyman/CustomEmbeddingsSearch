"""Phase 1 baseline evaluation: BM25 vs off-the-shelf dense on the constructed dataset.

Establishes the "before" picture for fine-tuning. Runs BM25 and
all-mpnet-base-v2 on the eval set produced by build_dataset.py.

Usage:
    python scripts/baseline_evaluation.py
    python scripts/baseline_evaluation.py --models all-mpnet-base-v2 all-MiniLM-L6-v2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_rankings
from src.evaluation.retrieval import bm25_retrieval, dense_retrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_eval_data(processed_dir: Path):
    """Load eval queries, corpus, and qrels from the constructed dataset."""
    # Eval queries
    queries = []
    with open(processed_dir / "eval_queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line))

    # Eval corpus
    corpus = []
    with open(processed_dir / "eval_corpus.jsonl") as f:
        for line in f:
            corpus.append(json.loads(line))

    # Qrels
    with open(processed_dir / "eval_qrels.json") as f:
        qrels = json.load(f)

    return queries, corpus, qrels


def build_relevance_mapping(queries, corpus, qrels):
    """Convert qrels (passage_id -> relevance) to index-based relevance lists."""
    # Build passage_id -> corpus_index mapping
    pid_to_idx = {p["passage_id"]: i for i, p in enumerate(corpus)}

    query_texts = []
    relevant_per_query = []
    query_metadata = []

    for q in queries:
        q_id = q["query_id"]
        if q_id not in qrels:
            continue

        relevant_pids = list(qrels[q_id].keys())
        relevant_indices = [pid_to_idx[pid] for pid in relevant_pids if pid in pid_to_idx]

        if not relevant_indices:
            continue

        query_texts.append(q["query"])
        relevant_per_query.append(relevant_indices)
        query_metadata.append(q)

    return query_texts, relevant_per_query, query_metadata


def print_results_table(all_results: list[dict]):
    """Print a formatted results table."""
    print(f"\n{'Method':<25} {'NDCG@10':>8} {'MRR@10':>8} {'R@10':>8} {'R@100':>8}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['method']:<25} "
            f"{r['ndcg@10']:>8.3f} "
            f"{r['mrr@10']:>8.3f} "
            f"{r['recall@10']:>8.3f} "
            f"{r['recall@100']:>8.3f}"
        )


def per_query_analysis(all_results: list[dict], query_texts: list[str], query_metadata: list[dict]):
    """Analyze per-query performance differences between methods."""
    if len(all_results) < 2:
        return

    bm25 = next((r for r in all_results if r["method"] == "BM25"), None)
    dense = next((r for r in all_results if r["method"] != "BM25"), None)

    if not bm25 or not dense:
        return

    print(f"\n--- Per-Query Analysis: BM25 vs {dense['method']} ---")

    bm25_ndcg = np.array(bm25["per_query_ndcg"])
    dense_ndcg = np.array(dense["per_query_ndcg"])

    bm25_wins = int(np.sum(bm25_ndcg > dense_ndcg + 0.01))
    dense_wins = int(np.sum(dense_ndcg > bm25_ndcg + 0.01))
    ties = len(bm25_ndcg) - bm25_wins - dense_wins

    print(f"BM25 wins: {bm25_wins}/{len(bm25_ndcg)}")
    print(f"Dense wins: {dense_wins}/{len(bm25_ndcg)}")
    print(f"Ties: {ties}/{len(bm25_ndcg)}")
    print(f"BM25 perfect (NDCG=1.0): {int(np.sum(bm25_ndcg >= 0.99))}/{len(bm25_ndcg)}")
    print(f"Dense perfect (NDCG=1.0): {int(np.sum(dense_ndcg >= 0.99))}/{len(bm25_ndcg)}")

    # Topic breakdown
    topics = {}
    for i, meta in enumerate(query_metadata):
        topic = meta.get("topic", "unknown")
        if topic not in topics:
            topics[topic] = {"bm25": [], "dense": []}
        topics[topic]["bm25"].append(bm25_ndcg[i])
        topics[topic]["dense"].append(dense_ndcg[i])

    print(f"\n--- Topic Breakdown (NDCG@10) ---")
    print(f"{'Topic':<20} {'Count':>6} {'BM25':>8} {'Dense':>8} {'Gap':>8}")
    print("-" * 50)
    for topic, scores in sorted(topics.items()):
        bm25_mean = float(np.mean(scores["bm25"]))
        dense_mean = float(np.mean(scores["dense"]))
        gap = dense_mean - bm25_mean
        print(
            f"{topic:<20} {len(scores['bm25']):>6} "
            f"{bm25_mean:>8.3f} {dense_mean:>8.3f} {gap:>+8.3f}"
        )

    # Worst queries for dense model
    print(f"\n--- Worst 5 queries for {dense['method']} ---")
    worst_indices = np.argsort(dense_ndcg)[:5]
    for idx in worst_indices:
        print(f"\n  Query: {query_texts[idx][:120]}")
        print(f"  Dense NDCG: {dense_ndcg[idx]:.3f}  BM25 NDCG: {bm25_ndcg[idx]:.3f}")
        print(f"  Topic: {query_metadata[idx].get('topic', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument(
        "--models", nargs="+", default=["all-mpnet-base-v2"],
        help="Dense models to evaluate",
    )
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed"

    print("=" * 60)
    print("Phase 1: Baseline Evaluation")
    print("=" * 60)

    # Load data
    queries, corpus, qrels = load_eval_data(processed_dir)
    print(f"Eval queries: {len(queries)}")
    print(f"Eval corpus: {len(corpus)} passages")

    # Build index-based relevance
    query_texts, relevant_per_query, query_metadata = build_relevance_mapping(
        queries, corpus, qrels
    )
    corpus_texts = [p["passage"] for p in corpus]
    print(f"Queries with relevance judgments: {len(query_texts)}")

    all_results = []

    # BM25
    print("\n--- BM25 Retrieval ---")
    bm25_rankings = bm25_retrieval(query_texts, corpus_texts)
    bm25_results = evaluate_rankings(bm25_rankings, relevant_per_query)
    bm25_results["method"] = "BM25"
    all_results.append(bm25_results)

    # Dense models
    for model_name in args.models:
        print(f"\n--- Dense Retrieval ({model_name}) ---")
        dense_rankings = dense_retrieval(query_texts, corpus_texts, model_name)
        dense_results = evaluate_rankings(dense_rankings, relevant_per_query)
        dense_results["method"] = model_name
        all_results.append(dense_results)

    # Results
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print_results_table(all_results)
    per_query_analysis(all_results, query_texts, query_metadata)

    # Save
    results_path = PROJECT_ROOT / "data" / "evaluation" / "baseline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "num_queries": len(query_texts),
        "corpus_size": len(corpus_texts),
        "methods": {r["method"]: r for r in all_results},
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
