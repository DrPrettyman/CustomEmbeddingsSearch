"""Pre-commitment Q3: Quick feasibility test.

Encodes ESGBench (query, passage) pairs with all-mpnet-base-v2 and BM25,
then computes retrieval metrics.

Go/no-go threshold: Off-the-shelf dense retrieval is imperfect (NDCG@10 < 0.9).
If it's already near-perfect, there's no room for fine-tuning to improve.
BM25 should be competitive but beatable on at least some queries.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.esgbench_parser import load_esgbench

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_ndcg_at_k(
    query_rankings: list[list[int]], relevant_indices: list[int], k: int = 10
) -> float:
    """Compute NDCG@k for a single query.

    query_rankings: sorted list of passage indices by score (highest first)
    relevant_indices: set of indices that are relevant
    """
    relevant_set = set(relevant_indices)
    dcg = 0.0
    for i, idx in enumerate(query_rankings[:k]):
        if idx in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at top
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_indices), k)))

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def compute_mrr(query_rankings: list[int], relevant_indices: list[int]) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    relevant_set = set(relevant_indices)
    for i, idx in enumerate(query_rankings):
        if idx in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(
    query_rankings: list[int], relevant_indices: list[int], k: int = 10
) -> float:
    """Compute Recall@k for a single query."""
    relevant_set = set(relevant_indices)
    retrieved_relevant = sum(1 for idx in query_rankings[:k] if idx in relevant_set)
    return retrieved_relevant / len(relevant_indices) if relevant_indices else 0.0


def run_bm25_retrieval(queries: list[str], corpus: list[str]) -> list[list[int]]:
    """Run BM25 retrieval. Returns ranked passage indices per query."""
    from rank_bm25 import BM25Okapi

    # Tokenize corpus
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    rankings = []
    for query in queries:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1].tolist()
        rankings.append(ranked_indices)

    return rankings


def run_dense_retrieval(
    queries: list[str], corpus: list[str], model_name: str
) -> list[list[int]]:
    """Run dense retrieval with a sentence-transformer. Returns ranked passage indices."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(corpus)} passages...")
    t0 = time.time()
    corpus_embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    t_corpus = time.time() - t0

    logger.info(f"Encoding {len(queries)} queries...")
    t0 = time.time()
    query_embeddings = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    t_query = time.time() - t0

    logger.info(
        f"Encoding done. Corpus: {t_corpus:.1f}s, Queries: {t_query:.1f}s"
    )

    # Compute similarities and rank
    similarities = cos_sim(query_embeddings, corpus_embeddings).numpy()

    rankings = []
    for i in range(len(queries)):
        ranked_indices = np.argsort(similarities[i])[::-1].tolist()
        rankings.append(ranked_indices)

    return rankings


def evaluate_method(
    rankings: list[list[int]],
    relevant_per_query: list[list[int]],
    method_name: str,
) -> dict:
    """Compute all metrics for a retrieval method."""
    ndcg_scores = []
    mrr_scores = []
    recall_10_scores = []
    recall_100_scores = []

    for ranking, relevant in zip(rankings, relevant_per_query):
        ndcg_scores.append(compute_ndcg_at_k(ranking, relevant, k=10))
        mrr_scores.append(compute_mrr(ranking, relevant))
        recall_10_scores.append(compute_recall_at_k(ranking, relevant, k=10))
        recall_100_scores.append(compute_recall_at_k(ranking, relevant, k=100))

    results = {
        "method": method_name,
        "ndcg@10": float(np.mean(ndcg_scores)),
        "mrr@10": float(np.mean(mrr_scores)),
        "recall@10": float(np.mean(recall_10_scores)),
        "recall@100": float(np.mean(recall_100_scores)),
        "per_query_ndcg": [float(x) for x in ndcg_scores],
    }
    return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "esgbench"

    print("=" * 60)
    print("Pre-commitment Q3: Feasibility Test")
    print("BM25 vs all-mpnet-base-v2 on ESGBench pairs")
    print("=" * 60)

    # Load ESGBench pairs
    pairs = load_esgbench(raw_dir)
    if len(pairs) < 10:
        print(f"FAIL: Only {len(pairs)} pairs. Need at least 10 for meaningful test.")
        return

    print(f"\nUsing {len(pairs)} ESGBench (query, passage) pairs")

    # Build corpus and queries
    # Each query's relevant passage is at the same index
    queries = [p.query for p in pairs]
    corpus = [p.passage for p in pairs]
    relevant_per_query = [[i] for i in range(len(pairs))]  # Each query has one relevant passage

    # Run BM25
    print("\n--- BM25 Retrieval ---")
    bm25_rankings = run_bm25_retrieval(queries, corpus)
    bm25_results = evaluate_method(bm25_rankings, relevant_per_query, "BM25")

    # Run dense retrieval with mpnet
    print("\n--- Dense Retrieval (all-mpnet-base-v2) ---")
    mpnet_rankings = run_dense_retrieval(queries, corpus, "all-mpnet-base-v2")
    mpnet_results = evaluate_method(mpnet_rankings, relevant_per_query, "all-mpnet-base-v2")

    # Results table
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<25} {'NDCG@10':>8} {'MRR@10':>8} {'R@10':>8} {'R@100':>8}")
    print("-" * 60)
    for results in [bm25_results, mpnet_results]:
        print(
            f"{results['method']:<25} "
            f"{results['ndcg@10']:>8.3f} "
            f"{results['mrr@10']:>8.3f} "
            f"{results['recall@10']:>8.3f} "
            f"{results['recall@100']:>8.3f}"
        )

    # Per-query analysis: where does each method fail?
    print("\n--- Per-Query Analysis ---")
    bm25_wins = 0
    dense_wins = 0
    ties = 0
    bm25_perfect = 0
    dense_perfect = 0

    for i in range(len(queries)):
        bm25_ndcg = bm25_results["per_query_ndcg"][i]
        dense_ndcg = mpnet_results["per_query_ndcg"][i]

        if bm25_ndcg > dense_ndcg + 0.01:
            bm25_wins += 1
        elif dense_ndcg > bm25_ndcg + 0.01:
            dense_wins += 1
        else:
            ties += 1

        if bm25_ndcg >= 0.99:
            bm25_perfect += 1
        if dense_ndcg >= 0.99:
            dense_perfect += 1

    print(f"BM25 wins: {bm25_wins}/{len(queries)}")
    print(f"Dense wins: {dense_wins}/{len(queries)}")
    print(f"Ties: {ties}/{len(queries)}")
    print(f"BM25 perfect (NDCG=1.0): {bm25_perfect}/{len(queries)}")
    print(f"Dense perfect (NDCG=1.0): {dense_perfect}/{len(queries)}")

    # Show worst queries for dense model
    print("\n--- Worst queries for dense model (where fine-tuning could help) ---")
    dense_ndcg_arr = np.array(mpnet_results["per_query_ndcg"])
    worst_indices = np.argsort(dense_ndcg_arr)[:5]
    for idx in worst_indices:
        print(f"\n  Query: {queries[idx][:100]}")
        print(f"  Passage: {corpus[idx][:100]}")
        print(f"  Dense NDCG: {mpnet_results['per_query_ndcg'][idx]:.3f}")
        print(f"  BM25 NDCG: {bm25_results['per_query_ndcg'][idx]:.3f}")
        print(f"  Topic: {pairs[idx].topic}")

    # Go/no-go assessment
    print("\n" + "=" * 60)
    dense_ndcg = mpnet_results["ndcg@10"]
    if dense_ndcg < 0.9:
        print(f"GO: Dense NDCG@10 = {dense_ndcg:.3f} < 0.9")
        print("There IS room for fine-tuning to improve retrieval.")
        gap = 1.0 - dense_ndcg
        print(f"Potential improvement gap: {gap:.1%}")
    else:
        print(f"WARNING: Dense NDCG@10 = {dense_ndcg:.3f} >= 0.9")
        print("Off-the-shelf model already near-perfect. Limited room for improvement.")
        print("Consider: is the evaluation task too easy? Are queries too similar to passages?")
    print("=" * 60)

    # Save results
    results_path = project_root / "data" / "evaluation" / "q3_feasibility_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {
                "num_pairs": len(pairs),
                "bm25": bm25_results,
                "mpnet": mpnet_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
