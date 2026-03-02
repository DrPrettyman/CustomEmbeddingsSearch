"""Retrieval evaluation metrics: NDCG, MRR, Recall.

Reusable across baseline evaluation, fine-tuned model evaluation,
and ablation studies.
"""

import numpy as np


def ndcg_at_k(
    ranked_indices: list[int], relevant_indices: list[int], k: int = 10
) -> float:
    """Compute NDCG@k for a single query."""
    relevant_set = set(relevant_indices)
    dcg = 0.0
    for i, idx in enumerate(ranked_indices[:k]):
        if idx in relevant_set:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_indices), k)))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def mrr(ranked_indices: list[int], relevant_indices: list[int]) -> float:
    """Compute Reciprocal Rank for a single query."""
    relevant_set = set(relevant_indices)
    for i, idx in enumerate(ranked_indices):
        if idx in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    ranked_indices: list[int], relevant_indices: list[int], k: int = 10
) -> float:
    """Compute Recall@k for a single query."""
    if not relevant_indices:
        return 0.0
    relevant_set = set(relevant_indices)
    retrieved_relevant = sum(1 for idx in ranked_indices[:k] if idx in relevant_set)
    return retrieved_relevant / len(relevant_indices)


def evaluate_rankings(
    rankings: list[list[int]],
    relevant_per_query: list[list[int]],
) -> dict:
    """Compute all metrics for a set of query rankings.

    Returns dict with aggregate metrics and per-query scores.
    """
    ndcg_scores = []
    mrr_scores = []
    r10_scores = []
    r100_scores = []

    for ranking, relevant in zip(rankings, relevant_per_query):
        ndcg_scores.append(ndcg_at_k(ranking, relevant, k=10))
        mrr_scores.append(mrr(ranking, relevant))
        r10_scores.append(recall_at_k(ranking, relevant, k=10))
        r100_scores.append(recall_at_k(ranking, relevant, k=100))

    return {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "mrr@10": float(np.mean(mrr_scores)),
        "recall@10": float(np.mean(r10_scores)),
        "recall@100": float(np.mean(r100_scores)),
        "per_query_ndcg": [float(x) for x in ndcg_scores],
        "per_query_mrr": [float(x) for x in mrr_scores],
    }
