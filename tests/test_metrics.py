"""Tests for retrieval evaluation metrics."""

import pytest

from src.evaluation.metrics import evaluate_rankings, mrr, ndcg_at_k, recall_at_k


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # Relevant doc at position 0
        assert ndcg_at_k([0, 1, 2], [0], k=10) == 1.0

    def test_relevant_at_position_two(self):
        # Relevant doc at position 1 (0-indexed)
        score = ndcg_at_k([1, 0, 2], [0], k=10)
        assert 0.0 < score < 1.0

    def test_no_relevant_in_top_k(self):
        # Relevant doc at position 100, k=10
        ranking = list(range(100))
        assert ndcg_at_k(ranking, [99], k=10) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k([0, 1, 2], [], k=10) == 0.0

    def test_multiple_relevant(self):
        # Both relevant docs at top
        score = ndcg_at_k([0, 1, 2, 3], [0, 1], k=10)
        assert score == 1.0


class TestMrr:
    def test_relevant_at_rank_1(self):
        assert mrr([0, 1, 2], [0]) == 1.0

    def test_relevant_at_rank_2(self):
        assert mrr([1, 0, 2], [0]) == 0.5

    def test_relevant_at_rank_3(self):
        assert mrr([2, 1, 0], [0]) == pytest.approx(1.0 / 3)

    def test_no_relevant(self):
        assert mrr([0, 1, 2], [5]) == 0.0


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k([0, 1, 2], [0, 1], k=10) == 1.0

    def test_partial_recall(self):
        assert recall_at_k([0, 1, 2], [0, 5], k=3) == 0.5

    def test_no_relevant_retrieved(self):
        assert recall_at_k([0, 1, 2], [5, 6], k=3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k([0, 1, 2], [], k=3) == 0.0


class TestEvaluateRankings:
    def test_returns_all_metrics(self):
        rankings = [[0, 1, 2], [1, 0, 2]]
        relevant = [[0], [0]]
        results = evaluate_rankings(rankings, relevant)
        assert "ndcg@10" in results
        assert "mrr@10" in results
        assert "recall@10" in results
        assert "recall@100" in results
        assert "per_query_ndcg" in results
        assert len(results["per_query_ndcg"]) == 2

    def test_perfect_rankings(self):
        rankings = [[0, 1, 2], [0, 1, 2]]
        relevant = [[0], [0]]
        results = evaluate_rankings(rankings, relevant)
        assert results["ndcg@10"] == 1.0
        assert results["mrr@10"] == 1.0
