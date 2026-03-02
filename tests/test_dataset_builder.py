"""Tests for dataset builder: deduplication, filtering, splitting."""

import pytest

from src.data.dataset_builder import (
    _truncate_at_sentence,
    build_eval_corpus,
    deduplicate_pairs,
    quality_filter,
    split_train_eval,
)
from src.data.schemas import QueryPassagePair


def _make_pair(query="What are Scope 1 emissions?", passage="In 2024, total Scope 1 " * 5,
               source="esgbench", topic="environmental"):
    return QueryPassagePair(
        query=query, passage=passage, source=source, topic=topic, doc_id="test"
    )


class TestDeduplicatePairs:
    def test_removes_exact_duplicates(self):
        p = _make_pair()
        result = deduplicate_pairs([p, p])
        assert len(result) == 1

    def test_keeps_same_query_different_passage(self):
        p1 = _make_pair(passage="Passage A " * 10)
        p2 = _make_pair(passage="Passage B " * 10)
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 2  # Same query, different passage — both valid

    def test_keeps_unique_pairs(self):
        p1 = _make_pair(query="Query one?", passage="Passage A " * 10)
        p2 = _make_pair(query="Query two?", passage="Passage B " * 10)
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 2

    def test_case_insensitive(self):
        p1 = _make_pair(query="What are SCOPE 1 emissions?")
        p2 = _make_pair(query="what are scope 1 emissions?")
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1


class TestTruncateAtSentence:
    def test_short_text_unchanged(self):
        assert _truncate_at_sentence("Short text.", 100) == "Short text."

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate_at_sentence(text, 35)
        assert result == "First sentence. Second sentence."

    def test_fallback_to_space(self):
        text = "A very long word sequence without periods that goes on and on"
        result = _truncate_at_sentence(text, 40)
        assert len(result) <= 40
        assert not result.endswith(" ")


class TestQualityFilter:
    def test_filters_short_queries(self):
        p = _make_pair(query="Short")
        result = quality_filter([p])
        assert len(result) == 0

    def test_filters_short_passages(self):
        p = _make_pair(passage="Short")
        result = quality_filter([p])
        assert len(result) == 0

    def test_keeps_good_pairs(self):
        p = _make_pair()
        result = quality_filter([p])
        assert len(result) == 1

    def test_filters_long_queries(self):
        p = _make_pair(query="A " * 200)
        result = quality_filter([p])
        assert len(result) == 0

    def test_truncates_long_passages(self):
        long_passage = ("This is a sentence about ESG reporting. " * 200)
        p = _make_pair(passage=long_passage)
        result = quality_filter([p])
        assert len(result) == 1
        assert len(result[0].passage) <= 4000


class TestSplitTrainEval:
    def test_split_preserves_all_pairs(self):
        pairs = [_make_pair(query=f"Query {i}?", passage=f"Passage {i} " * 10)
                 for i in range(100)]
        train, eval_ = split_train_eval(pairs, eval_fraction=0.2)
        assert len(train) + len(eval_) == len(pairs)

    def test_eval_fraction(self):
        pairs = [_make_pair(query=f"Query {i}?", passage=f"Passage {i} " * 10)
                 for i in range(100)]
        train, eval_ = split_train_eval(pairs, eval_fraction=0.2)
        assert len(eval_) == 20

    def test_prefers_esgbench_for_eval(self):
        esg_pairs = [_make_pair(query=f"ESG {i}?", passage=f"ESG passage {i} " * 10,
                                source="esgbench") for i in range(10)]
        synth_pairs = [_make_pair(query=f"Synth {i}?", passage=f"Synth passage {i} " * 10,
                                  source="synthetic") for i in range(90)]
        _, eval_ = split_train_eval(esg_pairs + synth_pairs, eval_fraction=0.15)
        esg_in_eval = sum(1 for p in eval_ if p.source == "esgbench")
        assert esg_in_eval == 10  # All ESGBench pairs should be in eval

    def test_deterministic(self):
        pairs = [_make_pair(query=f"Query {i}?", passage=f"Passage {i} " * 10)
                 for i in range(50)]
        t1, e1 = split_train_eval(pairs, seed=42)
        t2, e2 = split_train_eval(pairs, seed=42)
        assert [p.query for p in t1] == [p.query for p in t2]
        assert [p.query for p in e1] == [p.query for p in e2]


class TestBuildEvalCorpus:
    def test_creates_corpus_with_distractors(self):
        eval_pairs = [_make_pair(query=f"Eval {i}?", passage=f"Eval passage {i} " * 10)
                      for i in range(5)]
        train_pairs = [_make_pair(query=f"Train {i}?", passage=f"Train passage {i} " * 10)
                       for i in range(20)]
        queries, corpus = build_eval_corpus(eval_pairs, train_pairs, max_corpus_size=15)
        assert len(queries) == 5
        assert len(corpus) >= 5  # At least the eval passages
        assert len(corpus) <= 15  # Respects max_corpus_size

    def test_eval_passages_in_corpus(self):
        eval_pairs = [_make_pair(query="Find this?", passage="Target passage " * 10)]
        train_pairs = [_make_pair(query=f"Train {i}?", passage=f"Distractor {i} " * 10)
                       for i in range(10)]
        queries, corpus = build_eval_corpus(eval_pairs, train_pairs)
        corpus_texts = {p.passage for p in corpus}
        assert eval_pairs[0].passage in corpus_texts

    def test_query_has_relevant_passage_ids(self):
        eval_pairs = [_make_pair(query="Find this?", passage="Target passage " * 10)]
        queries, corpus = build_eval_corpus(eval_pairs, [])
        assert len(queries[0].relevant_passage_ids) == 1
        # The relevant passage should exist in the corpus
        relevant_id = queries[0].relevant_passage_ids[0]
        corpus_ids = {p.passage_id for p in corpus}
        assert relevant_id in corpus_ids
