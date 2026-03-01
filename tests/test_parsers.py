"""Tests for ESGBench parser."""

import pytest

from src.data.esgbench_parser import (
    CATEGORY_MAP,
    MIN_PASSAGE_LENGTH,
    parse_esgbench_record,
)
from src.data.schemas import QueryPassagePair


class TestParseEsgbenchRecord:
    """Tests for converting a single ESGBench record to (query, passage) pairs."""

    def _make_record(self, **overrides):
        """Helper to create a valid ESGBench record with sensible defaults."""
        base = {
            "esgbench_id": None,
            "company": "Test Corp",
            "doc_name": "TEST_CORP_2024_ESG",
            "category": "Environmental",
            "question": "What are the company's Scope 1 emissions?",
            "answer": "1000 tCO2e",
            "evidence": [
                {
                    "evidence_text": "In 2024, direct Scope 1 greenhouse gas emissions "
                    "totaled 1,000 tCO2e across all operations globally.",
                    "evidence_page_num": 42,
                    "evidence_doc_name": "TEST_CORP_2024_ESG",
                }
            ],
        }
        base.update(overrides)
        return base

    def test_basic_parsing(self):
        record = self._make_record()
        pairs = parse_esgbench_record(record)
        assert len(pairs) == 1
        assert isinstance(pairs[0], QueryPassagePair)
        assert pairs[0].query == "What are the company's Scope 1 emissions?"
        assert "Scope 1" in pairs[0].passage
        assert pairs[0].source == "esgbench"
        assert pairs[0].topic == "environmental"

    def test_empty_question_skipped(self):
        record = self._make_record(question="")
        assert parse_esgbench_record(record) == []

    def test_missing_question_skipped(self):
        record = self._make_record()
        del record["question"]
        assert parse_esgbench_record(record) == []

    def test_no_evidence_skipped(self):
        record = self._make_record(evidence=[])
        assert parse_esgbench_record(record) == []

    def test_short_evidence_filtered(self):
        record = self._make_record(
            evidence=[
                {
                    "evidence_text": "Short",  # Below MIN_PASSAGE_LENGTH
                    "evidence_page_num": 1,
                    "evidence_doc_name": "TEST",
                }
            ]
        )
        pairs = parse_esgbench_record(record)
        assert len(pairs) == 0

    def test_multiple_evidence_items(self):
        long_text = "A" * (MIN_PASSAGE_LENGTH + 10)
        record = self._make_record(
            evidence=[
                {
                    "evidence_text": long_text + " first",
                    "evidence_page_num": 10,
                    "evidence_doc_name": "DOC1",
                },
                {
                    "evidence_text": long_text + " second",
                    "evidence_page_num": 20,
                    "evidence_doc_name": "DOC1",
                },
            ]
        )
        pairs = parse_esgbench_record(record)
        assert len(pairs) == 2
        assert "first" in pairs[0].passage
        assert "second" in pairs[1].passage

    def test_category_mapping(self):
        for esg_cat, expected_topic in CATEGORY_MAP.items():
            record = self._make_record(category=esg_cat.title())
            pairs = parse_esgbench_record(record)
            if pairs:
                assert pairs[0].topic == expected_topic

    def test_unknown_category(self):
        record = self._make_record(category="NewCategory")
        pairs = parse_esgbench_record(record)
        assert pairs[0].topic == "unknown"

    def test_metadata_preserved(self):
        record = self._make_record()
        pairs = parse_esgbench_record(record)
        assert pairs[0].metadata["company"] == "Test Corp"
        assert pairs[0].metadata["page_num"] == 42
        assert pairs[0].doc_id == "TEST_CORP_2024_ESG"
