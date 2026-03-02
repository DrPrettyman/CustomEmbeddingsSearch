"""Tests for synthetic query generation."""

import json

import pytest

from src.data.schemas import QueryPassagePair
from src.data.synthetic_generator import (
    _build_full_prompt,
    _parse_query_response,
    generate_synthetic_pairs,
)


class TestBuildFullPrompt:
    def test_includes_passage_text(self):
        prompt = _build_full_prompt("This is an ESG passage about emissions.")
        assert "ESG passage about emissions" in prompt

    def test_includes_company(self):
        prompt = _build_full_prompt("Passage text.", company="Apple Inc")
        assert "Apple Inc" in prompt

    def test_includes_doc_type(self):
        prompt = _build_full_prompt("Passage text.", doc_type="CDP response")
        assert "CDP response" in prompt

    def test_includes_n_queries(self):
        prompt = _build_full_prompt("Passage text.", n_queries=5)
        assert "5" in prompt

    def test_truncates_long_passages(self):
        long_passage = "x" * 3000
        prompt = _build_full_prompt(long_passage)
        # Passage is truncated to 1500 chars in the template
        assert "x" * 1500 in prompt
        assert "x" * 1501 not in prompt


class TestParseQueryResponse:
    def test_parses_json_array(self):
        text = '["What are Scope 1 emissions?", "Carbon footprint data", "Direct GHG output"]'
        result = _parse_query_response(text, expected_count=3)
        assert len(result) == 3
        assert result[0] == "What are Scope 1 emissions?"

    def test_parses_markdown_code_block(self):
        text = '```json\n["Query one", "Query two"]\n```'
        result = _parse_query_response(text, expected_count=2)
        assert len(result) == 2
        assert result[0] == "Query one"

    def test_parses_bare_code_block(self):
        text = '```\n["Query one", "Query two"]\n```'
        result = _parse_query_response(text, expected_count=2)
        assert len(result) == 2

    def test_fallback_quoted_strings(self):
        text = 'Here are the queries:\n"First query here"\n"Second query here"'
        result = _parse_query_response(text, expected_count=2)
        assert len(result) == 2
        assert result[0] == "First query here"

    def test_fallback_numbered_lines(self):
        text = "1. What are emissions targets?\n2. Carbon reduction strategy\n3. Net zero commitments"
        result = _parse_query_response(text, expected_count=3)
        assert len(result) == 3

    def test_strips_whitespace(self):
        text = '["  Query with spaces  ", "  Another query  "]'
        result = _parse_query_response(text, expected_count=2)
        assert result[0] == "Query with spaces"

    def test_filters_empty_strings(self):
        text = '["Good query", "", "Another query"]'
        result = _parse_query_response(text, expected_count=3)
        assert all(q for q in result)

    def test_respects_expected_count(self):
        text = '"Q1"\n"Q2"\n"Q3"\n"Q4"\n"Q5"'
        result = _parse_query_response(text, expected_count=3)
        assert len(result) <= 3


class TestGenerateSyntheticPairs:
    def test_calls_backend_and_creates_pairs(self, monkeypatch):
        """Test that generate_synthetic_pairs creates QueryPassagePair objects."""
        def fake_generate(passage, company, doc_type, n_queries):
            return [f"Query about {company}" for _ in range(n_queries)]

        monkeypatch.setattr(
            "src.data.synthetic_generator.generate_queries_claude_code",
            fake_generate,
        )

        passages = [
            {
                "text": "Apple reduced Scope 1 emissions by 40% in 2024.",
                "company": "Apple",
                "doc_type": "sustainability report",
                "topic": "environmental",
                "source_file": "apple_esg.pdf",
            }
        ]

        pairs = generate_synthetic_pairs(
            passages, backend="claude-code", n_queries_per_passage=3, rate_limit_delay=0
        )

        assert len(pairs) == 3
        assert all(isinstance(p, QueryPassagePair) for p in pairs)
        assert all(p.source == "synthetic" for p in pairs)
        assert all(p.topic == "environmental" for p in pairs)
        assert all("Apple" in p.query for p in pairs)

    def test_handles_errors_gracefully(self, monkeypatch):
        """Test that errors in generation are caught and counted."""
        call_count = 0

        def failing_generate(passage, company, doc_type, n_queries):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("API error")
            return ["Fallback query"]

        monkeypatch.setattr(
            "src.data.synthetic_generator.generate_queries_claude_code",
            failing_generate,
        )

        passages = [
            {"text": f"Passage {i} about ESG reporting standards." * 5,
             "company": "Test", "doc_type": "report", "topic": "environmental"}
            for i in range(3)
        ]

        pairs = generate_synthetic_pairs(
            passages, backend="claude-code", n_queries_per_passage=1, rate_limit_delay=0
        )

        # First 2 fail, third succeeds
        assert len(pairs) == 1

    def test_max_passages_limit(self, monkeypatch):
        def fake_generate(passage, company, doc_type, n_queries):
            return ["Query"]

        monkeypatch.setattr(
            "src.data.synthetic_generator.generate_queries_claude_code",
            fake_generate,
        )

        passages = [
            {"text": f"Passage {i}" * 20, "company": "Test", "doc_type": "report",
             "topic": "environmental"}
            for i in range(10)
        ]

        pairs = generate_synthetic_pairs(
            passages, backend="claude-code", n_queries_per_passage=1,
            rate_limit_delay=0, max_passages=3,
        )

        assert len(pairs) == 3

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            generate_synthetic_pairs([], backend="invalid")
