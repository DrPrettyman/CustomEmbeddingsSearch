"""Tests for CDP questionnaire parser."""

import pytest

from src.data.cdp_parser import (
    MIN_ANSWER_CHARS,
    _get_topic_for_question,
    _is_substantive_answer,
    extract_cdp_qa_pairs,
)


class TestGetTopicForQuestion:
    def test_governance_questions(self):
        assert _get_topic_for_question("1.3") == "governance"
        assert _get_topic_for_question("4.1.1") == "governance"

    def test_environmental_questions(self):
        assert _get_topic_for_question("7.1") == "environmental"
        assert _get_topic_for_question("6.1") == "environmental"

    def test_invalid_question(self):
        assert _get_topic_for_question("invalid") == "unknown"


class TestIsSubstantiveAnswer:
    def test_checkbox_only(self):
        text = "Select from:\n☑ Yes\n☑ No"
        assert not _is_substantive_answer(text)

    def test_short_text(self):
        text = "We use renewable energy."
        assert not _is_substantive_answer(text)

    def test_substantive_text(self):
        text = (
            "Apple has implemented a comprehensive climate strategy that addresses "
            "both direct and indirect emissions across our value chain. Our approach "
            "includes setting science-based targets aligned with a 1.5 degree pathway, "
            "investing in renewable energy procurement for all global operations, and "
            "engaging with our supply chain partners to reduce their carbon footprint. "
            "We have achieved carbon neutrality for our corporate operations since 2020."
        )
        assert _is_substantive_answer(text)


class TestExtractCdpQaPairs:
    def _make_cdp_text(self, question_id, question, answer):
        return f"\n({question_id}) {question}\n{answer}\n"

    def test_extracts_substantive_pair(self):
        answer = (
            "Our organization has implemented a comprehensive environmental management "
            "system that covers all operational facilities worldwide. This system includes "
            "regular monitoring of greenhouse gas emissions, water usage, waste generation, "
            "and biodiversity impacts. We conduct annual environmental audits and report "
            "our findings through multiple frameworks including GRI, TCFD, and CDP."
        )
        text = self._make_cdp_text("1.3.3", "Describe your organization's approach.", answer)
        pairs = extract_cdp_qa_pairs(text, company="Test Corp", doc_name="test_cdp")
        assert len(pairs) == 1
        assert pairs[0].source == "cdp"
        assert pairs[0].metadata["question_id"] == "1.3.3"

    def test_filters_checkbox_answers(self):
        text = self._make_cdp_text(
            "1.1", "In which language are you submitting?", "Select from:\n☑ English"
        )
        pairs = extract_cdp_qa_pairs(text, company="Test", doc_name="test")
        assert len(pairs) == 0

    def test_filters_short_answers(self):
        text = self._make_cdp_text("1.2", "Select the currency.", "USD")
        pairs = extract_cdp_qa_pairs(text, company="Test", doc_name="test")
        assert len(pairs) == 0

    def test_skips_toc_entries(self):
        text = "\n(4.1.1) Is there board-level oversight .................. 23\nSome text\n"
        pairs = extract_cdp_qa_pairs(text, company="Test", doc_name="test")
        assert len(pairs) == 0

    def test_multiple_questions(self):
        answer = (
            "We have established a comprehensive governance framework for managing "
            "environmental issues across our organization. The board of directors maintains "
            "oversight through a dedicated sustainability committee that meets quarterly "
            "to review environmental performance metrics and strategic initiatives. "
            "Senior management is accountable for implementing environmental policies."
        )
        text = (
            self._make_cdp_text("4.1.1", "Is there board oversight?", answer)
            + self._make_cdp_text("4.1.2", "Identify the positions.", answer)
        )
        pairs = extract_cdp_qa_pairs(text, company="Test", doc_name="test")
        assert len(pairs) == 2
