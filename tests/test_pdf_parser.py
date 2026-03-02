"""Tests for PDF text extraction and passage chunking."""

import pytest

from src.data.pdf_parser import (
    MIN_CHUNK_CHARS,
    Passage,
    chunk_text,
    clean_page_text,
    is_section_header,
)


class TestCleanPageText:
    def test_strips_page_numbers(self):
        text = "Some content\n42\nMore content"
        cleaned = clean_page_text(text)
        assert "42" not in cleaned.split("\n")
        assert "Some content" in cleaned

    def test_strips_copyright(self):
        text = "Content here\n© 2024 Company Inc. All rights reserved.\nMore content"
        cleaned = clean_page_text(text)
        assert "©" not in cleaned
        assert "Content here" in cleaned

    def test_collapses_blank_lines(self):
        text = "Paragraph 1\n\n\n\n\nParagraph 2"
        cleaned = clean_page_text(text)
        assert "\n\n\n" not in cleaned
        assert "Paragraph 1" in cleaned
        assert "Paragraph 2" in cleaned

    def test_empty_input(self):
        assert clean_page_text("") == ""
        assert clean_page_text("   \n\n  ") == ""


class TestIsSectionHeader:
    def test_gri_header(self):
        assert is_section_header("GRI 305: Emissions 2024")

    def test_tcfd_header(self):
        assert is_section_header("TCFD Governance Recommendations")

    def test_scope_header(self):
        assert is_section_header("Scope 1 Direct Emissions")

    def test_governance_header(self):
        assert is_section_header("Governance")
        assert is_section_header("Strategy")
        assert is_section_header("Risk Management")

    def test_not_header(self):
        assert not is_section_header(
            "This is a normal paragraph with lots of words that describes emissions data."
        )

    def test_empty(self):
        assert not is_section_header("")
        assert not is_section_header("   ")


class TestChunkText:
    def _long_paragraph(self, n_chars: int) -> str:
        """Generate a paragraph of approximately n_chars."""
        sentence = "This is a sentence about ESG emissions reporting standards. "
        repeats = n_chars // len(sentence) + 1
        return (sentence * repeats)[:n_chars]

    def test_single_short_paragraph(self):
        text = self._long_paragraph(200)
        passages = chunk_text(text, "test.pdf", page_num=0)
        assert len(passages) == 1
        assert passages[0].source_file == "test.pdf"
        assert passages[0].page_num == 0
        assert isinstance(passages[0], Passage)

    def test_filters_below_min_chars(self):
        text = "Too short"
        passages = chunk_text(text, "test.pdf", page_num=0)
        assert len(passages) == 0

    def test_merges_short_paragraphs(self):
        para = self._long_paragraph(150)
        text = f"{para}\n\n{para}"
        passages = chunk_text(text, "test.pdf", page_num=0, target_chars=500)
        # Two 150-char paragraphs should merge into one chunk
        assert len(passages) == 1

    def test_splits_long_paragraphs(self):
        text = self._long_paragraph(3000)
        passages = chunk_text(text, "test.pdf", page_num=0, target_chars=500, max_chars=1000)
        assert len(passages) > 1
        for p in passages:
            assert len(p.text) <= 1500  # Some slack for overlap

    def test_respects_paragraph_boundaries(self):
        p1 = self._long_paragraph(400)
        p2 = self._long_paragraph(400)
        text = f"{p1}\n\n{p2}"
        passages = chunk_text(text, "test.pdf", page_num=0, target_chars=450)
        assert len(passages) == 2

    def test_passage_ids_are_deterministic(self):
        text = self._long_paragraph(300)
        p1 = chunk_text(text, "test.pdf", page_num=0)
        p2 = chunk_text(text, "test.pdf", page_num=0)
        assert p1[0].passage_id == p2[0].passage_id

    def test_different_content_different_ids(self):
        t1 = self._long_paragraph(200) + " alpha"
        t2 = self._long_paragraph(200) + " beta"
        p1 = chunk_text(t1, "test.pdf", page_num=0)
        p2 = chunk_text(t2, "test.pdf", page_num=0)
        assert p1[0].passage_id != p2[0].passage_id
