"""Offline unit tests for multi-strategy paper search helpers."""

from oignon.core.paper_search import (
    _author_match,
    _title_similarity,
    detect_work_id,
)
from tests.test_graph_search import make_paper


class TestDetectWorkId:
    def test_openalex_id(self):
        assert detect_work_id("W2159974629") == "W2159974629"
        assert detect_work_id("w2159974629") == "W2159974629"

    def test_bare_doi(self):
        assert detect_work_id("10.1038/nature12373") == "10.1038/nature12373"

    def test_doi_url(self):
        assert detect_work_id("https://doi.org/10.1038/nature12373") == (
            "10.1038/nature12373"
        )

    def test_plain_text_is_not_an_id(self):
        assert detect_work_id("nanoscale thermometry in cells") is None

    def test_partial_w_word_is_not_an_id(self):
        assert detect_work_id("Water dynamics") is None


class TestTitleSimilarity:
    def test_exact_title_scores_high(self):
        title = "Nanometre-scale thermometry in a living cell"
        assert _title_similarity(title, title) == 1.0

    def test_containment_when_title_has_extra_words(self):
        sim = _title_similarity(
            "ice crystal roughness",
            "Effects of ice crystal surface roughness on scattering",
        )
        assert sim == 1.0

    def test_unrelated_titles_score_low(self):
        sim = _title_similarity(
            "quantum thermometry",
            "A survey of deep learning for natural language",
        )
        assert sim < 0.5


class TestAuthorMatch:
    def test_surname_matches(self):
        paper = make_paper("W1", "Some title", authors=["Ping Yang", "Chao Liu"])
        assert _author_match("Yang", paper) == 1.0

    def test_initials_are_ignored(self):
        paper = make_paper("W1", "Some title", authors=["Ping Yang"])
        assert _author_match("P. Yang", paper) == 1.0

    def test_no_match(self):
        paper = make_paper("W1", "Some title", authors=["Ping Yang"])
        assert _author_match("Smith", paper) == 0.0
