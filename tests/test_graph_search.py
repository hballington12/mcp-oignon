"""Offline unit tests for ranked graph search (no network required)."""

from oignon.core.graph import (
    Author,
    FullPaper,
    Graph,
    GraphMetadata,
    PrimaryTopic,
)
from oignon.storage.memory import GraphStore
from oignon.storage.search import SearchIndex, pick_snippets, tokenize


def make_paper(
    paper_id: str,
    title: str,
    abstract: str = "",
    keywords: list[str] | None = None,
    topic: str | None = None,
    year: int = 2020,
    authors: list[str] | None = None,
    role: str | None = None,
) -> FullPaper:
    return FullPaper(
        id=paper_id,
        doi=None,
        title=title,
        authors=[Author(name=n) for n in (authors or ["A. Nonymous"])],
        year=year,
        citation_count=10,
        references_count=0,
        references=[],
        openalex_url=f"https://openalex.org/{paper_id}",
        abstract=abstract,
        keywords=keywords,
        primary_topic=PrimaryTopic(id="T1", name=topic) if topic else None,
        role=role,
    )


def make_graph(source: FullPaper, papers: list[FullPaper]) -> Graph:
    metadata = GraphMetadata(
        source_year=source.year,
        total_root_seeds=0,
        total_root_papers=0,
        total_branch_seeds=0,
        total_branch_papers=0,
        n_roots=0,
        n_branches=0,
        papers_in_graph=len(papers),
        edges_in_graph=0,
        build_time_seconds=0.0,
        api_calls=0,
        timestamp="",
    )
    return Graph(
        source_paper=source,
        root_seeds=[],
        branch_seeds=[],
        papers=papers,
        edges=[],
        metadata=metadata,
    )


def build_store() -> GraphStore:
    source = make_paper(
        "W1",
        "Light scattering by hexagonal ice crystals",
        abstract="We compute scattering phase functions for cirrus particles.",
        topic="Atmospheric Optics",
        year=2015,
    )
    papers = [
        make_paper(
            "W2",
            "Surface roughness effects on ice crystal scattering",
            abstract=(
                "Roughness of the crystal surface changes the asymmetry "
                "parameter. We model surface roughness with tilted facets."
            ),
            keywords=["surface roughness", "ice crystals"],
            topic="Atmospheric Optics",
            year=2018,
            authors=["C. Liu", "P. Yang"],
            role="branch",
        ),
        make_paper(
            "W3",
            "A review of climate feedback mechanisms",
            abstract=(
                "General circulation models show varied feedbacks. Ice "
                "clouds are mentioned briefly. Roughness is not discussed."
            ),
            topic="Climate Modeling",
            year=2019,
            role="branch",
        ),
        make_paper(
            "W4",
            "Geometric optics for large particles",
            abstract="Ray tracing methods for particles much larger than the wavelength.",
            topic="Atmospheric Optics",
            year=2010,
            authors=["A. Macke"],
            role="root",
        ),
    ]
    store = GraphStore()
    store.load(make_graph(source, papers))
    return store


class TestTokenize:
    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("The Effect of Ice") == ["effect", "ice"]

    def test_keeps_numbers(self):
        assert "2015" in tokenize("published in 2015")


class TestSearchIndex:
    def test_ranks_title_match_above_abstract_mention(self):
        index = SearchIndex()
        index.add("title_hit", {"title": "roughness models", "abstract": "other text"})
        index.add("abstract_hit", {"title": "other text", "abstract": "roughness models"})

        results = index.search("roughness")
        assert results[0][0] == "title_hit"

    def test_multi_term_beats_single_term(self):
        index = SearchIndex()
        index.add("both", {"abstract": "ice crystal scattering results"})
        index.add("one", {"abstract": "ice sheet melting results"})

        results = index.search("ice crystal")
        assert results[0][0] == "both"

    def test_empty_query_returns_nothing(self):
        index = SearchIndex()
        index.add("doc", {"title": "something"})
        assert index.search("the of and") == []


class TestGraphStoreSearch:
    def test_topic_phrase_ranks_specialist_paper_first(self):
        store = build_store()
        results = store.search("surface roughness")

        assert len(results) > 0
        assert results[0]["id"] == "W2"
        assert results[0]["score"] > 0

    def test_results_include_snippets(self):
        store = build_store()
        results = store.search("surface roughness")

        matched = results[0]["matched"]
        assert len(matched) > 0
        assert any("roughness" in s.lower() for s in matched)

    def test_role_filter(self):
        store = build_store()
        results = store.search("particles scattering", role="root")

        assert all(r["role"] == "root" for r in results)
        assert any(r["id"] == "W4" for r in results)

    def test_author_search(self):
        store = build_store()
        results = store.search("Macke")

        assert results[0]["id"] == "W4"

    def test_substring_fallback_finds_ids(self):
        store = build_store()
        results = store.search("W3")

        assert any(r["id"] == "W3" for r in results)

    def test_stats_include_topics(self):
        store = build_store()
        stats = store.get_stats()

        assert stats["by_topic"]["Atmospheric Optics"] == 3
        assert stats["by_topic"]["Climate Modeling"] == 1

    def test_clear_resets_index(self):
        store = build_store()
        store.clear()

        assert store.search("roughness") == []


class TestSnippets:
    def test_skips_title_observation(self):
        obs = ["Title: roughness study", "Roughness changes scattering."]
        snippets = pick_snippets(obs, "roughness")

        assert snippets == ["Roughness changes scattering."]
