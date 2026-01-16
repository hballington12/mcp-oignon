"""Integration tests against real OpenAlex API.

Reference paper: "Nanometre-scale thermometry in a living cell"
OpenAlex ID: W2159974629
DOI: 10.1038/nature12373

This is the default paper used in the Oignon web app.
"""

import json

import pytest

from oignon.core.builder import build_graph
from oignon.core.openalex import fetch_paper, search_papers
from oignon.storage.memory import GraphStore

# Reference paper details
REFERENCE_PAPER_ID = "W2159974629"
REFERENCE_PAPER_DOI = "10.1038/nature12373"
REFERENCE_PAPER_TITLE = "Nanometre-scale thermometry in a living cell"
REFERENCE_PAPER_YEAR = 2013
REFERENCE_PAPER_SOURCE = "Nature"


class TestOpenAlexAPI:
    """Test OpenAlex API client."""

    def test_fetch_paper_by_id(self):
        """Fetch reference paper by OpenAlex ID."""
        paper = fetch_paper(REFERENCE_PAPER_ID)

        assert paper is not None
        assert paper.id == REFERENCE_PAPER_ID
        assert paper.title == REFERENCE_PAPER_TITLE
        assert paper.year == REFERENCE_PAPER_YEAR
        assert paper.source_name == REFERENCE_PAPER_SOURCE
        assert paper.type == "article"
        assert paper.language == "en"

    def test_fetch_paper_by_doi(self):
        """Fetch reference paper by DOI."""
        paper = fetch_paper(REFERENCE_PAPER_DOI)

        assert paper is not None
        assert paper.id == REFERENCE_PAPER_ID
        assert paper.title == REFERENCE_PAPER_TITLE

    def test_fetch_paper_metadata(self):
        """Verify detailed metadata for reference paper."""
        paper = fetch_paper(REFERENCE_PAPER_ID)

        assert paper is not None

        # Authors - should have G. Kucsko as first author
        assert len(paper.authors) > 0
        author_names = [a.name for a in paper.authors]
        assert any("Kucsko" in name for name in author_names)

        # Citations - paper is highly cited (>1800 as of 2025)
        assert paper.citation_count > 1800

        # References - paper cites ~36 works
        assert paper.references_count >= 30
        assert paper.references_count <= 50

        # FWCI - field-weighted citation impact should be high
        assert paper.fwci is not None
        assert paper.fwci > 40

        # Citation percentile - should be top 1% (value is 0-1 scale)
        assert paper.citation_percentile is not None
        assert paper.citation_percentile.value > 0.99
        assert paper.citation_percentile.is_in_top_1_percent is True

        # Topic hierarchy
        assert paper.primary_topic is not None
        assert paper.primary_topic.domain is not None
        assert paper.primary_topic.domain.get("name") == "Physical Sciences"

    def test_search_papers(self):
        """Search should find the reference paper."""
        results = search_papers("Nanometre-scale thermometry living cell", limit=5)

        assert len(results) > 0

        # Reference paper should be in top results
        ids = [p.id for p in results]
        assert REFERENCE_PAPER_ID in ids

    def test_fetch_nonexistent_paper(self):
        """Fetching nonexistent paper returns None."""
        paper = fetch_paper("INVALID_ID_THAT_DOES_NOT_EXIST")
        assert paper is None


class TestGraphBuilder:
    """Test graph building algorithm."""

    @pytest.fixture(scope="class")
    def graph(self):
        """Build graph once for all tests in this class."""
        return build_graph(REFERENCE_PAPER_ID, n_roots=25, n_branches=25)

    def test_source_paper(self, graph):
        """Source paper should be correctly identified."""
        assert graph.source_paper.id == REFERENCE_PAPER_ID
        assert graph.source_paper.title == REFERENCE_PAPER_TITLE
        assert graph.source_paper.year == REFERENCE_PAPER_YEAR

    def test_root_seeds_count(self, graph):
        """Should fetch root seeds (papers the source cites)."""
        # Source cites ~36 papers, most should be fetched
        assert len(graph.root_seeds) >= 25
        assert len(graph.root_seeds) <= 50

    def test_branch_seeds_count(self, graph):
        """Should fetch branch seeds (papers citing the source)."""
        # Source is highly cited, should have many branch seeds
        assert len(graph.branch_seeds) >= 50

    def test_top_roots_count(self, graph):
        """Should select requested number of top roots."""
        roots = [p for p in graph.papers if p.role == "root"]
        assert len(roots) == 25

    def test_top_branches_count(self, graph):
        """Should select requested number of top branches."""
        branches = [p for p in graph.papers if p.role == "branch"]
        assert len(branches) == 25

    def test_total_papers(self, graph):
        """Total papers should be substantial."""
        total = 1 + len(graph.root_seeds) + len(graph.branch_seeds) + len(graph.papers)
        assert total > 50

    def test_edges_exist(self, graph):
        """Graph should have citation edges."""
        assert len(graph.edges) > 0

        # Edges should reference papers in the graph
        all_ids = (
            {graph.source_paper.id}
            | {p.id for p in graph.root_seeds}
            | {p.id for p in graph.branch_seeds}
            | {p.id for p in graph.papers}
        )

        for edge in graph.edges[:10]:  # Check first 10
            assert edge["source"] in all_ids
            assert edge["target"] in all_ids
            assert edge["type"] == "cites"

    def test_metadata(self, graph):
        """Metadata should be populated."""
        meta = graph.metadata

        assert meta.source_year == REFERENCE_PAPER_YEAR
        assert meta.n_roots == 25
        assert meta.n_branches == 25
        assert meta.papers_in_graph == 50
        assert meta.edges_in_graph > 0
        assert meta.build_time_seconds > 0
        assert meta.api_calls > 0
        assert meta.timestamp is not None

    def test_root_papers_are_older(self, graph):
        """Root papers should generally be older than source."""
        roots = [p for p in graph.papers if p.role == "root"]
        older_count = sum(1 for p in roots if p.year <= REFERENCE_PAPER_YEAR)

        # Most roots should be from before or same year as source
        assert older_count >= len(roots) * 0.8

    def test_branch_papers_are_newer(self, graph):
        """Branch papers should be newer than source."""
        branches = [p for p in graph.papers if p.role == "branch"]
        newer_count = sum(1 for p in branches if p.year > REFERENCE_PAPER_YEAR)

        # All branches should be newer (by algorithm design)
        assert newer_count == len(branches)

    def test_papers_have_ranks(self, graph):
        """Top papers should have rank scores."""
        for paper in graph.papers:
            assert paper.rank is not None
            assert paper.rank >= 0
            assert paper.rank_details is not None


class TestGraphStore:
    """Test in-memory graph storage."""

    @pytest.fixture(scope="class")
    def loaded_store(self):
        """Build graph and load into store."""
        graph = build_graph(REFERENCE_PAPER_ID, n_roots=25, n_branches=25)
        store = GraphStore()
        store.load(graph)
        return store

    def test_is_loaded(self, loaded_store):
        """Store should report as loaded."""
        assert loaded_store.is_loaded() is True

    def test_empty_store(self):
        """Empty store should report as not loaded."""
        store = GraphStore()
        assert store.is_loaded() is False

    def test_entity_count(self, loaded_store):
        """Should have many entities."""
        stats = loaded_store.get_stats()
        assert stats["entities"] > 50

    def test_relation_count(self, loaded_store):
        """Should have citation relations."""
        stats = loaded_store.get_stats()
        assert stats["relations"] > 0

    def test_get_source_entity(self, loaded_store):
        """Should retrieve source paper entity."""
        entity = loaded_store.get_entity(REFERENCE_PAPER_ID)

        assert entity is not None
        assert entity.name == REFERENCE_PAPER_ID
        assert entity.entity_type == "article"

        # Check observations contain expected data
        obs_text = " ".join(entity.observations)
        assert REFERENCE_PAPER_TITLE in obs_text
        assert "2013" in obs_text
        assert "source" in obs_text.lower()

    def test_search_by_title(self, loaded_store):
        """Search should find papers by title."""
        results = loaded_store.search("thermometry")

        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert REFERENCE_PAPER_ID in ids

    def test_search_by_year(self, loaded_store):
        """Search should find papers by year."""
        results = loaded_store.search("2013")

        assert len(results) > 0

    def test_search_by_topic(self, loaded_store):
        """Search should find papers by topic/field."""
        results = loaded_store.search("diamond")

        # May or may not have results depending on graph content
        assert isinstance(results, list)

    def test_get_citations_cited_by(self, loaded_store):
        """Should get papers citing the source."""
        citing = loaded_store.get_citations(REFERENCE_PAPER_ID, "cited_by")

        # Source is highly cited, should have papers citing it in graph
        assert len(citing) > 0

    def test_get_citations_cites(self, loaded_store):
        """Should get papers the source cites."""
        cites = loaded_store.get_citations(REFERENCE_PAPER_ID, "cites")

        # Source cites other papers
        assert len(cites) > 0

    def test_stats_year_distribution(self, loaded_store):
        """Stats should include year distribution."""
        stats = loaded_store.get_stats()

        assert "by_year" in stats
        assert len(stats["by_year"]) > 0
        assert "2013" in stats["by_year"]

    def test_stats_role_distribution(self, loaded_store):
        """Stats should include role distribution."""
        stats = loaded_store.get_stats()

        assert "by_role" in stats
        assert "source" in stats["by_role"]
        assert "root" in stats["by_role"]
        assert "branch" in stats["by_role"]

    def test_clear(self, loaded_store):
        """Clear should empty the store."""
        # Create a fresh store to test clear without affecting other tests
        graph = build_graph(REFERENCE_PAPER_ID, n_roots=5, n_branches=5)
        store = GraphStore()
        store.load(graph)

        assert store.is_loaded() is True

        store.clear()

        assert store.is_loaded() is False
        assert store.get_stats() == {}


class TestMCPTools:
    """Test MCP tool responses."""

    @pytest.fixture(scope="class")
    def built_graph(self):
        """Build graph via the tool interface."""
        from oignon.server import build_citation_graph, get_store

        import asyncio

        # Build graph
        result = asyncio.get_event_loop().run_until_complete(
            build_citation_graph(REFERENCE_PAPER_ID, n_roots=25, n_branches=25)
        )
        return json.loads(result)

    def test_search_paper_tool(self):
        """search_paper should return list of papers."""
        from oignon.server import search_paper

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            search_paper("Nanometre-scale thermometry")
        )
        data = json.loads(result)

        assert isinstance(data, list)
        assert len(data) > 0

        # Each result should have expected fields
        paper = data[0]
        assert "id" in paper
        assert "title" in paper
        assert "authors" in paper
        assert "year" in paper
        assert "citations" in paper

    def test_get_paper_tool(self):
        """get_paper should return paper details."""
        from oignon.server import get_paper

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_paper(REFERENCE_PAPER_ID)
        )
        data = json.loads(result)

        assert data["id"] == REFERENCE_PAPER_ID
        assert data["title"] == REFERENCE_PAPER_TITLE
        assert data["year"] == REFERENCE_PAPER_YEAR
        assert data["citations"] > 1800
        assert isinstance(data["authors"], list)

    def test_get_paper_tool_error(self):
        """get_paper should return error for invalid ID."""
        from oignon.server import get_paper

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_paper("INVALID_ID_THAT_DOES_NOT_EXIST")
        )
        data = json.loads(result)

        assert "error" in data

    def test_build_citation_graph_tool(self, built_graph):
        """build_citation_graph should return summary."""
        assert "source" in built_graph
        assert built_graph["source"]["id"] == REFERENCE_PAPER_ID
        assert built_graph["source"]["title"] == REFERENCE_PAPER_TITLE

        assert "counts" in built_graph
        assert built_graph["counts"]["entities"] > 50
        assert built_graph["counts"]["top_roots"] == 25
        assert built_graph["counts"]["top_branches"] == 25

        assert "top_roots" in built_graph
        assert "top_branches" in built_graph
        assert "metadata" in built_graph

    def test_search_graph_tool(self, built_graph):
        """search_graph should find papers in loaded graph."""
        from oignon.server import search_graph

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            search_graph("thermometry")
        )
        data = json.loads(result)

        assert "found" in data
        assert data["found"] > 0
        assert "papers" in data
        assert len(data["papers"]) > 0

    def test_get_graph_node_tool(self, built_graph):
        """get_graph_node should return entity details."""
        from oignon.server import get_graph_node

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_graph_node(REFERENCE_PAPER_ID)
        )
        data = json.loads(result)

        assert data["id"] == REFERENCE_PAPER_ID
        assert data["type"] == "article"
        assert "observations" in data
        assert isinstance(data["observations"], list)
        assert len(data["observations"]) > 0

    def test_get_graph_node_tool_error(self, built_graph):
        """get_graph_node should return error for unknown paper."""
        from oignon.server import get_graph_node

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_graph_node("W0000000000")
        )
        data = json.loads(result)

        assert "error" in data

    def test_get_citations_tool(self, built_graph):
        """get_citations should return connected papers."""
        from oignon.server import get_citations

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_citations(REFERENCE_PAPER_ID, "cited_by")
        )
        data = json.loads(result)

        assert data["paper_id"] == REFERENCE_PAPER_ID
        assert data["direction"] == "cited_by"
        assert "total" in data
        assert "papers" in data

    def test_get_graph_stats_tool(self, built_graph):
        """get_graph_stats should return statistics."""
        from oignon.server import get_graph_stats

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(get_graph_stats())
        data = json.loads(result)

        assert "entities" in data
        assert data["entities"] > 50
        assert "relations" in data
        assert data["relations"] > 0
        assert "by_year" in data
        assert "by_role" in data

    def test_get_all_papers_tool(self, built_graph):
        """get_all_papers should return all papers."""
        from oignon.server import get_all_papers

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_all_papers(sort_by="year")
        )
        data = json.loads(result)

        assert data["total"] > 50
        assert data["sort_by"] == "year"
        assert "papers" in data
        assert len(data["papers"]) > 50

        # Check paper structure
        paper = data["papers"][0]
        assert "id" in paper
        assert "title" in paper
        assert "year" in paper
        assert "role" in paper

    def test_get_all_papers_sorted_by_year(self, built_graph):
        """Papers should be sorted by year (newest first)."""
        from oignon.server import get_all_papers

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_all_papers(sort_by="year")
        )
        data = json.loads(result)

        years = [p["year"] for p in data["papers"] if p["year"]]
        # Check descending order
        assert years == sorted(years, reverse=True)

    def test_get_all_papers_sorted_by_role(self, built_graph):
        """Papers should be sorted by role."""
        from oignon.server import get_all_papers

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_all_papers(sort_by="role")
        )
        data = json.loads(result)

        roles = [p["role"] for p in data["papers"]]
        # Source should come first
        assert roles[0] == "source"
