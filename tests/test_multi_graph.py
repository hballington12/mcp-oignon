"""Offline unit tests for the multi-graph registry."""

from oignon.storage.registry import GraphRegistry
from tests.test_graph_search import make_graph, make_paper


def graph_one():
    source = make_paper(
        "W100",
        "Light scattering by ice crystals",
        abstract="Scattering phase functions for cirrus clouds.",
        topic="Atmospheric Optics",
    )
    papers = [
        make_paper(
            "W101",
            "Surface roughness of hexagonal columns",
            abstract="Roughness alters the phase function.",
            role="branch",
        ),
    ]
    return make_graph(source, papers)


def graph_two():
    source = make_paper(
        "W101",  # a paper that also exists inside graph one
        "Surface roughness of hexagonal columns",
        abstract="Roughness alters the phase function.",
        topic="Atmospheric Optics",
    )
    papers = [
        make_paper(
            "W201",
            "Facet tilting models for rough particles",
            abstract="Tilted facet statistics reproduce measured scattering.",
            role="branch",
        ),
    ]
    return make_graph(source, papers)


class TestGraphRegistry:
    def test_building_second_graph_keeps_first(self):
        registry = GraphRegistry()
        registry.load(graph_one())
        registry.load(graph_two())

        graphs = registry.list_graphs()
        assert len(graphs) == 2
        assert {g["graph_id"] for g in graphs} == {"W100", "W101"}

    def test_newest_graph_is_active_default(self):
        registry = GraphRegistry()
        registry.load(graph_one())
        registry.load(graph_two())

        assert registry.active_id == "W101"
        # Default get() targets the active graph
        store = registry.get()
        assert store is not None
        assert store.source_id == "W101"

    def test_explicit_graph_id_reaches_older_graph(self):
        registry = GraphRegistry()
        registry.load(graph_one())
        registry.load(graph_two())

        store = registry.get("W100")
        assert store is not None
        assert store.get_entity("W100") is not None

    def test_rebuilding_same_source_replaces_not_duplicates(self):
        registry = GraphRegistry()
        _, _, replaced_first = registry.load(graph_one())
        _, _, replaced_again = registry.load(graph_one())

        assert replaced_first is False
        assert replaced_again is True
        assert len(registry.list_graphs()) == 1

    def test_search_all_tags_results_with_graph_id(self):
        registry = GraphRegistry()
        registry.load(graph_one())
        registry.load(graph_two())

        results = registry.search_all("roughness")

        assert len(results) > 0
        graph_ids = {r["graph_id"] for r in results}
        assert graph_ids == {"W100", "W101"}

    def test_drop_active_falls_back_to_remaining(self):
        registry = GraphRegistry()
        registry.load(graph_one())
        registry.load(graph_two())

        assert registry.drop("W101") is True
        assert registry.active_id == "W100"
        assert registry.get() is not None

    def test_drop_last_graph_leaves_none_active(self):
        registry = GraphRegistry()
        registry.load(graph_one())

        registry.drop("W100")
        assert registry.active_id is None
        assert registry.get() is None
        assert registry.list_graphs() == []

    def test_unknown_graph_id_returns_none(self):
        registry = GraphRegistry()
        registry.load(graph_one())

        assert registry.get("W999") is None
        assert registry.drop("W999") is False
