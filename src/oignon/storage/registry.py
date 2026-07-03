"""Registry of multiple loaded citation graphs.

Graphs are keyed by their source paper's OpenAlex ID, so building a new
graph never overwrites an existing one (unless it has the same source,
in which case it replaces it). The most recently built graph is "active"
and is the default target for queries, so single-graph workflows behave
exactly as before.
"""

from oignon.core.graph import Graph
from oignon.storage.memory import GraphStore


class GraphRegistry:
    """Holds multiple loaded graphs, keyed by source paper ID."""

    def __init__(self):
        self._stores: dict[str, GraphStore] = {}
        self._active_id: str | None = None

    def clear(self) -> None:
        """Drop all graphs."""
        self._stores.clear()
        self._active_id = None

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def load(self, graph: Graph) -> tuple[str, dict, bool]:
        """Load a graph and make it active.

        Returns (graph_id, load summary, whether an existing graph with
        the same source was replaced).
        """
        graph_id = graph.source_paper.id
        replaced = graph_id in self._stores

        store = GraphStore()
        summary = store.load(graph)

        self._stores[graph_id] = store
        self._active_id = graph_id

        return graph_id, summary, replaced

    def get(self, graph_id: str = "") -> GraphStore | None:
        """Get a graph by ID, or the active graph if no ID given."""
        if graph_id:
            return self._stores.get(graph_id)
        if self._active_id:
            return self._stores[self._active_id]
        return None

    def list_graphs(self) -> list[dict]:
        """Summaries of all loaded graphs."""
        graphs = []
        for graph_id, store in self._stores.items():
            source = store.paper_summary(graph_id) or {}
            graphs.append(
                {
                    "graph_id": graph_id,
                    "source_title": source.get("title"),
                    "source_year": source.get("year"),
                    "entities": store.entity_count,
                    "relations": store.relation_count,
                    "active": graph_id == self._active_id,
                }
            )
        return graphs

    def drop(self, graph_id: str) -> bool:
        """Remove a graph. The most recently built remaining graph
        becomes active."""
        if graph_id not in self._stores:
            return False

        del self._stores[graph_id]
        if self._active_id == graph_id:
            self._active_id = next(reversed(self._stores), None)

        return True

    def search_all(
        self, query: str, limit: int = 15, role: str | None = None
    ) -> list[dict]:
        """Search every loaded graph, merging results by score."""
        results = []
        for graph_id, store in self._stores.items():
            for result in store.search(query, limit=limit, role=role):
                result["graph_id"] = graph_id
                results.append(result)

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[:limit]


# Module-level singleton for the MCP server
_registry = GraphRegistry()


def get_registry() -> GraphRegistry:
    """Get the global graph registry instance."""
    return _registry
