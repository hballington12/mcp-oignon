"""In-memory graph storage and search."""

from dataclasses import dataclass

from oignon.core.graph import FullPaper, Graph
from oignon.storage.observations import build_index_fields, build_observations
from oignon.storage.search import SearchIndex, pick_snippets


@dataclass
class Entity:
    """Paper entity for storage."""

    name: str  # OpenAlex ID
    entity_type: str  # paper type (article, review, etc.)
    observations: list[str]  # metadata as searchable strings


@dataclass
class Relation:
    """Citation relation."""

    from_entity: str
    to_entity: str
    relation_type: str  # "cites"


class GraphStore:
    """In-memory storage for a citation graph."""

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._source_id: str | None = None
        self._index = SearchIndex()
        self._meta: dict[str, dict] = {}

    def clear(self) -> None:
        """Clear all stored data."""
        self._entities.clear()
        self._relations.clear()
        self._source_id = None
        self._index.clear()
        self._meta.clear()

    def is_loaded(self) -> bool:
        """Check if a graph is loaded."""
        return len(self._entities) > 0

    def load(self, graph: Graph) -> dict:
        """Load a graph into storage. Returns summary."""
        self.clear()
        self._source_id = graph.source_paper.id

        # Convert papers to entities
        self._add_paper(graph.source_paper, role="source")
        for paper in graph.root_seeds:
            self._add_paper(paper, role="root_seed")
        for paper in graph.branch_seeds:
            self._add_paper(paper, role="branch_seed")
        for paper in graph.papers:
            self._add_paper(paper, role=paper.role or "ranked")

        # Convert edges to relations
        for edge in graph.edges:
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            if source_id in self._entities and target_id in self._entities:
                self._relations.append(Relation(source_id, target_id, "cites"))

        return self._build_summary(graph)

    def search(
        self, query: str, limit: int = 15, role: str | None = None
    ) -> list[dict]:
        """Ranked search over the loaded graph.

        BM25 over titles, keywords, topics, authors, venues, and abstract
        text, with a boost when the exact phrase appears verbatim. Falls
        back to substring matching (IDs, exact strings) when BM25 finds
        nothing.
        """
        hits = self._index.search(query, limit=max(limit * 3, 30))

        phrase = query.lower().strip()
        is_phrase = " " in phrase

        results = []
        for entity_id, score in hits:
            entity = self._entities.get(entity_id)
            if not entity:
                continue

            meta = self._meta.get(entity_id, {})
            if role and meta.get("role") != role:
                continue

            if is_phrase and any(phrase in obs.lower() for obs in entity.observations):
                score *= 1.3

            result = self._entity_to_result(entity)
            result["score"] = round(score, 2)
            result["citations"] = meta.get("citations")
            result["topic"] = meta.get("topic")
            result["matched"] = pick_snippets(entity.observations, query)
            results.append(result)

        results.sort(key=lambda r: r["score"], reverse=True)
        if results:
            return results[:limit]

        return self._substring_search(query, limit, role)

    def _substring_search(
        self, query: str, limit: int, role: str | None = None
    ) -> list[dict]:
        """Literal substring fallback (catches IDs and exact strings)."""
        query_lower = query.lower()
        matches = []

        for entity in self._entities.values():
            if role and self._meta.get(entity.name, {}).get("role") != role:
                continue
            if self._matches(entity, query_lower):
                matches.append(self._entity_to_result(entity))
                if len(matches) >= limit:
                    break

        return matches

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    @property
    def source_id(self) -> str | None:
        return self._source_id

    def paper_summary(self, paper_id: str) -> dict | None:
        """Basic metadata for one paper (title, year, role, citations, topic)."""
        meta = self._meta.get(paper_id)
        if not meta:
            return None
        return {"id": paper_id, **meta}

    def list_papers(self, sort_by: str = "year") -> list[dict]:
        """All papers with basic metadata, sorted by year or role."""
        papers = [{"id": pid, **meta} for pid, meta in self._meta.items()]

        if sort_by == "year":
            papers.sort(key=lambda p: p["year"], reverse=True)
        elif sort_by == "role":
            role_order = {
                "source": 0,
                "root": 1,
                "root_seed": 2,
                "branch": 3,
                "branch_seed": 4,
            }
            papers.sort(key=lambda p: role_order.get(p["role"], 99))

        return papers

    def get_citations(self, paper_id: str, direction: str) -> list[str]:
        """Get papers citing or cited by a paper."""
        if direction == "cites":
            return [r.to_entity for r in self._relations if r.from_entity == paper_id]
        else:  # cited_by
            return [r.from_entity for r in self._relations if r.to_entity == paper_id]

    def get_stats(self) -> dict:
        """Get statistics about loaded graph."""
        if not self.is_loaded():
            return {}

        years: dict[str, int] = {}
        roles: dict[str, int] = {}

        for entity in self._entities.values():
            # Extract year from observations
            for obs in entity.observations:
                if obs.startswith("Year: "):
                    year = obs.replace("Year: ", "")
                    years[year] = years.get(year, 0) + 1
                elif obs.startswith("Graph role: "):
                    role = obs.replace("Graph role: ", "")
                    roles[role] = roles.get(role, 0) + 1

        topics: dict[str, int] = {}
        for meta in self._meta.values():
            topic = meta.get("topic")
            if topic:
                topics[topic] = topics.get(topic, 0) + 1

        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "by_year": dict(sorted(years.items(), reverse=True)),
            "by_role": roles,
            "by_topic": dict(
                sorted(topics.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    def _add_paper(self, paper: FullPaper, role: str) -> None:
        """Convert paper to entity, store, and index it."""
        if paper.id in self._entities:
            return

        entity = Entity(
            name=paper.id,
            entity_type=paper.type or "article",
            observations=build_observations(paper, role),
        )
        self._entities[paper.id] = entity

        self._index.add(paper.id, build_index_fields(paper, role))

        self._meta[paper.id] = {
            "title": paper.title,
            "year": paper.year,
            "role": role,
            "citations": paper.citation_count,
            "topic": paper.primary_topic.name if paper.primary_topic else None,
        }

    def _matches(self, entity: Entity, query: str) -> bool:
        """Check if entity matches search query."""
        if query in entity.name.lower():
            return True
        if query in entity.entity_type.lower():
            return True
        for obs in entity.observations:
            if query in obs.lower():
                return True
        return False

    def _entity_to_result(self, entity: Entity) -> dict:
        """Convert entity to search result dict."""
        meta = self._meta.get(entity.name, {})
        return {
            "id": entity.name,
            "type": entity.entity_type,
            "title": meta.get("title", ""),
            "year": meta.get("year", ""),
            "role": meta.get("role", ""),
        }

    def _build_summary(self, graph: Graph) -> dict:
        """Build summary dict for load response."""
        source = graph.source_paper

        # Top papers by rank
        top_roots = [
            {"id": p.id, "title": p.title[:60], "year": p.year, "rank": p.rank}
            for p in graph.papers
            if p.role == "root"
        ][:5]

        top_branches = [
            {"id": p.id, "title": p.title[:60], "year": p.year, "rank": p.rank}
            for p in graph.papers
            if p.role == "branch"
        ][:5]

        return {
            "source": {
                "id": source.id,
                "title": source.title,
                "year": source.year,
                "citations": source.citation_count,
            },
            "counts": {
                "entities": len(self._entities),
                "relations": len(self._relations),
                "root_seeds": len(graph.root_seeds),
                "branch_seeds": len(graph.branch_seeds),
                "top_roots": len([p for p in graph.papers if p.role == "root"]),
                "top_branches": len([p for p in graph.papers if p.role == "branch"]),
            },
            "top_roots": top_roots,
            "top_branches": top_branches,
            "metadata": {
                "build_time_seconds": graph.metadata.build_time_seconds,
                "api_calls": graph.metadata.api_calls,
            },
        }
