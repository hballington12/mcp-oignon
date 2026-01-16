"""In-memory graph storage and search."""

import re
from dataclasses import dataclass

from oignon.core.graph import FullPaper, Graph


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

    def clear(self) -> None:
        """Clear all stored data."""
        self._entities.clear()
        self._relations.clear()
        self._source_id = None

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

    def search(self, query: str, limit: int = 15) -> list[dict]:
        """Search entities by query string."""
        query_lower = query.lower()
        matches = []

        for entity in self._entities.values():
            if self._matches(entity, query_lower):
                matches.append(self._entity_to_result(entity))
                if len(matches) >= limit:
                    break

        return matches

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

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

        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "by_year": dict(sorted(years.items(), reverse=True)),
            "by_role": roles,
        }

    def _add_paper(self, paper: FullPaper, role: str) -> None:
        """Convert paper to entity and store."""
        if paper.id in self._entities:
            return

        observations = []

        # Core metadata
        observations.append(f"Title: {paper.title}")
        observations.append(f"Year: {paper.year}")

        if paper.authors:
            author_names = [a.name for a in paper.authors[:5]]
            if len(paper.authors) > 5:
                author_names.append("et al.")
            observations.append(f"Authors: {', '.join(author_names)}")

        observations.append(f"Citations: {paper.citation_count}")
        observations.append(f"References: {paper.references_count}")

        if paper.doi:
            observations.append(f"DOI: {paper.doi}")

        observations.append(f"Graph role: {role}")

        if paper.source_name:
            observations.append(f"Published in: {paper.source_name}")

        if paper.open_access is not None:
            observations.append(f"Open access: {'yes' if paper.open_access else 'no'}")

        if paper.fwci:
            observations.append(f"Field-weighted citation impact: {paper.fwci:.2f}")

        if paper.citation_percentile:
            cp = paper.citation_percentile
            if cp.is_in_top_1_percent:
                observations.append("Highly cited: top 1% in field")
            elif cp.is_in_top_10_percent:
                observations.append("Highly cited: top 10% in field")

        if paper.primary_topic:
            topic = paper.primary_topic
            observations.append(f"Topic: {topic.name}")
            if topic.field:
                observations.append(f"Field: {topic.field.get('name', '')}")
            if topic.domain:
                observations.append(f"Domain: {topic.domain.get('name', '')}")

        if paper.keywords:
            observations.append(f"Keywords: {', '.join(paper.keywords[:10])}")

        if paper.sdgs:
            sdg_names = [s.name for s in paper.sdgs[:3]]
            observations.append(f"SDGs: {', '.join(sdg_names)}")

        # Abstract sentences
        if paper.abstract:
            sentences = self._split_abstract(paper.abstract)
            observations.extend(sentences)

        entity = Entity(
            name=paper.id,
            entity_type=paper.type or "article",
            observations=observations,
        )
        self._entities[paper.id] = entity

    def _split_abstract(self, abstract: str, max_sentences: int = 10) -> list[str]:
        """Split abstract into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences[:max_sentences]

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
        title = ""
        year = ""
        role = ""

        for obs in entity.observations:
            if obs.startswith("Title: "):
                title = obs.replace("Title: ", "")
            elif obs.startswith("Year: "):
                year = obs.replace("Year: ", "")
            elif obs.startswith("Graph role: "):
                role = obs.replace("Graph role: ", "")

        return {
            "id": entity.name,
            "type": entity.entity_type,
            "title": title,
            "year": year,
            "role": role,
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


# Module-level singleton for the MCP server
_store = GraphStore()


def get_store() -> GraphStore:
    """Get the global graph store instance."""
    return _store
