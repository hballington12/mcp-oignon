"""Data structures for citation graphs."""

from dataclasses import dataclass


@dataclass
class SlimPaper:
    """Minimal paper data for ranking."""

    id: str
    year: int
    citation_count: int
    references: list[str]


@dataclass
class Author:
    """Author with optional affiliation."""

    name: str
    orcid: str | None = None
    affiliation: str | None = None
    affiliation_country: str | None = None


@dataclass
class PrimaryTopic:
    """OpenAlex topic hierarchy."""

    id: str
    name: str
    subfield: dict | None = None
    field: dict | None = None
    domain: dict | None = None


@dataclass
class CitationPercentile:
    """Citation impact percentile."""

    value: float
    is_in_top_1_percent: bool = False
    is_in_top_10_percent: bool = False


@dataclass
class SDG:
    """UN Sustainable Development Goal."""

    id: str
    name: str
    score: float


@dataclass
class FullPaper:
    """Complete paper data for final graph."""

    id: str
    doi: str | None
    title: str
    authors: list[Author]
    year: int
    citation_count: int
    references_count: int
    references: list[str]
    openalex_url: str
    # Optional metadata
    type: str | None = None
    source_type: str | None = None
    source_name: str | None = None
    open_access: bool | None = None
    language: str | None = None
    abstract: str | None = None
    fwci: float | None = None
    citation_percentile: CitationPercentile | None = None
    primary_topic: PrimaryTopic | None = None
    sdgs: list[SDG] | None = None
    keywords: list[str] | None = None
    # Graph role
    role: str | None = None
    rank: float | None = None
    rank_details: dict | None = None


@dataclass
class GraphMetadata:
    """Build statistics."""

    source_year: int
    total_root_seeds: int
    total_root_papers: int
    total_branch_seeds: int
    total_branch_papers: int
    n_roots: int
    n_branches: int
    papers_in_graph: int
    edges_in_graph: int
    build_time_seconds: float
    api_calls: int
    timestamp: str


@dataclass
class Graph:
    """Complete citation network graph."""

    source_paper: FullPaper
    root_seeds: list[FullPaper]
    branch_seeds: list[FullPaper]
    papers: list[FullPaper]
    edges: list[dict]
    metadata: GraphMetadata
