"""MCP server for Oignon."""

import json

from mcp.server.fastmcp import FastMCP

from oignon.core.builder import build_graph
from oignon.core.openalex import fetch_paper
from oignon.core.paper_search import find_papers
from oignon.core.ratelimit import OpenAlexRateLimitError
from oignon.storage.registry import get_registry

mcp = FastMCP("oignon")


def _no_graph_error(graph_id: str = "") -> str:
    """Error JSON for a missing graph."""
    if graph_id:
        loaded = [g["graph_id"] for g in get_registry().list_graphs()]
        return json.dumps(
            {"error": f"No graph '{graph_id}'. Loaded graphs: {loaded or 'none'}."}
        )
    return json.dumps({"error": "No graph loaded. Call build_citation_graph first."})


@mcp.tool()
async def search_paper(query: str = "", author: str = "", year: str = "") -> str:
    """Search for academic papers on OpenAlex.

    Runs multiple search strategies (relevance, exact-title, author) in
    parallel and merges the rankings. IMPORTANT: put author names in the
    author parameter, NOT in the query - mixing them degrades results.

    Args:
        query: Title words or topic keywords (no author names). A DOI or
            OpenAlex ID here triggers a direct lookup.
        author: Author name(s), e.g. "Smith" or "J Smith"
        year: Publication year "2015" or range "2010-2020"

    Returns:
        Ranked papers with venue, topic, and DOI for disambiguation
    """
    if not query.strip() and not author.strip():
        return json.dumps({"error": "Provide a query and/or an author."})

    try:
        results = find_papers(query=query, author=author, year=year, limit=10)
    except OpenAlexRateLimitError as e:
        return json.dumps({"error": str(e)})

    return json.dumps(results, indent=2)


@mcp.tool()
async def get_paper(work_id: str) -> str:
    """Get details for a paper by OpenAlex ID or DOI.

    Args:
        work_id: OpenAlex ID (W1234567890) or DOI (10.1234/...)

    Returns:
        Paper details including title, authors, year, citations
    """
    try:
        paper = fetch_paper(work_id)
    except OpenAlexRateLimitError as e:
        return json.dumps({"error": str(e)})

    if not paper:
        return json.dumps({"error": f"Could not fetch paper: {work_id}"})

    authors = [a.name for a in paper.authors[:5]]

    result = {
        "id": paper.id,
        "title": paper.title,
        "authors": authors,
        "year": paper.year,
        "citations": paper.citation_count,
        "references": paper.references_count,
    }

    return json.dumps(result, indent=2)


@mcp.tool()
async def build_citation_graph(
    source_id: str,
    n_roots: int = 25,
    n_branches: int = 25,
) -> str:
    """Build a citation network graph around a source paper and load it into memory.

    Creates a "Local Citation Network" showing:
    - ROOTS: Historical lineage - foundational papers that led to this work
    - BRANCHES: Future influence - important papers that built on this work

    Multiple graphs can be loaded at once; each is keyed by its source
    paper ID and the newest becomes the active default for queries.
    After building, use search_graph and get_graph_node to explore.

    Args:
        source_id: OpenAlex work ID (W1234567890) or DOI
        n_roots: Number of top root papers to include (default 25)
        n_branches: Number of top branch papers to include (default 25)

    Returns:
        Summary of the built graph, including its graph_id
    """
    try:
        graph = build_graph(source_id, n_roots=n_roots, n_branches=n_branches)
    except OpenAlexRateLimitError as e:
        return json.dumps({"error": str(e)})

    graph_id, summary, replaced = get_registry().load(graph)

    summary["graph_id"] = graph_id
    summary["note"] = (
        f"Replaced previously loaded graph for {graph_id}."
        if replaced
        else "Graph loaded and set as active. Other loaded graphs are kept."
    )

    return json.dumps(summary, indent=2)


@mcp.tool()
async def search_graph(
    query: str, limit: int = 15, role: str = "", graph_id: str = ""
) -> str:
    """Ranked keyword search over loaded citation graphs.

    BM25 relevance ranking over titles, abstracts, keywords, topics,
    authors, and venues. Multi-word queries rank papers matching more
    (and rarer) terms higher; each result includes matched text snippets
    showing why it matched. Must call build_citation_graph first.

    Args:
        query: Topic words or phrases (e.g., "ice crystal surface roughness"),
            author names, or venue names
        limit: Maximum results (default 15)
        role: Optional filter - "source", "root", "branch", "root_seed",
            or "branch_seed"
        graph_id: Which graph to search (default: active graph, "all" for
            every loaded graph)

    Returns:
        Papers ranked by relevance, with scores and matched snippets
    """
    registry = get_registry()

    if graph_id == "all":
        results = registry.search_all(query, limit=limit, role=role or None)
        return json.dumps({"found": len(results), "papers": results}, indent=2)

    store = registry.get(graph_id)
    if not store:
        return _no_graph_error(graph_id)

    results = store.search(query, limit=limit, role=role or None)

    return json.dumps(
        {
            "found": len(results),
            "papers": results,
        },
        indent=2,
    )


@mcp.tool()
async def list_graphs() -> str:
    """List all citation graphs currently loaded in memory.

    Returns:
        Each graph's ID, source paper, size, and which one is active
    """
    registry = get_registry()
    graphs = registry.list_graphs()

    if not graphs:
        return json.dumps({"error": "No graphs loaded. Call build_citation_graph first."})

    return json.dumps(
        {"graphs": graphs, "active_graph_id": registry.active_id}, indent=2
    )


@mcp.tool()
async def get_graph_node(paper_id: str, graph_id: str = "") -> str:
    """Get full details for a paper in a loaded graph.

    Args:
        paper_id: OpenAlex ID (e.g., W1234567890)
        graph_id: Which graph to read (default: active graph)

    Returns:
        Full paper details including all observations
    """
    store = get_registry().get(graph_id)

    if not store:
        return _no_graph_error(graph_id)

    entity = store.get_entity(paper_id)

    if not entity:
        return json.dumps({"error": f"Paper {paper_id} not found in graph"})

    return json.dumps(
        {
            "id": entity.name,
            "type": entity.entity_type,
            "observations": entity.observations,
        },
        indent=2,
    )


@mcp.tool()
async def get_citations(
    paper_id: str, direction: str = "cited_by", graph_id: str = ""
) -> str:
    """Get papers that cite or are cited by a specific paper.

    Args:
        paper_id: OpenAlex ID (e.g., W1234567890)
        direction: "cited_by" (papers citing this one) or "cites" (papers this one cites)
        graph_id: Which graph to read (default: active graph)

    Returns:
        List of connected papers
    """
    store = get_registry().get(graph_id)

    if not store:
        return _no_graph_error(graph_id)

    target_ids = store.get_citations(paper_id, direction)

    papers = []
    for pid in target_ids[:15]:
        summary = store.paper_summary(pid)
        if summary:
            papers.append(
                {"id": pid, "title": summary["title"], "year": summary["year"]}
            )

    return json.dumps(
        {
            "paper_id": paper_id,
            "direction": direction,
            "total": len(target_ids),
            "showing": len(papers),
            "papers": papers,
        },
        indent=2,
    )


@mcp.tool()
async def get_graph_stats(graph_id: str = "") -> str:
    """Get statistics about a loaded citation graph.

    Args:
        graph_id: Which graph to read (default: active graph)

    Returns:
        Counts of entities, relations, year/role/topic distributions
    """
    store = get_registry().get(graph_id)

    if not store:
        return _no_graph_error(graph_id)

    stats = store.get_stats()

    return json.dumps(stats, indent=2)


@mcp.tool()
async def get_all_papers(sort_by: str = "year", graph_id: str = "") -> str:
    """Get all papers in a loaded graph.

    Args:
        sort_by: Sort order - "year" (default, newest first) or "role"
        graph_id: Which graph to read (default: active graph)

    Returns:
        All papers in the graph with basic metadata
    """
    store = get_registry().get(graph_id)

    if not store:
        return _no_graph_error(graph_id)

    papers = store.list_papers(sort_by)

    return json.dumps(
        {
            "total": len(papers),
            "sort_by": sort_by,
            "papers": papers,
        },
        indent=2,
    )
