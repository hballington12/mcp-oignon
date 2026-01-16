"""MCP server for Oignon."""

import json

from mcp.server.fastmcp import FastMCP

from oignon.core.builder import build_graph
from oignon.core.openalex import fetch_paper, search_papers
from oignon.storage.memory import get_store

mcp = FastMCP("oignon")


@mcp.tool()
async def search_paper(query: str) -> str:
    """Search for academic papers by title or keywords.

    Args:
        query: Search terms (title, author, keywords)

    Returns:
        List of matching papers with OpenAlex IDs
    """
    papers = search_papers(query, limit=10)

    results = []
    for p in papers:
        authors = [a.name for a in p.authors[:3]]
        results.append(
            {
                "id": p.id,
                "title": p.title,
                "authors": authors,
                "year": p.year,
                "citations": p.citation_count,
            }
        )

    return json.dumps(results, indent=2)


@mcp.tool()
async def get_paper(work_id: str) -> str:
    """Get details for a paper by OpenAlex ID or DOI.

    Args:
        work_id: OpenAlex ID (W1234567890) or DOI (10.1234/...)

    Returns:
        Paper details including title, authors, year, citations
    """
    paper = fetch_paper(work_id)

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

    After building, use search_graph and get_graph_node to explore.

    Args:
        source_id: OpenAlex work ID (W1234567890) or DOI
        n_roots: Number of top root papers to include (default 25)
        n_branches: Number of top branch papers to include (default 25)

    Returns:
        Summary of the built graph
    """
    graph = build_graph(source_id, n_roots=n_roots, n_branches=n_branches)

    store = get_store()
    summary = store.load(graph)

    return json.dumps(summary, indent=2)


@mcp.tool()
async def search_graph(query: str) -> str:
    """Search the loaded citation graph for papers.

    Searches paper titles, topics, years, and abstract content.
    Must call build_citation_graph first.

    Args:
        query: Search terms (e.g., "climate", "2020", topic name)

    Returns:
        Matching papers with IDs and titles
    """
    store = get_store()

    if not store.is_loaded():
        return json.dumps({"error": "No graph loaded. Call build_citation_graph first."})

    results = store.search(query)

    return json.dumps(
        {
            "found": len(results),
            "papers": results,
        },
        indent=2,
    )


@mcp.tool()
async def get_graph_node(paper_id: str) -> str:
    """Get full details for a paper in the loaded graph.

    Args:
        paper_id: OpenAlex ID (e.g., W1234567890)

    Returns:
        Full paper details including all observations
    """
    store = get_store()

    if not store.is_loaded():
        return json.dumps({"error": "No graph loaded. Call build_citation_graph first."})

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
async def get_citations(paper_id: str, direction: str = "cited_by") -> str:
    """Get papers that cite or are cited by a specific paper.

    Args:
        paper_id: OpenAlex ID (e.g., W1234567890)
        direction: "cited_by" (papers citing this one) or "cites" (papers this one cites)

    Returns:
        List of connected papers
    """
    store = get_store()

    if not store.is_loaded():
        return json.dumps({"error": "No graph loaded. Call build_citation_graph first."})

    target_ids = store.get_citations(paper_id, direction)

    papers = []
    for pid in target_ids[:15]:
        entity = store.get_entity(pid)
        if entity:
            title = ""
            year = ""
            for obs in entity.observations:
                if obs.startswith("Title: "):
                    title = obs.replace("Title: ", "")
                elif obs.startswith("Year: "):
                    year = obs.replace("Year: ", "")
            papers.append({"id": pid, "title": title, "year": year})

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
async def get_graph_stats() -> str:
    """Get statistics about the currently loaded citation graph.

    Returns:
        Counts of entities, relations, and year distribution
    """
    store = get_store()

    if not store.is_loaded():
        return json.dumps({"error": "No graph loaded. Call build_citation_graph first."})

    stats = store.get_stats()

    return json.dumps(stats, indent=2)


@mcp.tool()
async def get_all_papers(sort_by: str = "year") -> str:
    """Get all papers in the loaded graph.

    Args:
        sort_by: Sort order - "year" (default, newest first) or "role"

    Returns:
        All papers in the graph with basic metadata
    """
    store = get_store()

    if not store.is_loaded():
        return json.dumps({"error": "No graph loaded. Call build_citation_graph first."})

    papers = []
    for entity in store._entities.values():
        title = ""
        year = ""
        role = ""
        citations = ""

        for obs in entity.observations:
            if obs.startswith("Title: "):
                title = obs.replace("Title: ", "")
            elif obs.startswith("Year: "):
                year = obs.replace("Year: ", "")
            elif obs.startswith("Graph role: "):
                role = obs.replace("Graph role: ", "")
            elif obs.startswith("Citations: "):
                citations = obs.replace("Citations: ", "")

        papers.append(
            {
                "id": entity.name,
                "title": title,
                "year": year,
                "role": role,
                "citations": citations,
            }
        )

    # Sort
    if sort_by == "year":
        papers.sort(key=lambda p: p["year"], reverse=True)
    elif sort_by == "role":
        role_order = {"source": 0, "root": 1, "root_seed": 2, "branch": 3, "branch_seed": 4}
        papers.sort(key=lambda p: role_order.get(p["role"], 99))

    return json.dumps(
        {
            "total": len(papers),
            "sort_by": sort_by,
            "papers": papers,
        },
        indent=2,
    )
