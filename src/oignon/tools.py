"""Tool definitions for oignon - single source of truth.

This module defines tool metadata (names, descriptions, schemas) that are used by:
1. The MCP server (server.py)
2. The benchmark harness (benchmarks/harness.py)

Tool descriptions are part of what we benchmark - if descriptions are bad,
the agent won't use tools effectively.
"""

from typing import Any

# Tool definitions with descriptions and parameter schemas
TOOLS: dict[str, dict[str, Any]] = {
    "search_paper": {
        "description": (
            "Search for academic papers on OpenAlex. Runs multiple search "
            "strategies (relevance, exact-title, author) in parallel and merges "
            "the rankings. IMPORTANT: put author names in the 'author' parameter, "
            "NOT in 'query' - mixing them degrades results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Title words or topic keywords (no author names). "
                        "A DOI or OpenAlex ID triggers a direct lookup."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": "Author name(s), e.g. 'Smith' or 'J Smith'",
                },
                "year": {
                    "type": "string",
                    "description": "Publication year '2015' or range '2010-2020'",
                },
            },
            "required": [],
        },
    },
    "get_paper": {
        "description": "Get details for a paper by OpenAlex ID or DOI.",
        "parameters": {
            "type": "object",
            "properties": {
                "work_id": {
                    "type": "string",
                    "description": "OpenAlex ID (W1234567890) or DOI (10.1234/...)",
                },
            },
            "required": ["work_id"],
        },
    },
    "build_citation_graph": {
        "description": (
            "Build a citation network graph around a source paper and load it into memory. "
            "Creates a 'Local Citation Network' showing: "
            "ROOTS (historical lineage - foundational papers that led to this work) and "
            "BRANCHES (future influence - important papers that built on this work). "
            "Multiple graphs can be loaded at once; each is keyed by its source paper ID "
            "and the newest becomes the active default for queries. "
            "After building, use search_graph and get_graph_node to explore."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "OpenAlex work ID (W1234567890) or DOI",
                },
                "n_roots": {
                    "type": "integer",
                    "description": "Number of top root papers to include (default 25)",
                    "default": 25,
                },
                "n_branches": {
                    "type": "integer",
                    "description": "Number of top branch papers to include (default 25)",
                    "default": 25,
                },
            },
            "required": ["source_id"],
        },
    },
    "search_graph": {
        "description": (
            "Ranked keyword search over loaded citation graphs. "
            "BM25 relevance over titles, abstracts, keywords, topics, authors, "
            "and venues - multi-word queries rank papers matching more terms "
            "higher, and results include matched snippets showing why each "
            "paper matched. Must call build_citation_graph first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Topic words or phrases (e.g., 'ice crystal surface "
                        "roughness'), author names, or venue names"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 15)",
                    "default": 15,
                },
                "role": {
                    "type": "string",
                    "description": (
                        "Optional filter: 'source', 'root', 'branch', "
                        "'root_seed', or 'branch_seed'"
                    ),
                },
                "graph_id": {
                    "type": "string",
                    "description": (
                        "Which graph to search (default: active graph, "
                        "'all' for every loaded graph)"
                    ),
                },
            },
            "required": ["query"],
        },
    },
    "list_graphs": {
        "description": (
            "List all citation graphs currently loaded in memory: each graph's "
            "ID, source paper, size, and which one is active."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_graph_node": {
        "description": "Get full details for a paper in a loaded graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "OpenAlex ID (e.g., W1234567890)",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Which graph to read (default: active graph)",
                },
            },
            "required": ["paper_id"],
        },
    },
    "get_citations": {
        "description": "Get papers that cite or are cited by a specific paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "OpenAlex ID (e.g., W1234567890)",
                },
                "direction": {
                    "type": "string",
                    "description": "'cited_by' (papers citing this one) or 'cites' (papers this one cites)",
                    "default": "cited_by",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Which graph to read (default: active graph)",
                },
            },
            "required": ["paper_id"],
        },
    },
    "get_graph_stats": {
        "description": "Get statistics about a loaded citation graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "graph_id": {
                    "type": "string",
                    "description": "Which graph to read (default: active graph)",
                },
            },
            "required": [],
        },
    },
    "get_all_papers": {
        "description": "Get all papers in a loaded graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "description": "Sort order - 'year' (default, newest first) or 'role'",
                    "default": "year",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Which graph to read (default: active graph)",
                },
            },
            "required": [],
        },
    },
}


def get_tool_names() -> list[str]:
    """Get list of all tool names."""
    return list(TOOLS.keys())


def get_tool_description(name: str) -> str:
    """Get description for a specific tool."""
    return TOOLS[name]["description"]


def get_tool_parameters(name: str) -> dict[str, Any]:
    """Get parameter schema for a specific tool."""
    return TOOLS[name]["parameters"]


def get_tools_for_claude_api() -> list[dict[str, Any]]:
    """Get tool definitions formatted for Claude API."""
    return [
        {
            "name": name,
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for name, tool in TOOLS.items()
    ]
