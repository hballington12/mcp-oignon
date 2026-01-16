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
        "description": "Search for academic papers by title or keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (title, author, keywords)",
                },
            },
            "required": ["query"],
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
            "Search the loaded citation graph for papers. "
            "Searches paper titles, topics, years, and abstract content. "
            "Must call build_citation_graph first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms (e.g., 'climate', '2020', topic name)",
                },
            },
            "required": ["query"],
        },
    },
    "get_graph_node": {
        "description": "Get full details for a paper in the loaded graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "OpenAlex ID (e.g., W1234567890)",
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
            },
            "required": ["paper_id"],
        },
    },
    "get_graph_stats": {
        "description": "Get statistics about the currently loaded citation graph.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_all_papers": {
        "description": "Get all papers in the loaded graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "description": "Sort order - 'year' (default, newest first) or 'role'",
                    "default": "year",
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
