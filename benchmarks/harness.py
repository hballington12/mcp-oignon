"""Benchmark harness for oignon MCP tools.

This module wraps oignon's tools for use with the Claude agent SDK,
using tool definitions from oignon/tools.py as the single source of truth.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from oignon.tools import TOOLS

# Import the actual tool implementations from server.py
from oignon.server import (
    build_citation_graph as _build_citation_graph,
    get_all_papers as _get_all_papers,
    get_citations as _get_citations,
    get_graph_node as _get_graph_node,
    get_graph_stats as _get_graph_stats,
    get_paper as _get_paper,
    search_graph as _search_graph,
    search_paper as _search_paper,
)


# Wrap each oignon function with @tool decorator
# Descriptions come from oignon/tools.py (single source of truth)


@tool(
    "search_paper",
    TOOLS["search_paper"]["description"],
    {"query": str},
)
async def search_paper(args: dict[str, Any]) -> dict[str, Any]:
    result = await _search_paper(args["query"])
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "get_paper",
    TOOLS["get_paper"]["description"],
    {"work_id": str},
)
async def get_paper(args: dict[str, Any]) -> dict[str, Any]:
    result = await _get_paper(args["work_id"])
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "build_citation_graph",
    TOOLS["build_citation_graph"]["description"],
    {"source_id": str, "n_roots": int, "n_branches": int},
)
async def build_citation_graph(args: dict[str, Any]) -> dict[str, Any]:
    result = await _build_citation_graph(
        args["source_id"],
        args.get("n_roots", 25),
        args.get("n_branches", 25),
    )
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "search_graph",
    TOOLS["search_graph"]["description"],
    {"query": str},
)
async def search_graph(args: dict[str, Any]) -> dict[str, Any]:
    result = await _search_graph(args["query"])
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "get_graph_node",
    TOOLS["get_graph_node"]["description"],
    {"paper_id": str},
)
async def get_graph_node(args: dict[str, Any]) -> dict[str, Any]:
    result = await _get_graph_node(args["paper_id"])
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "get_citations",
    TOOLS["get_citations"]["description"],
    {"paper_id": str, "direction": str},
)
async def get_citations(args: dict[str, Any]) -> dict[str, Any]:
    result = await _get_citations(
        args["paper_id"],
        args.get("direction", "cited_by"),
    )
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "get_graph_stats",
    TOOLS["get_graph_stats"]["description"],
    {},
)
async def get_graph_stats(args: dict[str, Any]) -> dict[str, Any]:
    result = await _get_graph_stats()
    return {"content": [{"type": "text", "text": result}]}


@tool(
    "get_all_papers",
    TOOLS["get_all_papers"]["description"],
    {"sort_by": str},
)
async def get_all_papers(args: dict[str, Any]) -> dict[str, Any]:
    result = await _get_all_papers(args.get("sort_by", "year"))
    return {"content": [{"type": "text", "text": result}]}


# Bundle tools into an SDK MCP server
OIGNON_SERVER = create_sdk_mcp_server(
    name="oignon",
    version="0.1.0",
    tools=[
        search_paper,
        get_paper,
        build_citation_graph,
        search_graph,
        get_graph_node,
        get_citations,
        get_graph_stats,
        get_all_papers,
    ],
)

# Explicitly allow only oignon MCP tools (prevents cheating with built-in tools)
ALLOWED_TOOLS = [
    "mcp__oignon__search_paper",
    "mcp__oignon__get_paper",
    "mcp__oignon__build_citation_graph",
    "mcp__oignon__search_graph",
    "mcp__oignon__get_graph_node",
    "mcp__oignon__get_citations",
    "mcp__oignon__get_graph_stats",
    "mcp__oignon__get_all_papers",
]


@dataclass
class ToolCall:
    """Record of a single tool call."""

    turn: int
    tool: str
    input: dict[str, Any]
    output: str


@dataclass
class TaskResult:
    """Result of running a benchmark task."""

    final_response: str | None
    tool_calls: list[ToolCall]
    turns: int
    transcript: list[dict[str, Any]]
    error: str | None = None


def reset_graph_store():
    """Reset the graph store between tasks."""
    from oignon.storage.memory import get_store

    store = get_store()
    store._entities = {}
    store._relations = []
    store._source_id = None


async def run_task(
    task: str,
    model: str = "claude-haiku-4-5",
    max_turns: int = 10,
    system: str | None = None,
) -> TaskResult:
    """Run a single benchmark task using claude-agent-sdk.

    Args:
        task: The task prompt
        model: Claude model to use
        max_turns: Maximum agentic turns
        system: Optional system prompt

    Returns:
        TaskResult
    """
    reset_graph_store()

    tool_calls: list[ToolCall] = []
    transcript: list[dict[str, Any]] = []
    final_response = None
    error = None
    turns = 0
    current_turn = 0

    default_system = (
        "You are a research assistant with access to academic paper tools. "
        "Use the available tools to help answer questions about academic literature."
    )

    options = ClaudeAgentOptions(
        model=model,
        mcp_servers={"oignon": OIGNON_SERVER},
        allowed_tools=ALLOWED_TOOLS,
        system_prompt=system or default_system,
        max_turns=max_turns,
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(task)

            async for msg in client.receive_messages():
                # Store raw message in transcript
                if hasattr(msg, "model_dump"):
                    transcript.append(msg.model_dump())
                else:
                    transcript.append({"type": type(msg).__name__, "raw": str(msg)})

                # Track tool calls from assistant messages
                if isinstance(msg, AssistantMessage):
                    current_turn += 1
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                ToolCall(
                                    turn=current_turn,
                                    tool=block.name,
                                    input=dict(block.input) if block.input else {},
                                    output="",
                                )
                            )

                # Extract final response from result
                if isinstance(msg, ResultMessage):
                    final_response = msg.result
                    if hasattr(msg, "num_turns"):
                        turns = msg.num_turns
                    break

    except Exception as e:
        error = str(e)

    return TaskResult(
        final_response=final_response,
        tool_calls=tool_calls,
        turns=turns,
        transcript=transcript,
        error=error,
    )


def print_tool_definitions():
    """Print all tool definitions for inspection."""
    print("Oignon Tool Definitions")
    print("=" * 60)
    for name, tool_def in TOOLS.items():
        print(f"\n{name}:")
        print(f"  Description: {tool_def['description'][:80]}...")
        params = tool_def["parameters"].get("properties", {})
        if params:
            print(f"  Parameters: {list(params.keys())}")
        else:
            print("  Parameters: none")


if __name__ == "__main__":
    print_tool_definitions()
    print("\n" + "=" * 60)
    print(f"Total tools: {len(TOOLS)}")
