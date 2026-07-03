# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`mcp-oignon` is an MCP server for exploring academic literature and building citation networks using the OpenAlex API. Published to PyPI as `mcp-oignon`; the Python package is `oignon` under `src/`. Managed with uv.

## Commands

- Run the MCP server (stdio transport): `uv run mcp-oignon`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_integration.py::TestOpenAlexAPI::test_fetch_paper_by_id`
- Lint: `uv run ruff check`
- Type check: `uv run pyright`
- Run benchmarks: `uv run python -m benchmarks.run` (options: `--model`, `--category`, `--limit`, `--parallel`, `--max-turns`, `-v`; requires the `bench` extra for `claude-agent-sdk` and an Anthropic API key)
- Build for PyPI: `uv build` (publish token lives in `.env` as `UV_PUBLISH_TOKEN`)

Tests are integration tests that hit the real OpenAlex API (no mocks, network required). The reference paper is W2159974629 ("Nanometre-scale thermometry in a living cell").

Set `OPENALEX_EMAIL` to join the OpenAlex polite pool (read in `core/openalex.py`).

## Architecture

Data flows in one direction: OpenAlex API -> graph builder -> in-memory store -> MCP tools.

- `src/oignon/core/openalex.py` — pyalex client. Two fetch tiers: "slim" (id, year, citations, references; used for ranking thousands of candidates cheaply) and "full" (complete metadata; only for papers that make the final graph). Batches IDs (100 per filter) and fetches batches in parallel threads. Fetch errors are swallowed and return empty results.
- `src/oignon/core/graph.py` — dataclasses: `SlimPaper`, `FullPaper`, `Graph`, `GraphMetadata`, etc.
- `src/oignon/core/builder.py` — the core algorithm. Builds a "Local Citation Network" around a source paper with two halves:
  - ROOTS (historical lineage): root_seeds = the source's references; root_papers = references of those seeds. Ranked by citedCount + coCitedCount + coCitingCount relative to the seeds.
  - BRANCHES (future influence): branch_seeds = papers citing the source (must be published >= source year + 1, citations > 0); branch_papers = references of branch seeds, pre-filtered to those referenced by >= 2 seeds. Ranked with a recency-weighted co-citation score.
  - Top N of each are fetched with full metadata and wired into edges.
- `src/oignon/storage/memory.py` — `GraphStore`, a module-level singleton holding exactly one loaded graph at a time. Papers become `Entity` objects whose metadata is flattened into searchable "observation" strings (`"Title: ..."`, `"Year: ..."`, `"Graph role: ..."`, abstract sentences). Search is substring matching over these observations; consumers parse fields back out by prefix.
- `src/oignon/server.py` — FastMCP tool definitions (`search_paper`, `get_paper`, `build_citation_graph`, `search_graph`, `get_graph_node`, `get_citations`, `get_graph_stats`, `get_all_papers`). Graph-query tools error until `build_citation_graph` has loaded the store.
- `src/oignon/tools.py` — tool metadata (descriptions, JSON schemas) as the single source of truth shared by both the MCP server and the benchmark harness. Tool descriptions are themselves benchmarked, so edit them here.
- `src/oignon/cli.py` — `mcp-oignon` entry point.

## Benchmarks

`benchmarks/` evaluates how well an agent uses the tools via the Claude Agent SDK:

- `harness.py` wraps the server's tool implementations into an SDK MCP server using metadata from `oignon/tools.py`.
- Tasks are YAML files under `benchmarks/tasks/<category>/` with prompts and ground truth; `graders.py` scores results.
- Results are written to `benchmarks/results/` as timestamped JSON.
- Call `reset_graph_store()` between trials since the store is a global singleton.
