"""CLI entry point for mcp-oignon."""

import click


@click.command()
def main() -> None:
    """Oignon - MCP server for exploring related literature."""
    from oignon.server import mcp

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
