"""CLI entry point for mcp-oignon."""

import click


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport protocol (default: stdio)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host for HTTP transport (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port for HTTP transport (default: 8000)",
)
def main(transport: str, host: str, port: int) -> None:
    """Oignon - MCP server for exploring related literature."""
    from oignon.server import mcp

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
