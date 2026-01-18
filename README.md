# mcp-oignon

A lightweight MCP server for exploring academic literature and building citation networks using OpenAlex.

Install with `pip install mcp-oignon` or `uv pip install mcp-oignon`

Add to your MCP client config:

```json
{
  "mcpServers": {
    "oignon": {
      "command": "mcp-oignon"
    }
  }
}
```

Config file locations:
- Claude Desktop (macOS): `~/Library/Application Support/Claude/claude_desktop_config.json`
- Claude Desktop (Windows): `%APPDATA%\Claude\claude_desktop_config.json`
- Claude Desktop (Linux): `~/.config/Claude/claude_desktop_config.json`
