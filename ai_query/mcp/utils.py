"""MCP utility functions."""

from __future__ import annotations

from ai_query.types import ToolSet
from ai_query.mcp.types import MCPServer


def merge_tools(*tool_sources: ToolSet | MCPServer) -> ToolSet:
    """Merge multiple tool sources into a single ToolSet.

    This is useful when you want to combine tools from multiple MCP servers
    or mix MCP tools with locally defined tools.

    Args:
        *tool_sources: ToolSet dicts or MCPServer instances to merge.

    Returns:
        A merged ToolSet containing all tools.

    Example:
        >>> local_tools = {"calculator": calc_tool}
        >>> async with mcp("python", "weather_server.py") as weather:
        ...     async with mcp("python", "search_server.py") as search:
        ...         all_tools = merge_tools(local_tools, weather, search)
        ...         result = await generate_text(
        ...             model=openai("gpt-4o"),
        ...             prompt="Search and calculate",
        ...             tools=all_tools,
        ...         )
    """
    merged: ToolSet = {}
    for source in tool_sources:
        if isinstance(source, MCPServer):
            merged.update(source.tools)
        else:
            merged.update(source)
    return merged
