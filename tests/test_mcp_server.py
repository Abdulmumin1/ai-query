"""Test MCP Server for ai-query integration testing.

A simple MCP server with basic tools for testing purposes.

Run directly:
    python test_mcp_server.py

Or use with ai-query:
    async with mcp("python", "test_mcp_server.py") as server:
        result = await generate_text(
            model=google("gemini-2.0-flash"),
            prompt="Calculate 25 * 4 and get the weather in Tokyo",
            tools=server.tools,
        )
"""

import asyncio
import json
import random
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create server instance
server = Server("test-server")


# --- Tool Definitions ---

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name="calculator",
            description="Perform basic math calculations. Supports +, -, *, /, and parentheses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate (e.g., '2 + 2', '10 * 5', '(3 + 4) * 2')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="get_weather",
            description="Get the current weather for a city. Returns temperature and conditions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city (e.g., 'Tokyo', 'New York', 'London')"
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="echo",
            description="Echo back the input message. Useful for testing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="random_number",
            description="Generate a random number between min and max (inclusive).",
            inputSchema={
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "Minimum value (default: 1)",
                        "default": 1
                    },
                    "max": {
                        "type": "integer",
                        "description": "Maximum value (default: 100)",
                        "default": 100
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="lookup_user",
            description="Look up a user by ID and return their profile information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The user ID to look up"
                    }
                },
                "required": ["user_id"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "calculator":
        expression = arguments.get("expression", "")
        try:
            # Only allow safe math operations
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return [TextContent(type="text", text=f"Error: Invalid characters in expression")]
            result = eval(expression)
            return [TextContent(type="text", text=f"Result: {expression} = {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error calculating: {e}")]

    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        # Simulated weather data
        weather_data = {
            "tokyo": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65},
            "new york": {"temp": 18, "condition": "Sunny", "humidity": 45},
            "london": {"temp": 14, "condition": "Rainy", "humidity": 80},
            "paris": {"temp": 16, "condition": "Cloudy", "humidity": 70},
            "sydney": {"temp": 25, "condition": "Sunny", "humidity": 55},
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            w = weather_data[city_lower]
            return [TextContent(
                type="text",
                text=f"Weather in {city}: {w['temp']}°C, {w['condition']}, Humidity: {w['humidity']}%"
            )]
        else:
            # Generate random weather for unknown cities
            temp = random.randint(10, 30)
            conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Rainy", "Clear"]
            condition = random.choice(conditions)
            humidity = random.randint(30, 90)
            return [TextContent(
                type="text",
                text=f"Weather in {city}: {temp}°C, {condition}, Humidity: {humidity}%"
            )]

    elif name == "echo":
        message = arguments.get("message", "")
        return [TextContent(type="text", text=f"Echo: {message}")]

    elif name == "random_number":
        min_val = arguments.get("min", 1)
        max_val = arguments.get("max", 100)
        result = random.randint(min_val, max_val)
        return [TextContent(type="text", text=f"Random number between {min_val} and {max_val}: {result}")]

    elif name == "lookup_user":
        user_id = arguments.get("user_id", 0)
        # Simulated user database
        users = {
            1: {"name": "Alice Johnson", "email": "alice@example.com", "role": "Admin"},
            2: {"name": "Bob Smith", "email": "bob@example.com", "role": "User"},
            3: {"name": "Charlie Brown", "email": "charlie@example.com", "role": "Moderator"},
        }
        if user_id in users:
            user = users[user_id]
            return [TextContent(
                type="text",
                text=f"User {user_id}: {user['name']} ({user['email']}) - Role: {user['role']}"
            )]
        else:
            return [TextContent(type="text", text=f"User {user_id} not found")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
