import pytest
import os
import sys
from ai_query.mcp import mcp
from ai_query.types import Tool

# Add project root to path so we can import the server script if needed, 
# although we run it via subprocess
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
server_script = os.path.join(project_root, "tests/test_mcp_server.py")

@pytest.mark.asyncio
async def test_mcp_server_connection():
    """Test connecting to the local MCP server and listing tools."""
    print(f"Connecting to server at: {server_script}")
    
    # Connect to the local python server
    async with mcp("python", server_script) as server:
        # Check tools are discovered
        tools = server.tools
        assert len(tools) >= 5
        
        # Check specific tool existence
        assert "calculator" in tools
        assert "get_weather" in tools
        assert "echo" in tools
        assert "random_number" in tools
        assert "lookup_user" in tools
        
        # Check tool details
        calc = tools["calculator"]
        assert isinstance(calc, Tool)
        assert "Perform basic math calculations" in calc.description

@pytest.mark.asyncio
async def test_calculator_tool():
    """Test the calculator tool execution."""
    async with mcp("python", server_script) as server:
        tools = server.tools
        
        # Test addition
        result = await tools["calculator"].run(expression="10 + 5")
        assert "15" in str(result)
        
        # Test multiplication
        result = await tools["calculator"].run(expression="3 * 7")
        assert "21" in str(result)
        
        # Test error handling
        result = await tools["calculator"].run(expression="invalid")
        assert "Error" in str(result)

@pytest.mark.asyncio
async def test_get_weather_tool():
    """Test the get_weather tool execution."""
    async with mcp("python", server_script) as server:
        tools = server.tools
        
        # Test known city
        result = await tools["get_weather"].run(city="Tokyo")
        assert "Tokyo" in str(result)
        assert "Humidity" in str(result)
        
        # Test unknown city (random data)
        result = await tools["get_weather"].run(city="Atlantis")
        assert "Atlantis" in str(result)

@pytest.mark.asyncio
async def test_echo_tool():
    """Test the echo tool execution."""
    async with mcp("python", server_script) as server:
        tools = server.tools
        
        msg = "Hello MCP!"
        result = await tools["echo"].run(message=msg)
        assert f"Echo: {msg}" in str(result)

@pytest.mark.asyncio
async def test_lookup_user_tool():
    """Test the lookup_user tool execution."""
    async with mcp("python", server_script) as server:
        tools = server.tools
        
        # Test existing user
        result = await tools["lookup_user"].run(user_id=1)
        assert "Alice Johnson" in str(result)
        assert "Admin" in str(result)
        
        # Test non-existing user
        result = await tools["lookup_user"].run(user_id=999)
        assert "not found" in str(result)
