import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import json
from agent.tools import mcp_client

# Note: These tests interact with the real MCP server subprocess and create a DB file.
# In a real CI env, we'd mock the subprocess or use a temporary DB path.

def test_mcp_client_add_and_search():
    # 1. Add Memory
    content = "Integration Test Content"
    category = "test"
    result_str = mcp_client.call_tool("add_memory", {"content": content, "category": category})
    
    # Expecting: Saved to test: Integration Test Content...
    assert "Saved to test" in result_str
    
    # 2. Search Memory
    search_res_str = mcp_client.call_tool("search_memory", {"query": "Integration"})
    
    # Expecting JSON string of results
    # The tool returns just the text, but let's check if the content is there.
    assert "Integration Test Content" in search_res_str
