"""
Agent Tests.

This module provides unit tests for the LangGraph multi-agent system.
It tests the supervisor node and its routing logic.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from agent.graph import supervisor_node
from langchain_core.messages import HumanMessage

def test_supervisor_routing_research():
    state = {"messages": [HumanMessage(content="Research LangGraph")], "plan": "Research LangGraph", "is_plan_done": False}
    result = supervisor_node(state)
    assert result["next"] == "Researcher"

def test_supervisor_routing_paper():
    state = {"messages": [HumanMessage(content="Find paper about LLMs")], "plan": "Find paper about LLMs", "is_plan_done": False}
    result = supervisor_node(state)
    assert result["next"] == "Researcher"

def test_supervisor_routing_save():
    # Supervisor requires 'SELECTED:' to route to Formatter
    context = "Here is the content. SELECTED: The Link - http://example.com"
    state = {"messages": [HumanMessage(content="Save this link"), HumanMessage(content=context)], "plan": "Save this link", "is_plan_done": False}
    result = supervisor_node(state)
    assert result["next"] == "Formatter"

def test_supervisor_routing_format():
    # Supervisor requires 'SELECTED:' to route to Formatter
    context = "Here is the content. SELECTED: The Link - http://example.com"
    state = {"messages": [HumanMessage(content="Write a report about this"), HumanMessage(content=context)], "plan": "Write a report about this", "is_plan_done": False}
    result = supervisor_node(state)
    assert result["next"] == "Formatter"

def test_supervisor_routing_chat():
    state = {"messages": [HumanMessage(content="Hi there")], "plan": "Hi there", "is_plan_done": False}
    result = supervisor_node(state)
    assert result["next"] == "ChatAgent"

def test_supervisor_routing_fallback():
    # Small talk bubbles up to ChatAgent now, not FINISH directly
    state = {"messages": [HumanMessage(content="Blah blah blah")], "plan": "Blah blah blah", "is_plan_done": False}
    result = supervisor_node(state)
    # The supervisor logic routes small talk to ChatAgent
    assert result["next"] == "ChatAgent"
