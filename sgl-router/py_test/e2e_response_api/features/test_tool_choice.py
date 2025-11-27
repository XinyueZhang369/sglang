"""
Test class for tool calling tests.

This module provides test cases for tool choices across different backends.
"""

import sys
from pathlib import Path

import pytest

# Add current directory for local imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.parametrize("setup_backend", ["grpc", "grpc_harmony"], indirect=True)
class TestToolChoice:

    # Shared function tool definitions
    SYSTEM_DIAGNOSTICS_FUNCTION = {
        "type": "function",
        "name": "get_system_diagnostics",
        "description": "Retrieve real-time diagnostics for a spacecraft system.",
        "parameters": {
            "type": "object",
            "properties": {
                "system_name": {
                    "type": "string",
                    "description": "Name of the spacecraft system to query. "
                    "Example: 'Astra-7 Core Reactor'.",
                }
            },
            "required": ["system_name"],
        },
    }

    GET_WEATHER_FUNCTION = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g., San Francisco",
                }
            },
            "required": ["location"],
        },
    }

    CALCULATE_FUNCTION = {
        "type": "function",
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }

    SEARCH_WEB_FUNCTION = {
        "type": "function",
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    LOCAL_SEARCH_FUNCTION = {
        "type": "function",
        "name": "local_search",
        "description": "Search local database",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    DEEPWIKI_MCP_TOOL = {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "require_approval": "never",
    }

    def test_tool_choice_auto(self, setup_backend):
        """
        Test tool_choice="auto" allows model to decide whether to use tools.

        The model should be able to choose to call a tool or not.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.GET_WEATHER_FUNCTION]

        # Query that should trigger tool use
        resp = client.responses.create(
            model=model,
            input="What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        assert len(output) > 0

        # With auto, model should choose to call get_weather for this query
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Model should choose to call function with tool_choice='auto'"

    def test_tool_choice_required(self, setup_backend):
        """
        Test tool_choice="required" forces the model to call at least one tool.

        The model must make at least one function call.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Must have at least one function call
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "tool_choice='required' must force at least one function call"

    def test_tool_choice_specific_function(self, setup_backend):
        """
        Test tool_choice with specific function name forces that function to be called.

        The model must call the specified function.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.SEARCH_WEB_FUNCTION, self.GET_WEATHER_FUNCTION]

        # Force specific function call
        resp = client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Must have function call
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, "Must call the specified function"

        # Must be the specified function
        called_function = function_calls[0]
        assert (
            called_function.name == "search_web"
        ), "Must call the function specified in tool_choice"

    def test_tool_choice_streaming(self, setup_backend):
        """
        Test tool_choice parameter works correctly with streaming.

        Verifies that tool_choice constraints are applied in streaming mode.
        """
        backend, model, client = setup_backend

        if backend in ["openai", "grpc"]:
            pytest.skip("skip for openai")

        tools = [self.CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="Calculate 42 * 17",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        events = [event for event in resp]
        assert len(events) > 0

        event_types = [e.type for e in events]

        # Should have function call events
        assert (
            "response.function_call_arguments.delta" in event_types
        ), "Should have function_call_arguments.delta events"

        # Verify completed event has function call
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        output = completed_events[0].response.output

        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Streaming with tool_choice='required' must produce function call"

    def test_tool_choice_with_mcp_tools(self, setup_backend):
        """
        Test tool_choice parameter works with MCP tools.

        Verifies that tool_choice can control MCP tool usage.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.DEEPWIKI_MCP_TOOL]

        # With tool_choice="auto", should allow MCP tool calls
        resp = client.responses.create(
            model=model,
            input="What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Should have mcp_call with auto
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "tool_choice='auto' should allow MCP tool calls"

    def test_tool_choice_mixed_function_and_mcp(self, setup_backend):
        """
        Test tool_choice with mixed function and MCP tools.

        Verifies tool_choice can select specific tools when both function and MCP tools are available.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.DEEPWIKI_MCP_TOOL, self.LOCAL_SEARCH_FUNCTION]

        # Force specific function call
        resp = client.responses.create(
            model=model,
            input="Search for information about Python",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "local_search"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Must call local_search, not MCP
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "local_search"

        # Should not have mcp_call
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) == 0, "Should only call specified function, not MCP tools"
