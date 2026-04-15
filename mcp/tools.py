# agent/mcp/tools.py
"""MCP tools - convert MCP tools to LangChain tools"""

import json
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import create_model, Field
from pydantic.fields import FieldInfo

from mcp.manager import get_mcp_manager


def create_mcp_tool(server_name: str, mcp_tool) -> BaseTool:
    """Convert a single MCP tool to a LangChain tool

    Args:
        server_name: MCP server name
        mcp_tool: MCPTool object

    Returns:
        LangChain BaseTool with proper args schema
    """
    tool_name = mcp_tool.name
    tool_description = mcp_tool.description or f"MCP tool: {tool_name}"

    # Generate function name: mcp_{server}_{tool}
    func_name = f"mcp_{server_name}_{tool_name}".replace("-", "_")

    # Build args schema from MCP tool's input_schema
    args_schema_class = _build_args_schema(mcp_tool.input_schema)

    # Create a closure to capture the variables
    _server_name = server_name
    _tool_name = tool_name

    class MCPTool(BaseTool):
        name: str = func_name
        description: str = f"[MCP:{server_name}] {tool_description}"
        args_schema: type = args_schema_class

        def _run(self, *args, **kwargs) -> dict[str, Any]:
            """Execute the MCP tool"""
            # Build tool_input from kwargs (preferred) or first positional arg
            if kwargs:
                tool_input = kwargs
            elif args and len(args) > 0:
                tool_input = args[0]
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input) if tool_input else {}
                    except json.JSONDecodeError:
                        return {
                            "success": False,
                            "error": "Invalid JSON arguments",
                        }
            else:
                tool_input = {}

            manager = get_mcp_manager()
            return manager.call_tool(_server_name, _tool_name, tool_input)

        async def _arun(self, *args, **kwargs) -> dict[str, Any]:
            """Async execute the MCP tool"""
            return self._run(*args, **kwargs)

    return MCPTool()


def _build_args_schema(input_schema):
    """Build Pydantic model from MCP input schema

    Args:
        input_schema: MCPToolInputSchema object

    Returns:
        Pydantic model class for tool arguments
    """
    properties = input_schema.properties or {}
    required = set(input_schema.required or [])

    fields = {}
    for prop_name, prop_def in properties.items():
        # Determine Python type
        prop_type = prop_def.get("type", "string")
        description = prop_def.get("description", "")
        default = ... if prop_name in required else prop_def.get("default", None)

        py_type = _json_schema_type_to_python(prop_type, prop_def)

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            fields[prop_name] = (py_type, Field(default=default, description=description))

    # Create a dynamic Pydantic model
    if fields:
        return create_model(
            "MCPToolArgs",
            __base__=None,
            **fields
        )
    else:
        # No arguments
        return create_model("MCPToolArgs")


def _json_schema_type_to_python(json_type: str, prop_def: dict) -> type:
    """Convert JSON Schema type to Python type"""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    base_type = type_map.get(json_type, str)

    # Handle optional/nullable
    if prop_def.get("nullable") or prop_def.get("default") is not None:
        from typing import Optional
        return Optional[base_type]

    return base_type


def load_mcp_tools() -> list:
    """Load all tools from connected MCP servers

    Returns:
        List of LangChain tools
    """
    manager = get_mcp_manager()
    tools = []

    for server_name, mcp_tool in manager.list_all_tools():
        try:
            langchain_tool = create_mcp_tool(server_name, mcp_tool)
            tools.append(langchain_tool)
        except Exception as e:
            print(f"[MCP] Failed to convert tool {server_name}/{mcp_tool.name}: {e}")

    return tools


def get_mcp_tool_info() -> list[dict]:
    """Get MCP tool information for debugging/display"""
    manager = get_mcp_manager()
    info = []

    for server_name, mcp_tool in manager.list_all_tools():
        info.append({
            "server": server_name,
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "input_schema": mcp_tool.input_schema.model_dump(),
        })

    return info
