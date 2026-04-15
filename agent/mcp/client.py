"""MCP Client - Connect to MCP servers via HTTP/SSE"""

import json
import httpx
from typing import Any

from agent.mcp.models import (
    JSONRPCRequest,
    JSONRPCResponse,
    InitializeResult,
    MCPTool,
    ToolCallResult,
)


class MCPClientError(Exception):
    """MCP client error"""
    pass


class MCPClient:
    """MCP Client - supports SSE, HTTP, and Streamable HTTP transport"""

    def __init__(
        self,
        name: str,
        transport: str,  # "sse" | "http" | "streamable"
        url: str,
        timeout: int = 30,
        headers: dict | None = None,
    ):
        self.name = name
        self.transport = transport
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}

        self._initialized = False
        self._capabilities: InitializeResult | None = None
        self._tools: list[MCPTool] = []
        self._request_id = 0
        self._http_client: httpx.Client | None = None

    # ============ Lifecycle ============

    def connect(self) -> bool:
        """Connect and initialize MCP server"""
        # Streamable HTTP requires both Accept types
        accept_header = "application/json"
        if self.transport == "streamable":
            accept_header = "application/json, text/event-stream"
        elif self.transport == "sse":
            # SSE endpoint is typically /sse
            if not self.url.endswith("/sse"):
                self.url = f"{self.url}/sse"

        self._http_client = httpx.Client(
            timeout=self.timeout,
            headers={
                **self.headers,
                "Content-Type": "application/json",
                "Accept": accept_header,
            },
        )

        return self._initialize()

    def disconnect(self) -> None:
        """Disconnect from server"""
        self._initialized = False
        if self._http_client:
            self._http_client.close()
            self._http_client = None

    def _initialize(self) -> bool:
        """Execute MCP initialization handshake"""
        try:
            # 1. Send initialize request
            result = self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-dev-tool", "version": "1.0.0"},
            })

            if not result:
                return False

            self._capabilities = InitializeResult(**result)

            # 2. Send initialized notification
            self._notify("notifications/initialized")

            # 3. Load tools if supported
            if self._capabilities.capabilities.has_tools():
                self._load_tools()

            self._initialized = True
            return True

        except MCPClientError:
            return False

    # ============ Tools ============

    def _load_tools(self) -> None:
        """Load tools from server"""
        result = self._request("tools/list")
        if result and "tools" in result:
            self._tools = [MCPTool(**t) for t in result["tools"]]

    def list_tools(self) -> list[MCPTool]:
        """Get available tools"""
        return self._tools

    def call_tool(self, name: str, arguments: dict) -> ToolCallResult:
        """Call a tool"""
        result = self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        if result:
            return ToolCallResult(**result)

        return ToolCallResult(
            content=[{"type": "text", "text": "Tool call failed"}],
            is_error=True,
        )

    # ============ Low-level communication ============

    def _parse_sse_response(self, response_text: str) -> dict | None:
        """Parse SSE response and extract JSON-RPC result"""
        result = None
        for line in response_text.split('\n'):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])
                    if 'result' in data:
                        result = data['result']
                    elif 'error' in data:
                        raise MCPClientError(data['error'].get('message', 'Unknown error'))
                except json.JSONDecodeError:
                    continue
        return result

    def _request(self, method: str, params: dict | None = None) -> dict | None:
        """Send JSON-RPC request"""
        if not self._http_client:
            raise MCPClientError("Client not connected")

        self._request_id += 1
        request = JSONRPCRequest(
            id=self._request_id,
            method=method,
            params=params,
        )

        try:
            response = self._http_client.post(
                self.url,
                json=request.model_dump(),
            )
            response.raise_for_status()

            content_type = response.headers.get('content-type', '')

            # Handle SSE response (Streamable HTTP)
            if 'text/event-stream' in content_type:
                return self._parse_sse_response(response.text)

            # Handle JSON response
            data = response.json()
            rpc_response = JSONRPCResponse.from_dict(data)

            if rpc_response.is_error():
                raise MCPClientError(rpc_response.get_error_message())

            return rpc_response.result

        except httpx.HTTPError as e:
            raise MCPClientError(f"HTTP error: {e}")
        except Exception as e:
            raise MCPClientError(f"Request failed: {e}")

    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send JSON-RPC notification (no response expected)"""
        if not self._http_client:
            return

        request = JSONRPCRequest(
            id=None,  # Notifications have no id
            method=method,
            params=params,
        )

        try:
            self._http_client.post(self.url, json=request.model_dump())
        except Exception:
            pass  # Notifications are fire-and-forget

    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._initialized

    @property
    def capabilities(self) -> InitializeResult | None:
        """Get server capabilities"""
        return self._capabilities
