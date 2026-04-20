"""MCP Client - Connect to MCP servers via HTTP/SSE or Stdio"""

import json
import subprocess
import threading
import httpx
from typing import Any
from pathlib import Path

from mcp.models import (
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
    """MCP Client - supports SSE, HTTP, Streamable HTTP, and Stdio transport"""

    def __init__(
        self,
        name: str,
        transport: str,  # "sse" | "http" | "streamable" | "stdio"
        url: str = "",
        timeout: int = 30,
        headers: dict | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ):
        self.name = name
        self.transport = transport
        self.url = url.rstrip("/") if url else ""
        self.timeout = timeout
        self.headers = headers or {}
        self.command = command or []
        self.env = env or {}
        self.cwd = str(cwd) if cwd else None

        self._initialized = False
        self._capabilities: InitializeResult | None = None
        self._tools: list[MCPTool] = []
        self._request_id = 0

        # HTTP transport
        self._http_client: httpx.Client | None = None

        # Stdio transport
        self._process: subprocess.Popen | None = None
        self._read_lock = threading.Lock()

    # ============ Lifecycle ============

    def connect(self) -> bool:
        """Connect and initialize MCP server"""
        if self.transport == "stdio":
            return self._connect_stdio()
        else:
            return self._connect_http()

    def _connect_http(self) -> bool:
        """Connect via HTTP/SSE/Streamable"""
        accept_header = "application/json"
        if self.transport == "streamable":
            accept_header = "application/json, text/event-stream"
        elif self.transport == "sse":
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

    def _connect_stdio(self) -> bool:
        """Connect via stdio (subprocess)"""
        if not self.command:
            raise MCPClientError("stdio transport requires 'command'")

        import os
        env = os.environ.copy()
        env.update(self.env)

        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=env,
                bufsize=0,  # unbuffered
            )
            return self._initialize()
        except FileNotFoundError as e:
            raise MCPClientError(f"Failed to start process: {e}")

    def disconnect(self) -> None:
        """Disconnect from server"""
        self._initialized = False
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

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
        if self.transport == "stdio":
            return self._request_stdio(method, params)
        else:
            return self._request_http(method, params)

    def _request_http(self, method: str, params: dict | None = None) -> dict | None:
        """Send JSON-RPC request via HTTP"""
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

    def _request_stdio(self, method: str, params: dict | None = None) -> dict | None:
        """Send JSON-RPC request via stdio"""
        if not self._process or not self._process.stdin:
            raise MCPClientError("Process not running")

        self._request_id += 1
        request = JSONRPCRequest(
            id=self._request_id,
            method=method,
            params=params,
        )

        try:
            # Write request to stdin (newline-delimited JSON)
            request_line = json.dumps(request.model_dump()) + "\n"
            with self._read_lock:
                self._process.stdin.write(request_line.encode("utf-8"))
                self._process.stdin.flush()

                # Read response from stdout
                response_line = self._process.stdout.readline()
                if not response_line:
                    raise MCPClientError("Empty response from server")

                data = json.loads(response_line.decode("utf-8").strip())
                rpc_response = JSONRPCResponse.from_dict(data)

                if rpc_response.is_error():
                    raise MCPClientError(rpc_response.get_error_message())

                return rpc_response.result

        except json.JSONDecodeError as e:
            raise MCPClientError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise MCPClientError(f"Stdio request failed: {e}")

    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send JSON-RPC notification (no response expected)"""
        request = JSONRPCRequest(
            id=None,  # Notifications have no id
            method=method,
            params=params,
        )

        if self.transport == "stdio":
            self._notify_stdio(request)
        else:
            self._notify_http(request)

    def _notify_http(self, request: JSONRPCRequest) -> None:
        """Send notification via HTTP"""
        if not self._http_client:
            return
        try:
            self._http_client.post(self.url, json=request.model_dump())
        except Exception:
            pass  # Notifications are fire-and-forget

    def _notify_stdio(self, request: JSONRPCRequest) -> None:
        """Send notification via stdio"""
        if not self._process or not self._process.stdin:
            return
        try:
            request_line = json.dumps(request.model_dump()) + "\n"
            self._process.stdin.write(request_line.encode("utf-8"))
            self._process.stdin.flush()
        except Exception:
            pass  # Notifications are fire-and-forget

    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        if self.transport == "stdio":
            return self._initialized and self._process is not None and self._process.poll() is None
        return self._initialized

    @property
    def capabilities(self) -> InitializeResult | None:
        """Get server capabilities"""
        return self._capabilities
