import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from app.mini_agent.tools.base import Tool, ToolResult

ConnectionType = Literal["stdio", "sse", "http", "streamable_http"]


@dataclass
class MCPTimeoutConfig:
    connect_timeout: float = 10.0
    execute_timeout: float = 60.0
    sse_read_timeout: float = 120.0


_default_timeout_config = MCPTimeoutConfig()


def set_mcp_timeout_config(
    connect_timeout: float | None = None,
    execute_timeout: float | None = None,
    sse_read_timeout: float | None = None,
) -> None:
    global _default_timeout_config
    if connect_timeout is not None:
        _default_timeout_config.connect_timeout = connect_timeout
    if execute_timeout is not None:
        _default_timeout_config.execute_timeout = execute_timeout
    if sse_read_timeout is not None:
        _default_timeout_config.sse_read_timeout = sse_read_timeout


def get_mcp_timeout_config() -> MCPTimeoutConfig:
    return _default_timeout_config


class MCPTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        session: ClientSession,
        execute_timeout: float | None = None,
    ):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._session = session
        self._execute_timeout = execute_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        timeout = self._execute_timeout or _default_timeout_config.execute_timeout

        try:
            async with asyncio.timeout(timeout):
                result = await self._session.call_tool(self._name, arguments=kwargs)

            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                else:
                    content_parts.append(str(item))

            content_str = "\n".join(content_parts)
            is_error = result.isError if hasattr(result, "isError") else False

            return ToolResult(
                success=not is_error,
                content=content_str,
                error=None if not is_error else "Tool returned error",
            )

        except TimeoutError:
            return ToolResult(
                success=False,
                content="",
                error=f"MCP tool execution timed out after {timeout}s.",
            )
        except Exception as e:
            return ToolResult(
                success=False, content="", error=f"MCP tool execution failed: {str(e)}"
            )


class MCPServerConnection:
    def __init__(
        self,
        name: str,
        connection_type: ConnectionType = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        connect_timeout: float | None = None,
        execute_timeout: float | None = None,
        sse_read_timeout: float | None = None,
    ):
        self.name = name
        self.connection_type = connection_type
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url
        self.headers = headers or {}
        self.connect_timeout = connect_timeout
        self.execute_timeout = execute_timeout
        self.sse_read_timeout = sse_read_timeout
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack | None = None
        self.tools: list[MCPTool] = []

    def _get_connect_timeout(self) -> float:
        return self.connect_timeout or _default_timeout_config.connect_timeout

    def _get_sse_read_timeout(self) -> float:
        return self.sse_read_timeout or _default_timeout_config.sse_read_timeout

    def _get_execute_timeout(self) -> float:
        return self.execute_timeout or _default_timeout_config.execute_timeout

    async def connect(self) -> bool:
        connect_timeout = self._get_connect_timeout()

        try:
            self.exit_stack = AsyncExitStack()

            async with asyncio.timeout(connect_timeout):
                if self.connection_type == "stdio":
                    read_stream, write_stream = await self._connect_stdio()
                elif self.connection_type == "sse":
                    read_stream, write_stream = await self._connect_sse()
                else:
                    read_stream, write_stream = await self._connect_streamable_http()

                session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                self.session = session

                await session.initialize()

                tools_list = await session.list_tools()

            execute_timeout = self._get_execute_timeout()
            for tool in tools_list.tools:
                parameters = tool.inputSchema if hasattr(tool, "inputSchema") else {}
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=parameters,
                    session=session,
                    execute_timeout=execute_timeout,
                )
                self.tools.append(mcp_tool)

            conn_info = self.url if self.url else self.command
            print(
                f"[OK] Connected to MCP server '{self.name}' ({self.connection_type}: {conn_info}) - loaded {len(self.tools)} tools"
            )
            return True

        except TimeoutError:
            print(
                f"[X] Connection to MCP server '{self.name}' timed out after {connect_timeout}s"
            )
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            return False

        except Exception as e:
            print(f"[X] Failed to connect to MCP server '{self.name}': {e}")
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            return False

    async def _connect_stdio(self):
        server_params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env if self.env else None
        )
        return await self.exit_stack.enter_async_context(stdio_client(server_params))

    async def _connect_sse(self):
        connect_timeout = self._get_connect_timeout()
        sse_read_timeout = self._get_sse_read_timeout()
        return await self.exit_stack.enter_async_context(
            sse_client(
                url=self.url,
                headers=self.headers if self.headers else None,
                timeout=connect_timeout,
                sse_read_timeout=sse_read_timeout,
            )
        )

    async def _connect_streamable_http(self):
        connect_timeout = self._get_connect_timeout()
        sse_read_timeout = self._get_sse_read_timeout()
        read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(
                url=self.url,
                headers=self.headers if self.headers else None,
                timeout=connect_timeout,
                sse_read_timeout=sse_read_timeout,
            )
        )
        return read_stream, write_stream

    async def disconnect(self):
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            except Exception:
                pass
            finally:
                self.exit_stack = None
                self.session = None


_mcp_connections: list[MCPServerConnection] = []


def _determine_connection_type(server_config: dict) -> ConnectionType:
    explicit_type = server_config.get("type", "").lower()
    if explicit_type in ("stdio", "sse", "http", "streamable_http"):
        return explicit_type
    if server_config.get("url"):
        return "streamable_http"
    return "stdio"


def _resolve_mcp_config_path(config_path: str) -> Path | None:
    config_file = Path(config_path)
    if config_file.exists():
        return config_file
    if config_file.name == "mcp.json":
        example_file = config_file.parent / "mcp-example.json"
        if example_file.exists():
            return example_file
    return None


async def load_mcp_tools_async(config_path: str = "mcp.json") -> list[Tool]:
    global _mcp_connections

    config_file = _resolve_mcp_config_path(config_path)

    if config_file is None:
        print(f"MCP config not found: {config_path}")
        return []

    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        mcp_servers = config.get("mcpServers", {})

        if not mcp_servers:
            print("No MCP servers configured")
            return []

        all_tools = []

        for server_name, server_config in mcp_servers.items():
            if server_config.get("disabled", False):
                print(f"Skipping disabled server: {server_name}")
                continue

            conn_type = _determine_connection_type(server_config)
            url = server_config.get("url")
            command = server_config.get("command")

            if conn_type == "stdio" and not command:
                print(f"No command specified for STDIO server: {server_name}")
                continue
            if conn_type in ("sse", "http", "streamable_http") and not url:
                print(f"No url specified for {conn_type.upper()} server: {server_name}")
                continue

            connection = MCPServerConnection(
                name=server_name,
                connection_type=conn_type,
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                url=url,
                headers=server_config.get("headers", {}),
                connect_timeout=server_config.get("connect_timeout"),
                execute_timeout=server_config.get("execute_timeout"),
                sse_read_timeout=server_config.get("sse_read_timeout"),
            )
            success = await connection.connect()

            if success:
                _mcp_connections.append(connection)
                all_tools.extend(connection.tools)

        print(f"\nTotal MCP tools loaded: {len(all_tools)}")

        return all_tools

    except Exception as e:
        print(f"Error loading MCP config: {e}")
        return []


async def cleanup_mcp_connections():
    global _mcp_connections
    for connection in _mcp_connections:
        await connection.disconnect()
    _mcp_connections.clear()
