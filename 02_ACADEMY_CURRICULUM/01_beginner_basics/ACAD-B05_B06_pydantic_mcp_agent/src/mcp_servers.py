"""Factory helpers for initializing the Time, Airbnb, and Weather MCP servers."""

from __future__ import annotations

from typing import Optional

from pydantic_ai.mcp import MCPServerStdio

from src.config import DEFAULT_TIMEZONE, AgentConfig


def create_time_server(
    local_timezone: str = DEFAULT_TIMEZONE,
    *,
    python_executable: str = "python",
) -> MCPServerStdio:
    """Create the Time MCP Server (`mcp-server-time`) over stdio."""
    timezone = (local_timezone or "").strip() or DEFAULT_TIMEZONE
    return MCPServerStdio(
        python_executable,
        args=["-m", "mcp_server_time", f"--local-timezone={timezone}"],
        id="time_server",
    )


def create_airbnb_server(*, npx_executable: str = "npx") -> MCPServerStdio:
    """Create the Airbnb MCP Server (`@openbnb/mcp-server-airbnb`) via `npx`."""
    return MCPServerStdio(
        npx_executable,
        args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
        id="airbnb_server",
    )


def create_weather_server(*, python_executable: str = "python") -> MCPServerStdio:
    """Create the Weather MCP Server (`mcp-weather-server`) over stdio.

    Uses `tool_prefix="weather"` to prevent a `convert_time` tool-name collision
    with `mcp-server-time` when PydanticAI combines all MCP toolsets.
    """
    return MCPServerStdio(
        python_executable,
        args=["-m", "mcp_weather_server"],
        tool_prefix="weather",
        id="weather_server",
    )


def create_mcp_servers(
    config: Optional[AgentConfig] = None,
    *,
    local_timezone: Optional[str] = None,
) -> list[MCPServerStdio]:
    """Build and return `[time_server, airbnb_server, weather_server]`."""
    resolved_timezone = (
        local_timezone
        or (config.local_timezone if config is not None else None)
        or DEFAULT_TIMEZONE
    )

    time_server = create_time_server(local_timezone=resolved_timezone)
    airbnb_server = create_airbnb_server()
    weather_server = create_weather_server()

    return [time_server, airbnb_server, weather_server]
