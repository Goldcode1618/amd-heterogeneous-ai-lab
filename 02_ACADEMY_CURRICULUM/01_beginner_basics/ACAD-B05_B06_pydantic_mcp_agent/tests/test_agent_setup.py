"""Unit tests for configuration, Multi-MCP server factories, and PydanticAI Agent setup."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from src.agent import SYSTEM_PROMPT, create_agent
from src.config import AgentConfig, load_config
from src.mcp_servers import create_mcp_servers


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear configuration environment variables before each test."""
    for key in ("BASE_URL", "OPENAI_API_KEY", "MODEL_NAME", "LOCAL_TIMEZONE"):
        monkeypatch.delenv(key, raising=False)


def test_load_config_from_env_file(
    tmp_path: Path,
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration should load and validate values from a `.env` file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BASE_URL=http://localhost:11434/v1",
                "OPENAI_API_KEY=ollama",
                "MODEL_NAME=qwen3:30b",
                "LOCAL_TIMEZONE=Asia/Jakarta",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Program Files\nodejs\npx.CMD" if name == "npx" else None)

    config = load_config(env_file=env_file, override=True)

    assert isinstance(config, AgentConfig)
    assert config.base_url == "http://localhost:11434/v1"
    assert config.openai_api_key == "ollama"
    assert config.model_name == "qwen3:30b"
    assert config.local_timezone == "Asia/Jakarta"
    assert config.npx_path == r"C:\Program Files\nodejs\npx.CMD"


@pytest.mark.parametrize(
    "missing_key",
    ["BASE_URL", "OPENAI_API_KEY", "MODEL_NAME"],
)
def test_load_config_rejects_empty_required_variables(
    tmp_path: Path,
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
) -> None:
    """Validation must fail when any required variable is empty."""
    values = {
        "BASE_URL": "http://localhost:8000/v1",
        "OPENAI_API_KEY": "abc-123",
        "MODEL_NAME": "Qwen3-30B-A3B",
        "LOCAL_TIMEZONE": "Asia/Jakarta",
    }
    values[missing_key] = ""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(f"{key}={val}" for key, val in values.items()),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda name: "npx" if name == "npx" else None)

    with pytest.raises(ValueError, match=missing_key):
        load_config(env_file=env_file, override=True)


def test_load_config_fails_when_npx_missing(
    tmp_path: Path,
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation must fail cleanly if `npx` cannot be found via `shutil.which`."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BASE_URL=http://localhost:8000/v1",
                "OPENAI_API_KEY=abc-123",
                "MODEL_NAME=Qwen3-30B-A3B",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(ValueError, match="npx"):
        load_config(env_file=env_file, override=True)


def test_create_mcp_servers_builds_three_stdio_servers_without_spawning() -> None:
    """`create_mcp_servers` should construct M=3 `MCPServerStdio` objects lazily."""
    config = AgentConfig(
        base_url="http://localhost:8000/v1",
        openai_api_key="abc-123",
        model_name="Qwen3-30B-A3B",
        local_timezone="Asia/Tokyo",
        npx_path="npx",
    )

    servers = create_mcp_servers(config)

    assert len(servers) == 3
    assert all(isinstance(server, MCPServerStdio) for server in servers)

    time_server, airbnb_server, weather_server = servers

    assert time_server.command == "python"
    assert list(time_server.args) == ["-m", "mcp_server_time", "--local-timezone=Asia/Tokyo"]
    assert time_server.tool_prefix is None

    assert airbnb_server.command == "npx"
    assert list(airbnb_server.args) == ["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
    assert airbnb_server.tool_prefix is None

    assert weather_server.command == "python"
    assert list(weather_server.args) == ["-m", "mcp_weather_server"]
    assert weather_server.tool_prefix == "weather"


def test_create_agent_initializes_openai_chat_model_and_mcp_toolsets() -> None:
    """`create_agent` should attach `OpenAIChatModel` and all 3 MCP server toolsets."""
    config = AgentConfig(
        base_url="http://localhost:1234/v1",
        openai_api_key="lm-studio",
        model_name="qwen2.5-coder-32b-instruct",
        local_timezone="Asia/Jakarta",
        npx_path="npx",
    )

    agent = create_agent(config)

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "qwen2.5-coder-32b-instruct"
    assert str(agent.model.base_url).rstrip("/") == "http://localhost:1234/v1"

    mcp_toolsets = [t for t in agent.toolsets if isinstance(t, MCPServerStdio)]
    assert len(mcp_toolsets) == 3
    assert [t.id for t in mcp_toolsets] == ["time_server", "airbnb_server", "weather_server"]

    assert "get_current_time" in SYSTEM_PROMPT
    assert "airbnb_search" in SYSTEM_PROMPT
    assert "airbnb_listing_details" in SYSTEM_PROMPT


def test_combined_mcp_toolsets_avoid_convert_time_collision() -> None:
    """Prefixing weather tools prevents `convert_time` conflict with time server."""

    def _build_time_tools() -> FunctionToolset[None]:
        def get_current_time(timezone: str) -> str:
            return timezone

        def convert_time(source_timezone: str, time: str, target_timezone: str) -> str:
            return time

        return FunctionToolset(tools=[get_current_time, convert_time])

    def _build_weather_tools() -> FunctionToolset[None]:
        def get_weather_byDateTimeRange(city: str) -> str:
            return city

        def convert_time(source_timezone: str) -> str:
            return source_timezone

        return FunctionToolset(tools=[get_weather_byDateTimeRange, convert_time])

    time_mock = _build_time_tools()
    weather_mock = _build_weather_tools()

    prefixed_weather: AbstractToolset[None] = weather_mock.prefixed("weather")
    combined = CombinedToolset([time_mock, prefixed_weather])

    run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tools = asyncio.run(combined.get_tools(run_ctx))
    assert "get_current_time" in tools
    assert "convert_time" in tools
    assert "weather_get_weather_byDateTimeRange" in tools
    assert "weather_convert_time" in tools
