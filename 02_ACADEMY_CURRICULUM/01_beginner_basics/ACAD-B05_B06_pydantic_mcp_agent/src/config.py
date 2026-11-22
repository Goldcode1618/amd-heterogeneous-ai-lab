"""Configuration loader and validator for the PydanticAI Multi-MCP Agent."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TIMEZONE = "Asia/Jakarta"


class NpxNotFoundError(RuntimeError, ValueError):
    """Raised when the `npx` binary is not available on the system PATH."""


class AgentConfig(BaseModel):
    """Validated runtime configuration for the LLM endpoint and MCP servers."""

    base_url: str = Field(..., description="OpenAI-compatible API base URL")
    openai_api_key: str = Field(..., description="API key for the OpenAI-compatible endpoint")
    model_name: str = Field(..., description="Target model identifier served by the endpoint")
    local_timezone: str = Field(
        default=DEFAULT_TIMEZONE,
        description="IANA local timezone passed to mcp-server-time",
    )
    npx_path: Optional[str] = Field(
        default=None,
        description="Resolved path to the `npx` executable on the host system",
    )

    @field_validator("base_url", "openai_api_key", "model_name")
    @classmethod
    def _validate_non_empty(cls, value: str, info) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            env_key = info.field_name.upper()
            raise ValueError(f"Konfigurasi wajib `{env_key}` tidak boleh kosong.")
        return cleaned

    @field_validator("local_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        cleaned = (value or "").strip()
        return cleaned or DEFAULT_TIMEZONE


def validate_npx_available() -> str:
    """Verify that `npx` is installed and accessible on PATH."""
    npx_executable = shutil.which("npx")
    if not npx_executable:
        raise NpxNotFoundError(
            "Biner `npx` tidak ditemukan di PATH sistem. "
            "Instal Node.js (termasuk npm/npx) agar Airbnb MCP Server dapat dijalankan."
        )
    return npx_executable


def load_config(
    env_file: Optional[str | Path] = None,
    *,
    check_npx: bool = True,
    override: bool = False,
) -> AgentConfig:
    """Load environment variables from `.env` and return a validated `AgentConfig`.

    Args:
        env_file: Optional explicit path to a `.env` file. Defaults to `<project_root>/.env`.
        check_npx: Whether to verify that `npx` is available via `shutil.which("npx")`.
        override: Whether values in `env_file` override existing environment variables.
    """
    target_env = Path(env_file) if env_file is not None else DEFAULT_ENV_PATH
    if target_env.is_file():
        load_dotenv(dotenv_path=target_env, override=override)

    npx_path = validate_npx_available() if check_npx else shutil.which("npx")

    return AgentConfig(
        base_url=os.getenv("BASE_URL", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", ""),
        local_timezone=os.getenv("LOCAL_TIMEZONE", DEFAULT_TIMEZONE),
        npx_path=npx_path,
    )
