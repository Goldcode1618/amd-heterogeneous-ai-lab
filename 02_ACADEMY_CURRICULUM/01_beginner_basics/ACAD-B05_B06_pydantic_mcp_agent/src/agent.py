"""PydanticAI Agent orchestration for OpenAI-compatible LLMs and Multi-MCP servers."""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.config import AgentConfig, load_config
from src.mcp_servers import create_mcp_servers

SYSTEM_PROMPT = """Anda adalah asisten perjalanan AI otonom yang menggunakan pola ReAct (Reasoning + Acting).
Setiap kali pengguna meminta rencana perjalanan, rekomendasi tanggal menginap, atau pencarian penginapan,
WAJIB ikuti urutan langkah berikut secara berurutan sebelum memberikan jawaban akhir:

1. **Cek Waktu Saat Ini Terlebih Dahulu**:
   - Selalu panggil tool waktu (`get_current_time`) untuk mengetahui tanggal, hari, dan zona waktu saat ini secara akurat.
2. **Cek Prakiraan Cuaca Destinasi**:
   - Panggil tool prakiraan cuaca (`weather_get_weather_byDateTimeRange`, `weather_get_current_weather`, atau `weather_get_weather_details`)
     untuk kota/destinasi tujuan guna menentukan rentang tanggal perjalanan terbaik (cuaca paling mendukung).
3. **Cari & Periksa Detail Penginapan Airbnb**:
   - Setelah menentukan tanggal rekomendasi dari langkah 1 dan 2, panggil `airbnb_search` sesuai destinasi,
     tanggal check-in/check-out hasil rekomendasi, dan jumlah tamu yang diminta.
   - Lanjutkan dengan memanggil `airbnb_listing_details` pada kandidat penginapan terbaik untuk memverifikasi
     fasilitas, harga, ulasan, dan kebijakan sebelum menyusun ringkasan rekomendasi kepada pengguna.

Jika salah satu tool mengembalikan error atau data tidak lengkap, jelaskan keterbatasan tersebut secara jujur tanpa mengarang data."""


class AgentExecutionError(RuntimeError):
    """Raised with diagnostic context when LLM or MCP server execution fails."""


def create_agent(
    config: Optional[AgentConfig] = None,
    mcp_servers: Optional[Sequence[MCPServerStdio]] = None,
) -> Agent[None, str]:
    """Create a configured PydanticAI `Agent` connected to an OpenAI-compatible endpoint."""
    resolved_config = config or load_config()
    servers = list(mcp_servers) if mcp_servers is not None else create_mcp_servers(resolved_config)

    provider = OpenAIProvider(
        base_url=resolved_config.base_url,
        api_key=resolved_config.openai_api_key,
    )
    model = OpenAIChatModel(
        resolved_config.model_name,
        provider=provider,
    )

    return Agent(
        model=model,
        toolsets=servers,
        system_prompt=SYSTEM_PROMPT,
    )


def _format_execution_error(exc: BaseException, config: AgentConfig) -> str:
    """Build a clear diagnostic message for endpoint or MCP server failures."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException, APITimeoutError)):
        return (
            f"Timeout saat menjalankan agen atau menunggu respons dari `{config.base_url}` / MCP server. "
            f"Detail: {exc}"
        )
    if isinstance(exc, (httpx.ConnectError, APIConnectionError, ConnectionError)):
        return (
            f"Gagal terhubung ke endpoint LLM `{config.base_url}` (model: `{config.model_name}`). "
            f"Pastikan server vLLM/Ollama/LM Studio atau API tujuan aktif dan dapat diakses. Detail: {exc}"
        )
    if isinstance(exc, (httpx.HTTPStatusError, APIStatusError)):
        return (
            f"Endpoint LLM `{config.base_url}` mengembalikan error status untuk model `{config.model_name}`. "
            f"Periksa nama model, API key, dan dukungan tool calling pada server. Detail: {exc}"
        )
    if isinstance(exc, (FileNotFoundError, OSError)):
        return (
            "Gagal menjalankan proses salah satu MCP server (pastikan `python`, `npx`, "
            "`mcp-server-time`, dan `mcp-weather-server` tersedia). "
            f"Detail: {exc}"
        )
    if isinstance(exc, UserError):
        return f"Konfigurasi atau toolset agen bermasalah: {exc}"
    return f"Eksekusi agen gagal pada `{config.base_url}` atau salah satu MCP server: {exc}"


async def run_agent(
    prompt: str,
    *,
    config: Optional[AgentConfig] = None,
    agent: Optional[Agent[None, str]] = None,
) -> str:
    """Run a user prompt asynchronously with all MCP servers active."""
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise ValueError("Prompt tidak boleh kosong.")

    resolved_config = config or load_config()
    active_agent = agent or create_agent(resolved_config)

    try:
        async with active_agent.run_mcp_servers():
            result = await active_agent.run(cleaned_prompt)
            return str(result.output)
    except Exception as exc:
        raise AgentExecutionError(_format_execution_error(exc, resolved_config)) from exc
