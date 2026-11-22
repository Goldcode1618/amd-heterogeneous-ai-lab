"""CLI entrypoint for the Standalone PydanticAI Multi-MCP Server Agent."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Optional

import httpx

from src.agent import AgentExecutionError, create_agent, run_agent
from src.config import DEFAULT_ENV_PATH, PROJECT_ROOT, AgentConfig, load_config

EXAMPLE_ENV_PATH = PROJECT_ROOT / ".env.example"


def _load_config_for_check() -> tuple[AgentConfig | None, str]:
    """Load configuration from `.env` or fall back to `.env.example` during `--check`."""
    if DEFAULT_ENV_PATH.is_file():
        return load_config(env_file=DEFAULT_ENV_PATH, check_npx=False), ".env"

    try:
        return load_config(check_npx=False), "environment variables"
    except ValueError:
        if EXAMPLE_ENV_PATH.is_file():
            return load_config(env_file=EXAMPLE_ENV_PATH, check_npx=False), ".env.example (fallback)"
        return None, "missing"


def check_environment() -> int:
    """Verify configuration, `python`, `npx`, MCP modules, and `BASE_URL/models` reachability."""
    print("=== Pemeriksaan Lingkungan PydanticAI Multi-MCP Agent ===")
    has_missing_binary = False

    python_path = shutil.which("python")
    if python_path:
        print(f"[OK]   python ditemukan: {python_path}")
    else:
        print("[FAIL] biner `python` tidak ditemukan di PATH.")
        has_missing_binary = True

    npx_path = shutil.which("npx")
    if npx_path:
        print(f"[OK]   npx ditemukan: {npx_path}")
    else:
        print("[FAIL] biner `npx` tidak ditemukan di PATH (dibutuhkan oleh Airbnb MCP Server).")
        has_missing_binary = True

    for module_name in ("pydantic_ai", "mcp_server_time", "mcp_weather_server"):
        if importlib.util.find_spec(module_name) is not None:
            print(f"[OK]   modul Python `{module_name}` tersedia.")
        else:
            print(f"[FAIL] modul Python `{module_name}` belum terpasang.")
            has_missing_binary = True

    config, source = _load_config_for_check()
    if config is None:
        print("[WARN] Konfigurasi `.env` belum lengkap (`BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME`).")
        return 1 if has_missing_binary else 0

    print(
        f"[OK]   konfigurasi dimuat dari `{source}` "
        f"(BASE_URL={config.base_url}, MODEL_NAME={config.model_name}, LOCAL_TIMEZONE={config.local_timezone})"
    )

    models_url = f"{config.base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {config.openai_api_key}"}
    try:
        response = httpx.get(models_url, headers=headers, timeout=5.0)
        if 200 <= response.status_code < 300:
            print(f"[OK]   endpoint `{models_url}` dapat dijangkau (HTTP {response.status_code}).")
        else:
            print(
                f"[WARN] endpoint `{models_url}` merespons HTTP {response.status_code}. "
                "Periksa BASE_URL atau API key Anda."
            )
    except httpx.HTTPError as exc:
        print(
            f"[WARN] endpoint `{models_url}` belum dapat dijangkau ({type(exc).__name__}: {exc}). "
            "Pastikan vLLM, Ollama, LM Studio, atau API target sedang berjalan sebelum menjalankan prompt."
        )

    return 1 if has_missing_binary else 0


async def _run_interactive(config: AgentConfig) -> int:
    """Start an interactive REPL loop for conversing with the agent."""
    agent = create_agent(config)
    print("Mode interaktif siap. Ketik permintaan Anda, atau ketik `exit` / `quit` untuk keluar.")
    print(f"Endpoint: {config.base_url} | Model: {config.model_name} | Timezone: {config.local_timezone}")

    while True:
        try:
            user_input = input("\nAnda> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari sesi interaktif.")
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Sampai jumpa.")
            return 0

        try:
            reply = await run_agent(user_input, config=config, agent=agent)
            print(f"\nAgen>\n{reply}")
        except AgentExecutionError as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"\n[ERROR] Terjadi kegagalan tak terduga: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Standalone AI Agent berbasis PydanticAI dengan Time, Airbnb, dan Weather MCP Server."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Jalankan agen satu kali (one-shot) dengan prompt tertentu.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Periksa keterjangkauan BASE_URL/models serta ketersediaan `python` dan `npx`.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path opsional ke file `.env` kustom.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check:
        return check_environment()

    try:
        config = load_config(env_file=args.env_file)
    except ValueError as exc:
        print(f"[ERROR] Konfigurasi tidak valid: {exc}", file=sys.stderr)
        print("Salin `.env.example` menjadi `.env` dan pastikan seluruh variabel wajib terisi.", file=sys.stderr)
        return 1

    if args.prompt is not None:
        try:
            output = asyncio.run(run_agent(args.prompt, config=config))
            print(output)
            return 0
        except AgentExecutionError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    return asyncio.run(_run_interactive(config))


if __name__ == "__main__":
    raise SystemExit(main())
