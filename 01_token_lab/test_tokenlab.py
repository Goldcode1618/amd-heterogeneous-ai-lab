"""Lightweight AMD Token Lab API client verification script.

Loads credentials exclusively from environment variables or a local .env file
(never hardcoded) and verifies connectivity to https://api.tokenlab.amd.com/v1
and inference against Gemma-4-31B-IT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError as exc:
    print(
        "Missing 'openai' package. Install dependencies with:\n"
        "  pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def load_local_env() -> None:
    """Load .env from the script directory if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(dotenv_path=env_path)
    elif env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_local_env()

    api_key = os.getenv("AMD_TOKENLAB_API_KEY", "").strip()
    base_url = os.getenv("AMD_TOKENLAB_BASE_URL", "https://api.tokenlab.amd.com/v1").strip()
    target_model = os.getenv("AMD_TOKENLAB_MODEL", "Gemma-4-31B-IT").strip()

    if not api_key or api_key == "your_amd_tokenlab_api_key_here":
        print(
            "ERROR: AMD_TOKENLAB_API_KEY is not set.\n"
            "Copy .env.example to .env and insert your personal AMD Token Lab API key.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"[1/2] Querying active models at {base_url} ...")
    try:
        models_resp = client.models.list()
        model_ids = sorted(m.id for m in models_resp.data)
        print(f"Found {len(model_ids)} active model(s):")
        for model_id in model_ids:
            marker = " <-- target" if model_id == target_model else ""
            print(f"  - {model_id}{marker}")
    except Exception as exc:
        print(f"ERROR querying /v1/models: {exc}", file=sys.stderr)
        return 2

    print(f"\n[2/2] Sending test completion to {target_model} ...")
    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise AMD AI Developer Program assistant.",
                },
                {
                    "role": "user",
                    "content": "Reply with 'TOKENLAB_OK' and name two benefits of ROCm & XDNA NPU.",
                },
            ],
            max_tokens=128,
            temperature=0.2,
        )
        reply = response.choices[0].message.content
        print(f"Model Response:\n{reply}")
        if response.usage:
            print(
                f"\nToken Usage -> prompt: {response.usage.prompt_tokens}, "
                f"completion: {response.usage.completion_tokens}, "
                f"total: {response.usage.total_tokens}"
            )
    except Exception as exc:
        print(f"ERROR calling chat completion on {target_model}: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
