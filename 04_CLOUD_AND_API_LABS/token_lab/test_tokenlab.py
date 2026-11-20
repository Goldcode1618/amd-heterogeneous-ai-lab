"""AMD Token Lab API client verification & latency/throughput logger.

Loads credentials exclusively from environment variables or a local .env file
(never hardcoded), verifies connectivity to https://api.tokenlab.amd.com/v1,
measures wall-clock latency and token throughput, and records results into
00_RESEARCH_LIBRARY via scripts/log_experiment.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError as exc:
    print(
        "Missing 'openai' package. Run inside 04_CLOUD_AND_API_LABS\\token_lab\\.venv:\n"
        "  .venv\\Scripts\\python.exe test_tokenlab.py",
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


def benchmark_model(client: OpenAI, target_model: str, base_url: str, available_count: int) -> int:
    """Run timed completion against target_model and log metrics to 00_RESEARCH_LIBRARY."""
    print(f"\n[2/2] Sending timed test completion to {target_model} ...")
    t0 = time.perf_counter()
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
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"ERROR calling chat completion on {target_model}: {exc}", file=sys.stderr)
        record_experiment(
            track="04_cloud_token_lab",
            hardware_tier="Cloud_API",
            device_name="AMD Token Lab Inference Endpoint",
            execution_provider="OpenAI_Compatible_API",
            model_or_workload=target_model,
            precision="BF16/FP8_API",
            latency_ms_p50=elapsed_ms,
            latency_ms_p95=elapsed_ms,
            status="FAIL",
            script_path=__file__,
            notes=f"base_url={base_url} error={exc}",
        )
        return 3

    elapsed_sec = max(time.perf_counter() - t0, 1e-6)
    elapsed_ms = elapsed_sec * 1000.0
    reply = response.choices[0].message.content
    print(f"Model Response:\n{reply}")

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    tps = 0.0
    if response.usage:
        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0
        total_tokens = response.usage.total_tokens or (prompt_tokens + completion_tokens)
        tps = completion_tokens / elapsed_sec if completion_tokens > 0 else 0.0
        print(
            f"\nToken Usage -> prompt: {prompt_tokens}, "
            f"completion: {completion_tokens}, total: {total_tokens}"
        )
    print(f"Wall Latency -> {elapsed_ms:.2f} ms | Throughput -> {tps:.2f} tokens/s")

    entry = record_experiment(
        track="04_cloud_token_lab",
        hardware_tier="Cloud_API",
        device_name="AMD Token Lab Inference Endpoint",
        execution_provider="OpenAI_Compatible_API",
        model_or_workload=target_model,
        precision="API_Hosted",
        batch_size=1,
        latency_ms_p50=elapsed_ms,
        latency_ms_p95=elapsed_ms,
        throughput_val=tps,
        throughput_unit="tokens/s",
        status="PASS",
        script_path=__file__,
        notes=(
            f"active_models={available_count} prompt_tok={prompt_tokens} "
            f"completion_tok={completion_tokens}"
        ),
        extra_metadata={
            "base_url": base_url,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    )
    print(f"Logged to 00_RESEARCH_LIBRARY -> experiment_id={entry['experiment_id']}")
    return 0


def main() -> int:
    load_local_env()

    parser = argparse.ArgumentParser(description="AMD Token Lab verification & latency benchmark.")
    parser.add_argument(
        "--model",
        default=os.getenv("AMD_TOKENLAB_MODEL", "Gemma-4-31B-IT").strip(),
        help="Target model ID on AMD Token Lab.",
    )
    args = parser.parse_args()

    api_key = os.getenv("AMD_TOKENLAB_API_KEY", "").strip()
    base_url = os.getenv("AMD_TOKENLAB_BASE_URL", "https://api.tokenlab.amd.com/v1").strip()
    target_model = args.model.strip()

    if not api_key or api_key == "your_amd_tokenlab_api_key_here":
        print(
            "ERROR: AMD_TOKENLAB_API_KEY is not set.\n"
            "Copy .env.example to .env and insert your personal AMD Token Lab API key.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"[1/2] Querying active models at {base_url} ...")
    t_list = time.perf_counter()
    try:
        models_resp = client.models.list()
        list_ms = (time.perf_counter() - t_list) * 1000.0
        model_ids = sorted(m.id for m in models_resp.data)
        print(f"Found {len(model_ids)} active model(s) in {list_ms:.1f} ms:")
        for model_id in model_ids:
            marker = " <-- target" if model_id == target_model else ""
            print(f"  - {model_id}{marker}")
    except Exception as exc:
        print(f"ERROR querying /v1/models: {exc}", file=sys.stderr)
        return 2

    return benchmark_model(client, target_model, base_url, len(model_ids))


if __name__ == "__main__":
    raise SystemExit(main())
