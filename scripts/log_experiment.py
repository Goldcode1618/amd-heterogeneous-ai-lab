#!/usr/bin/env python3
"""Centralized publication-grade experiment logger for AMD AI Developer Program.

Appends immutable JSON records to 00_RESEARCH_LIBRARY/MASTER_EXPERIMENT_LOG.jsonl,
maintains structured rows in 00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv, and exports
publication-ready LaTeX booktabs tables to 05_PUBLICATION_DRAFTS/tables/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_DIR = WORKSPACE_ROOT / "00_RESEARCH_LIBRARY"
JSONL_PATH = LIBRARY_DIR / "MASTER_EXPERIMENT_LOG.jsonl"
CSV_PATH = LIBRARY_DIR / "MASTER_BENCHMARK.csv"
LATEX_PATH = WORKSPACE_ROOT / "05_PUBLICATION_DRAFTS" / "tables" / "benchmark_summary.tex"

CSV_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "track",
    "hardware_tier",
    "device_name",
    "execution_provider",
    "model_or_workload",
    "precision",
    "batch_size",
    "latency_ms_p50",
    "latency_ms_p95",
    "throughput_val",
    "throughput_unit",
    "memory_mb",
    "status",
    "git_commit",
    "script_sha256",
    "notes",
]


def _get_git_commit() -> str:
    """Return short HEAD commit hash (with dirty suffix if working tree modified)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return "uninitialized"
        commit = proc.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(WORKSPACE_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            return f"{commit}-dirty"
        return commit
    except OSError:
        return "unknown"


def _compute_sha256(script_path: str | Path | None, fallback_payload: dict[str, Any]) -> str:
    """Compute SHA-256 digest of the executing script or deterministic JSON payload."""
    if script_path:
        path_obj = Path(script_path)
        if not path_obj.is_absolute():
            path_obj = WORKSPACE_ROOT / path_obj
        if path_obj.is_file():
            return hashlib.sha256(path_obj.read_bytes()).hexdigest()
    canonical = json.dumps(fallback_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _format_opt_float(value: float | None, decimals: int = 3) -> str:
    """Format optional floating-point metric for CSV storage."""
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def record_experiment(
    *,
    track: str,
    hardware_tier: str,
    device_name: str,
    execution_provider: str,
    model_or_workload: str,
    precision: str = "FP32",
    batch_size: int = 1,
    latency_ms_p50: float | None = None,
    latency_ms_p95: float | None = None,
    throughput_val: float | None = None,
    throughput_unit: str = "items/s",
    memory_mb: float | None = None,
    status: str = "PASS",
    script_path: str | Path | None = None,
    notes: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an experiment record to MASTER_EXPERIMENT_LOG.jsonl and MASTER_BENCHMARK.csv."""
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git_commit = _get_git_commit()

    rel_script = ""
    if script_path:
        p = Path(script_path)
        try:
            rel_script = str(p.resolve().relative_to(WORKSPACE_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            rel_script = str(p)

    core_payload: dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "experiment_id": experiment_id,
        "host_node": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "track": track,
        "hardware_tier": hardware_tier,
        "device_name": device_name,
        "execution_provider": execution_provider,
        "model_or_workload": model_or_workload,
        "precision": precision,
        "batch_size": int(batch_size),
        "latency_ms_p50": round(float(latency_ms_p50), 3) if latency_ms_p50 is not None else None,
        "latency_ms_p95": round(float(latency_ms_p95), 3) if latency_ms_p95 is not None else None,
        "throughput_val": round(float(throughput_val), 3) if throughput_val is not None else None,
        "throughput_unit": throughput_unit,
        "memory_mb": round(float(memory_mb), 2) if memory_mb is not None else None,
        "status": status,
        "git_commit": git_commit,
        "script_path": rel_script,
        "notes": notes,
    }
    if extra_metadata:
        core_payload["extra_metadata"] = extra_metadata

    script_sha256 = _compute_sha256(script_path, core_payload)
    core_payload["script_sha256"] = script_sha256

    with JSONL_PATH.open("a", encoding="utf-8", newline="\n") as jf:
        jf.write(json.dumps(core_payload, ensure_ascii=False) + "\n")

    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    csv_row = {
        "timestamp_utc": timestamp_utc,
        "experiment_id": experiment_id,
        "track": track,
        "hardware_tier": hardware_tier,
        "device_name": device_name,
        "execution_provider": execution_provider,
        "model_or_workload": model_or_workload,
        "precision": precision,
        "batch_size": str(int(batch_size)),
        "latency_ms_p50": _format_opt_float(latency_ms_p50, 3),
        "latency_ms_p95": _format_opt_float(latency_ms_p95, 3),
        "throughput_val": _format_opt_float(throughput_val, 3),
        "throughput_unit": throughput_unit,
        "memory_mb": _format_opt_float(memory_mb, 2),
        "status": status,
        "git_commit": git_commit,
        "script_sha256": script_sha256,
        "notes": notes,
    }

    with CSV_PATH.open("a", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    return core_payload


def _load_csv_rows() -> list[dict[str, str]]:
    """Read all recorded rows from MASTER_BENCHMARK.csv."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as cf:
        return list(csv.DictReader(cf))


def print_summary() -> int:
    """Print a concise summary table of all logged experiments."""
    rows = _load_csv_rows()
    if not rows:
        print("No experiments recorded yet in MASTER_BENCHMARK.csv.")
        return 0

    print(f"Total Logged Experiments: {len(rows)}")
    print(f"JSONL Audit Log : {JSONL_PATH}")
    print(f"CSV Benchmark   : {CSV_PATH}\n")
    header = (
        f"{'Timestamp (UTC)':<22} | {'Tier':<14} | {'Provider':<24} | "
        f"{'Workload':<20} | {'p50 (ms)':>10} | {'Throughput':>18} | {'Status':<6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ts = r.get("timestamp_utc", "")[:19]
        tier = r.get("hardware_tier", "")[:14]
        prov = r.get("execution_provider", "")[:24]
        work = r.get("model_or_workload", "")[:20]
        p50 = r.get("latency_ms_p50", "-") or "-"
        tput_v = r.get("throughput_val", "")
        tput_u = r.get("throughput_unit", "")
        tput = f"{tput_v} {tput_u}".strip()[:18] if tput_v else "-"
        status = r.get("status", "")[:6]
        print(
            f"{ts:<22} | {tier:<14} | {prov:<24} | "
            f"{work:<20} | {p50:>10} | {tput:>18} | {status:<6}"
        )
    return 0


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters for safe table compilation."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def export_latex() -> int:
    """Generate a publication-ready LaTeX booktabs table from MASTER_BENCHMARK.csv."""
    rows = _load_csv_rows()
    LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "% Auto-generated by scripts/log_experiment.py export-latex",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-Tier Empirical Performance Summary Across the AMD AI Ecosystem (GOLDENRATIO1618 \& Cloud).}",
        r"\label{tab:amd_ecosystem_benchmark}",
        r"\begin{tabular}{lllllrrr}",
        r"\toprule",
        r"\textbf{Tier} & \textbf{Device} & \textbf{Provider / Stack} & \textbf{Workload} & \textbf{Prec.} & \textbf{p50 (ms)} & \textbf{p95 (ms)} & \textbf{Throughput} \\",
        r"\midrule",
    ]

    if not rows:
        lines.append(r"\multicolumn{8}{c}{\textit{No experiment entries recorded yet.}} \\")
    else:
        for r in rows:
            tier = _escape_latex(r.get("hardware_tier", ""))
            dev = _escape_latex(r.get("device_name", ""))
            prov = _escape_latex(r.get("execution_provider", ""))
            work = _escape_latex(r.get("model_or_workload", ""))
            prec = _escape_latex(r.get("precision", ""))
            p50 = r.get("latency_ms_p50", "") or "--"
            p95 = r.get("latency_ms_p95", "") or "--"
            tv = r.get("throughput_val", "")
            tu = _escape_latex(r.get("throughput_unit", ""))
            tput = f"{tv} {tu}".strip() if tv else "--"
            lines.append(
                f"{tier} & {dev} & {prov} & {work} & {prec} & {p50} & {p95} & {tput} \\\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    LATEX_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Exported LaTeX table ({len(rows)} row(s)) to: {LATEX_PATH}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Centralized experiment logger for AMD AI Developer Program."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record an experiment run to JSONL and CSV.")
    rec.add_argument("--track", required=True, help="Research track (e.g. edge_npu, cloud_api, mi300x).")
    rec.add_argument("--hardware-tier", required=True, help="Hardware tier (e.g. Edge_NPU, Edge_iGPU, Cloud_API).")
    rec.add_argument("--device-name", required=True, help="Target hardware device name.")
    rec.add_argument("--execution-provider", required=True, help="Runtime provider or API backend.")
    rec.add_argument("--model-or-workload", required=True, help="Model name or benchmark workload.")
    rec.add_argument("--precision", default="FP32", help="Numerical precision (INT8, BF16, FP16, FP32).")
    rec.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    rec.add_argument("--latency-ms-p50", type=float, default=None, help="Median latency in ms.")
    rec.add_argument("--latency-ms-p95", type=float, default=None, help="95th percentile latency in ms.")
    rec.add_argument("--throughput-val", type=float, default=None, help="Throughput numeric value.")
    rec.add_argument("--throughput-unit", default="items/s", help="Throughput unit.")
    rec.add_argument("--memory-mb", type=float, default=None, help="Memory / VRAM footprint in MB.")
    rec.add_argument("--status", default="PASS", help="PASS or FAIL.")
    rec.add_argument("--script-path", default="", help="Source script path for SHA-256 provenance.")
    rec.add_argument("--notes", default="", help="Free-form experiment notes.")

    sub.add_parser("summary", help="Print summary table of all logged experiments.")
    sub.add_parser("export-latex", help="Export LaTeX booktabs table to 05_PUBLICATION_DRAFTS/tables/.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "record":
        entry = record_experiment(
            track=args.track,
            hardware_tier=args.hardware_tier,
            device_name=args.device_name,
            execution_provider=args.execution_provider,
            model_or_workload=args.model_or_workload,
            precision=args.precision,
            batch_size=args.batch_size,
            latency_ms_p50=args.latency_ms_p50,
            latency_ms_p95=args.latency_ms_p95,
            throughput_val=args.throughput_val,
            throughput_unit=args.throughput_unit,
            memory_mb=args.memory_mb,
            status=args.status,
            script_path=args.script_path or None,
            notes=args.notes,
        )
        print(f"Recorded experiment_id={entry['experiment_id']} sha256={entry['script_sha256'][:12]}")
        return 0
    if args.command == "summary":
        return print_summary()
    if args.command == "export-latex":
        return export_latex()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
