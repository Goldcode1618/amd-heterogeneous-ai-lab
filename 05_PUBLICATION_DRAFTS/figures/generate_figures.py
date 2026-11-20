#!/usr/bin/env python3
"""Publication-ready figure generator from 00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv.

Automatically switches to a logarithmic latency axis when cross-tier latencies
span multiple orders of magnitude (e.g. millisecond edge inference vs multi-second
cloud LLM API calls), and exports both vector SVG and 300-DPI PNG (if matplotlib
is available).
"""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = WORKSPACE_ROOT / "00_RESEARCH_LIBRARY" / "MASTER_BENCHMARK.csv"
FIG_DIR = Path(__file__).resolve().parent
SVG_OUT = FIG_DIR / "edge_vs_cloud_benchmark.svg"
PNG_OUT = FIG_DIR / "edge_vs_cloud_benchmark.png"

TIER_COLORS = {
    "Edge_NPU": "#ED1C24",
    "Edge_iGPU": "#0071C5",
    "Edge_CPU": "#5A6577",
    "Cloud_API": "#00897B",
    "Cloud_MI300X": "#6A1B9A",
}


def load_latest_by_provider() -> list[dict[str, str]]:
    """Load latest PASS record per (hardware_tier, execution_provider, model_or_workload)."""
    if not CSV_PATH.is_file():
        return []
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    with CSV_PATH.open("r", encoding="utf-8", newline="") as cf:
        for row in csv.DictReader(cf):
            if row.get("status") != "PASS":
                continue
            if not row.get("latency_ms_p50"):
                continue
            key = (
                row.get("hardware_tier", ""),
                row.get("execution_provider", ""),
                row.get("model_or_workload", ""),
            )
            latest[key] = row
    rows = list(latest.values())
    rows.sort(key=lambda r: float(r["latency_ms_p50"]))
    return rows


def render_svg(rows: list[dict[str, str]], out_path: Path) -> None:
    """Render a clean, IEEE-style horizontal bar chart of p50/p95 latencies in SVG."""
    width = 1040
    left_pad = 310
    right_pad = 290
    top_pad = 92
    row_h = 58
    height = max(290, top_pad + len(rows) * row_h + 70)
    plot_w = width - left_pad - right_pad

    p50s = [max(float(r["latency_ms_p50"]), 0.01) for r in rows]
    p95s = [max(float(r.get("latency_ms_p95") or r["latency_ms_p50"]), 0.01) for r in rows]
    min_val = min(p50s)
    max_val = max(p95s)
    use_log = (max_val / max(min_val, 0.01)) >= 20.0

    if use_log:
        log_min = math.floor(math.log10(min_val))
        log_max = math.ceil(math.log10(max_val))
        if log_min == log_max:
            log_max += 1
        span = float(log_max - log_min)

        def scale_x(v: float) -> float:
            lv = math.log10(max(v, 10 ** log_min))
            return ((lv - log_min) / span) * plot_w

        ticks = [10 ** e for e in range(log_min, log_max + 1)]
        scale_note = "Log10 Latency Scale (ms)"
    else:
        def scale_x(v: float) -> float:
            return (v / max_val) * plot_w if max_val > 0 else 0.0

        ticks = [max_val * f for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        scale_note = "Linear Latency Scale (ms)"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "  .title { font: 700 18px 'Segoe UI', Arial, sans-serif; fill: #111827; }",
        "  .subtitle { font: 400 12px 'Segoe UI', Arial, sans-serif; fill: #4B5563; }",
        "  .label { font: 600 12px 'Segoe UI', Arial, sans-serif; fill: #1F2937; }",
        "  .sublabel { font: 400 11px 'Segoe UI', Arial, sans-serif; fill: #6B7280; }",
        "  .val { font: 600 11px 'Consolas', monospace; fill: #111827; }",
        "  .grid { stroke: #E5E7EB; stroke-width: 1; stroke-dasharray: 4 3; }",
        "  .axis { stroke: #374151; stroke-width: 1.5; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#FFFFFF" />',
        '<text x="28" y="36" class="title">AMD Ecosystem Cross-Tier Latency Benchmark (p50 Median vs p95 Tail)</text>',
        f'<text x="28" y="56" class="subtitle">{html.escape(scale_note)} · Source: 00_RESEARCH_LIBRARY/MASTER_BENCHMARK.csv (Host: GOLDENRATIO1618)</text>',
    ]

    for tv in ticks:
        x = left_pad + scale_x(tv)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_pad - 12}" x2="{x:.1f}" y2="{height - 48}" class="grid" />'
        )
        lbl = f"{tv:g} ms" if tv >= 1 else f"{tv:.2f} ms"
        parts.append(
            f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" class="sublabel">{lbl}</text>'
        )

    parts.append(
        f'<line x1="{left_pad}" y1="{top_pad - 12}" x2="{left_pad}" y2="{height - 48}" class="axis" />'
    )

    for idx, r in enumerate(rows):
        y = top_pad + idx * row_h
        tier = r.get("hardware_tier", "")
        prov = r.get("execution_provider", "")
        work = r.get("model_or_workload", "")
        p50 = float(r["latency_ms_p50"])
        p95 = float(r.get("latency_ms_p95") or p50)
        tput_v = r.get("throughput_val", "")
        tput_u = r.get("throughput_unit", "")
        color = TIER_COLORS.get(tier, "#334155")

        w_p50 = max(4.0, scale_x(p50))
        w_p95 = max(4.0, scale_x(p95))

        parts.append(
            f'<text x="{left_pad - 12}" y="{y + 15}" text-anchor="end" class="label">'
            f'{html.escape(tier)} · {html.escape(prov)}</text>'
        )
        parts.append(
            f'<text x="{left_pad - 12}" y="{y + 31}" text-anchor="end" class="sublabel">'
            f'{html.escape(work)} ({html.escape(r.get("precision", ""))})</text>'
        )
        parts.append(
            f'<rect x="{left_pad}" y="{y + 4}" width="{w_p95:.1f}" height="24" '
            f'rx="3" fill="{color}" fill-opacity="0.28" />'
        )
        parts.append(
            f'<rect x="{left_pad}" y="{y + 8}" width="{w_p50:.1f}" height="16" '
            f'rx="2" fill="{color}" />'
        )
        tput_str = f" | {tput_v} {tput_u}" if tput_v else ""
        parts.append(
            f'<text x="{left_pad + w_p95 + 8:.1f}" y="{y + 20}" class="val">'
            f'p50={p50:.2f}ms / p95={p95:.2f}ms{html.escape(tput_str)}</text>'
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def try_render_png(rows: list[dict[str, str]], out_path: Path) -> bool:
    """Export 300-DPI PNG with auto-log axis and direct annotations if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [f"{r['hardware_tier']}\n{r['execution_provider']}\n({r['model_or_workload']})" for r in rows]
    p50s = [float(r["latency_ms_p50"]) for r in rows]
    p95s = [float(r.get("latency_ms_p95") or r["latency_ms_p50"]) for r in rows]
    colors = [TIER_COLORS.get(r["hardware_tier"], "#334155") for r in rows]
    use_log = (max(p95s) / max(min(p50s), 0.01)) >= 20.0

    fig, ax = plt.subplots(figsize=(11, max(3.8, 0.95 * len(rows) + 1.4)), dpi=300)
    y_pos = list(range(len(rows)))
    ax.barh(y_pos, p95s, height=0.62, color=colors, alpha=0.28, label="p95 tail")
    ax.barh(y_pos, p50s, height=0.38, color=colors, alpha=0.95, label="p50 median")
    if use_log:
        ax.set_xscale("log")
        ax.set_xlim(left=max(0.5, min(p50s) * 0.4), right=max(p95s) * 120.0)
    else:
        ax.set_xlim(right=max(p95s) * 1.45)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms, log scale)" if use_log else "Latency (ms)")
    ax.set_title("AMD Ecosystem Cross-Tier Latency Benchmark (GOLDENRATIO1618 & Token Lab)")
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.35)

    for idx, (r, p50, p95) in enumerate(zip(rows, p50s, p95s)):
        tput_v = r.get("throughput_val", "")
        tput_u = r.get("throughput_unit", "")
        tput = f" | {tput_v} {tput_u}" if tput_v else ""
        ax.text(
            p95 * 1.15 if use_log else p95 + max(p95s) * 0.02,
            idx,
            f"p50={p50:.2f}ms (p95={p95:.2f}ms){tput}",
            va="center",
            fontsize=8,
            fontfamily="monospace",
        )

    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def main() -> int:
    rows = load_latest_by_provider()
    if not rows:
        print("WARNING: No PASS benchmark rows with latency found in MASTER_BENCHMARK.csv.")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render_svg(rows, SVG_OUT)
    print(f"Generated publication SVG ({len(rows)} series): {SVG_OUT}")

    if try_render_png(rows, PNG_OUT):
        print(f"Generated 300-DPI PNG: {PNG_OUT}")
    else:
        print("matplotlib not present in active Python; vector SVG generated cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
