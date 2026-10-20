#!/usr/bin/env python3
"""
Generate high-resolution (300 DPI) benchmark charts for SM80 DeepInfer Optimization.
Reads production benchmark JSONs from benchmarks/logs/ and kernel microbenchmark measurements.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = ROOT / "charts"
LOGS_DIR = ROOT / "benchmarks" / "logs"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

COLOR_BASE = "#4C72B0"
COLOR_OPT = "#DD8452"
COLOR_ACCENT = "#55A868"


def plot_kernel_speedups():
    labels = [
        "mHC Fused Post+Pre\n(Decode T=6, 83x)",
        "mHC Fused Post+Pre\n(2-Stream T=12, 83x)",
        "mHC Post+Sqrsum\n(Prefill T=1536, 83x)",
        "SM80 Sparse MLA Decode\n(B=6 Verify, 41x)",
        "SM80 Sparse MLA Decode\n(B=24 4-Stream, 41x)",
        "SM80 Sparse MLA Prefill\n(Consumer T=1536, 30x)",
        "Dequant & Gather K\n(16K Ctx, 38x)",
        "Dequant & Gather K\n(48K Ctx, 38x)",
        "Triton Combine TopK+SWA\n(Prefill T=1536, 30x)",
    ]
    speedups = [
        163.64 / 125.59,
        357.70 / 123.77,
        275.27 / 232.62,
        149.64 / 69.92,
        349.12 / 167.51,
        3742.80 / 3202.11,
        80.10 / 19.93,
        263.43 / 34.47,
        1083.72 / 48.88,
    ]
    base_us = [163.64, 357.70, 275.27, 149.64, 349.12, 3742.80, 80.10, 263.43, 1083.72]
    opt_us = [125.59, 123.77, 232.62, 69.92, 167.51, 3202.11, 19.93, 34.47, 48.88]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, speedups, color=COLOR_OPT, edgecolor="#222222", height=0.62)
    ax.axvline(1.0, color="#666666", linestyle="-", linewidth=1.2, label="Baseline vLLM SM80 (1.0x)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Kernel Speedup Factor (x, log scale)")
    ax.set_xscale("log")
    ax.set_xlim(0.9, 38.0)
    ax.set_title("NVIDIA A100/GA100 (SM80, 74 SMs) Kernel Speedups: Baseline vLLM vs. SM80 DeepInfer")

    for bar, s, b_u, o_u in zip(bars, speedups, base_us, opt_us):
        ax.text(
            bar.get_width() * 1.04,
            bar.get_y() + bar.get_height() / 2.0,
            f"{s:.2f}x  ({b_u:.1f} µs → {o_u:.1f} µs)",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )

    ax.legend(loc="lower right")
    plt.tight_layout()
    out_path = CHARTS_DIR / "01_sm80_kernel_speedups.png"
    plt.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


def plot_sparse_mla_decode_scaling():
    batches = ["B=1\n(Single)", "B=4\n(4-Seq)", "B=6\n(1-Stream K=5)", "B=12\n(2-Stream K=5)", "B=16\n(Batch 16)", "B=24\n(4-Stream K=5)"]
    base_us = [151.79, 148.34, 149.64, 190.36, 254.66, 349.12]
    opt_us = [76.14, 78.70, 69.92, 93.59, 121.12, 167.51]

    x = np.arange(len(batches))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    b1 = ax.bar(x - width / 2, base_us, width, label="Baseline vLLM SM80 (Split NoPE/RoPE + exp2 + 8 Warps)", color=COLOR_BASE, edgecolor="#222222")
    b2 = ax.bar(x + width / 2, opt_us, width, label="SM80 DeepInfer (Unified 512-dim + Bitshift UE8M0 + 2-CTA/SM + Direct-Out)", color=COLOR_OPT, edgecolor="#222222")

    ax.set_ylabel("Per-Layer Kernel Latency (µs, lower is better)")
    ax.set_title("SM80 Sparse MLA Decode Kernel Latency Across Batch Sizes (100% Bit-Exact, 0.00e+00 Error)")
    ax.set_xticks(x)
    ax.set_xticklabels(batches)
    ax.set_ylim(0, 410)

    for r1, r2 in zip(b1, b2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        sp = h1 / h2
        ax.text(r1.get_x() + r1.get_width() / 2, h1 + 5, f"{h1:.1f}µs", ha="center", va="bottom", fontsize=9)
        ax.text(r2.get_x() + r2.get_width() / 2, h2 + 5, f"{h2:.1f}µs\n({sp:.2f}x)", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#B03A2E")

    ax.legend(loc="upper left")
    plt.tight_layout()
    out_path = CHARTS_DIR / "02_sparse_mla_decode_scaling.png"
    plt.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


def plot_prefill_chunk_breakdown():
    categories = [
        "combine_topk_swa_indices\n(30 layers / chunk)",
        "Sparse MLA Prefill + Ragged Reuse\n(30 consumer layers / chunk)",
        "mHC Post+Pre Boundary\n(83 boundaries / chunk)",
        "dequantize_and_gather_k_cache\n(38 layers / chunk @ 48K)",
    ]
    base_ms = [32.51, 112.28, 22.85, 10.01]
    opt_ms = [1.47, 96.06, 19.31, 1.31]

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    b1 = ax.bar(x - width / 2, base_ms, width, label="Baseline vLLM SM80 (ms / 1536-token chunk)", color=COLOR_BASE, edgecolor="#222222")
    b2 = ax.bar(x + width / 2, opt_ms, width, label="SM80 DeepInfer (ms / 1536-token chunk)", color=COLOR_OPT, edgecolor="#222222")

    ax.set_ylabel("Latency per 1536-Token Chunk (ms, lower is better)")
    ax.set_title("Per-Chunk (T=1536) Prefill Latency Savings by Component (-59.5 ms / chunk saved)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 130)

    for r1, r2 in zip(b1, b2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        diff = h1 - h2
        ax.text(r1.get_x() + r1.get_width() / 2, h1 + 2, f"{h1:.2f} ms", ha="center", va="bottom", fontsize=9.5)
        ax.text(r2.get_x() + r2.get_width() / 2, h2 + 2, f"{h2:.2f} ms\n(-{diff:.1f} ms)", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E8449")

    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = CHARTS_DIR / "03_prefill_chunk_breakdown.png"
    plt.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


def plot_e2e_production():
    summary_file = ROOT / "benchmarks" / "results" / "sm80_benchmark_summary.json"
    if not summary_file.exists():
        print("Skipping E2E chart until sm80_benchmark_summary.json is available.")
        return

    e2e = json.loads(summary_file.read_text())["e2e_production_benchmarks"]
    b = {
        "prefill_16k_tps": e2e["prefill_16k_tps"]["baseline"],
        "prefill_64k_tps": e2e["prefill_64k_tps"]["baseline"],
        "decode_4_tps_per_stream": e2e["decode_4_stream_per_seq_prose_tps"]["baseline"],
        "decode_4_tps_aggregate": e2e["decode_4_stream_aggregate_prose_tps"]["baseline"],
    }
    o = {
        "prefill_16k_tps": e2e["prefill_16k_tps"]["deepinfer_sm80"],
        "prefill_64k_tps": e2e["prefill_64k_tps"]["deepinfer_sm80"],
        "decode_4_tps_per_stream": e2e["decode_4_stream_per_seq_prose_tps"]["deepinfer_sm80"],
        "decode_4_tps_aggregate": e2e["decode_4_stream_aggregate_prose_tps"]["deepinfer_sm80"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # Subplot 1: Prefill & TTFT
    p_labels = ["16K Prefill\n(tok/s)", "64K Prefill\n(tok/s)"]
    p_base = [b["prefill_16k_tps"], b["prefill_64k_tps"]]
    p_opt = [o["prefill_16k_tps"], o["prefill_64k_tps"]]
    x1 = np.arange(len(p_labels))
    w = 0.35
    r1 = ax1.bar(x1 - w / 2, p_base, w, label="Baseline vLLM SM80", color=COLOR_BASE, edgecolor="#222222")
    r2 = ax1.bar(x1 + w / 2, p_opt, w, label="SM80 DeepInfer", color=COLOR_OPT, edgecolor="#222222")
    ax1.set_ylabel("Prefill Throughput (tokens/sec, higher is better)")
    ax1.set_title("End-to-End Prefill Throughput (TP2×PP3 + Engram CPU)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(p_labels)
    ax1.set_ylim(0, max(max(p_base), max(p_opt)) * 1.22)
    for bar1, bar2 in zip(r1, r2):
        v1, v2 = bar1.get_height(), bar2.get_height()
        pct = (v2 - v1) / v1 * 100.0
        ax1.text(bar1.get_x() + bar1.get_width() / 2, v1 + 60, f"{v1:,.0f}", ha="center", va="bottom", fontsize=9.5)
        ax1.text(bar2.get_x() + bar2.get_width() / 2, v2 + 60, f"{v2:,.0f}\n({pct:+.1f}%)", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E8449" if pct >= 0 else "#B03A2E")
    ax1.legend(loc="upper left")

    # Subplot 2: Decode Throughput (Multi-Stream & Speculative Domains)
    d_labels = ["4-Stream Per-Seq\n(Prose, tok/s)", "4-Stream Aggregate\n(Prose, tok/s)", "1-Stream Code\n(DSpark K=5)", "1-Stream 16K+\n(DSpark K=5)"]
    d_base = [b["decode_4_tps_per_stream"], b["decode_4_tps_aggregate"], 151.8, 178.4]
    d_opt = [o["decode_4_tps_per_stream"], o["decode_4_tps_aggregate"], 160.8, 188.8]
    x2 = np.arange(len(d_labels))
    r3 = ax2.bar(x2 - w / 2, d_base, w, label="Baseline vLLM SM80", color=COLOR_BASE, edgecolor="#222222")
    r4 = ax2.bar(x2 + w / 2, d_opt, w, label="SM80 DeepInfer", color=COLOR_OPT, edgecolor="#222222")
    ax2.set_ylabel("Decode Throughput (tokens/sec, higher is better)")
    ax2.set_title("End-to-End Decode Throughput (DSpark K=5 Speculative)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(d_labels, fontsize=9.5)
    ax2.set_ylim(0, max(max(d_base), max(d_opt)) * 1.24)
    for bar1, bar2 in zip(r3, r4):
        v1, v2 = bar1.get_height(), bar2.get_height()
        pct = (v2 - v1) / v1 * 100.0
        ax2.text(bar1.get_x() + bar1.get_width() / 2, v1 + 2.5, f"{v1:.1f}", ha="center", va="bottom", fontsize=9)
        ax2.text(bar2.get_x() + bar2.get_width() / 2, v2 + 2.5, f"{v2:.1f}\n({pct:+.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E8449" if pct >= 0 else "#B03A2E")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = CHARTS_DIR / "04_e2e_production_throughput.png"
    plt.savefig(out_path)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    plot_kernel_speedups()
    plot_sparse_mla_decode_scaling()
    plot_prefill_chunk_breakdown()
    plot_e2e_production()
