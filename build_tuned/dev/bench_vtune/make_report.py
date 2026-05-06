#!/usr/bin/env python3
"""
make_report.py — compose a per-run VTune markdown report.

Reads from a VTune result dir:
  - bench_output.txt         (bench stdout — per-cell timings)
  - summary.csv              (vtune -report summary -format=csv)
  - topdown.csv              (vtune -report top-down -format=csv -group-by task) — uarch-exploration only
  - hotspots.csv             (vtune -report hotspots -format=csv -group-by task)

Writes:
  <result_dir>/report.md     — composed markdown summary

Usage:
  python make_report.py <result_dir> [--collect-mode <mode>]
"""

from __future__ import annotations
import argparse
import csv
import datetime as dt
import os
import platform
import re
import sys
from pathlib import Path


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-16")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = read_text(path)
    if not raw.strip():
        return []
    # VTune CSVs use comma or tab depending on locale; sniff
    sample = raw.splitlines()[0]
    delim = "\t" if "\t" in sample else ","
    return list(csv.DictReader(raw.splitlines(), delimiter=delim))


def fmt_pct(s):
    """VTune emits percentages like '12.3%' or '12.3' or 'N/A'."""
    if not s:
        return "—"
    s = str(s).strip().rstrip("%")
    try:
        return f"{float(s):.1f}%"
    except ValueError:
        return s or "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--collect-mode", default="hotspots")
    args = ap.parse_args()

    rd = args.result_dir
    if not rd.is_dir():
        print(f"error: {rd} not a directory", file=sys.stderr)
        sys.exit(1)

    bench_text = read_text(rd / "bench_output.txt")
    summary    = read_text(rd / "summary.csv")
    topdown    = read_csv_rows(rd / "topdown.csv")
    hotspots   = read_csv_rows(rd / "hotspots.csv")

    # ── Header ────────────────────────────────────────────────────────
    out = []
    out.append(f"# VTune profile report — {args.collect_mode}")
    out.append("")
    out.append(f"**Generated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    out.append(f"**Host:** {platform.node()}  ")
    out.append(f"**CPU:** {platform.processor()}  ")
    out.append(f"**Result dir:** `{rd.name}`  ")
    out.append(f"**Collection:** `vtune -collect {args.collect_mode}`")
    out.append("")

    # ── Bench output (per-cell timings) ───────────────────────────────
    out.append("## Per-cell wall time + GFLOP/s")
    out.append("")
    if bench_text.strip():
        # Pluck just the table portion of bench_output.txt
        lines = bench_text.splitlines()
        in_table = False
        out.append("```")
        for ln in lines:
            if "category" in ln and "vfft_ns" in ln:
                in_table = True
            if in_table:
                out.append(ln)
        out.append("```")
    else:
        out.append("_bench_output.txt missing or empty_")
    out.append("")

    # ── Top-down per task (microarchitecture breakdown) ───────────────
    if topdown:
        out.append("## Microarchitectural breakdown per cell")
        out.append("")
        out.append(
            "Top-down classification per ITT task. Look for elevated "
            "**Backend Bound — Memory Bound** on CLOSE cells (= the cost "
            "model picked a factorization that hits memory hard) and "
            "high **Retiring** on DECISIVE cells (= codelets fully fed)."
        )
        out.append("")

        # VTune CSV column names vary by version. Probe the first row.
        if topdown:
            cols = list(topdown[0].keys())
            # Find best-effort matches
            task_col   = next((c for c in cols if "Task" in c or "Function" in c), cols[0])
            retire_col = next((c for c in cols if "Retiring" in c), None)
            be_col     = next((c for c in cols if "Backend" in c and "Bound" in c), None)
            fe_col     = next((c for c in cols if "Frontend" in c and "Bound" in c), None)
            bs_col     = next((c for c in cols if "Bad" in c and "Spec" in c), None)

            out.append(f"| Task | Retiring | Backend Bound | Frontend Bound | Bad Spec |")
            out.append(f"|------|---------:|--------------:|---------------:|---------:|")
            for row in topdown:
                task = row.get(task_col, "?")
                if not task or task == "?":
                    continue
                # Filter to our ITT-named tasks
                if not (task.startswith("VFFT_") or task.startswith("MKL_")):
                    continue
                out.append(
                    f"| `{task}` "
                    f"| {fmt_pct(row.get(retire_col, ''))} "
                    f"| {fmt_pct(row.get(be_col, ''))} "
                    f"| {fmt_pct(row.get(fe_col, ''))} "
                    f"| {fmt_pct(row.get(bs_col, ''))} |"
                )
        out.append("")

    # ── Top hotspot functions ─────────────────────────────────────────
    if hotspots:
        out.append("## Top hotspot functions")
        out.append("")
        out.append(
            "Function-level CPU time. Validates which codelet variants "
            "(`radix*_t1_dit_*`, `radix*_t1s_*`) actually run for each cell."
        )
        out.append("")

        cols = list(hotspots[0].keys())
        fn_col   = next((c for c in cols if "Function" in c), cols[0])
        time_col = next((c for c in cols if "CPU Time" in c or "Time" in c), None)
        task_col = next((c for c in cols if "Task" in c), None)

        # Group rows by task; show top 5 functions per task
        from collections import defaultdict
        by_task = defaultdict(list)
        for row in hotspots:
            t = row.get(task_col, "global") if task_col else "global"
            by_task[t].append(row)

        for task, rows in by_task.items():
            if not (task.startswith("VFFT_") or task.startswith("MKL_") or task == "global"):
                continue
            rows_sorted = sorted(
                rows,
                key=lambda r: -float(str(r.get(time_col, "0")).replace("s", "").strip() or 0)
            )
            out.append(f"### `{task}`")
            out.append("")
            out.append("| Function | Time |")
            out.append("|----------|-----:|")
            for r in rows_sorted[:5]:
                fn   = r.get(fn_col, "?")
                tval = r.get(time_col, "?")
                out.append(f"| `{fn}` | {tval} |")
            out.append("")

    # ── VTune summary (raw paste) ─────────────────────────────────────
    if summary.strip():
        out.append("## Raw `vtune -report summary` output")
        out.append("")
        out.append("<details><summary>Click to expand</summary>")
        out.append("")
        out.append("```")
        out.append(summary.strip())
        out.append("```")
        out.append("</details>")
        out.append("")

    # ── Cross-reference to existing memory entries ───────────────────
    out.append("## See also")
    out.append("")
    out.append(
        "- [VTune profile index in MEMORY.md](../../../../../.claude/projects/"
        "c--Users-Tugbars-Desktop-highSpeedFFT/memory/MEMORY.md) — historic "
        "per-codelet VTune profiles for direct comparison"
    )
    out.append("- [bench_vtune README](README.md) — what to look for per category")
    out.append("- [v1.0 perf doc](../../../../docs/performance/v1_0_results.md) — "
               "where these cells came from")

    report_path = rd / "report.md"
    report_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
