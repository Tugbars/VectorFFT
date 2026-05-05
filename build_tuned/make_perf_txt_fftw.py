#!/usr/bin/env python3
"""
Convert vfft_perf_tuned_1d_fftw.csv → vfft_perf_tuned_1d_fftw.txt
in the same human-readable format as vfft_perf_tuned_1d.txt (MKL).

Usage:  python make_perf_txt_fftw.py [csv_path] [txt_path]

Defaults to build_tuned/vfft_perf_tuned_1d_fftw.{csv,txt}.
Re-run any time after the bench appends rows to the CSV.
"""
import csv, sys, statistics
from pathlib import Path

CATEGORIES = [
    ("small",      "Small (N<=128)"),
    ("pow2",       "Power-of-2"),
    ("composite",  "Composite"),
    ("odd_comp",   "Odd composite"),
    ("mixed_deep", "Mixed deep"),
    ("prime_pow",  "Prime powers"),
    ("genfft",     "Genfft (R=11/13)"),
    ("rader",      "Rader primes"),
    ("bluestein",  "Bluestein primes"),
]

SEP = "=" * 92
DASH_CAT = "-" * 60
DASH_CELL = "-" * 92

def fmt_ratio(r):
    return f"{r:.2f}x"

def main():
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else "build_tuned/vfft_perf_tuned_1d_fftw.csv")
    txt_path = Path(sys.argv[2] if len(sys.argv) > 2
                    else "build_tuned/vfft_perf_tuned_1d_fftw.txt")

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["N"]        = int(r["N"])
            r["K"]        = int(r["K"])
            r["vfft_ns"]  = float(r["vfft_ns"])
            r["fftw_ns"]  = float(r["fftw_ns"])
            r["vfft_GF"]  = float(r["vfft_gflops"])
            r["fftw_GF"]  = float(r["fftw_gflops"])
            r["ratio"]    = float(r["ratio_vs_fftw"])
            rows.append(r)

    out = []
    out.append(SEP)
    out.append("VectorFFT new core - bench vs FFTW3")
    out.append(f"Wisdom: vfft_wisdom_tuned.txt    Cells: {len(rows)}")
    out.append(SEP)
    out.append("")

    # Category summary
    out.append(f"{'Category':<22} {'Cells':>6} {'Min':>7} {'Median':>8} {'Max':>7} {'Mean':>7}")
    out.append(DASH_CAT)
    total_ratios = []
    for code, name in CATEGORIES:
        rs = [r["ratio"] for r in rows if r["category"] == code and r["ratio"] > 0]
        if not rs:
            continue
        rs_sorted = sorted(rs)
        mn = rs_sorted[0]
        md = statistics.median(rs_sorted)
        mx = rs_sorted[-1]
        mean = sum(rs_sorted) / len(rs_sorted)
        out.append(f"{name:<22} {len(rs):>6} {fmt_ratio(mn):>7} "
                   f"{fmt_ratio(md):>8} {fmt_ratio(mx):>7} {fmt_ratio(mean):>7}")
        total_ratios.extend(rs_sorted)

    if total_ratios:
        ts = sorted(total_ratios)
        out.append(DASH_CAT)
        out.append(f"{'OVERALL':<22} {len(ts):>6} {fmt_ratio(ts[0]):>7} "
                   f"{fmt_ratio(statistics.median(ts)):>8} "
                   f"{fmt_ratio(ts[-1]):>7} {fmt_ratio(sum(ts)/len(ts)):>7}")
        out.append("")
        wins = sum(1 for x in ts if x > 1.0)
        pct = 100.0 * wins / len(ts)
        out.append(f"Wins (vfft faster than FFTW3): {wins}/{len(ts)} ({pct:.1f}%)")
        out.append("")

    # Per-category sections
    for code, name in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == code]
        if not cat_rows:
            continue
        cat_rows.sort(key=lambda r: (r["N"], r["K"]))
        out.append("")
        out.append(SEP)
        out.append(f"  {name.upper()}")
        out.append(SEP)
        out.append(f"{'N':>7} {'K':>4} {'factors':<30} "
                   f"{'vfft_ns':>12} {'fftw_ns':>12}  "
                   f"{'vfft_GF':>7}  {'fftw_GF':>7}  {'ratio':>6}")
        out.append(DASH_CELL)
        for r in cat_rows:
            out.append(f"{r['N']:>7d} {r['K']:>4d} {r['factors']:<30s} "
                       f"{r['vfft_ns']:>12.0f} {r['fftw_ns']:>12.0f}  "
                       f"{r['vfft_GF']:>7.2f}  {r['fftw_GF']:>7.2f}  "
                       f"{fmt_ratio(r['ratio']):>6}")

    txt_path.write_text("\n".join(out) + "\n")
    print(f"wrote {txt_path}  ({len(rows)} cells)")

if __name__ == "__main__":
    main()
