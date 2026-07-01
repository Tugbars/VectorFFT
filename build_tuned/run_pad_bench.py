"""
run_pad_bench.py — paper-grade padding GATE: our_pad & our_tight vs MKL, ISOLATED per cell.

Same discipline as run_bench.py (which it mirrors): ONE FRESH PROCESS per (cell, engine),
pinned core 2 / High priority, engine-order ALTERNATES per cell (flip=i%2), inter-engine
cooldown (--cool-ms) + inter-cell cooldown so no engine is measured on a warmer core and no
cross-cell carryover survives (the bench itself says trust only isolated numbers). See the
[[canonical-mkl-bench]] methodology (bench_1d_vs_mkl.c).

Scope: only the PAD-WINNER cells in spike_wisdom_padded.txt (exec_me == Kp). For each, benches
BOTH engines isolated: pad (Kp-plan, me=Kp, baked/JIT) and tight (K-plan, me=K, SSE2 tail). Parses
the bench's 'mkl/ours=X.XXx' line, aggregates mkl/pad, mkl/tight, and the uplift = tight_ns/pad_ns
= (mkl/pad)/(mkl/tight). The GATE passes if our_pad is competitive-with / beats MKL where padding
wins (and mkl/pad > mkl/tight — padding closed the gap).

usage:
  python build_tuned/run_pad_bench.py                    # all pad-winners, build + run
  python build_tuned/run_pad_bench.py --skip-build
  python build_tuned/run_pad_bench.py --cool-ms 250
"""
from __future__ import annotations
import argparse, os, re, statistics, subprocess, sys, time
from pathlib import Path

import calibrate as cal   # verify_power(), HIGH_PRIORITY discipline

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DAG  = ROOT / 'src' / 'dag-fft-compiler'
BENCH_SRC = HERE / 'benches' / 'bench_pad_vs_mkl.c'
BENCH_EXE = HERE / 'benches' / ('bench_pad_vs_mkl.exe' if os.name == 'nt' else 'bench_pad_vs_mkl')
PAD_WIS   = DAG / 'generator' / 'generated' / 'spike_wisdom_padded.txt'

MKL_BIN   = r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin'
MINGW_BIN = r'C:\mingw152\mingw64\bin'
VW = 4
_RATIO = re.compile(r'mkl/ours=([0-9.]+)x')
_OURS  = re.compile(r'ours\s+([0-9.]+)')
_MKL   = re.compile(r'mkl\s+([0-9.]+)')


def roundup_vw(k): return (k + (VW - 1)) & ~(VW - 1)


def pad_winners(path: Path):
    """(N,K) cells whose padded-wisdom exec_me == Kp (pad won). Smallest N*K first."""
    cells = []
    if not path.exists():
        return cells
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines():
        t = ln.strip()
        if not t or t[0] in '#@':
            continue
        p = t.split()
        if len(p) < 3:
            continue
        try:
            N, K, nf = int(p[0]), int(p[1]), int(p[2])
            exec_me = int(p[-1])   # trailing v6 field
        except ValueError:
            continue
        if exec_me == roundup_vw(K):
            cells.append((N, K))
    return sorted(set(cells), key=lambda nk: (nk[0] * nk[1], nk[1], nk[0]))


def build():
    print('[build] bench_pad_vs_mkl (--mkl --jit) ...', flush=True)
    r = subprocess.run([sys.executable, str(HERE / 'build.py'),
                        '--src', str(BENCH_SRC), '--mkl', '--jit', '--compile'])
    if r.returncode != 0 or not BENCH_EXE.exists():
        print('[build] FAILED', file=sys.stderr); sys.exit(1)


def run_one(N, K, engine, flip, cool_ms, core):
    """isolated bench of ONE engine vs MKL; returns (mkl_over_ours, ours_ns, mkl_ns) or (0,0,0)."""
    env = os.environ.copy()
    env['PATH'] = MKL_BIN + os.pathsep + MINGW_BIN + os.pathsep + env.get('PATH', '')
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    # bench args: N K engine flip cool_ms core
    p = subprocess.run([str(BENCH_EXE), str(N), str(K), engine, str(flip), str(cool_ms), str(core)],
                       env=env, creationflags=flags, capture_output=True, text=True)
    out = p.stdout or ''
    sys.stdout.write(out)
    mr, our, mk = _RATIO.search(out), _OURS.search(out), _MKL.search(out)
    return (float(mr.group(1)) if mr else 0.0,
            float(our.group(1)) if our else 0.0,
            float(mk.group(1)) if mk else 0.0)


def cooldown_for(nk): return 3.0 if nk < 100_000 else 6.0 if nk < 1_000_000 else 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--core', type=int, default=2)
    ap.add_argument('--cool-ms', type=int, default=250, help='inter-engine idle (order-bias)')
    ap.add_argument('--wisdom', default=str(PAD_WIS))
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cells = pad_winners(Path(args.wisdom))
    print('=' * 72)
    print(f' run_pad_bench — GATE  core={args.core}  cool-ms={args.cool_ms}')
    print(f' wisdom={args.wisdom}')
    print(f' pad-winner cells: {len(cells)}  {cells}')
    print(' order: N*K ascending; engine order ALTERNATES per cell (flip=i%2)')
    print('=' * 72)
    if not cells:
        print('[warn] no pad-winners in the padded wisdom — run calibrate_pad.py first.'); return
    if args.dry_run:
        return

    cal.verify_power()
    if not args.skip_build:
        build()

    rows = []
    for i, (N, K) in enumerate(cells):
        nk = N * K; flip = i % 2
        print(f'\n----- [{i+1}/{len(cells)}] N={N} K={K} (N*K={nk}) flip={flip} -----', flush=True)
        rp, pad_ns, mkl_p = run_one(N, K, 'pad',   flip, args.cool_ms, args.core)
        time.sleep(cooldown_for(nk))
        rt, tit_ns, mkl_t = run_one(N, K, 'tight', flip, args.cool_ms, args.core)
        uplift = (tit_ns / pad_ns) if (pad_ns > 0 and tit_ns > 0) else 0.0
        rows.append((N, K, rp, rt, uplift))
        if i < len(cells) - 1:
            time.sleep(cooldown_for(nk))

    print('\n' + '=' * 72)
    print(' PADDING GATE SUMMARY   (mkl/x > 1 = we beat MKL; uplift = tight_ns/pad_ns > 1 = pad faster)')
    print('=' * 72)
    print('  N      K    mkl/pad  mkl/tight  uplift(pad vs our tail)')
    padbeat = 0
    for N, K, rp, rt, up in rows:
        if rp >= 1.0:
            padbeat += 1
        print(f'  {N:<6} {K:<4} {rp:6.2f}x   {rt:6.2f}x     {up:5.2f}x')
    mp = [r for _, _, r, _, _ in rows if r > 0]
    ups = [u for *_, u in rows if u > 0]
    if mp:
        print(f'\n  our_pad >= MKL: {padbeat}/{len(rows)}   median mkl/pad={statistics.median(mp):.2f}x'
              f'   median uplift={statistics.median(ups):.2f}x')


if __name__ == '__main__':
    main()
