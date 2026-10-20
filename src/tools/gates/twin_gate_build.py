#!/usr/bin/env python3
"""twin_gate_build.py - builds src/tools/gates/twin_gate.exe, the AVX-512 twin gate.

The twin gate links BOTH codelet libraries, which no single gauntlet/build.py run
does (build.py builds for one ISA, VFFT_ISA, read once at import). So build.py is
loaded once per ISA, and each instance supplies what it supplies for its own
builds: the driver flags with its -I set, and its cached codelet library
(src/dag-fft-compiler/.obj/<isa>/libdagcodelets.a, compiled where stale):

  the gate         gcc <avx2 flags>                    -c twin_gate.c
  the avx2 side    gcc <avx2 flags>   -DVFFT_TWIN_SIDE -c twin_gate.c   (twin_side_avx2)
  the avx512 side  gcc <avx512 flags> -DVFFT_TWIN_SIDE -c twin_gate.c   (twin_side_avx512)
  link             the three objects + the avx2 library + the avx512 library

The gate's own object carries no AVX-512 code: on a host without it, it says so
and exits 2 (run it under Intel SDE there). run_gates.py calls this for
twin_gate; by hand, from the repo root:

  python src/tools/gates/twin_gate_build.py
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = HERE / 'twin_gate.c'


def load_build(isa):
    """gauntlet/build.py as a module whose VFFT_ISA is isa."""
    prev = os.environ.get('VFFT_ISA')
    os.environ['VFFT_ISA'] = isa
    try:
        spec = importlib.util.spec_from_file_location('vfft_build_' + isa,
                                                      ROOT / 'gauntlet' / 'build.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if prev is None:
            os.environ.pop('VFFT_ISA', None)
        else:
            os.environ['VFFT_ISA'] = prev
    return mod


def main():
    t0 = time.time()
    b2, b5 = load_build('avx2'), load_build('avx512')
    tc = b2.detect_toolchain()
    if tc['is_msvc_style']:
        print('[twin] the twin gate builds with a gcc-style compiler only', file=sys.stderr)
        return 1
    out = HERE / ('twin_gate.exe' if tc['is_windows'] else 'twin_gate')
    # each call builds (or finds cached) its ISA's codelet library
    cf2, _, la2 = b2.build_cmd(tc, SRC, out, split=True)
    cf5, _, la5 = b5.build_cmd(tc, SRC, out, split=True)
    lib2 = [a for a in la2 if a.endswith('.a')]
    lib5 = [a for a in la5 if a.endswith('.a')]
    if not lib2 or not lib5:
        print('[twin] a codelet library did not build', file=sys.stderr)
        return 1
    # the avx512 side must BE avx512 (build_isa.h reads -mavx512f -mavx512dq) on any host
    for f in ('-mavx512f', '-mavx512dq'):
        if f not in cf5:
            cf5 = cf5 + [f]
    objdir = b2.HERE / '.obj' / 'twin_gate'
    objdir.mkdir(parents=True, exist_ok=True)
    units = [(cf2, [], objdir / 'main.o'),
             (cf2, ['-DVFFT_TWIN_SIDE'], objdir / 'side_avx2.o'),
             (cf5, ['-DVFFT_TWIN_SIDE'], objdir / 'side_avx512.o')]
    env = b2.build_env(tc)
    procs = [subprocess.Popen([tc['cc']] + cf + extra + ['-c', str(SRC), '-o', str(obj)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              encoding='utf-8', errors='replace', env=env)
             for (cf, extra, obj) in units]
    ok = True
    for (cf, extra, obj), p in zip(units, procs):
        _, err = p.communicate()
        if p.returncode != 0:
            print(f'[twin] compile FAILED: {obj.name}\n{err[:4000]}', file=sys.stderr)
            ok = False
    if not ok:
        return 1
    link = ([tc['cc']] + [str(o) for (_, _, o) in units] + ['-o', str(out)]
            + lib2 + lib5 + ['-lm'])
    r = subprocess.run(link, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', env=env)
    if r.returncode != 0:
        print(f'[twin] link FAILED:\n{r.stderr[:4000]}', file=sys.stderr)
        return 1
    print(f'[twin] {out} OK ({time.time() - t0:.1f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
