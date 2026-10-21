#!/usr/bin/env python3
"""twin_gate_build.py - builds src/tools/gates/twin_gate.exe, the AVX-512 twin gate.

The twin gate links BOTH codelet libraries, which no single gauntlet/build.py run
does (build.py builds for one ISA, VFFT_ISA, read once at import). So build.py is
loaded once per ISA, and each instance supplies what it supplies for its own
builds: the driver flags with its -I set, and its cached codelet library
(src/dag-fft-compiler/.obj/<isa>/libdagcodelets.a, compiled where stale):

  the gate         gcc <avx2 flags>                    -c twin_gate.c
  the avx2 side    gcc <avx2 flags>   -DVFFT_TWIN_SIDE -DTWIN_REG_NKERN=n2 -c twin_gate.c   (twin_side_avx2)
  the avx512 side  gcc <avx512 flags> -DVFFT_TWIN_SIDE -DTWIN_REG_NKERN=n5 -c twin_gate.c   (twin_side_avx512)
  link             the three objects + the avx2 library + the avx512 library

TWIN_REG_NKERN is the number of kernels the registry's FWD and BWD radix lists
hold at that side's flags (registry_kernels: the preprocessor's own view of
il_isa.h, so the registry counted is the one the side includes). The side's
kernel table must expand exactly that many: a kind the registry gains and the
table does not name stops the compile (twin_gate.c's _Static_assert).

The gate's own object carries no AVX-512 code: on a host without it, it says so
and exits 2 (run it under Intel SDE there). run_gates.py calls this for
twin_gate; by hand, from the repo root:

  python src/tools/gates/twin_gate_build.py
"""
import importlib.util
import os
import re
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


def registry_kernels(tc, cflags, env):
    """The kernels the IL registry's FWD and BWD radix lists hold at cflags'
    ISA: the X(R) entries of every VFFT_IL_<KIND>_{FWD,BWD}_RADICES(X) macro
    il_isa.h defines there (the _PAIR_ lists repeat them). None if the
    preprocessor fails."""
    r = subprocess.run([tc['cc']] + cflags + ['-E', '-dM', '-x', 'c', '-'],
                       input='#include "il_isa.h"\n', capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=env)
    if r.returncode != 0:
        print(f'[twin] the registry count failed:\n{r.stderr[:4000]}', file=sys.stderr)
        return None
    n = 0
    for line in r.stdout.splitlines():
        m = re.match(r'#define VFFT_IL_\w+_(?:FWD|BWD)_RADICES\(X\)(.*)', line)
        if m:
            n += len(re.findall(r'\bX\(', m.group(1)))
    return n


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
    env = b2.build_env(tc)
    n2, n5 = registry_kernels(tc, cf2, env), registry_kernels(tc, cf5, env)
    if n2 is None or n5 is None:
        return 1
    units = [(cf2, [], objdir / 'main.o'),
             (cf2, ['-DVFFT_TWIN_SIDE', f'-DTWIN_REG_NKERN={n2}'], objdir / 'side_avx2.o'),
             (cf5, ['-DVFFT_TWIN_SIDE', f'-DTWIN_REG_NKERN={n5}'], objdir / 'side_avx512.o')]
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
