"""build_cil_ab.py — direct-gcc build for cil_ab.c (+ cil_ab_arms/*.c).

Mirrors build_vtune.py's recipe (which mirrors build.py): cil_ab.c #includes
src/core/vfft.c itself to see struct vfft_plan_s and swap il2p form pointers,
so build.py --vfft would link a second copy and collide. No MKL, no ITT.

Every candidate .c in cil_ab_arms/ is compiled into the exe; each must carry
a UNIQUE symbol (sed-rename) and a matching DECL + ARMS[] row in cil_ab.c.

Usage:  python build_cil_ab.py
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parents[1]                       # .../highSpeedFFT
DAG  = ROOT / 'src' / 'dag-fft-compiler'
DAG_CORE = ROOT / 'src' / 'core'
DAG_GEN  = DAG / 'generator' / 'generated'
ISA  = os.environ.get('VFFT_ISA', 'avx2')
LIB  = DAG / '.obj' / ISA / 'libdagcodelets.a'

CC   = os.environ.get('CC', r'C:\mingw152\mingw64\bin\gcc.exe')
SRC  = HERE / 'cil_ab.c'
ARMS = sorted((HERE / 'cil_ab_arms').glob('*.c'))
OUT  = HERE / 'cil_ab.exe'


def includes() -> list[str]:
    core = [DAG_CORE] + sorted(d for d in DAG_CORE.rglob('*') if d.is_dir())
    inc = [ROOT / 'include', DAG, DAG_GEN, DAG / 'jit'] + core
    return [f'-I{p}' for p in inc]


def main() -> int:
    if not LIB.is_file():
        print(f'[error] cached codelet lib missing: {LIB}', file=sys.stderr)
        print('        run  python build.py --src <anything> --compile  from '
              'build_tuned/ first (SERIAL builds only)', file=sys.stderr)
        return 2
    if not ARMS:
        print('[warn] cil_ab_arms/ is empty — only baseline lanes will link')

    flags = ['-O3', '-mavx2', '-mfma', '-march=native', '-fpermissive',
             '-D_CRT_SECURE_NO_WARNINGS',
             '-Wno-overflow', '-Wno-implicit-function-declaration',
             '-Wno-unused-function', '-Wno-unknown-argument',
             '-Wno-incompatible-pointer-types', '-Wno-deprecated-declarations']
    cmd = ([CC] + flags + includes()
           + [str(SRC)] + [str(a) for a in ARMS]
           + [str(LIB), '-static', '-o', str(OUT), '-lm'])

    print('=' * 78)
    print('EXACT BUILD COMMAND')
    print('=' * 78)
    print(' '.join(f'"{a}"' if ' ' in a else a for a in cmd))
    print('=' * 78, flush=True)

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(f'[compile] FAILED ({time.time()-t0:.1f}s)')
        print(r.stdout[-4000:])
        print(r.stderr[-12000:])
        return 1
    print(f'[compile] OK ({time.time()-t0:.1f}s) -> {OUT}')
    if r.stderr.strip():
        print('[compile] warnings (first 20 lines):')
        print('\n'.join(r.stderr.splitlines()[:20]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
