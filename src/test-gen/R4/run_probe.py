"""R=4 large-ios probe runner.

Usage: python run_probe.py

Generates radix4 AVX2 header, compiles probe, runs it.
Works on Linux (gcc) and Windows (icx or gcc).
"""
import subprocess, sys, os, platform
from pathlib import Path

HERE = Path(__file__).parent.resolve()
CC = os.environ.get('CC', 'icx' if platform.system() == 'Windows' else 'gcc')

def run(cmd, **kw):
    print(f">>> {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)

# Generate (force UTF-8 for Windows cp1252 compatibility)
# The generator emits Unicode box-drawing chars in comments.
with open(HERE / 'radix4_avx2.h', 'w', encoding='utf-8') as f:
    subprocess.run([sys.executable, str(HERE / 'gen_radix4.py'), 'avx2'],
                   stdout=f, check=True,
                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})

# Compile
exe_ext = '.exe' if platform.system() == 'Windows' else ''
exe = HERE / f'r4_probe{exe_ext}'
flags = ['-O3', '-mavx2', '-mfma', '-march=native']
if CC == 'icx' and platform.system() == 'Windows':
    flags += ['-fuse-ld=lld']
run([CC, *flags, '-o', str(exe), str(HERE / 'r4_probe.c')])

# Run
run([str(exe)])