#!/usr/bin/env bash
# half_store_bench.sh -- builds and runs gauntlet/half_store_bench.c: the AVX-512 forward column leaf
# with its plane stores as one 512-bit store (as shipped), two 256-bit halves and four 128-bit
# quarters, at the 2D real r2c column pass's odd pitch (the question and the arms: the .c's header).
# On an AVX-512 machine, from bash (Git Bash or MSYS2 on Windows), from any directory:
#
#   ./gauntlet/half_store_bench.sh            the run: correctness, then 72 cells x 25 rounds
#   ./gauntlet/half_store_bench.sh --quick    one cell, 2 rounds: proves the build and the run
#
# Further arguments go to the bench (--leaves all, --rounds N, ...). Results land in
# gauntlet/results/half_store_<cpu folder>_<date>/ (_quick for --quick): table.txt, rows.csv
# (every sample) and run.log; the cpu folder is the library's own (gauntlet/wisdom_folder.py).
# Nothing is written into src/wisdom. The compiler is build.py's (CC, else the usual MinGW-w64 and
# MSYS2 folders), else the gcc on PATH.
#
# SDE=<path to sde.exe> runs the bench under Intel SDE on a machine without AVX-512: the
# correctness check is exact there, the timings mean nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
suffix=""
for a in "$@"; do [ "$a" = "--quick" ] && suffix="_quick"; done

# 1. THE BUILD. gauntlet/build.py loaded as a module at VFFT_ISA=avx512 (one source, many
# translation units, as src/tools/gates/twin_gate_build.py builds the twin gate):
#   the bench   gcc <build.py's flags, AVX-512 off: no AVX-512 before its CPU check> -c half_store_bench.c
#   the arms    gcc <the codelet library's flags> -DHALF_STORE_ARM=<512|256|128> -DHALF_STORE_KERNEL="<kernel>"
#                   -D<symbol>=half_store_st<width>_<symbol> -c half_store_bench.c    (13 kernels x 3)
#   link        + src/core/vfft.c (build.py's cached object) + the avx512 codelet library
# then each arm's machine code is read back (objdump): st512 stores its plane rows with unmasked
# 512-bit stores, st256 and st128 with none, through 256-bit / 128-bit extracts to memory.
"$py" - <<'PY'
import concurrent.futures, importlib.util, os, re, shutil, subprocess, sys
from pathlib import Path
ROOT = Path.cwd()
SRC = ROOT / 'gauntlet' / 'half_store_bench.c'
ZIL = ROOT / 'src' / 'dag-fft-compiler' / 'codelets' / 'zil' / 'avx512'
os.environ['VFFT_ISA'] = 'avx512'                       # build.py reads it once, at import
spec = importlib.util.spec_from_file_location('vfft_build_avx512', ROOT / 'gauntlet' / 'build.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
tc = b.detect_toolchain()
if tc['is_msvc_style']:
    sys.exit('half_store_bench builds with a gcc-style compiler only')
if not os.environ.get('CC') and not Path(tc['cc']).is_file() and shutil.which('gcc'):
    tc['cc'] = shutil.which('gcc')                      # none of build.py's folders has one: the gcc on PATH
ver = subprocess.run([tc['cc'], '--version'], capture_output=True, text=True, errors='replace').stdout
print('[half_store] compiler: %s (%s)' % (tc['cc'], ver.splitlines()[0] if ver else '?'), flush=True)
exe = '.exe' if tc['is_windows'] else ''
vfft_c = ROOT / 'src' / 'core' / 'vfft.c'

# the probe gauntlet/wisdom_folder.py asks for this CPU's folder, when this checkout has none
probe = ROOT / 'gauntlet' / ('recal_1d_probe' + exe)
if not probe.is_file():
    print('[half_store] building gauntlet/recal_1d_probe (wisdom_folder.py asks it for the cpu folder)', flush=True)
    cf, bs, la = b.build_cmd(tc, ROOT / 'gauntlet' / 'recal_1d_probe.c', probe, extra_srcs=[vfft_c], split=True)
    if not b.build_gcc(tc, probe, cf, bs, la):
        sys.exit('[half_store] recal_1d_probe did not build')

out = ROOT / 'gauntlet' / ('half_store_bench' + exe)
cflags, _, link = b.build_cmd(tc, SRC, out, extra_srcs=[vfft_c], split=True)   # builds or finds the avx512 codelet library
if not any(a.endswith('.a') for a in link):
    sys.exit('[half_store] the avx512 codelet library did not build')
main_flags = [f for f in cflags if f not in ('-mavx512f', '-mavx512dq')] + ['-mno-avx512f']
arm_flags = ['-O3', '-mavx2', '-mfma', '-march=native', *b.DAG_ISA_FLAGS, '-fpermissive', '-w']   # build.py's codelet library flags
kernels = re.findall(r'X\(\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*(\w+)\s*\)', SRC.read_text())
if not kernels:
    sys.exit('[half_store] no kernel table in %s' % SRC)
objdir = ROOT / 'gauntlet' / '.obj' / 'half_store'
objdir.mkdir(parents=True, exist_ok=True)
units = [(main_flags, [], objdir / 'bench.o', None, None)]
for R, f, sym in kernels:
    for w in (512, 256, 128):
        units.append((arm_flags, ['-DHALF_STORE_ARM=%d' % w, '-DHALF_STORE_KERNEL="%s"' % (ZIL / f).as_posix(),
                                  '-D%s=half_store_st%d_%s' % (sym, w, sym)],
                      objdir / ('st%d_%s.o' % (w, sym)), int(R), w))
env = b.build_env(tc)

def compile_unit(u):
    fl, extra, obj = u[0], u[1], u[2]
    return obj, subprocess.run([tc['cc']] + fl + extra + ['-c', str(SRC), '-o', str(obj)], capture_output=True,
                               text=True, encoding='utf-8', errors='replace', env=env)

print('[half_store] compiling the bench and %d arms' % (len(units) - 1), flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
    done = list(ex.map(compile_unit, units))
bad = [(o, r) for o, r in done if r.returncode]
if bad:
    sys.exit('[half_store] compile FAILED: %s\n%s' % (bad[0][0].name, bad[0][1].stderr[:4000]))
if not b.build_gcc(tc, out, cflags, [str(vfft_c)], [str(u[2]) for u in units] + link):
    sys.exit('[half_store] link FAILED')

objdump = Path(tc['cc']).with_name('objdump' + exe)
objdump = str(objdump) if objdump.is_file() else shutil.which('objdump')
if not objdump:
    print('[half_store] no objdump beside the compiler or on PATH: the arms\' machine code is not read back')
else:
    ST512 = re.compile(r'\svmovupd\s+%zmm\d+,[-0-9a-fx]*\((%\w+)[^)]*\)\s*$')      # unmasked, to memory (spills: %rsp)
    X256 = re.compile(r'\svextract[fi](?:64x4|32x8)\s+\$0x1,%zmm\d+,[-0-9a-fx]*\(')
    X128 = re.compile(r'\svextract[fi](?:128|64x2|32x4)\s+\$0x[123],%[yz]mm\d+,[-0-9a-fx]*\(')
    wrong = []
    for fl, extra, obj, R, w in units[1:]:
        txt = subprocess.run([objdump, '-d', '--no-show-raw-insn', str(obj)], capture_output=True, text=True,
                             errors='replace').stdout.splitlines()
        z = sum(1 for l in txt if ST512.search(l) and ST512.search(l).group(1) != '%rsp')
        x256 = sum(1 for l in txt if X256.search(l))
        x128 = sum(1 for l in txt if X128.search(l))
        ok = (z >= R) if w == 512 else (z == 0 and x256 >= R) if w == 256 else (z == 0 and x128 >= 3 * R)
        if not ok:
            wrong.append('%s: %d unmasked 512-bit stores, %d 256-bit and %d 128-bit extracts to memory'
                         % (obj.name, z, x256, x128))
    if wrong:
        sys.exit('[half_store] an arm is not the store width it is named for:\n  ' + '\n  '.join(wrong))
    print('[half_store] the arms\' machine code: st512 stores with 512-bit stores, st256 and st128 with none '
          '(%d kernels)' % len(kernels))
print('[half_store] built %s' % out.relative_to(ROOT).as_posix(), flush=True)
PY

# 2. THE RESULTS DIRECTORY: gauntlet/results/half_store_<cpu folder>_<date>[_quick][-n]
folder=$("$py" -c "import os, sys; sys.path.insert(0, 'gauntlet'); from wisdom_folder import shipped_folder; print(os.path.basename(shipped_folder()[0]))" | tr -d '\r')
dir="gauntlet/results/half_store_${folder}_$(date +%Y-%m-%d)${suffix}"
base="$dir"; n=2
while [ -e "$dir" ]; do dir="${base}-$n"; n=$((n + 1)); done
mkdir -p "$dir"

# 3. THE RUN
bin=./gauntlet/half_store_bench
[ -f "$bin.exe" ] && bin="$bin.exe"
if [ -n "${SDE:-}" ]; then
    "$SDE" -spr -- "$bin" --out "$dir" "$@" 2>&1 | tee "$dir/run.log"
else
    "$bin" --out "$dir" "$@" 2>&1 | tee "$dir/run.log"
fi
echo "results: $dir (table.txt, rows.csv, run.log)"
