#!/usr/bin/env bash
# half_store_bench.sh -- builds and runs gauntlet/half_store_bench.c: the AVX-512 forward column leaf
# with its plane stores as one 512-bit store (as shipped), two 256-bit halves and four 128-bit
# quarters, at the 2D real r2c column pass's odd pitch (the question and the arms: the .c's header).
# On an AVX-512 machine, from bash (Git Bash or MSYS2 on Windows), from any directory:
#
#   ./gauntlet/half_store_bench.sh            the run: correctness, then 99 cells x 25 rounds
#   ./gauntlet/half_store_bench.sh --quick    one cell, 2 rounds: proves the build and the run
#
# Further arguments go to the bench (--leaves all, --rounds N, ...). Results land in
# gauntlet/results/half_store_<cpu>_<date>/ (_quick for --quick; <cpu> is the CPU's own name):
# table.md, rows.csv (every sample) and run.log. Nothing is written into src/wisdom. The compiler
# is build.py's (CC, else the usual MinGW-w64 and MSYS2 folders), else the gcc on PATH.
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
#   the copies  each kernel's shipped AVX-512 source copied into gauntlet/.obj/half_store/src/ with
#               its plane stores spelled HALF_STORE_PLANE; proved, kernel by kernel, against the AVX2
#               half twin: the rewritten stores are, one for one by destination, the stores the twin
#               splits, and the 512-bit stores left alone (into S[]) the ones it keeps whole
#   the bench   gcc <build.py's flags, AVX-512 off: no AVX-512 before its CPU check> -c half_store_bench.c
#   the arms    gcc <the codelet library's flags> -DHALF_STORE_ARM=<512|256|128> -DHALF_STORE_KERNEL="<copy>"
#                   -D<symbol>=half_store_st<width>_<symbol> -c half_store_bench.c    (32 kernels x 3)
#   link        + src/core/vfft.c (build.py's cached object) + the avx512 codelet library
# then each arm's machine code is read back (objdump) for the width of its plane stores.
"$py" - <<'PY'
import concurrent.futures, importlib.util, os, re, shutil, subprocess, sys
from pathlib import Path
ROOT = Path.cwd()
SRC = ROOT / 'gauntlet' / 'half_store_bench.c'
ZIL = ROOT / 'src' / 'dag-fft-compiler' / 'codelets' / 'zil'
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
objdir = ROOT / 'gauntlet' / '.obj' / 'half_store'
(objdir / 'src').mkdir(parents=True, exist_ok=True)

# THE COPIES AND THEIR PROOF. In the AVX-512 kernel the wide loop runs from its 4-column header to
# the masked 3-column pass; every 512-bit store must sit there and go either into the plane
# (&zout[E], rewritten) or into S[] (left alone). In the AVX2 half twin the wide loop runs from its
# 2-column header to the VEX-128 tail; every store there must be a split plane store (two 128-bit
# halves of &zout[E]) or a whole 256-bit store into S[]. Then the plane destinations E must be the
# same multiset, and the S[] slots too (an AVX-512 slot is twice the AVX2 one: twice the doubles).
P5 = re.compile(r'^\s*_mm512_storeu_pd\(&zout\[([^\]]+)\], (\w+)\);\s*$')
S5 = re.compile(r'^\s*_mm512_storeu_pd\(&S\[(\d+)\], (\w+)\);\s*$')
H2 = re.compile(r'^\s*_mm_storeu_pd\(&zout\[([^\]]+)\], _mm256_castpd256_pd128\((\w+)\)\); '
                r'_mm_storeu_pd\(&zout\[([^\]]+)\] \+ 2, _mm256_extractf128_pd\((\w+), 1\)\);\s*$')
S2 = re.compile(r'^\s*_mm256_storeu_pd\(&S\[(\d+)\], (\w+)\);\s*$')

def wide_loop(lines, head, tail):
    a = next((i for i, l in enumerate(lines) if re.match(head, l)), None)
    z = None if a is None else next((i for i in range(a + 1, len(lines)) if re.match(tail, lines[i])), None)
    return None if z is None else (a, z)

def copy_and_prove(f5, sym, f2):
    """the copy's text, its plane-store count and its S[]-store count; ValueError on any difference"""
    t5 = (ZIL / 'avx512' / f5).read_text(encoding='utf-8')
    t2 = (ZIL / 'avx2' / f2).read_text(encoding='utf-8')
    l5, l2 = t5.split('\n'), t2.split('\n')
    if not re.search(r'^void %s\(' % re.escape(sym), t5, re.M):
        raise ValueError('%s does not define %s' % (f5, sym))
    w5 = wide_loop(l5, r'^\s*for \(; k \+ 4 <= count; k \+= 4\) \{', r'^\s*if \(count - k == 3\)')
    w2 = wide_loop(l2, r'^\s*for \(; k \+ 2 <= count; k \+= 2\) \{', r'^\s*(for \(; k < count; \+\+k\)|if \(k < count\))')
    if not w5 or not w2:
        raise ValueError('no wide loop found in %s' % (f5 if not w5 else f2))
    plane5, scr5, at = [], [], []
    for i, l in enumerate(l5):
        if '_mm512_storeu_pd(' not in l:
            continue
        m, s = P5.match(l), S5.match(l)
        if not w5[0] < i < w5[1]:
            raise ValueError('%s:%d: a 512-bit store outside the wide loop' % (f5, i + 1))
        if m:
            plane5.append(m.group(1))
            at.append(i)
        elif s:
            scr5.append(int(s.group(1)))
        else:
            raise ValueError('%s:%d: a 512-bit store neither into the plane nor into S[]: %s' % (f5, i + 1, l.strip()))
    split2, whole2 = [], []
    for i, l in enumerate(l2):
        m = H2.match(l)
        if m:
            if not w2[0] < i < w2[1]:
                raise ValueError('%s:%d: a split plane store outside the wide loop' % (f2, i + 1))
            if m.group(1) != m.group(3) or m.group(2) != m.group(4):
                raise ValueError('%s:%d: the two halves are not one vector\'s' % (f2, i + 1))
            split2.append(m.group(1))
        elif w2[0] < i < w2[1]:
            s = S2.match(l)
            if s:
                whole2.append(2 * int(s.group(1)))
            elif re.search(r'_mm(256)?_(mask)?stor', l):
                raise ValueError('%s:%d: a wide-loop store neither split into the plane nor whole into S[]: %s'
                                 % (f2, i + 1, l.strip()))
    if sorted(plane5) != sorted(split2):
        raise ValueError('its %d plane stores are not, one for one, the %d stores the AVX2 twin %s splits'
                         % (len(plane5), len(split2), f2))
    if sorted(scr5) != sorted(whole2):
        raise ValueError('its %d S[] stores are not, one for one, the %d the AVX2 twin %s keeps whole'
                         % (len(scr5), len(whole2), f2))
    out = list(l5)
    for i in at:
        out[i] = out[i].replace('_mm512_storeu_pd(', 'HALF_STORE_PLANE(', 1)
    text = '\n'.join(out)
    if [i for i, (x, y) in enumerate(zip(l5, out)) if x != y] != at or text.count('HALF_STORE_PLANE(') != len(at) \
       or text.count('_mm512_storeu_pd(') != len(scr5):
        raise ValueError('the copy differs from %s in more than its plane stores' % f5)
    return text, len(plane5), len(scr5)

table = re.sub(r'\\\r?\n', ' ', SRC.read_text(encoding='utf-8'))  # the macro's lines joined
kernels = re.findall(r'X\(\s*(\d+)\s*,\s*"(\w+)"\s*,\s*"([^"]+)"\s*,\s*(\w+)\s*,\s*"([^"]+)"\s*\)', table)
if not kernels or len(kernels) != len(re.findall(r'\bX\(\s*\d', table)):
    sys.exit('[half_store] the kernel table in %s does not read as X(radix, "name", "file", symbol, "twin")' % SRC)
plane, scratch, copy = {}, {}, {}
for R, name, f5, sym, f2 in kernels:
    try:
        text, plane[sym], scratch[sym] = copy_and_prove(f5, sym, f2)
    except (ValueError, OSError) as e:
        sys.exit('[half_store] %s (%s): %s' % (name, f5, e))
    copy[sym] = objdir / 'src' / Path(f5).name
    if not copy[sym].is_file() or copy[sym].read_text(encoding='utf-8') != text:
        copy[sym].write_text(text, encoding='utf-8', newline='\n')
print('[half_store] %d kernels copied: their 512-bit plane stores are, one for one, the stores the AVX2 half '
      'twin splits, and their S[] stores the ones it keeps whole (%d plane stores, %d kernels with S[])'
      % (len(kernels), sum(plane.values()), sum(1 for s in scratch.values() if s)), flush=True)

out = ROOT / 'gauntlet' / ('half_store_bench' + exe)
cflags, _, link = b.build_cmd(tc, SRC, out, extra_srcs=[vfft_c], split=True)   # builds or finds the avx512 codelet library
if not any(a.endswith('.a') for a in link):
    sys.exit('[half_store] the avx512 codelet library did not build')
main_flags = [f for f in cflags if f not in ('-mavx512f', '-mavx512dq')] + ['-mno-avx512f']
arm_flags = ['-O3', '-mavx2', '-mfma', '-march=native', *b.DAG_ISA_FLAGS, '-fpermissive', '-w']   # build.py's codelet library flags
units = [(main_flags, [], objdir / 'bench.o', None, None)]
for R, name, f5, sym, f2 in kernels:
    for w in (512, 256, 128):
        units.append((arm_flags, ['-DHALF_STORE_ARM=%d' % w, '-DHALF_STORE_KERNEL="%s"' % copy[sym].as_posix(),
                                  '-D%s=half_store_st%d_%s' % (sym, w, sym)],
                      objdir / ('st%d_%s.o' % (w, sym)), sym, w))
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

# THE STORE WIDTHS. Every unmasked store of a vector register to memory is counted by width,
# whatever instruction makes it (vmovupd/vmovups/vmovapd/vmovaps/vmovdqu*/vmovdqa* of an xmm, ymm
# or zmm register; vextractf128/vextracti128, vextract[fi]64x2/32x4 to memory are 128-bit,
# vextract[fi]64x4/32x8 256-bit); stores to the stack are left out (S[], spills): the base %rsp,
# and %rbp when the function realigns its stack (and $..,%rsp), which makes %rbp its frame pointer;
# anywhere else %rbp is one more register a plane address may sit in.
# st512 must keep its plane stores whole (unmasked 512-bit stores >= the kernel's plane stores);
# st256 and st128 must keep no unmasked 512-bit store outside the stack, and add, over the st512
# arm's own (its tail's), at least 2 (st256) or 4 (st128) stores of their width per plane store:
# a plane store split only partly (a 256-bit half beside two 128-bit quarters) fails here, as the
# bitwise check cannot see it.
objdump = Path(tc['cc']).with_name('objdump' + exe)
objdump = str(objdump) if objdump.is_file() else shutil.which('objdump')
if not objdump:
    print('[half_store] no objdump beside the compiler or on PATH: the arms\' machine code is not read back')
else:
    MEM = r'[-0-9a-fx]*\((%\w+)?(?:,%\w+,\d)?\)(\{%k\d\})?\s*$'
    VMOV = re.compile(r':\s+vmov(?:up[sd]|ap[sd]|dq[ua](?:8|16|32|64)?)\s+%([xyz])mm\d+,' + MEM)
    VEXT = re.compile(r':\s+vextract[fi](64x4|32x8|64x2|32x4|128)\s+\$0x[0-9a-f]+,%[yz]mm\d+,' + MEM)
    REALIGN = re.compile(r':\s+and\s+\$0x[0-9a-f]+,%rsp\s*$')
    WIDTH = {'z': 512, 'y': 256, 'x': 128, '64x4': 256, '32x8': 256, '64x2': 128, '32x4': 128, '128': 128}

    def widths(obj):
        """unmasked stores to memory other than the stack, by width"""
        txt = subprocess.run([objdump, '-d', '--no-show-raw-insn', str(obj)], capture_output=True, text=True,
                             errors='replace').stdout.splitlines()
        stack = {'%rsp', '%rbp'} if any(map(REALIGN.search, txt)) else {'%rsp'}
        n = {512: 0, 256: 0, 128: 0}
        for l in txt:
            m = VMOV.search(l) or VEXT.search(l)
            if m and not m.group(3) and m.group(2) not in stack:
                n[WIDTH[m.group(1)]] += 1
        return n

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        got = dict(zip([u[2] for u in units[1:]], ex.map(widths, [u[2] for u in units[1:]])))
    wrong, report = [], []
    for R, name, f5, sym, f2 in kernels:
        P = plane[sym]
        n = {w: got[objdir / ('st%d_%s.o' % (w, sym))] for w in (512, 256, 128)}
        report.append('%-7s plane stores %3d | st512: %4d x512 | st256: %4d x512 %4d x256 (st512 %d) | '
                      'st128: %4d x512 %4d x128 (st512 %d)' % (name, P, n[512][512], n[256][512], n[256][256],
                                                              n[512][256], n[128][512], n[128][128], n[512][128]))
        if n[512][512] < P:
            wrong.append('%s st512: %d unmasked 512-bit stores outside the stack, %d plane stores' % (name, n[512][512], P))
        for w in (256, 128):
            if n[w][512] or n[w][w] - n[512][w] < (512 // w) * P:
                wrong.append('%s st%d: %d unmasked 512-bit stores outside the stack (0 expected); %d %d-bit stores, '
                             'st512 %d, so %d split stores for %d plane stores (at least %d)'
                             % (name, w, n[w][512], n[w][w], w, n[512][w], n[w][w] - n[512][w], P, (512 // w) * P))
    (objdir / 'store_widths.log').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if wrong:
        sys.exit('[half_store] an arm is not the store width it is named for (all counts: %s):\n  '
                 % (objdir / 'store_widths.log').relative_to(ROOT).as_posix() + '\n  '.join(wrong))
    print('[half_store] the arms\' machine code: st512 keeps its plane stores 512-bit, st256 and st128 split them '
          '(%d kernels; counts in %s)' % (len(kernels), (objdir / 'store_widths.log').relative_to(ROOT).as_posix()))
print('[half_store] built %s' % out.relative_to(ROOT).as_posix(), flush=True)
PY

# 2. THE RESULTS DIRECTORY: gauntlet/results/half_store_<cpu>_<date>[_quick][-n], <cpu> the CPU's
# name as the OS reports it (Windows: the registry's ProcessorNameString; Linux: /proc/cpuinfo)
cpu=$("$py" - <<'PY' | tr -d '\r'
import os, re
name = ''
if os.name == 'nt':
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
        name = winreg.QueryValueEx(k, 'ProcessorNameString')[0]
    except OSError:
        pass
if not name:
    try:
        name = next((l.split(':', 1)[1] for l in open('/proc/cpuinfo', errors='replace') if l.startswith('model name')), '')
    except OSError:
        pass
name = re.sub(r'\((?:R|TM|tm)\)|\bCPU\b|\bProcessor\b|@.*$|\s(?:w/|with)\s.*$', ' ', name)
print(re.sub(r'[^A-Za-z0-9.-]+', '_', name).strip('_') or 'cpu')
PY
)
dir="gauntlet/results/half_store_${cpu}_$(date +%Y-%m-%d)${suffix}"
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
echo "results: $dir (table.md, rows.csv, run.log)"
