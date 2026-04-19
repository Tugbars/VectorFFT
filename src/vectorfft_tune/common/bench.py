"""
bench.py — radix-generic bench orchestrator.

Pipeline (one radix):
  1. generate: for each ISA that any candidate targets, run the radix's
     generator with that ISA. Write `fft_radix<R>_<isa>.h` to staging/.
  2. compile: compile the harness with staging/ on the include path, one
     binary per bench run. The harness includes all generator headers and
     links to the runtime candidate registration table.
  3. run: for each enumerated candidate and each (me, ios, direction) in
     its sweep, invoke the harness binary once. Append the resulting JSON
     line to measurements.jsonl.

Portability:
  - Uses pathlib throughout; no forward slashes in exec args.
  - Compiler + flags routed via common.compiler.
  - Subprocess encoding forced to utf-8 so Windows Python doesn't default
    to cp1252 (burned by this before with R=4 Phase 2).
  - AVX-512 host skips are done at runtime by the harness, not by the
    driver — the driver runs the harness and lets it print a skip JSON.

Resume / incremental re-run:
  The `.jsonl` is read before running. For each candidate × (me, ios, dir),
  if an entry already exists in the file, it's skipped. Delete the file to
  force a fresh run.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Resolve sibling modules.
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
import compiler as _cc  # noqa: E402
import protocols as _protos  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def _utf8_env():
    env = dict(os.environ)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    return env


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True,
         capture: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess portably (utf-8 env, string cmd, stringified Path args)."""
    cmd_str = [str(x) for x in cmd]
    r = subprocess.run(
        cmd_str,
        cwd=str(cwd) if cwd else None,
        env=_utf8_env(),
        capture_output=capture,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if check and r.returncode != 0:
        msg = f'command failed (exit {r.returncode}): {" ".join(cmd_str)}'
        if r.stdout: msg += f'\nstdout:\n{r.stdout}'
        if r.stderr: msg += f'\nstderr:\n{r.stderr}'
        raise RuntimeError(msg)
    return r


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass (and other machinery using sys.modules
    # at class-body time) can find the module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════
# Phase: generate codelet headers
# ═══════════════════════════════════════════════════════════════

def phase_generate(candidates_mod, staging: Path):
    staging.mkdir(parents=True, exist_ok=True)
    isas_needed = sorted({c.isa for c in candidates_mod.enumerate_all()})
    R = candidates_mod.RADIX
    for isa in isas_needed:
        out_path = staging / f'fft_radix{R}_{isa}.h'
        r = _run([sys.executable, candidates_mod.GEN_SCRIPT, '--isa', isa])
        out_path.write_text(r.stdout, encoding='utf-8')
        print(f'  [gen] {out_path.name}: {out_path.stat().st_size} bytes')


# ═══════════════════════════════════════════════════════════════
# Phase: emit candidate registration + compile harness
# ═══════════════════════════════════════════════════════════════

_CANDS_PER_ISA_TEMPLATE = '''\
/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix{R}_{isa}.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {{
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
}} candidate_t;

const candidate_t CANDIDATES_{ISA}[] = {{
{rows}
}};
const size_t N_CANDIDATES_{ISA} = sizeof(CANDIDATES_{ISA}) / sizeof(CANDIDATES_{ISA}[0]);
'''

_CANDS_MASTER_TEMPLATE = '''\
/* Auto-generated candidate aggregator. DO NOT EDIT. */
#include <stddef.h>

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {{
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
}} candidate_t;

{externs}

/* Public view seen by the harness: concatenates every per-ISA array.
 * We build it at startup via a constructor-like initializer below, but
 * since C doesn't allow initializing an array from other arrays at file
 * scope we expose an indexer function instead. */
{lengths_const}

static candidate_t _flat[{total_len}];
static size_t _flat_n = 0;
static int _flat_init = 0;

static void _init_flat(void) {{
  if (_flat_init) return;
  _flat_init = 1;
  size_t j = 0;
{copies}
  _flat_n = j;
}}

const candidate_t *candidate_at(size_t i) {{
  _init_flat();
  if (i >= _flat_n) return NULL;
  return &_flat[i];
}}

size_t candidate_count(void) {{
  _init_flat();
  return _flat_n;
}}
'''


def emit_candidate_table(candidates_mod, staging: Path, build: Path) -> list[Path]:
    """Emit one candidates_<isa>.c per ISA + one master aggregator. Return all
    .c paths so the compile step can pass them to the compiler."""
    R = candidates_mod.RADIX
    cands = candidates_mod.enumerate_all()
    isas = sorted({c.isa for c in cands})

    emitted: list[Path] = []
    per_isa_counts: dict[str, int] = {}

    for isa in isas:
        # Dedup: handicap candidates (log1_tight) share symbols with the base
        # variant; we only need the C symbol declared once per (variant, isa).
        seen: set[tuple[str, str]] = set()
        rows = []
        for c in cands:
            if c.isa != isa:
                continue
            key = (c.variant, c.isa)
            if key in seen:
                continue
            seen.add(key)
            proto = candidates_mod.protocol(c.variant)
            fwd_sym = candidates_mod.function_name(c.variant, c.isa, 'fwd')
            bwd_sym = candidates_mod.function_name(c.variant, c.isa, 'bwd')
            req512 = 1 if c.requires_avx512 else 0
            rows.append(f'  {{"{c.variant}", "{c.isa}", "{proto}", '
                        f'{fwd_sym}, {bwd_sym}, {req512}}},')
        code = _CANDS_PER_ISA_TEMPLATE.format(
            R=R, isa=isa, ISA=isa.upper(),
            rows='\n'.join(rows))
        path = build / f'vfft_harness_candidates_{isa}.c'
        path.write_text(code, encoding='utf-8')
        emitted.append(path)
        per_isa_counts[isa] = len(rows)
        print(f'  [cands] {path.name}: {len(rows)} symbols')

    # Master aggregator: declares externs for all per-ISA arrays, exposes
    # candidate_at()/candidate_count() accessors. The harness uses these
    # (not a single flat CANDIDATES[] array) since the totals are only
    # known at runtime after summing per-ISA lengths.
    externs = '\n'.join(
        f'extern const candidate_t CANDIDATES_{isa.upper()}[];\n'
        f'extern const size_t N_CANDIDATES_{isa.upper()};'
        for isa in isas)
    total_len = sum(per_isa_counts.values())
    lengths_const = f'/* compile-time total: {total_len} */'
    copies = '\n'.join(
        f'  for (size_t i = 0; i < N_CANDIDATES_{isa.upper()}; i++) '
        f'_flat[j++] = CANDIDATES_{isa.upper()}[i];'
        for isa in isas)
    master_code = _CANDS_MASTER_TEMPLATE.format(
        externs=externs, total_len=total_len,
        lengths_const=lengths_const, copies=copies)
    master_path = build / 'vfft_harness_candidates.c'
    master_path.write_text(master_code, encoding='utf-8')
    emitted.append(master_path)
    print(f'  [cands] {master_path.name}: aggregator for {total_len} total')
    return emitted


def phase_compile(candidates_mod, staging: Path, build: Path,
                  harness_c: Path, cand_c_files: list[Path]) -> Path:
    build.mkdir(parents=True, exist_ok=True)
    spec = _cc.detect()

    # Compile with the widest ISA flag set any candidate needs. Individual
    # codelet functions use __attribute__((target(...))) for function-level
    # gating, so global flags just enable intrinsic headers.
    isas = sorted({c.isa for c in candidates_mod.enumerate_all()})
    isa_flags: list[str] = []
    seen_flags: set[str] = set()
    for isa in isas:
        for f in spec.isa_flags(isa):
            if f in seen_flags:
                continue
            seen_flags.add(f)
            isa_flags.append(f)
    if spec.kind == 'msvc':
        # MSVC only accepts one /arch: at a time; widest wins.
        if 'avx512' in isas:
            isa_flags = spec.isa_flags('avx512')
        elif 'avx2' in isas:
            isa_flags = spec.isa_flags('avx2')
        else:
            isa_flags = []

    cmd = [spec.cc] + spec.base_flags('O3') + isa_flags
    cmd += spec.include(str(staging))
    cmd += [str(harness_c)]
    cmd += [str(p) for p in cand_c_files]
    cmd += spec.link_math()

    exe_name = 'harness.exe' if spec.on_windows else 'harness'
    exe_path = build / exe_name
    cmd += spec.output_flag(str(exe_path))

    print(f'  [cc] {spec.kind} ({spec.cc})')
    _run(cmd)
    return exe_path


# ═══════════════════════════════════════════════════════════════
# Phase: run the bench
# ═══════════════════════════════════════════════════════════════

def _bench_key(m: dict) -> str:
    """Identifier for a single measurement entry (for resume / dedup)."""
    return (f"{m.get('variant')}|{m.get('isa')}|{m.get('protocol')}"
            f"|{m.get('radix')}|{m.get('me')}|{m.get('ios')}|{m.get('dir')}")


def _load_done(jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl.exists():
        return done
    with jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
                done.add(_bench_key(m))
            except Exception:
                pass
    return done


def phase_run(candidates_mod, exe: Path, measurements: Path):
    cands = candidates_mod.enumerate_all()
    R = candidates_mod.RADIX
    done = _load_done(measurements)

    total_jobs = 0
    for c in cands:
        for (me, ios) in candidates_mod.sweep_grid(c.variant):
            total_jobs += 2  # fwd + bwd
    print(f'  [run] {total_jobs} jobs ({len(done)} already done — will skip)')

    completed = 0
    skipped_done = 0
    skipped_host = 0
    measurements_dir = measurements.parent
    measurements_dir.mkdir(parents=True, exist_ok=True)

    with measurements.open('a', encoding='utf-8') as out:
        for c in cands:
            proto = c.protocol_override or candidates_mod.protocol(c.variant)
            for (me, ios) in candidates_mod.sweep_grid(c.variant):
                for direction in ('fwd', 'bwd'):
                    key = f"{c.variant}|{c.isa}|{proto}|{R}|{me}|{ios}|{direction}"
                    if key in done:
                        skipped_done += 1
                        completed += 1
                        continue
                    cmd = [str(exe),
                           '--variant',  c.variant,
                           '--isa',      c.isa,
                           '--protocol', proto,
                           '--dir',      direction,
                           '--radix',    str(R),
                           '--me',       str(me),
                           '--ios',      str(ios)]
                    r = _run(cmd, check=False)
                    if r.returncode != 0:
                        print(f'    [warn] harness exit {r.returncode} on '
                              f'{c.id} me={me} ios={ios} dir={direction}')
                        continue
                    line = (r.stdout or '').strip()
                    if not line:
                        print(f'    [warn] empty harness output for {c.id} '
                              f'me={me} ios={ios} dir={direction}')
                        continue
                    try:
                        m = json.loads(line)
                    except Exception as e:
                        print(f'    [warn] bad json from harness: {e}: {line!r}')
                        continue
                    # Dedup-ish protection
                    if m.get('skipped'):
                        skipped_host += 1
                    out.write(line + '\n')
                    out.flush()
                    done.add(key)
                    completed += 1
                    if completed % 25 == 0:
                        pct = 100.0 * completed / total_jobs
                        print(f'    [run] {completed}/{total_jobs} '
                              f'({pct:.0f}%)  skipped_done={skipped_done} '
                              f'skipped_host={skipped_host}')
    print(f'  [run] done: {completed}/{total_jobs}  '
          f'skipped_done={skipped_done}  skipped_host={skipped_host}')


def phase_validate(candidates_mod, staging: Path, generated: Path,
                   build: Path, validate_c: Path) -> int:
    """Compile the validator (once per ISA due to scalar-symbol collisions),
    run it, return total failure count."""
    R = candidates_mod.RADIX
    spec = _cc.detect()
    isas = sorted({c.isa for c in candidates_mod.enumerate_all()})

    total_failures = 0
    for isa in isas:
        if isa == 'scalar':
            continue  # validator covers SIMD ISAs only
        isa_flags = spec.isa_flags(isa)
        if not isa_flags:
            continue
        if isa == 'avx512' and not _cc.host_supports_avx512():
            print(f'  [validate] skipping {isa}: host lacks AVX-512')
            continue

        define_flag = 'VALIDATE_' + isa.upper()
        cmd = [spec.cc] + spec.base_flags('O3') + isa_flags
        cmd += spec.define('RADIX', str(R))
        cmd += spec.define(define_flag)
        cmd += spec.include(str(staging))
        cmd += spec.include(str(generated))
        cmd += spec.include(str(validate_c.parent))   # where fft_radix_include.h lives
        cmd += [str(validate_c)]
        cmd += spec.link_math()
        exe_name = f'validate_{isa}.exe' if spec.on_windows else f'validate_{isa}'
        exe_path = build / exe_name
        cmd += spec.output_flag(str(exe_path))

        print(f'  [cc validate] {isa}')
        _run(cmd)

        print(f'  [run validate] {isa}')
        r = _run([str(exe_path)], check=False)
        print((r.stdout or '').rstrip())
        if r.returncode != 0:
            # Collect failure lines from stderr for a compact summary.
            for line in (r.stderr or '').splitlines():
                if '[FAIL]' in line:
                    print(f'    {line}')
            total_failures += 1
    return total_failures


# ═══════════════════════════════════════════════════════════════
# Top-level driver
# ═══════════════════════════════════════════════════════════════

def main_driver(radix_dir: Path, out_root: Path, phases: list[str],
                generated_root: Path | None = None):
    candidates_py = radix_dir / 'candidates.py'
    if not candidates_py.exists():
        raise SystemExit(f'no candidates.py at {candidates_py}')
    candidates_mod = _load_module(candidates_py, 'candidates')

    R = candidates_mod.RADIX
    staging = out_root / f'r{R}' / 'staging'
    build   = out_root / f'r{R}' / 'build'
    measurements = out_root / f'r{R}' / 'measurements.jsonl'
    harness_c = HERE / 'harness.c'
    validate_c = HERE / 'validate.c'
    if generated_root is None:
        generated_root = Path('generated').resolve()
    generated = generated_root / f'r{R}'

    t0 = time.time()
    if 'generate' in phases or 'all' in phases:
        print('[phase] generate')
        phase_generate(candidates_mod, staging)
    if 'compile'  in phases or 'all' in phases:
        print('[phase] compile')
        build.mkdir(parents=True, exist_ok=True)
        cand_c_files = emit_candidate_table(candidates_mod, staging, build)
        phase_compile(candidates_mod, staging, build, harness_c, cand_c_files)
    if 'run'      in phases or 'all' in phases:
        print('[phase] run')
        spec = _cc.detect()
        exe_name = 'harness.exe' if spec.on_windows else 'harness'
        exe = build / exe_name
        if not exe.exists():
            raise SystemExit(f'no harness at {exe}; run --phase compile first')
        phase_run(candidates_mod, exe, measurements)
    if 'emit'     in phases or 'all' in phases:
        print('[phase] emit')
        # select_and_emit lives in the same package; invoke it as a module.
        import select_and_emit as _sae
        generated.mkdir(parents=True, exist_ok=True)
        host_desc = os.environ.get('VFFT_HOST_DESC', '')
        if not host_desc:
            try:
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if line.startswith('model name'):
                            host_desc = line.split(':', 1)[1].strip()
                            break
            except Exception:
                host_desc = ''
        _sae.emit_all(candidates_mod, measurements, generated,
                      host_desc=host_desc or 'unknown')
    if 'validate' in phases or 'all' in phases:
        print('[phase] validate')
        n_fail = phase_validate(candidates_mod, staging, generated,
                                build, validate_c)
        if n_fail:
            raise SystemExit(f'validation failed in {n_fail} ISA(s)')
    dt = time.time() - t0
    print(f'[done] elapsed {dt:.1f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--radix-dir', required=True,
                    help='path to radixes/rN directory')
    ap.add_argument('--out', default='bench_out',
                    help='output root (default: bench_out in CWD)')
    ap.add_argument('--generated', default='generated',
                    help='output dir for emitted dispatcher headers')
    ap.add_argument('--phase', action='append', default=None,
                    choices=['generate', 'compile', 'run', 'emit',
                             'validate', 'all'],
                    help='pipeline phase(s) to run; repeat or omit for all')
    args = ap.parse_args()
    phases = args.phase or ['all']
    main_driver(Path(args.radix_dir).resolve(),
                Path(args.out).resolve(),
                phases,
                generated_root=Path(args.generated).resolve())