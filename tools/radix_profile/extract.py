"""
extract.py — extract per-radix arithmetic count + register-pressure profile
from VectorFFT's generated codelets.

For each (radix R, ISA, variant) it locates the codelet
`radix{R}_{variant}_fwd_{isa}` (n1) or `radix{R}_{variant}_dit_fwd_{isa}`
(t1, t1s) inside src/vectorfft_tune/generated/r{R}/, parses the function body,
and counts:
  - n_load   : SIMD loads (direct intrinsics + LD()/R*_LD() macro aliases)
  - n_store  : SIMD stores (direct intrinsics + ST()/R*_ST() macro aliases)
  - n_add    : adds + subs
  - n_mul    : standalone muls (FMAs counted separately)
  - n_fma    : fused multiply-add variants
  - n_blend  : permute / unpack / blend / shuffle ops
  - n_set1   : broadcast (typically constant load)
  - n_xor    : SIMD xor (sign-flip helpers)
  - n_decls  : SIMD register declarations (proxy for liveness peak)

Three codelet variants are profiled per (R, ISA):
  n1  — no-twiddle (stage 0 of DIT plans).         radix{R}_n1_fwd_{isa}
  t1  — DIT twiddle codelet (stages 1+, default).  radix{R}_t1_dit_fwd_{isa}
  t1s — DIT scalar-twiddle variant.                radix{R}_t1s_dit_fwd_{isa}
log3 is intentionally skipped (too experimental for the cost model).

Outputs:
  profile_avx2.csv     — per-radix per-variant table for AVX2
  profile_avx512.csv   — per-radix per-variant table for AVX-512
  radix_profile.h      — auto-generated C header consumed by the cost model

Run from the repo root or from this directory:
    python tools/radix_profile/extract.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parents[1]                           # repo root
GEN  = ROOT / 'src' / 'vectorfft_tune' / 'generated'

RADIXES  = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13,
            16, 17, 19, 20, 25, 32, 64]
ISAS     = ['avx2', 'avx512']
VARIANTS = ['n1', 't1', 't1s']                   # log3 intentionally omitted


def _fn_name(R: int, variant: str, isa: str) -> str:
    """Mangled symbol of the codelet definition we're after."""
    if variant == 'n1':
        return f'radix{R}_n1_fwd_{isa}'
    return f'radix{R}_{variant}_dit_fwd_{isa}'   # t1, t1s


def find_codelet_file(R: int, variant: str, isa: str) -> Path | None:
    """Return the .h file containing the function definition."""
    rdir = GEN / f'r{R}'
    if not rdir.is_dir():
        return None
    fn = _fn_name(R, variant, isa)
    target = re.compile(rf'^\s*{re.escape(fn)}\s*\(', re.M)
    for f in sorted(rdir.glob('*.h')):
        try:
            txt = f.read_text(errors='replace')
        except Exception:
            continue
        if target.search(txt):
            return f
    return None


def extract_function_body(src: str, fn_name: str) -> str | None:
    """Find `fn_name(` and return everything from the opening { of its body
    to the matching closing }. Returns None if not found."""
    m = re.search(rf'\b{re.escape(fn_name)}\s*\(', src)
    if not m:
        return None
    # advance past argument list
    i = m.end()
    paren = 1
    while i < len(src) and paren > 0:
        c = src[i]
        if c == '(': paren += 1
        elif c == ')': paren -= 1
        i += 1
    # find opening brace
    while i < len(src) and src[i] not in '{;':
        i += 1
    if i >= len(src) or src[i] != '{':
        return None
    start = i
    brace = 1
    i += 1
    while i < len(src) and brace > 0:
        c = src[i]
        if c == '{': brace += 1
        elif c == '}': brace -= 1
        i += 1
    return src[start:i]


# ──────────────────────────────────────────────────────────────────
# Intrinsic + macro-alias patterns.
#
# Codelets reach SIMD loads/stores either via the direct _mm{256,512}_loadu_pd
# intrinsic OR via aliases like LD()/ST(), R2A_LD(), R11A_CT_LD(), R8_256_LD(),
# R11L_CT_LD() — all of which expand to the same _mm*_loadu_pd. Counting
# either form is equivalent (each load = exactly one source-text occurrence).
#
# The R-prefixed variants use uppercase letters/digits/underscores after R and
# end in _LD or _ST: R2A_LD, R8_256_LD, R8_512_LD, R2Z_LD, R{R}A_CT_LD,
# R{R}L_CT_LD. Lowercase is rejected to avoid false positives like Run_LD.
# ──────────────────────────────────────────────────────────────────

LOAD_PAT  = re.compile(
    r'(?:_mm(?:256|512)?_load[au]?_pd'
    r'|\bLD'
    r'|\bR[A-Z0-9_]*_LD'
    r')\s*\('
)
STORE_PAT = re.compile(
    r'(?:_mm(?:256|512)?_store[au]?_pd'
    r'|\bST'
    r'|\bR[A-Z0-9_]*_ST'
    r')\s*\('
)
ADD_PAT   = re.compile(r'_mm(?:256|512)?_(?:add|sub|addsub)_pd\b')
MUL_PAT   = re.compile(r'_mm(?:256|512)?_mul_pd\b')
FMA_PAT   = re.compile(r'_mm(?:256|512)?_(?:fmadd|fmsub|fnmadd|fnmsub)_pd\b')
BLEND_PAT = re.compile(
    r'_mm(?:256|512)?_(?:permute|permutex|permutex2var|unpack(?:hi|lo)|'
    r'blend|shuffle|movedup)_pd\b'
)
SET1_PAT  = re.compile(r'_mm(?:256|512)?_set1_pd\b')
XOR_PAT   = re.compile(r'_mm(?:256|512)?_xor_pd\b')

# Register declarations: count distinct identifiers in lines like
#   __m256d a, b, c;
DECL_TYPE_AVX2   = re.compile(r'\b__m256d\s+([^;]+);')
DECL_TYPE_AVX512 = re.compile(r'\b__m512d\s+([^;]+);')


def count_decls(body: str, isa: str) -> int:
    """Count distinct SIMD register declarations as a register-pressure proxy.
    Splits comma-separated declarations and treats `name = expr` as one decl."""
    pat = DECL_TYPE_AVX2 if isa == 'avx2' else DECL_TYPE_AVX512
    total = 0
    for m in pat.finditer(body):
        decls = m.group(1)
        for piece in decls.split(','):
            name = piece.strip().split('=')[0].strip()
            # skip array syntax like "spill[64]" — those are stack arrays,
            # not SIMD registers
            if not name or '[' in name:
                continue
            total += 1
    return total


def _empty_row(R: int, variant: str, isa: str, fname: str) -> dict:
    return {'radix': R, 'variant': variant, 'isa': isa, 'present': 0,
            'n_add': 0, 'n_mul': 0, 'n_fma': 0,
            'n_load': 0, 'n_store': 0, 'n_blend': 0,
            'n_set1': 0, 'n_xor': 0,
            'n_decls': 0, 'file': fname}


def profile(R: int, variant: str, isa: str) -> dict:
    f = find_codelet_file(R, variant, isa)
    if f is None:
        return _empty_row(R, variant, isa, '')

    src = f.read_text(errors='replace')
    body = extract_function_body(src, _fn_name(R, variant, isa))
    if body is None:
        return _empty_row(R, variant, isa, f.name)

    return {
        'radix':   R,
        'variant': variant,
        'isa':     isa,
        'present': 1,
        'n_add':   len(ADD_PAT.findall(body)),
        'n_mul':   len(MUL_PAT.findall(body)),
        'n_fma':   len(FMA_PAT.findall(body)),
        'n_load':  len(LOAD_PAT.findall(body)),
        'n_store': len(STORE_PAT.findall(body)),
        'n_blend': len(BLEND_PAT.findall(body)),
        'n_set1':  len(SET1_PAT.findall(body)),
        'n_xor':   len(XOR_PAT.findall(body)),
        'n_decls': count_decls(body, isa),
        'file':    f.name,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ['radix', 'variant', 'present', 'n_add', 'n_mul', 'n_fma',
            'n_load', 'n_store', 'n_blend', 'n_set1', 'n_xor',
            'n_decls', 'file']
    with path.open('w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')


# ──────────────────────────────────────────────────────────────────
# Header generation.
# Three static const tables per ISA: stride_radix_profile_{n1,t1,t1s}_{isa}.
# ──────────────────────────────────────────────────────────────────

HEADER_TEMPLATE = """\
/* radix_profile.h — auto-generated by tools/radix_profile/extract.py.
 * DO NOT EDIT BY HAND. Regenerate by running the extractor.
 *
 * Per-radix arithmetic count + register-pressure proxy for three codelet
 * variants used by VectorFFT's executor:
 *   n1   no-twiddle codelet (DIT stage 0).         radix{R}_n1_fwd_{isa}
 *   t1   DIT twiddle codelet (stages 1+).          radix{R}_t1_dit_fwd_{isa}
 *   t1s  scalar-twiddle variant of t1.             radix{R}_t1s_dit_fwd_{isa}
 *
 * Fields:
 *   n_add    SIMD add/sub ops (incl. addsub)
 *   n_mul    standalone SIMD muls (FMA counted separately)
 *   n_fma    fused multiply-add variants (fmadd/fmsub/fnmadd/fnmsub)
 *   n_load   SIMD loads (direct intrinsic + LD()/R*_LD() macro aliases)
 *   n_store  SIMD stores (direct intrinsic + ST()/R*_ST() macro aliases)
 *   n_blend  permute / unpack / shuffle / blend ops
 *   n_set1   broadcast (typically constant load)
 *   n_xor    sign-flip helpers
 *   n_decls  __m256d / __m512d declarations (peak-liveness proxy)
 *
 * Lookup: stride_radix_profile_{n1,t1,t1s}_{avx2,avx512}[R].
 * Entries for radixes / variants we don't generate codelets for are zero.
 */
#ifndef STRIDE_RADIX_PROFILE_H
#define STRIDE_RADIX_PROFILE_H

#define STRIDE_RADIX_PROFILE_MAX_R @@MAX_R@@

typedef struct {
    int n_add;
    int n_mul;
    int n_fma;
    int n_load;
    int n_store;
    int n_blend;
    int n_set1;
    int n_xor;
    int n_decls;
} stride_radix_profile_t;

@@TABLES@@

/* Convenience: total SIMD op count for radix R on the given ISA + variant.
 * Used by the cost model as a per-radix base cost multiplier. */
static inline int _stride_total_ops(const stride_radix_profile_t *p) {
    return p->n_add + p->n_mul + p->n_fma + p->n_load
         + p->n_store + p->n_blend + p->n_set1 + p->n_xor;
}

static inline int stride_radix_total_ops_avx2(int R) {
    if (R <= 0 || R >= STRIDE_RADIX_PROFILE_MAX_R) return 0;
    return _stride_total_ops(&stride_radix_profile_n1_avx2[R]);
}
static inline int stride_radix_total_ops_avx512(int R) {
    if (R <= 0 || R >= STRIDE_RADIX_PROFILE_MAX_R) return 0;
    return _stride_total_ops(&stride_radix_profile_n1_avx512[R]);
}

#endif /* STRIDE_RADIX_PROFILE_H */
"""


def emit_table(rows: list[dict], variant: str, isa: str, max_r: int) -> str:
    by_r = {r['radix']: r for r in rows
            if r['variant'] == variant and r['isa'] == isa}
    lines = [f"static const stride_radix_profile_t "
             f"stride_radix_profile_{variant}_{isa}"
             f"[STRIDE_RADIX_PROFILE_MAX_R] = {{"]
    for R in range(0, max_r):
        r = by_r.get(R)
        if r is None or not r['present']:
            continue  # zeros via aggregate-init zeroing
        lines.append(
            f"    [{R:>2}] = {{ "
            f".n_add = {r['n_add']:>3}, "
            f".n_mul = {r['n_mul']:>3}, "
            f".n_fma = {r['n_fma']:>3}, "
            f".n_load = {r['n_load']:>3}, "
            f".n_store = {r['n_store']:>3}, "
            f".n_blend = {r['n_blend']:>3}, "
            f".n_set1 = {r['n_set1']:>3}, "
            f".n_xor = {r['n_xor']:>3}, "
            f".n_decls = {r['n_decls']:>4} }},"
        )
    lines.append('};')
    return '\n'.join(lines)


def main() -> int:
    print(f'[radix_profile] scanning {GEN}')
    if not GEN.is_dir():
        print(f'[error] generated dir not found: {GEN}', file=sys.stderr)
        return 1

    rows_by_isa = {isa: [] for isa in ISAS}
    for isa in ISAS:
        for variant in VARIANTS:
            for R in RADIXES:
                rows_by_isa[isa].append(profile(R, variant, isa))

    for isa, rows in rows_by_isa.items():
        csv_path = HERE / f'profile_{isa}.csv'
        write_csv(rows, csv_path)
        print(f'  wrote {csv_path.name}')

    max_r = max(RADIXES) + 1
    blocks = []
    for isa in ISAS:
        for variant in VARIANTS:
            blocks.append(emit_table(rows_by_isa[isa], variant, isa, max_r))
    tables = '\n\n'.join(blocks)

    header = (HEADER_TEMPLATE
              .replace('@@MAX_R@@', str(max_r))
              .replace('@@TABLES@@', tables))
    h_path = HERE / 'radix_profile.h'
    h_path.write_text(header, encoding='utf-8')
    print(f'  wrote {h_path.name}')

    # Console summary table — one row per (R, variant), AVX2 numbers.
    print()
    print(f'  {"R":>3} {"var":>4} | {"add":>4} {"mul":>4} {"fma":>4} '
          f'{"ld":>4} {"st":>4} {"blnd":>4} {"set1":>4} {"xor":>3} '
          f'{"dcl":>4}   file')
    print('  ' + '-' * 90)
    for R in RADIXES:
        for variant in VARIANTS:
            r = next((x for x in rows_by_isa['avx2']
                      if x['radix'] == R and x['variant'] == variant), None)
            if r is None or not r['present']:
                print(f'  {R:>3} {variant:>4} | (missing)')
                continue
            print(f'  {R:>3} {variant:>4} | '
                  f'{r["n_add"]:>4} {r["n_mul"]:>4} {r["n_fma"]:>4} '
                  f'{r["n_load"]:>4} {r["n_store"]:>4} {r["n_blend"]:>4} '
                  f'{r["n_set1"]:>4} {r["n_xor"]:>3} '
                  f'{r["n_decls"]:>4}   {r["file"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
