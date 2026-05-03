# 02 — Static profile (`radix_profile.h`)

What `tools/radix_profile/extract.py` produces and what's in
`src/core/generated/radix_profile.h`.

## What it measures

For each `(radix R, codelet variant, ISA)` triple where a codelet exists,
`extract.py` parses the C source and counts:

| Field | What it counts |
|-------|----------------|
| `n_add` | SIMD add/sub/addsub ops |
| `n_mul` | standalone SIMD muls (FMAs counted separately) |
| `n_fma` | fused multiply-add variants (`fmadd`, `fmsub`, `fnmadd`, `fnmsub`) |
| `n_load` | SIMD loads — direct intrinsic AND macro aliases (see below) |
| `n_store` | SIMD stores — direct intrinsic AND macro aliases |
| `n_blend` | permute / unpack / shuffle / blend ops |
| `n_set1` | `_mm*_set1_pd` broadcasts |
| `n_xor` | sign-flip helpers (`_mm*_xor_pd`) |
| `n_decls` | `__m256d`/`__m512d` declarations — peak register pressure proxy |

## Variants covered

Per ISA (avx2, avx512), three variants:

- **n1** — no-twiddle codelet (`radix{R}_n1_fwd_{isa}`), used at stage 0
- **t1** — DIT twiddle codelet (`radix{R}_t1_dit_fwd_{isa}`), used at stages 1+
- **t1s** — scalar-twiddle variant (`radix{R}_t1s_dit_fwd_{isa}`), used at stages 1+ when `wisdom_bridge`'s `prefer_t1s` predicate fires

`log3` codelets exist in the registry but are not statically profiled —
the cost model uses CPE numbers for log3 instead of op counts (see
[03_dynamic_cpe.md](03_dynamic_cpe.md)).

## Macro alias detection (the non-obvious part)

Codelets reach SIMD loads/stores through three layers of macros:

```
_mm256_loadu_pd     # direct intrinsic — easy to count
LD(...)             # bare alias  →  expands to R*A_LD or similar
R8_256_LD(...)      # per-radix alias  →  expands to _mm256_loadu_pd
R11A_CT_LD(...)     # CT-variant alias  →  expands to _mm256_loadu_pd
R11L_CT_LD(...)     # AVX-512 CT-variant alias  →  expands to _mm512_loadu_pd
```

The early version of `extract.py` only recognized direct intrinsics, so
R=2/4/8 — which use macro aliases exclusively — reported zero loads.
This made R=2 look free in the cost model and broke pick decisions.

The current regex catches all three macro shapes:

```python
LOAD_PAT  = re.compile(
    r'(?:_mm(?:256|512)?_load[au]?_pd'
    r'|\bLD'
    r'|\bR[A-Z0-9_]*_LD'
    r')\s*\('
)
```

Each macro call expands to exactly one SIMD load at preprocessor time,
so counting them is equivalent to counting expanded loads.

## What's in the emitted header

Three tables per ISA — one per variant:

```c
static const stride_radix_profile_t stride_radix_profile_n1_avx2[]  = { ... };
static const stride_radix_profile_t stride_radix_profile_t1_avx2[]  = { ... };
static const stride_radix_profile_t stride_radix_profile_t1s_avx2[] = { ... };

/* + same three for avx512 */
```

Indexed by `[R]`, with zero-initialized entries for radixes the codelet
generator hasn't emitted. Caller checks `_stride_total_ops(p) == 0` as
a "this slot is empty" sentinel.

## How the cost model uses it

The cost model prefers measured cycles (`radix_cpe.h`) and only falls
back to op counts when a CPE slot is empty. Specifically:

```c
double cpe = _radix_cpe_lookup(R, variant_idx, isa_avx512);
if (cpe > 0.0) return cpe;
/* No measured CPE — fall through to static op count. */
int ops = _stride_total_ops(p);
return (double)ops / (double)simd_width;
```

This means `radix_profile.h` is mostly a **safety net** today. It's
load-bearing only for radixes that exist in the registry but weren't
seen by `measure_cpe` (rare — would happen if a new radix were added
between regen passes).

## Reasons the static count diverges from real cost

A pure ops-based model has known limitations the CPE numbers correct
for:

| Phenomenon | Static op count says | Reality |
|------------|---------------------|---------|
| Decoder pressure on huge codelets | proportional to ops | super-linear above ~200 ops |
| DTLB capacity overflow on R=32/64 | ignored | dominant bottleneck at K≥256 |
| Dependency chain length on Winograd primes | ignored | dominates R=10/11/12/13 |
| Compiler register-spill effectiveness | underestimated by `n_decls` proxy | ICX handles 94 decls on AVX2 fine |

The bench history records this: pure-ops scoring gave a 1.69× mean
estimate-vs-wisdom ratio, while CPE-based scoring brought it to 1.19×.

## Regeneration

```
python tools/radix_profile/extract.py
```

Output goes to `src/core/generated/radix_profile.h`. CSV side-products
(`profile_avx2.csv`, `profile_avx512.csv`) stay in
`tools/radix_profile/` as raw data for inspection.

Runs in <1 second. Deterministic — same codelets give same output. No
host dependencies — works on Linux, Windows, macOS.

Re-run when:

- A codelet was regenerated (new intrinsics, added variant, etc.)
- A new radix was added to the registry
- The intrinsic regex needs updating (new macro alias pattern)

The script also prints a console table summarizing per-radix per-variant
op counts, useful as a sanity check after a regen.

## See also

- [03_dynamic_cpe.md](03_dynamic_cpe.md) — the dynamic CPE table that mostly supersedes this one
- [`tools/radix_profile/extract.py`](../../tools/radix_profile/extract.py) — the source
- [`src/core/generated/radix_profile.h`](../../src/core/generated/radix_profile.h) — the output
