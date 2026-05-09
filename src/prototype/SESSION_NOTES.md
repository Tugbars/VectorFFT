# Session: FFTW-style algsimp ported, FMA-aware throughout

## What we built this session

1. **`NK_Fma` IR node + `fma_lift` pass** — first-class FMA atoms in the
   algsimp IR. Codegen renders as `_mm512_fmadd_pd` / `fmsub_pd` /
   `fnmadd_pd` / `fnmsub_pd`. `dag_stats` reports `fmas` separately
   so vector-instruction count matches asm.

2. **`factor_by_atom` pass** — complement to `factor_common_muls`.
   Where `factor_common_muls` buckets by Mul's *constant* operand
   (FFTW's `collectM` first-direction), `factor_by_atom` buckets by
   the *non-constant* operand (FFTW's `collectM` second-direction).
   When `c1·a + c2·a + ... + cN·a` appears, the constants compile-time-fold
   and N muls collapse to 1.

3. **Fixed-point pipeline** in prime_opcount — FFTW-style loop that runs
   factor + factor_atom + share + transpose to convergence. Stops when
   op count stops decreasing.

4. **Treat `Fma` as opaque** in `factor_common_muls`, `factor_by_atom`,
   `share_subsums`, `transpose` — once a Mul is fused into an FMA, it's
   claimed and can't be unbundled by other passes. This prevents passes
   from un-fusing FMAs to factor or share their components.

## What we measured

### FMA lift is asm-equivalent to GCC auto-fusion

Direct asm comparison on R=11 t1_dit, with vs without `fma_lift`:

|              | with lift | without lift |
|--------------|---:|---:|
| vfmadd       | 167 | 168 |
| vmul         | 55  | 55  |
| vadd         | 69  | 68  |
| **Total**    | 335 | 335 |

GCC -O3 -mfma already fuses Mul+Add → FMA reliably. Our explicit lift
emits the same asm. We keep `fma_lift` for IR cleanliness and metric
accuracy, not for codegen perf.

### Aggressive passes save real asm instructions

A/B test, R=11 t1_dit asm instruction count:

|                | aggressive ON | aggressive OFF |
|----------------|---:|---:|
| R=5 t1_dit     | 68  | 81  |
| R=7 t1_dit     | 130 | 192 |
| R=11 t1_dit    | 335 | 543 |

Aggressive saves 16-38% asm instructions. The factor+share+transpose
pipeline does real work; it isn't net-zero against GCC's fusion.

### `factor_by_atom` doesn't fire on prime DFTs (verified)

Tested on synthetic input `0.3·x + 0.5·x + 0.7·x → 1.5·x` — pass fires
correctly, collapses 3 muls to 1.

Tested on R=5/7/11 prime DFTs — pass NEVER fires. The pattern
"same atom × different constants in same flat sum" simply does not
arise in direct DFT for primes. Each (input, output) pair has exactly
one coefficient by construction.

This is a definitive structural finding: **no amount of algebraic
simplification can reach Rader's mul count starting from direct DFT**.
Rader uses primitive-root reordering of Z/N*Z — a number-theoretic
substitution, not an algebraic identity.

## Current op counts (FMA-aware)

| R    | n1 (m, fma) | t1_dit (m, fma) | t1_dif (m, fma) |
|------|---|---|---|
| R=3  | 18 (4, 0)   | 26 (4, 0)   | 26 (4, 0) |
| R=5  | 64 (16, 0)  | 82 (16, 0)  | 82 (16, 0) |
| R=7  | 126 (30, 14)| 150 (30, 14)| 150 (30, 14) |
| R=11 | 342 (82, 60)| 389 (87, 55)| 382 (82, 60) |

CT-decomposed (passes default to no-op):

| R    | n1 (m, fma) | t1_dit (m, fma) |
|------|---|---|
| R=4  | 16 (0, 0)    | 28 (0, 0) |
| R=8  | 57 (4, 0)    | 85 (4, 0) |
| R=16 | 162 (16, 8)  | 222 (16, 8) |
| R=32 | 438 (56, 32) | 562 (56, 32) |
| R=64 | 1106 (160, 88) | 1358 (160, 88) |

## Asm-level perf gap to hand-coded (R=11 t1_dit)

| | Hand | OCaml (aggressive) | Gap |
|---|---:|---:|---:|
| vfmadd      | 52  | 167 | +115 |
| vfnmadd     | 48  | 14  |  -34 |
| vfmsub      | 10  | 0   |  -10 |
| vmul        | 30  | 55  |  +25 |
| vadd        | 30  | 69  |  +39 |
| vsub        | 20  | 30  |  +10 |
| **Total**   | **190** | **335** | **+145 (76%)** |

The hand-coded uses fnmadd/fmsub aggressively for sign tricks; OCaml
uses mostly fmadd. Plus more standalone muls and adds. All of this is
algorithmic — Rader-Winograd produces fewer arith ops at the math
layer; algsimp can't recover this from direct DFT.

## Next concrete steps (when you pursue this)

The path to closing the algorithmic gap is clear and has two options:

### Option A: FFTW codelet ingest

Parse FFTW's emitted .c codelets (small fixed dialect: `LD`, `ST`,
`ADD`, `SUB`, `MUL`, `FMA`, `FNMA`, `FMS`, `FNMS`, `K(literal)`) into
our Algsimp.t IR. Then run our existing pipeline (factor, share,
transpose, fma_lift, scheduler, spill controller, register allocator,
emit_c) on top.

This is what your Python pipeline does. ~300 lines of OCaml for the
parser, then leverage everything we've built. The pitch:
"FFTW's algorithmic core is excellent, but its emit phase leaves cycles
on modern hardware. I built a re-optimization layer that ingests FFTW's
output, applies microarchitecture-aware scheduling and spill control,
and produces faster codelets."

This is the lowest-effort path to better-than-FFTW perf on every prime.

### Option B: Rader at the math layer

Write `dft_rader_winograd N` in `lib/dft.ml` for each prime that
matters. ~300 lines per radix. More elegant (self-contained) but much
slower to ship, and you have to derive each one. Not obviously better
than Option A unless the pitch specifically requires "no FFTW
dependency."

### Recommendation

**Option A**. You've already validated the architecture in Python.
Reimplementing it in OCaml gives you a single binary, better
maintainability, and access to all the downstream work that's already
in place (the SU scheduler, the BB spill controller, the register
allocator, the emit_c with K-strided layout). You also retain the
FMA-aware metric so you can measure improvements rigorously.

The interview pitch is sharp: a code re-optimizer for FFT codelets
that beats FFTW's own codegen on modern hardware.

---

## Update: conjugate-pair construction (this session)

The "algorithmic gap" framing in the section above was wrong. R=11 hand
is just direct DFT with conjugate-pair sum/diff factoring — no Rader,
no Winograd. The gap was structural CSE that our binary IR + pair-fold
couldn't find.

**Fix**: `dft_direct_conjugate_pair` in `lib/dft.ml` constructs the
shared subexpressions (s_jk, d_jk, p_re_m, p_im_m, q_re_m, q_im_m)
explicitly. Hash-cons preserves them. Used for odd primes (n≥3, odd).

**Results**:
- R=11 n1 dropped 300 → 190 ops (matches hand)
- R=11 t1_dit dropped 336 → 230 ops
- vmovapd in asm dropped 252 → 69 (register pressure resolved)
- Bench R=11 t1_dit @ K=4096: G/H = **1.00-1.07** (parity)
- No regression on pow2

See `docs/23_conjugate_pair.md` for the full diagnosis, the
forward/backward sign convention, and the supporting changes
(`needs_reassoc`, `gen_radix` FP loop alignment).
