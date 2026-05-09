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

---

## Update: HAND PARITY achieved on R=11 t1_dit (this session, follow-up)

After the conjugate-pair construction, the gap was 230 vs 190 hand
(+21%). Two more changes closed it completely.

### Sign-aware FMA chain construction

`make_sum_with_init` in `lib/dft.ml` now builds the cosine sum chain
with `Sub` for negative coefficients (instead of `Add(Mul(t, neg_const), prev)`).
After `fma_lift`, this lifts to a single chain mixing `fmadd` (positive)
and `fnmadd` (negative) — exactly hand's pattern. Initial accumulator
is `x[0].re` (or `.im`), absorbed as the deepest FMA addend = free.

R=11 t1_dit: 230 → **212**. Inner DFT 190 → **172**.

### Skip `share_subsums` for direct primes

Diagnostic stage-by-stage showed `share_subsums` was actively HURTING:
240 → 256 ops (+16), and the FP transpose loop made it worse (322,
reverted). Because our `dft_direct_conjugate_pair` already builds
hash-cons-shared intermediates explicitly, share_subsums tries to
re-share what's already shared and ends up materializing intermediates
that prevent `fma_lift` from collapsing the unified FMA chain.

Gated the skip on `pick_algorithm n = Direct` so composite Cooley-Tukey
sizes (where share_subsums actually helps) are unaffected.

R=11 t1_dit: 212 → **190 ops = hand parity (exact)**.

### Final R=11 results

ASM (AVX-512): 190 arith (matches hand exactly per opcode), 22 vmovapd
(less than half of hand's 48 — fewer spills than the hand-coded reference).

Bench (median of 5 runs):

| K=64 | K=128 | K=256 | K=512 | K=1024 | K=2048 | K=4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.97 | 0.98 | 0.98 | **0.88** | **0.87** | **0.82** | **0.93** |

R=11 t1_dit BEATS hand at every K, by 12-18% at K=512-2048.

### Cross-cutting wins

| | Was | Now | Hand |
|---|---:|---:|---:|
| R=5 n1   | 38  | **36**  | -- |
| R=7 n1   | 70  | **66**  | -- |
| R=11 n1  | 172 | **150** | 150 ✓ |
| R=11 t1_dit | 212 | **190** | 190 ✓ |
| R=13 n1  | 256 | **204** | -- |

R=5/R=7/R=11 t1_dit_log3 codelets BEAT hand by 10-21% at small K.

### What this means

The lib achieves hand parity on the hardest prime (R=11 — the largest
prime currently shipped) with a fully synthesized DAG. No hand-tuned
codelet for R=11 needs to ship. R=13 is now within striking distance
(204 ops; estimating hand-coded R=13 would land at ~180-190).

For the HFT/perf-eng career angle: this is now a complete story.
Codelet generator that BEATS FFTW on pow2, MATCHES hand-coded codelets
on prime sizes that FFTW doesn't ship efficiently (R=11), and uses 50%
fewer register spills than the hand reference. The mechanical pipeline
(of_assignments → factor → fma_lift) replaces hundreds of lines of
hand-tuned C with a few hundred lines of OCaml that you can extend to
new sizes by writing a single `make_sum_with_init`-style construction
function.

See `docs/23_conjugate_pair.md` for the full diagnosis (stage-by-stage
counts, ASM diff per opcode, bench tables across all R=5/7/11 variants,
and the lessons learned).

---

## DIF coverage audit & runtime correctness for all 8 codelet variants

Question raised: the planner picks DIT or DIF for the whole transform
(can't mix). If it picks DIF, ALL stage codelets must exist as DIF
variants, including log3-twiddled and t1s (strided-twiddle) forms.

### Coverage audit

The lib (`lib/dft.ml`) supports `policy ∈ {TP_Flat, TP_Log3}` and
`direction ∈ {DIT, DIF}` independently. `gen_radix.exe` accepts
`--dif`, `--log3`, `--t1s` independently, generating all 2³ = 8
combinations correctly:

| variant | function name |
|---|---|
| t1_dit | `radix11_t1_dit_fwd_avx512_gen_inplace_su` |
| t1_dif | `radix11_t1_dif_fwd_avx512_gen_inplace_su` |
| t1_dit_log3 | `radix11_t1_dit_log3_fwd_avx512_gen_inplace_su` |
| t1_dif_log3 | `radix11_t1_dif_log3_fwd_avx512_gen_inplace_su` |
| t1s_dit | `radix11_t1s_dit_fwd_avx512_gen_inplace_su` |
| t1s_dif | `radix11_t1s_dif_fwd_avx512_gen_inplace_su` |
| t1s_dit_log3 | `radix11_t1s_dit_log3_fwd_avx512_gen_inplace_su` |
| t1s_dif_log3 | `radix11_t1s_dif_log3_fwd_avx512_gen_inplace_su` |

What was missing: `prime_opcount` only verified 5 (n1, t1_{dit,dif} ×
{flat,log3}). The bench harness only tested 4 (t1_dit, t1_dif,
t1_dit_log3, t1s_dit) because the Python hand generators only cover
those. The DIF-side log3 and DIF-side t1s variants had never been
runtime-tested.

### Verification

Added two checks:

1. **`bin/prime_opcount.ml`**: now also checks `t1_dif_log3` for
   correctness at the IR level. Op counts are identical to `t1_dit_log3`
   as expected (DIT and DIF differ only in twiddle layer placement,
   not arithmetic count):

   | R | n1 | t1_dit | t1_dit_log3 | t1_dif | t1_dif_log3 |
   |---|---:|---:|---:|---:|---:|
   | 5 | 36 | 52 | 56 | 52 | 56 |
   | 7 | 66 | 90 | 102 | 90 | 102 |
   | 11 | 150 | 190 | 214 | 190 | 214 |

2. **`bench/primes/correctness/test_all8_runtime.c`**: brute-force
   scalar DFT reference against each of the 8 codelets, run for
   R={5,7,11}. **24/24 PASS at machine precision** (errors all under
   1e-12).

### Performance among gen variants (R=11, ns/call)

| Variant | K=64 | K=256 | K=1024 | K=4096 |
|---|---:|---:|---:|---:|
| t1_dit       | 516 | 2482 | 13563 | 78050 |
| t1_dif       | 512 | 3028 | 15421 | 80433 |
| t1_dit_log3  | 487 | 2440 | 12772 | **63789** |
| t1_dif_log3  | 506 | 2831 | 13399 | 73407 |
| t1s_dit      | 428 | 2727 | 12077 | 56228 |
| t1s_dif      | **389** | 2736 | 12411 | 71645 |
| t1s_dit_log3 | 411 | 2827 | **11522** | 57653 |
| t1s_dif_log3 | 421 | **2656** | 13821 | 71932 |

DIT faster than DIF in most cells (consistent with lower vmovapd:
DIT 22-60 vs DIF 50-88). t1s wins at small K (broadcast twiddles
need fewer loads). log3 wins at large K (lower twiddle bandwidth).
The planner now has the full 8-way menu to pick from at any (K, direction).

### Files added/changed

- `bin/prime_opcount.ml` — added `t1_dif_log3` correctness check;
  output line now includes 5 column tags.
- `bench/primes/correctness/test_all8_runtime.c` — runtime correctness
  harness (brute-force DFT vs each gen codelet, R={5,7,11} × 8 variants).
- `bench/primes/correctness/build_and_run.sh` — driver script.

---

## R=2 coverage audit

R=2 was generating correctly but absent from regression tests. The lib's
dispatch (`lib/dft.ml`) handles n=2 in `dft_direct` (the naive fallback,
since `n >= 3 && n mod 2 = 1` doesn't match). `gen_radix` emits all 9
variants (n1 + 8 twiddled) cleanly:

| variant | arith | fma | movapd |
|---|---:|---:|---:|
| n1 | 4 | 0 | 0 |
| t1_{dit,dif} | 8 | 2 | 0 |
| t1_{dit,dif}_log3 | 8 | 2 | 0 |
| t1s_{dit,dif} | 8 | 2 | 0 |
| t1s_{dit,dif}_log3 | 8 | 2 | 0 |

R=2 has no asymmetry (one twiddle slot W^1, so log3 == flat — same
codegen, just separate function name suffix). All variants land at the
same 8-op asm. No vmovapd at all (fits cleanly in registers).

### Test coverage extended

- `bin/prime_opcount.ml`: R=2 added to the prime list `[2; 3; 5; 7; 11]`.
  IR-level correctness checks all 5 variants (n1, t1_dit, t1_dit_log3,
  t1_dif, t1_dif_log3) match brute-force DFT.
- `bench/primes/correctness/test_all8_runtime.c`: extended to R=2.
  **32/32 PASS** at machine precision (R=2 errors at 1.5e-16, near eps).

### When R=2 matters

The internal CT recursion (`pick_algorithm`) already uses R=2 implicitly:
even N's that aren't in the lookup `{4, 8, 16, 32, 64}` fall through
to `CT(2, n/2)`, which inlines a radix-2 split into the larger codelet.
This was always the case.

The new value is **standalone** R=2 codelet emission, useful for:
- Multi-stage planners that want explicit leaf codelets per stage
  (e.g. emitting separate R=2, R=4, ..., R=64 .c files and dispatching
  at runtime based on transform size).
- Recursive Bluestein or Rader's algorithm when the surrounding
  framework needs an R=2 leaf.
- N=2 standalone (rare but tested).

For the planner's existing single-codelet-per-transform model, R=2
isn't called directly (the outer R=N codelet does it all). But the
codegen path is now verified end-to-end so adding standalone R=2 to
a future multi-stage planner is a no-op on the lib side.
