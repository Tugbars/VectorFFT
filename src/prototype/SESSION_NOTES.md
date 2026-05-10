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

---

## Single-use inlining + Pass 1/2 store split (this session)

### TL;DR

Two changes landed:

1. **Single-use inlining** in `lib/emit_c.ml`'s SU-emit path — values
   with exactly one consumer get inlined into the consumer's C
   expression instead of being emitted as `const __m512d t<N> = ...;`.
   Closes the structural gap to hand-coded FFTW codelets:
   - R=13 t1_dif nested intrinsic patterns: **24 → 102** (hand: 120)
   - R=13 t1_dit movapd: **75 → 55** (-20)
   - R=17 t1_dit movapd: **157 → 138** (-19)
   - R=11/13/17 t1_dit bench beats hand at K≥1024 (R=13 K=1024: 0.83 G/H)

2. **Pass 1 / Pass 2 output store split** in the spill-emit path —
   pre-existing bug surfaced while testing: composite codelets
   (R=32/64 t1_dit/t1_dif) had been failing to compile with
   `t<N> undeclared`. Output values whose dep chain didn't cross the
   spill boundary were Pass 1, but the safety net emitted their stores
   inside Pass 2's nested scope where the value was out of scope. Fixed
   by splitting `assigns` by classification and emitting each set in
   the matching scope. R=32/64 t1_dit/t1_dif now compile and pass
   machine-precision correctness.

### The DIF-prime spill investigation

The motivation: R=13/17 t1_dif codelets were 5–20% slower than hand.
R=13 t1_dif had 81 movapd vs hand's 31; R=17 had 176 vs 115.

I tested several hypotheses:

| Approach | Result |
|---|---|
| Sink-preference in scheduler | Reorders C; GCC re-allocates anyway. No asm change. |
| `--annotate` (Annotated_SU/topological) | **Worse**: R=11 t1_dif 50 → 93 movapd. Forward-decl of mutable `__m512d` defeats SSA without giving slot-reuse. |
| FMA-fusion priority tweaks | No-op. |
| Single-use inlining | **Win** for DIT, structural improvement for DIF. |

Why annotate was a dead end: `annotate.ml` declares all cross-scope
values at the outer level (`__m512d t1, t2, ..., t64;`) and assigns
them later. Each is still a single-assignment SSA value, but the
forward-declaration confuses GCC's lifetime analysis. Hand only uses
forward-declared mutables for the **slot variables** (`x0_re ... x12_im`)
that get *reassigned* through `input → intermediate → output → cmul-output`.
That's the real FFTW idiom; annotate doesn't capture it.

What actually closed the gap: hand uses ~120 nested intrinsic patterns
in R=13 t1_dif (`_mm512_fmadd_pd(K1, T1, _mm512_fmadd_pd(K2, T2, ...))`).
Our generator linearized everything into separate SSA temporaries —
~250 unique `const __m512d t<N>` declarations vs hand's ~130 named
values. GCC's allocator handles fewer SSA names better.

### Single-use inlining design

`compute_inline_set` walks the DAG once, counting users per tag:

```ocaml
let compute_inline_set ?fused_muls assigns =
  let nodes = topo_sort_reachable (List.map snd assigns) in
  (* count users: DAG predecessors + output-assignment refs *)
  let use_count = Hashtbl.create 256 in
  List.iter (fun n -> List.iter (bump n.tag) (preds_of n)) nodes;
  List.iter (fun (_, e) -> bump e.tag) assigns;
  (* a tag is inlinable iff:
   *   count = 1   AND   not a sink (output)   AND
   *   not in fused_muls   AND   kind allows inlining
   *   (Const inlined as broadcast; Load not inlined to avoid duplicate
   *   memory ops; Cmul not inlined — paired emit semantics) *)
  ... build set
```

`render_node_def` extended with optional `?inline_set`. The body
renderer uses `render_operand` for predecessors:

```ocaml
let rec render_operand depth n =
  if depth >= inline_max_depth || not (should_inline n) then v n
  else render_inlined depth n
and render_inlined depth n = match n.node with
  | NK_Add (a, b) -> Isa.add_pd isa (render_operand (depth+1) a) (render_operand (depth+1) b)
  | NK_Sub ... (* same for all op kinds *)
  | NK_CmulRe _ | NK_CmulIm _ | NK_Load _ -> v n  (* never inline *)
```

`inline_max_depth = 32` — single-use chain length is bounded by the
predecessor chain length, so the constant matters only as a sanity
limit. Multi-use nodes act as natural stop points.

The SU emission path is wired up at `lib/emit_c.ml` line ~1185:

```ocaml
| SU uarch ->
  let scheduled = Schedule.su_schedule uarch assigns in
  let inline_set = compute_inline_set assigns in
  let is_inlined e = Hashtbl.mem inline_set e.tag in
  List.iter (fun (oref_opt, e) ->
    match oref_opt with
    | None ->
      (* skip emission for inlined values — consumer will inline them *)
      if not (is_inlined e) && not (Hashtbl.mem defined e.tag) then begin
        ... render_node_def ~inline_set:(Some inline_set) e
      end
    | Some oref ->
      (* sinks always emit standalone; they're excluded from inline_set *)
      ... emit_store buf oref e
  ) scheduled
```

Other emit paths (Topological, Bisection, Annotated_*, spill PASS 1/PASS 2)
don't pass `inline_set`, so they fall back to the old behavior. This
keeps the change contained to the prime-codelet path that benefits.

### IR-level results (movapd, AVX-512 GCC 13.3 -O3)

| Codelet | Before | After | Hand |
|---|---:|---:|---:|
| R=11 t1_dit | 22 | **21** | n/a (we already beat hand) |
| R=11 t1_dif | 50 | **49** | 50 |
| R=13 t1_dit | 75 | **55** | n/a |
| R=13 t1_dif | 81 | 83 | 31 |
| R=17 t1_dit | 157 | **138** | n/a |
| R=17 t1_dif | 176 | 180 | 115 |
| R=11 t1_dit_log3 | (n/a) | 20 | n/a |
| R=17 t1_dit_log3 | 156 | 150 | n/a |

DIT improvements are large and clean (-19, -20). DIF is roughly neutral
because the structural change does help (raw nested-intrinsic counts go
up significantly), but DIF's specific bottleneck is the cmul layer where
each raw output has 2 uses (CmulRe + CmulIm) — single-use inlining
explicitly excludes those.

### Bench (5-run median, virt-Skylake-X)

DIT primes (the wins):

| Codelet | K=512 | K=1024 | K=2048 | K=4096 |
|---|---:|---:|---:|---:|
| R=11 t1_dit | 0.93 | 0.88 | 0.91 | 0.86 |
| R=13 t1_dit | 0.97 | **0.83** | 0.96 | 1.05 |
| R=17 t1_dit | **0.86** | 0.95 | 0.96 | 1.17 |
| R=11 t1_dit_log3 | 0.92 | 0.93 | 0.87 | 0.92 |
| R=17 t1_dit_log3 | 0.96 | 0.89 | 0.97 | 1.03 |

R=13 K=1024 at 0.83 G/H means we're 17% faster than hand. R=17 K=512
at 0.86 means 14% faster.

DIF primes (still trailing hand, structural improvement didn't translate
to bench wins because the cmul layer is the bottleneck):

| Codelet | K=512 | K=1024 | K=2048 |
|---|---:|---:|---:|
| R=11 t1_dif | 1.06 | 1.10 | 1.12 |
| R=13 t1_dif | 1.03 | 1.16 | 1.19 |
| R=17 t1_dif | 1.11 | 1.17 | 1.16 |

Closing this gap requires destructive-update emission (FFTW slot-reuse
style: `tr = x_re; x_re = fmsub(x_re, wr, mul(x_im, wi))` reusing the
register slot). That's a substantial emit_c.ml rewrite — left for a
future session.

### The composite codelet bug

Side discovery: R=32/64 t1_dit/t1_dif had been failing to compile with
errors like `'t918' undeclared (first use in this function)`. I'd
initially assumed the inlining work caused this, but a clean test
(setting `inline_set = None`) showed the bug existed independently.

Diagnosis:

```c
{                                        /* PASS 1 scope opens */
    const __m512d t918 = _mm512_sub_pd(t902, t917);  /* defined PASS 1 */
    /* ... PASS 1 stores spilled values ... */
}                                        /* PASS 1 closes — t918 dies */
{                                        /* PASS 2 scope opens */
    /* ... PASS 2 reloads spills, computes ... */
    _mm512_storeu_pd(&rio_re[31*ios + k], t918);  /* ERROR: undeclared */
}
```

t918 was an output value with no internal consumers (only the eventual
store). The forward pass classified it Pass 1 (no spilled ancestors).
The backward pass requires non-empty internal consumers to reclassify,
so it stayed Pass 1. But the output-store emit ran inside PASS 2's
scope — `safety net` at the end iterated over all `assigns` regardless
of classification.

Wrong fix attempt: I added an `output_tags` hint to `classify_passes`
to count output stores as PASS 2 consumers, pushing output-only nodes
to Pass 2. This worked for t918 but broke other codelets — promoting
an output to Pass 2 doesn't bring its preds along, so Pass 2 then
references Pass 1 values that aren't spilled (`'t13' undeclared`).
Reverted.

Correct fix: emit each store in the scope where its value lives.

```ocaml
let pass1_assigns = List.filter (fun (_, e) ->
  Hashtbl.find_opt cls e.tag = Some `Pass1
) assigns in
let pass2_assigns = List.filter (fun (_, e) ->
  Hashtbl.find_opt cls e.tag = Some `Pass2
) assigns in
```

At end of PASS 1's `{ ... }`, before the closing brace, emit
`pass1_assigns` stores. The PASS 2 store machinery (`assigns_by_cluster`,
the cluster flush loop, the safety net) already worked — just had to
restrict its iteration target from `assigns` to `pass2_assigns` so it
doesn't try to re-store Pass 1 outputs in the wrong scope.

Verification (machine-precision DFT-vs-brute-force):

| Codelet | Error | Result |
|---|---|---|
| R=32 t1_dit | 1.58e-14 | PASS |
| R=32 t1_dif | 1.58e-14 | PASS |
| R=64 t1_dit | 3.79e-14 | PASS |
| R=64 t1_dif | 3.79e-14 | PASS |

R=32 t1_dit bench: median T/H=1.04 at K=1024, parity at K=2048+, 1.027
at K=4096. Slightly behind hand at small K (K=128 worst at 1.39); at the
sizes anyone cares about (K≥1024) it's within 5% of hand. Acceptable.

### Files changed

- `lib/emit_c.ml`:
  - **Added** `inline_max_depth = 32` constant.
  - **Added** `compute_inline_set` helper (walks DAG, counts uses,
    returns `(int, unit) Hashtbl.t` of inlinable tags).
  - **Extended** `render_node_def` with `?inline_set` parameter +
    mutually-recursive `render_operand` / `render_inlined` for
    expression-tree inlining.
  - **Wired** `inline_set` through the SU emission path.
  - **Split** `assigns` into `pass1_assigns` / `pass2_assigns` in the
    spill-emit path.
  - **Emit** Pass 1 stores at end of PASS 1 scope (new line).
  - **Restricted** safety net + `assigns_by_cluster` to iterate
    `pass2_assigns`.

`classify_passes` itself is unchanged from prior session. The
`output_tags` hint experiment was reverted.

### Coverage check

Comprehensive compile sweep: R={5, 7, 11, 13, 16, 17, 32, 64} × {t1_dit,
t1_dif, t1_dit_log3, n1} all compile cleanly. 32/32 prime correctness
PASS at machine precision. Composite codelets pass machine-precision
DFT-vs-brute-force.

### What's next

Worth noting for future work:

- **Destructive-update emission for DIF cmul layer** would close the
  remaining DIF gap (5–20%). Requires emit_c.ml to recognize "raw
  output → cmul output" patterns and emit FFTW-style mutable slot
  reassignment (`tr = x_re; x_re = fmsub(...); x_im = fmadd(tr, ...)`).
  Substantial rewrite; not done.

- **Spill-path inlining**: current `compute_inline_set` is computed
  globally and only used by the SU non-spill path. The spill path
  (PASS 1 / PASS 2 emission, where composite codelets live) doesn't
  benefit. Plumbing `inline_set` through PASS 1 and PASS 2 emission
  loops would extend the gain to composite codelets, but the cluster
  boundary makes it tricky — a node inlined in one pass but referenced
  in another would break. Worth attempting once the DIF bottleneck is
  closed.

- **Restoring inlining for `t1s` codelets**: t1s twiddles are scalar
  broadcasts (`set1_pd`), already implicitly inlined by the constant
  path. No change needed.

---

## R=19 + extended test coverage (this session, follow-up)

### TL;DR

R=19 already generated correctly via the existing
`dft_direct_conjugate_pair` (the construction is generic for any odd
prime ≥ 3). What was missing was test infrastructure: the prime
op-count tool and the runtime correctness harness only covered
R={2, 5, 7, 11}. Extended both to cover R={2, 5, 7, 11, 13, 17, 19}.
**56/56 codelet variants now PASS at machine precision** (up from
32/32). All 8 R=19 variants verified against brute-force scalar DFT.

### Op counts (FMA-aware, post algsimp + factor + share)

```
R= 2: n1=4    t1_dit=8    t1_dit_log3=8    t1_dif=8    t1_dif_log3=8    (fma 0)
R= 3: n1=12   t1_dit=20   t1_dit_log3=20   t1_dif=20   t1_dif_log3=20   (fma 6)
R= 5: n1=36   t1_dit=52   t1_dit_log3=56   t1_dif=52   t1_dif_log3=56   (fma 12)
R= 7: n1=66   t1_dit=90   t1_dit_log3=102  t1_dif=90   t1_dif_log3=102  (fma 30)
R=11: n1=150  t1_dit=190  t1_dit_log3=214  t1_dif=190  t1_dif_log3=214  (fma 90)
R=13: n1=204  t1_dit=252  t1_dit_log3=284  t1_dif=252  t1_dif_log3=284  (fma 132)
R=17: n1=336  t1_dit=400  t1_dit_log3=444  t1_dif=400  t1_dif_log3=444  (fma 240)
R=19: n1=414  t1_dit=486  t1_dit_log3=538  t1_dif=486  t1_dif_log3=538  (fma 306)
```

For comparison, the hand-coded gen_radix19.py reports 542 ops at the
genfft DAG level (314 add + 114 mul + 114 FMA — pre-fusion counting,
each FMA still contains a mul + an add as separate atoms). After our
algsimp + FMA fusion, n1 is 414 ops total (108 add + 306 FMA) — 23%
fewer atoms than hand. t1_dit at 486 reflects the same direct-DFT
inner butterfly plus 18 cmuls for input twiddling.

R=19 has very high register pressure (19 inputs + 18 twiddles + 26
constants ≫ 32 ZMM), so heavy spilling is structural — both ours and
hand spill aggressively. AVX-512 R=19 t1_dit asm shows ~170 vmovapd
with 51 spill slots (3272 bytes of stack). Our share-skip path saves
108 ops vs the share-aggressive variant — bigger savings than R=17
(-98) because more shared subexpressions exist to NOT pessimize.

### R=19 correctness errors (vs brute-force scalar DFT, K=8)

| Variant | Error |
|---|---|
| R=19 t1_dit | 1.6e-13 |
| R=19 t1_dif | 1.9e-12 |
| R=19 t1_dit_log3 | 1.9e-13 |
| R=19 t1_dif_log3 | 1.9e-12 |
| R=19 t1s_dit / dif / *_log3 | same as flat (deterministic) |

All under 1e-10 threshold. DIF errors slightly larger than DIT —
expected because DIF post-multiplies outputs (extra mul layer
accumulates rounding) but well within machine-precision bounds.

### Files changed

- `bin/prime_opcount.ml`: prime list extended `[2; 3; 5; 7; 11]` →
  `[2; 3; 5; 7; 11; 13; 17; 19]`. Header comment updated.
- `bin/dump_stages.ml`: stage-dump prime list extended
  `[3; 5; 7; 11; 13]` → `[3; 5; 7; 11; 13; 17; 19]`.
- `bench/primes/correctness/test_all8_runtime.c`: `MAX_N` 11 → 19,
  `MAX_W` 10 → 18, `DECL_ALL` for R=13/17/19, `RUN8` for the new
  radixes.
- `bench/primes/correctness/build_and_run.sh`: codelet generation
  loop extended to `for R in 2 5 7 11 13 17 19`.

The OCaml generator itself (`lib/dft.ml`, `lib/algsimp.ml`,
`lib/emit_c.ml`) needed no changes — `dft_direct_conjugate_pair` is
generic in N, so adding a new prime is a test-coverage addition only.

### Why R=19 likely won't get bench wins vs hand at AVX-512

For R=11/13/17 our inner DFT competes well with hand. R=19 has the
same structural advantages on op count but the spill pressure is so
extreme (51 slots) that GCC's allocator has limited room to maneuver
in either direction. Hand uses heavily nested intrinsic expressions
to lower SSA name count — a similar approach to what closed the gap
for R=13/17 in the inlining work — but the 19-input DAG creates many
long dependency chains that resist inlining (multi-use intermediates
break inline chains). DIT primes will likely be at parity with hand;
DIF primes at 5-20% behind, similar to R=17. No bench data captured
yet (would require a hand R=19 reference, which means running
gen_radix19.py through Python — not done in this session).

---

## emit_c.ml refactor — three reviewer-suggested cleanups

A separate review of `lib/emit_c.ml` flagged dead machinery and
duplication. Three changes landed; all three preserve byte-identical
asm output (verified for R=17 t1_dit and R=32 t1_dit), with the only
asm difference being the `.file` directive (source filename header).

### #1 — Delete dead FMA fusion machinery

The file carried a full source-level FMA fusion pipeline (`fused_muls`
hashtbl, `_is_fusable_mul`, `as_fused_mul` in `render_node_def`,
`through_fused_mul` in PASS 2 reload tracking, the `~fused`
declarator-skip branch in node emission). The fusion table was
created empty and never populated — a comment block at the spill-path
entry explained the table was disabled because GCC's `-mfma` at `-O3`
fuses contracted patterns automatically and explicit fmadd emission
slightly hurt by constraining GCC's variant selection.

Net effect: every `as_fused_mul` returned `None`, every
`through_fused_mul` returned `[n]`, every `~fused` rendering branch
was unreachable. ~140 lines of code that looked active but did
nothing — exactly the kind of trap that lulls a future contributor
into incorrect reasoning about emission.

Removed:
- `?fused_muls` parameter from `render_node_def` and
  `compute_inline_set`
- `as_fused_mul` helper (and its NK_Add/NK_Sub fusion arms in both
  the body and `render_inlined`)
- `through_fused_mul` walker in PASS 2 reload tracking
- `fused_muls` hashtbl creation, `_is_fusable_mul`, the `ignore`
  line, `fused_muls_opt`
- `is_spill_target` and `is_fused_pass1_tag` (only fed
  `_is_fusable_mul`)
- `use_count` and `bump_use` in the spill path (also only fed
  `_is_fusable_mul`)
- The `Hashtbl.mem fused_muls e.tag` skip in PASS 1 and PASS 2
  emission

Renamed: the surviving `~fused` parameter became `~no_declarator`.
The original name was overloaded for two unrelated concepts — FMA
fusion (now gone) and the spill-fused-slot forward-declaration mode
(real, kept). Disambiguating made the surviving path readable.

### #2 — Consolidate `preds_of` definitions

The IR-predecessor walk was redefined locally in seven places:

- `lib/emit_c.ml` × 4 (classify_passes, block-sequential reorder,
  cluster propagation as `preds_of_general`, PASS 2 reload tracking)
- `lib/schedule.ml` × 2 (`predecessor_exprs`, `preds_of`)
- `lib/bb.ml` × 1 (`preds_of`, exposed via `Bb.preds_of` for the
  bb_diagnostic CLI)

Plus an eighth in `lib/annotate.ml` that the review didn't catch.

All eight bodies were pixel-identical (modulo whitespace and the
parameter type annotation). Consolidated into one canonical
`Algsimp.preds : t -> t list` defined immediately after the `t`
record type. All seven other locations now call `Algsimp.preds` (or
unqualified `preds` where Algsimp is opened); `Bb.preds_of` is kept
as `let preds_of = Algsimp.preds` to preserve the exported symbol
for the diagnostic CLI without forcing that file to change.

The `through_fused_mul` walker that was the only meaningful variation
between the seven definitions disappeared with #1, so consolidation
became fully mechanical.

### #4 — Tighten the PASS 2 cluster-flush state machine

The cluster-boundary detection in PASS 2 emission was a three-arm
match:

```ocaml
(match cur_cluster, !last_pass2_cluster with
 | Some c, Some prev when c <> prev ->
   flush_cluster_stores prev;
   last_pass2_cluster := Some c
 | Some c, None ->
   last_pass2_cluster := Some c
 | _ -> ())
```

Two arms had the same `last_pass2_cluster := Some c` update and the
remaining `_ -> ()` case made it visually ambiguous which scenarios
were no-ops. Easy to break on a future refactor (e.g., adding a
fourth arm for some new cluster category).

Collapsed to a two-arm match (only the prev≠cur transition does
work) plus a single unconditional update:

```ocaml
let cur_cluster = Hashtbl.find_opt cluster_of_pass2_node e.tag in
(match !last_pass2_cluster, cur_cluster with
 | Some prev, Some now when prev <> now -> flush_cluster_stores prev
 | _ -> ());
(match cur_cluster with
 | Some _ -> last_pass2_cluster := cur_cluster
 | None -> ())
```

Same logic, smaller surface area. Verified asm-equivalent.

### Verification

- 56/56 prime correctness PASS at machine precision (R={2,5,7,11,13,17,19} × 8 variants)
- 42/42 prime spot-check compile sweep
- R=32/64 n1/t1_dit/t1_dif composite codelets compile
- R=17 t1_dit and R=32 t1_dit asm byte-identical to baseline (only
  difference: `.file` directive with source filename)

### Line count delta

| File | Before | After | Δ |
|---|---|---|---|
| lib/emit_c.ml | 1327 | 1166 | **−161** |
| lib/schedule.ml | 785 | 768 | −17 |
| lib/algsimp.ml | 1707 | 1720 | +13 (canonical preds) |
| lib/annotate.ml | 313 | 305 | −8 |
| lib/bb.ml | 310 | 305 | −5 |
| **Total** | **4442** | **4264** | **−178** |

### What the review missed (logged for later)

- The `lib/annotate.ml` `preds` definition was an eighth duplicate;
  the review listed five.
- `lib/bb.ml` exports `preds_of` for `bin/bb_diagnostic.ml`; the
  cleanest path was to keep the symbol as a re-export of
  `Algsimp.preds` rather than rename the public API.
- The empirical concern in the review's #3 (constants inside the
  for-loop not being hoisted by GCC's LICM) was empirically wrong at
  GCC 13.3 / `-O3 -mavx512f -mfma`: `vbroadcastsd` instructions for
  literal constants land at lines 361–381 of the R=32 t1_dit asm,
  before the loop label `.L7` at line 385. The structural critique
  (cleaner to emit constants outside the loop) stands but not on
  perf grounds.
- The Annotate emission path in `lib/emit_c.ml` (~300 lines, the
  `--annotate` flag) is also dead — prior session established it
  makes spills 86% worse because forward-declaring mutables defeats
  SSA. Same kind of "scaffolded experiment with no current value"
  that #1 addressed, but at a larger blast radius. Left for a future
  cleanup.
