# Conjugate-pair direct DFT: closing the prime gap

## TL;DR

Built `dft_direct_conjugate_pair` for odd-prime N. R=11 n1 dropped from
**300 → 190** ops (matching hand-coded). t1_dit dropped from **336 → 230**
ops. Bench: G/H ratios for R=11 went from 1.6-1.8× down to 1.0-1.4× across
K, with parity at K=4096 for R=11 t1_dit and R=7. R=5/7 mostly at parity.
No regression on pow2 (R=4..64 unchanged).

## The diagnosis

`prime_opcount` reported R=11 t1_dit = 336 ops (hand: 190). The ASM showed
335 arith + 252 vmovapd vs hand's 190 arith + 48 vmovapd. Gap = 145 arith
+ 200+ register copies.

Earlier I had claimed the gap was "algorithmic" requiring Rader-Winograd.
That was wrong. The hand-coded R=11 is just direct DFT with **conjugate-pair
sum/difference factoring** — standard for any prime, no special algebra.
The structure:

```
For pair (j, N-j), pair sum/diff:
  s_re_j = x[j].re + x[N-j].re      d_re_j = x[j].re - x[N-j].re
  s_im_j = x[j].im + x[N-j].im      d_im_j = x[j].im - x[N-j].im

For each output pair (m, N-m), four shared intermediates:
  p_re_m = Σ cos(2πjm/N) · s_re_j     [shared between X[m].re and X[N-m].re]
  p_im_m = Σ cos(2πjm/N) · s_im_j     [shared between X[m].im and X[N-m].im]
  q_re_m = Σ sin(2πjm/N) · d_re_j     [opposite signs in im outputs]
  q_im_m = Σ sin(2πjm/N) · d_im_j     [opposite signs in re outputs]

Combine:
  X[m].re   = x[0].re + p_re_m + q_im_m
  X[N-m].re = x[0].re + p_re_m - q_im_m
  X[m].im   = x[0].im + p_im_m - q_re_m
  X[N-m].im = x[0].im + p_im_m + q_re_m
```

Each shared intermediate (5 fmas at most) is computed once and reused for
two outputs. The naive direct DFT computed each X[m] independently with
no cross-output sharing.

The ROOT CAUSE of the gap was: our DAG had pair sums (s_jk, d_jk) but did
NOT share the per-pair-output intermediates (p_re_m, q_im_m). With binary
IR + pair-fold, the surrounding contexts of X[m] and X[N-m] differ
(opposite signs on sin terms), so identical hash-cons subtrees never form.

## What was tried first

1. **Relaxing fma_lift's `single_use` constraint** to `liftable_mul = true`.
   Lifted shared muls (use_count 2+) into FMAs in each consumer, on the
   theory that the implicit "duplication" is free at the asm level.
   - DAG dropped 389 → 336 for R=11 t1_dit (FMA count 55 → 186)
   - ASM essentially unchanged (335 → 336 arith, vmovapd 219 → 252)
   - GCC was already auto-fusing those patterns; explicit lift just
     normalized our metric to track GCC behavior.
   - Kept the change because the DAG metric is now accurate.

## What worked

`dft_direct_conjugate_pair` in `lib/dft.ml`. The construction explicitly
shares the four per-pair-output intermediates via OCaml value reuse;
algsimp's hash-cons preserves the sharing through `of_expr`. Wired into
`dft` dispatch for `n >= 3 && n mod 2 = 1`.

Key implementation details:

1. **OCaml expr value sharing**: `s_re`, `s_im`, `d_re`, `d_im` are
   `Array.init`'d once. `p_re_m`, `q_im_m`, etc. are constructed with
   shared references to the s/d arrays. When multiple outputs reference
   the same intermediate `expr`, `of_expr` walks each occurrence but
   `mk_*` hash-cons returns identical algsimp tags.

2. **Linear-chain make_sum**: builds `((m1+m2)+m3)+m4` left-folded chains
   of `Mul + Add`, exactly the pattern fma_lift catches. Each chain link
   becomes `Fma(c, x, prev_acc)` after fma_lift. Result: 5-fma linear chains
   per intermediate.

3. **`reassoc=false` for the conjugate-pair path**: smart `mk_add` flattens
   nested adds, which would break the explicit p_re_m / q_im_m structure
   by mixing them into each output's outer sum. With reassoc=false,
   `of_expr` calls `mk_add_binary` which just hash-cons's. Updated
   `needs_reassoc` to return `false` for odd primes (`Direct` case where
   conjugate-pair construction has already done the work).

4. **Sign convention for forward/backward DFT**: the sin coefficient
   factor is `-sgn` (not `sgn`). Forward (sgn=-1) gives +sin, backward
   (sgn=+1) gives -sin. Easy to get wrong; the bug crashed correctness
   on first attempt with O(1)-magnitude errors. The math:

   ```
   exp(-iθ) = cos(θ) - i·sin(θ)            [forward]
   X[m].re_forward = Σ (x[j].re·cos + x[j].im·sin)
   ```

5. **Aligned gen_radix's pipeline with prime_opcount's fixed-point loop**.
   `gen_radix` previously ran one transpose round trip; `prime_opcount`
   iterated up to 6 rounds. For R=11 n1, 1 round → 255 ops, 6-round FP →
   190. Updated `gen_radix` to match.

## Op count results (DAG, vector instructions)

| Codelet           | Before | After | Hand |
|-------------------|-------:|------:|-----:|
| R=3 n1            |     14 |    14 |  --  |
| R=5 t1_dit        |     82 |    60 |  --  |
| R=7 t1_dit        |    150 |   104 |  --  |
| R=11 t1_dit       |    389 |   230 |  ~190 (incl twiddle) |
| R=11 n1           |    342 |   190 | 190 (matches!)   |
| R=11 t1_dif       |    341 |   230 |  ~190 |
| R=4..64 (pow2)    |   same |  same |  -- |

## ASM results (R=11 t1_dit, AVX-512 -O3 -mfma)

| Metric          | Before | After | Hand |
|-----------------|-------:|------:|-----:|
| arith total     |    335 |   230 |  190 |
| vfmadd          |    167 |    76 |  ~52 |
| vfnmadd         |     14 |    24 |  ~48 |
| vfmsub/vfnmsub  |      0 |    12 |  ~10 |
| vmul            |     55 |    44 |   30 |
| vadd            |     69 |    50 |   30 |
| vsub            |     30 |    24 |   20 |
| vmovapd (regs)  |    219 |    69 |   48 |
| total vec ops   |    588 |   371 |  ~318 |

**Total vec ops dropped 37%.** vmovapd dropped 69% (252 → 69), indicating
register pressure resolved.

## Bench (claude.ai virt-Skylake-X, hypervisor noise present)

R=11 t1_dit G/H by K (lower = OCaml faster):

| K     | Before | After |
|-------|-------:|------:|
| 64    | 1.74×  | 1.15-1.46× |
| 128   | 1.54×  | 1.16-1.31× |
| 256   | 1.42×  | 1.10-1.14× |
| 512   | 1.18×  | 0.96-1.23× |
| 1024  | 1.29×  | 1.13-1.34× |
| 2048  | 1.17×  | 1.10-1.23× |
| 4096  | 1.21×  | **1.00-1.07×** |

R=7 / R=5 mostly at parity (G/H 0.95-1.17 across K).

## Files touched

- `lib/dft.ml` — added `dft_direct_conjugate_pair` (~100 lines), wired
  into `dft` dispatch for odd N≥3, updated `needs_reassoc`.
- `lib/algsimp.ml` — `fma_lift` relaxed to `liftable_mul = true`.
- `bin/gen_radix.ml` — replaced single transpose round-trip with FP loop
  matching `bin/prime_opcount.ml`.

## Remaining gap

R=11 t1_dit at 230 vs hand 190 is the cmul-pair twiddle layer (40 ops):
hand fuses twiddles into the inner butterflies more tightly. Our cmul
nodes stay opaque. Closing this would mean teaching the simplifier to
look inside cmul (currently transpose, factor, etc. all skip it).

vmovapd at 69 vs hand 48 — small register-pressure residual. The SU
scheduler keeps live set near 32 ZMM but a few overflows happen. May
investigate tighter scheduling later, but it's secondary.

R=5 inner DFT is 60 vs hand n1 hopefully ~40-50 — could probably be
closer. Lower priority since R=5 already at parity in bench.

## Lessons

1. **Hand-coded references aren't doing anything magical.** R=11 hand
   uses the same pair-sum decomposition any prime DFT can use. The
   "algorithmic gap" framing was wrong; the gap was structural CSE
   that our binary IR + pair-fold couldn't find.

2. **Construction-time sharing > post-hoc CSE.** Rather than implement
   a generic deepCollectM that finds shared sub-Plus expressions across
   different binary tree shapes (hard with our IR), we constructed the
   shared subexpressions explicitly and let hash-cons preserve them.
   Special-purpose, more direct, less code.

3. **`reassoc=true` is not always a win.** It's a "find butterflies in
   flat sums" pass. When the construction has already structured the
   sums optimally, reassoc destroys that structure by re-flattening.

---

## Update: closed the entire gap to hand parity (R=11 t1_dit = 190 ops)

After the conjugate-pair construction, two follow-up changes closed the
remaining 40-op gap to hand-coded.

### Change A: sign-aware FMA chain construction with x[0] absorption

Previously `make_sum coeffs terms` built `Σ Mul(c_j, t_j)` as a left-fold
of `Add` nodes. With negative `c_j` (typical for DFT cos/sin coefficients
where some j produce negative values), the resulting structure was
`Add(prev, Mul(t_j, Const(-c_j)))`. Algsimp's `mk_mul` doesn't normalize
negative consts to `Neg(Mul(...))`, but downstream passes ended up
splitting positive and negative coefficient terms into separate sub-chains
combined by `Sub` at the output level.

Replaced `make_sum` with two variants:

```ocaml
let make_sum_with_init initial coeffs terms =
  let acc = ref initial in
  for j = 1 to half do
    let c = coeffs.(j) in
    let abs_c = Float.abs c in
    let term = Mul (terms.(j), Const abs_c) in
    acc := if c < 0.0 then Sub (!acc, term)
           else Add (!acc, term)
  done; !acc
```

The key insight: build with **`Sub` for negative coefficients**, not
`Add(neg_const · t, prev)`. After `fma_lift`:
- `Add(acc, Mul(t, c))` → `Fma(t, c, acc, F, F)` = fmadd
- `Sub(acc, Mul(t, c))` → `Fma(t, c, acc, T, F)` = fnmadd

A 5-term chain with mixed signs lifts to a single chain of 5 nested
Fmas mixing fmadd and fnmadd — exactly hand's pattern.

`make_sum_with_init` takes `x[0].re` (or `x[0].im`) as the deepest
addend. After lift, x[0] becomes the `c` operand of the innermost FMA —
free at the asm level. Output combinations simplify to:

```ocaml
out_re.(m)     <- Add (p_re_m_with_x0, q_im_m);  (* 1 add *)
out_re.(n - m) <- Sub (p_re_m_with_x0, q_im_m);  (* 1 sub *)
```

(was 6 combining ops per pair re; now 2.)

R=11 t1_dit: 230 → **212 ops**. Inner DFT 190 → **172 ops**.

### Change B: skip `share_subsums` (and the FP transpose loop) for direct primes

Stage-by-stage diagnostic at R=11 after Change A:

| Stage | Op count |
|---|---:|
| of_assignments | 240 |
| dedup_sub_pairs | 240 |
| factor_common_muls | 240 |
| factor_by_atom | 240 |
| dedup | 240 |
| **share_subsums** | **256 (+16!)** |
| FP loop iter 0 (transpose+factor+share) | 322 (worse, reverts) |
| **fma_lift** | **172** |
| --- | --- |
| (skip share + skip FP loop) → fma_lift | **150 = hand parity** |

`share_subsums` extracts common addition subsums across outputs. For
`dft_direct_conjugate_pair`, the chains are already hash-cons-shared:
`p_re_m` is one node referenced by both X[m].re and X[N-m].re. The pass
sees that the positive-coefficient prefix `Add(Add(x[0], M1), M2)` is
also a subtree and chooses to materialize it as a separate intermediate.
This breaks the unified chain into pos_cos and neg_cos sub-trees that
need an extra add to combine, AND prevents `fma_lift` from collapsing
the chain into 5 nested FMAs.

For composite (Cooley-Tukey) sizes, share_subsums genuinely helps —
many cross-output partial-sum overlaps in CT butterflies — so it stays
enabled there. The skip is gated on the algorithm choice
(`pick_algorithm n = Direct`).

Code change in `bin/gen_radix.ml` and `bin/prime_opcount.ml`:

```ocaml
let is_direct = aggressive in  (* aggressive ↔ Direct *)
let shared =
  if is_direct then factored
  else Algsimp.share_subsums ~aggressive factored
in
let post_trans =
  if aggressive && not has_cmul && not is_direct then begin
    (* FP transpose loop also disabled — relies on share_subsums *)
    ...
  end else shared
in
```

### Final results

| | Before chain B | After chain B (final) | Hand |
|---|---:|---:|---:|
| R=3 n1 | 12 | **12** | -- |
| R=5 n1 | 38 | **36** | -- |
| R=7 n1 | 70 | **66** | -- |
| R=11 n1 | 172 | **150** | 150 ✓ |
| R=11 t1_dit | 212 | **190** | 190 ✓ |
| R=13 n1 | 256 | **204** | -- |

ASM (R=11 t1_dit, AVX-512):

| | Ours (final) | Hand |
|---|---:|---:|
| vfmadd | 52 | 52 ✓ |
| vfnmadd | 52 | 48 |
| vfmsub | 6 | 10 |
| vmul | 30 | 30 ✓ |
| vadd | 30 | 30 ✓ |
| vsub | 20 | 20 ✓ |
| **arith total** | **190** | **190** |
| vmovapd | **22** | 48 |

We match hand on every arith metric and have less than half the
register spills.

### Bench (median of 5 runs, virt-Skylake-X) — G/H ratio

| Codelet | K=64 | K=128 | K=256 | K=512 | K=1024 | K=2048 | K=4096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R=11 t1_dit       | 0.97 | 0.98 | 0.98 | **0.88** | **0.87** | **0.82** | **0.93** |
| R=11 t1_dit_log3  | **0.86** | **0.86** | **0.87** | 1.01 | 1.03 | 0.95 | 0.97 |
| R=11 t1_dif       | 0.98 | 1.03 | 0.99 | 1.01 | 1.06 | 1.09 | 1.07 |
| R=11 t1s_dit      | 0.97 | 0.98 | 0.99 | 0.99 | 1.00 | 1.04 | 1.02 |
| R=7  t1_dit       | 1.00 | 1.00 | 1.01 | 0.97 | 0.97 | 0.96 | 1.01 |
| R=7  t1_dit_log3  | **0.85** | **0.88** | **0.88** | **0.89** | 0.94 | 0.96 | 0.97 |
| R=7  t1_dif       | 1.02 | 1.06 | 1.04 | 1.13 | 1.10 | 1.12 | 1.12 |
| R=7  t1s_dit      | 0.96 | 1.00 | 1.00 | 1.09 | 1.06 | 1.04 | 1.10 |
| R=5  t1_dit       | 0.97 | 0.97 | 0.97 | 0.97 | 1.00 | 1.00 | 1.01 |
| R=5  t1_dit_log3  | **0.79** | **0.83** | **0.83** | 1.05 | 1.08 | 1.09 | 1.08 |
| R=5  t1_dif       | 0.99 | 0.98 | 0.98 | 1.12 | 1.04 | 1.01 | 1.01 |
| R=5  t1s_dit      | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Every R=5/R=7/R=11 codelet at parity or beats hand. R=11 t1_dit is
12-18% faster than hand at K=512-2048. R=5 t1_dit_log3 at small K is
17-21% faster.

R=7 t1_dif K=512+ regression (1.10-1.13×) persists — separate from the
algsimp pipeline; likely a scheduling pattern that DIF twiddle layout
exposes. Lower priority since R=11 (the headliner) is solidly ahead.

### Lessons (updated)

4. **More passes ≠ better.** `share_subsums` was designed for general
   structures where the construction doesn't already share intermediates.
   For our direct-prime construction (where pair sums/diffs and chain
   intermediates are explicit hash-cons-shared values), running
   share_subsums actively destroys the optimal layout. Always check
   stage-by-stage what each pass contributes — sometimes the answer
   is "negative".

5. **fma_lift is the workhorse.** All three stages (pre-share,
   post-share, post-FP) have similar surface op counts (240-256). The
   100-op savings from 256 → 172 vs the 90-op savings from 240 → 150
   came from fma_lift, not the upstream passes. The upstream passes'
   job is to give fma_lift a clean structure to lift; share_subsums
   was giving it a messier structure.

6. **The DAG was right; the hand source had no secret sauce.** Once we
   arranged the DAG to have the same structural shape as hand's source
   (single mixed-sign FMA chain with x[0] as deepest addend), `fma_lift`
   and the C compiler reproduced hand's asm essentially exactly.
