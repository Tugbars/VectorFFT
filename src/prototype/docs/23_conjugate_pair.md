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
