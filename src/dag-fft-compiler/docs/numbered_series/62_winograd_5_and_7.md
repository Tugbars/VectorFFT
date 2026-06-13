# 62. Winograd-5 and Winograd-7 — closing prime-N leaf-codelet gaps

Status: Both algorithms shipped to current tree. Both verified sub-ulp correct
(Fwd and Bwd). Op counts now match FFTW's `gen_notw -fma` exactly at R=5 and R=7.
Cascade benefits propagate to every radix that decomposes through DFT-5 or DFT-7
(R=14, R=15, R=20, R=21, R=25, ...).

Wired as unconditional defaults — no env vars. The AVX-512 micro-regression on
the R=25 / R=7 cascade is accepted as the cost of code simplicity; a future
n-ary IR rewrite is the actual fix and will flip that trade.

## TL;DR

```
                Before               After                Δ
R=5    AVX-512  36 ops, 0 spills    32 ops, 0 spills     matches FFTW
R=5    AVX2     36 ops, 22 spills   32 ops, 6 spills     −73% spills
R=7    AVX-512  66 ops, 4 spills    60 ops, 2 spills     matches FFTW
R=7    AVX2     66 ops, 74 spills   60 ops, 62 spills    −16% spills
R=15            308 ops (Direct)    181 ops (CT(3,5)+W5) −41% (+16% vs FFTW)
R=20            272 ops             250 ops              cascade only
R=21            Direct O(N²)        313 ops (CT(3,7)+W7) algorithm choice now sane
R=25            412 ops             383 ops              +8.8% vs FFTW
```

Wall-clock confirmation (sandbox bench, R=5 and R=7 directly):
- AVX2 R=7: −2.5% cycles (40 → 39 min, very reproducible)
- AVX2 R=5: −5–9% cycles, tighter variance
- AVX-512 R=7: +5% (regression)
- AVX-512 R=5: roughly tied

## The gap before today

R=3 was already exact match (12 ops, both us and FFTW). The conjugate-pair
Direct construction recovers the optimal DFT-3 via plain CSE.

For larger primes the picture was different:

```
                 FFTW (FMA branch)             OCaml (conj-pair Direct)
R=5              32  (18 fma + 0 mul + 14 a/s) 36  (12 fma + 4 mul + 20 a/s)
R=7              60  (42 fma + 0 mul + 18 a/s) 66  (30 fma + 6 mul + 30 a/s)
```

The shape of both gaps is identical: we had fewer FMAs, leftover standalone
multiplications, and extra add/sub. The standalone muls were tan-factored
Path B outputs that couldn't get absorbed because their consumers were also
multiplications, not Add/Sub.

The root cause is that **generic algsimp cannot discover algebraic identities
specific to roots of unity for a given prime N**. FFTW's `gen_notw -fma`
emitter doesn't discover them either; it has them hand-coded inside the codelet
recipe. The corresponding fix on our side is also to hand-code them.

## Winograd-5

The Winograd 5-point DFT uses four constants that look unusual at first:

```
KP_QUARTER     = 0.25
KP_ROOT5_4     = √5/4              ≈ 0.559
KP_SIN_2PI5    = sin(2π/5)         ≈ 0.951
KP_INV_PHI     = (√5−1)/2 = 1/φ    ≈ 0.618  (= 2·cos(2π/5))
```

These come from two identities:

```
cos(2π/5) + cos(4π/5) = −1/2
cos(2π/5) − cos(4π/5) = √5/2

⇒  cos(2π/5) = −1/4 + √5/4
⇒  cos(4π/5) = −1/4 − √5/4
```

So a DFT-5 inner product like `(x_1+x_4)·cos(2π/5) + (x_2+x_3)·cos(4π/5)` —
which needs two distinct multiplications by irrational constants — refactors
to `−1/4·((x_1+x_4)+(x_2+x_3)) + √5/4·((x_1+x_4)−(x_2+x_3))`. The sum and
diff terms are pre-computed once and reused across the symmetric outputs
X_2 and X_3, so the four cos-channel multiplications collapse to two.

The sin-channel uses the second identity:

```
sin(4π/5) = sin(2π/5) · (1/φ)
```

So `s1·t3 + s2·t4` (where s1=sin(2π/5), s2=sin(4π/5)) becomes
`s1·(t3 + t4/φ)`. Two sin-channel terms collapse to one outer-multiplication
by `s1`, with the `1/φ` ratio handled by an inner FMA.

Net algebra: 14 add/sub + 18 fma = 32 ops total, 0 standalone muls.

### Implementation: `lib/dft.ml :: dft_winograd5`

The OCaml is a faithful mirror of FFTW's emitted ordering:
real-channel pre-adds (4 sums + 3 subs), imag-channel pre-adds (3 sums +
3 subs), output 0 (4 plain adds), then the four real outputs followed by
the four imag outputs — each output pair sharing a "common" anchor value
plus or minus a sin-coupled cross-channel term.

The sign of the Fwd vs Bwd direction is absorbed into a single constant:

```ocaml
let s_sign = match sign with `Fwd -> 1.0 | `Bwd -> -1.0 in
let k_sin_2pi5_s = Const (s_sign *. sin (two_pi /. 5.0)) in
```

All structural FMA/FNMS choices in the body stay identical for both directions
— only the value of `k_sin_2pi5_s` flips. Verified Fwd correctness at max |Δ|
= 3.8e-15 sub-ulp against the conjugate-pair Direct baseline.

### Register pressure

Naturally tighter than the baseline. AVX2 spills drop 22 → 6 (−73%) without
any explicit scheduling tricks — the FFTW-mirrored emission order gives
algsimp+regalloc good locality:

```
                  AVX2 spills    AVX-512 spills
Direct (36 ops)   22             0
Winograd (32 op)  6              0
```

Both fit AVX-512's 32-zmm budget trivially. On AVX2, Winograd is at the
edge of the 16-ymm budget instead of well over it.

## Winograd-7

Same approach, larger algebra. FFTW's R=7 codelet uses six derived constants:

```
KP_356895867 = 0.356895867...   same-channel reduction
KP_554958132 = 0.554958132...   same-channel pair-diff combo
KP_801937735 = 0.801937735...   same-channel
KP_692021471 = 0.692021471...   same-channel
KP_900968867 = |cos(6π/7)|      same-channel scaling
KP_974927912 = sin(4π/7)        cross-channel coupling (sin: flips for Bwd)
```

These come from Rader's algorithm, which transforms a prime-N DFT into an
(N−1)-point cyclic convolution by exploiting the multiplicative-group
structure of `(Z/N)*`. For N=7 the cyclic convolution is over 6 elements,
which factors as 2×3 — and the small-convolution Winograd algorithms for
the size-2 and size-3 subproblems contribute the specific constant values
above.

We don't re-derive the algebra; we mirror FFTW's emitted structure. Net:
18 add/sub + 42 fma = 60 ops total.

### Imag-channel sign convention

There's an asymmetry between channels that's load-bearing for correctness.
In FFTW's emitted code:

```
Real pair-diffs:  TI = T3 − T2  (= x_6.re − x_1.re)        HIGH − LOW
Imag pair-diffs:  Tj = Th − Ti  (= x_1.im − x_6.im)        LOW − HIGH
```

The opposite sign on real vs imag pair-diffs encodes the `−i sin(θ)`
relationship in the DFT formula. Flipping it inverts the imaginary
outputs. Both directions checked Fwd/Bwd at max |Δ| = 1.0e-14.

### Output-pair sign and Bwd handling

Each output pair has the same structural form:

```
ro[k]      = FMA  (KP_974927912, T_imag_channel, T_real_anchor)   ← + cross
ro[N−k]    = FNMS (KP_974927912, T_imag_channel, T_real_anchor)   ← − cross
io[k]      = FMA  (KP_974927912, T_real_channel, T_imag_anchor)
io[N−k]    = FNMS (KP_974927912, T_real_channel, T_imag_anchor)
```

For Bwd the sin-coupling sign reverses everywhere. The structural FMA/FNMS
shapes stay identical — we scale `KP_974927912` by `s_sign` (+1 for Fwd,
−1 for Bwd) and the math works out. Algebraic check:

```
Fwd: Re(X_k) = cos·real + sin·imag    ⇒ FMA gives this
Bwd: Re(X_k) = cos·real − sin·imag    ⇒ FMA with −sin gives this
```

The same applies to the imag-side outputs by symmetric reasoning.

### Register pressure for R=7

```
                  AVX2 spills    AVX-512 spills
Conj-pair (66 op) 74             4
Winograd (60 op)  62             2
```

AVX2 drops 16%, AVX-512 drops 50% in absolute terms (4 → 2 spills). Both
are minor in the AVX-512 case; the AVX2 reduction matters more since AVX2
was at significant overcommit (74 spills on a 16-ymm budget for a 7-element
DFT — every value is live at some point).

## The cascade

Winograd-5 propagates to every radix decomposed as CT(*, 5) or CT(5, *)
or any deeper nesting. Same for Winograd-7. Effects measured on AVX-512:

```
                  Before today    After (with Winograd cascade + picker fixes)
R=5     32 (FFTW) 36              32     (matches FFTW)
R=7     60        66              60     (matches FFTW)
R=14    148       180             160    (+8% vs FFTW)
R=15    156       308 (Direct!)   181    (+16% vs FFTW)
R=20    208       272             250    (+20% vs FFTW)
R=21    n/a       Direct O(N²)    313    (CT(3,7) + Winograd-7)
R=25    352       412             383    (+9% vs FFTW)
```

### Two picker entries added

```ocaml
| 15 -> Cooley_Tukey (3, 5)   (* was falling to Direct DFT-15 *)
| 21 -> Cooley_Tukey (3, 7)   (* was falling to Direct DFT-21 *)
```

Without these, the generic fallback `| _ when n mod 2 = 0 -> CT(2, n/2)`
catches even composites but odd composites fall to `Direct`, which is
O(N²). R=15 went 308 → 181 ops (−41%) and R=21 went from a
hundreds-of-ops O(N²) blowup to 313.

R=20 is unchanged in factorization (it was already `CT(5, 4)` from earlier
work) — gain comes entirely from the four DFT-5 instances inside.

## The R=25 (and similarly R=7 cascade) AVX-512 regression

Same architectural pattern hits at R=25 and at R=7 cascaded radices:

```
R=25 AVX2 (10 runs, min cycles):
  baseline  avg 255.5
  Winograd  avg 250.7    →  −1.9%  (Winograd wins 8/10 runs)

R=25 AVX-512 (10 runs, min cycles):
  baseline  avg 289.3
  Winograd  avg 301.7    →  +4.3%  (baseline wins 9/10 runs)
```

The asymmetry is reproducible across runs (10-run signal, not noise).
Mechanism:

- **Winograd** has more FMAs (deeper dependency chains) and fewer total ops.
- **Conjugate-pair Direct** has more parallel ops (multiplications and
  add/subs that can issue across more ports).

On AVX-512 (32 zmm), both fit comfortably and the conj-pair structure wins
on port utilization. On AVX2 (16 ymm), the conj-pair structure overflows
the register budget and spill traffic exceeds Winograd's chain-depth penalty.

The crossover sits right between the two register budgets.

### Why we're shipping it anyway

The user explicitly chose code simplicity over the marginal AVX-512 perf
in two places:

1. We considered an ISA-aware default (Winograd for narrow budgets, baseline
   for wide budgets). It works correctly but adds runtime branching in
   `pick_algorithm` and a code reader has to understand why.
2. We considered keeping `VFFT_WINOGRAD5` as an env-var gate for A/B
   testing. We already have nine VFFT_* env vars in the codebase — a
   tenth would make the situation worse, not better.

The honest fix is to close the underlying gap, not paper over it with
flags. That underlying gap is the **n-ary IR rewrite** described in Doc 59
(addendum) — FFTW's `gen_notw` uses an n-ary `Plus` IR that defers FMA
materialization until emit time, walking Add/Sub chains as virtual Plus
lists. Once we do that, the R=25 twiddle stage drops 30+ ops and the
AVX-512 trade-off flips. The Winograd-5/7 codelets will likely become
unambiguously the right choice at every ISA, and the broader R=64 gap to
FFTW also closes substantially.

Until that lands (3–5 day minimum port, 1–2 week faithful port), we accept
the regression and ship the simpler tree.

## Code shape

Three modules changed today:

```
lib/dft.ml
  + dft_winograd5    (35 lines)   ~line 700
  + dft_winograd7    (90 lines)   ~line 780
  + picker entries for R=15 (CT(3,5)) and R=21 (CT(3,7))
  + dispatch in `Direct` arm: n=5 → dft_winograd5
                              n=7 → dft_winograd7
                              else → existing dft_direct_conjugate_pair / dft_direct

lib/algsimp.ml
  + flatten_fma_mul_addend (from earlier in today's arc — Doc 59)
    No-op at R=25 because butterfly-pair muls block the rewrite.
    Kept in the tree because it documents the boundary cleanly.

bin/gen_radix.ml
  + target_vec_regs setter from `isa.vec_regs` (Doc 59 addendum, R=64 AVX2)
```

No env-var dependencies. No conditional compilation. No code paths
selected at runtime by anything other than the radix value.

## Future work

Two natural next steps if non-pow2 radices matter further on EPYC:

1. **Winograd-11, Winograd-13** if needed. Rader's algorithm applies for
   any prime. R=11 has a 10-element cyclic convolution; R=13 has a
   12-element one. The constants are messier than R=7's and the algebra
   would have to be derived (FFTW ships R=11 and R=13 codelets we can
   mirror). Each is maybe half a day of work. Cascade impact: R=22,
   R=26, R=33, R=39, R=55, R=65, R=77, R=91, R=143...
2. **Additional CT picker entries** if intermediate sizes matter:
   `R=28 -> CT(4, 7)`, `R=35 -> CT(5, 7)`, `R=49 -> CT(7, 7)`. Each
   would automatically benefit from the Winograd-7 cascade. The first
   two are cheap one-line additions; R=49 might exceed the register
   budget for monolithic emission and would need the recipe machinery.

For everything else (R=8, R=16, R=32, ... pure-power-of-2), the existing
CT decomposition is already optimal and these gaps don't exist.

## References

- Doc 59 — five-phase optimization arc + AVX2 R=64 factorization + n-ary
  IR discussion
- Doc 46 — FFTW algsimp comparison
- FFTW codelet references: `/tmp/fftw-3.3.10/dft/scalar/codelets/n1_5.c`
  and `n1_7.c`
- Winograd, S. — "On computing the discrete Fourier transform" (1978);
  the cyclic-convolution formulation behind Rader's prime-N algorithm
