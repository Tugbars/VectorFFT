# Cross-uarch check + correct fence/pin bench recipe (addendum)

Addendum to `fence_pin_decomposition.md`. Records a cross-microarchitecture
re-measurement of the fence/pin policy and pins down the exact bench
methodology so the result is reproducible and not re-derived from scratch.

## TL;DR

On a **Cascade Lake KVM guest** (single vCPU, the container this was run
in), the fence/pin policy from `fence_pin_decomposition.md` **does not
reproduce**. The fence is neutral-to-harmful across both n1 and t1
codelets at R ∈ {8,16,32,64} AVX-512; in particular the doc's flagship
case (R64 t1 AVX-512, reported -19% from the fence) measures **dead even
(-0.7% / +0.9%)** here. This is consistent with the doc's own §6
portability caveat ("older uarchs may differ; not tested"). The two-rule
policy should be treated as **host-specific** (validated on a
Sapphire-Rapids-class host) and re-checked on each target uarch before
being trusted there.

This is NOT a correctness problem (all variants are bit-identical math,
see gate below) and NOT a codelet bug. It is a measurement-host result.

## Data (this container: Cascade Lake KVM, gcc-13, in-place rio codelets)

ns per transform; min-of-11 trials, median-of-3 whole-binary runs,
1000-call warmup, reps auto-scaled so the codelet body dominates,
200 ms pacing between runs.

n1 codelets:

| me  | R   | M-off  | M-fence | M-on   | fence vs off | best   |
|----:|----:|-------:|--------:|-------:|-------------:|--------|
| 64  | 8   | 1.240  | 1.241   | 1.242  | +0.1%        | tie    |
| 64  | 16  | 3.562  | 3.575   | 3.893  | +0.3%        | M-off  |
| 64  | 32  | 11.112 | 10.835  | 10.961 | -2.5%        | M-fence|
| 64  | 64  | 38.149 | 44.174  | 39.409 | +15.8%       | M-off  |
| 256 | 8   | 1.251  | 1.251   | 1.244  | -0.0%        | tie    |
| 256 | 16  | 10.482 | 11.605  | 11.626 | +10.7%       | M-off  |
| 256 | 32  | 24.535 | 26.515  | 22.170 | +8.1%        | M-on   |
| 256 | 64  | 66.470 | 73.160  | 68.259 | +10.1%       | M-off  |

t1 codelets (the doc's strong-fence kind):

| me  | R   | M-off  | M-fence | M-on   | fence vs off | best   |
|----:|----:|-------:|--------:|-------:|-------------:|--------|
| 64  | 16  | 5.204  | 5.565   | 5.409  | +6.9%        | M-off  |
| 64  | 32  | 17.399 | 17.825  | 17.717 | +2.4%        | M-off  |
| 64  | 64  | 47.822 | 47.499  | 51.091 | -0.7%        | M-fence|
| 256 | 16  | 11.712 | 11.006  | 11.425 | -6.0%        | M-fence|
| 256 | 32  | 27.604 | 29.820  | 28.365 | +8.0%        | M-off  |
| 256 | 64  | 81.721 | 82.443  | 81.766 | +0.9%        | M-off  |

Reading: no consistent fence win in either kind; the largest fence
effects are ±6-8% and flip sign with working-set size (R16 t1: +6.9% at
me=64, -6.0% at me=256). R64 t1 AVX-512, the doc's -19% headline,
is within noise here.

## Why it diverges (most likely)

Ranked:

1. **Host / uarch.** The doc's host is Sapphire-Rapids-class (per the OOP
   codelet provenance header: `Tier C uarch: sapphire_rapids_avx512`).
   This container is Cascade Lake (Skylake-X family) under KVM, single
   vCPU. The doc's §6 explicitly lists this uarch class as untested. The
   fence works by constraining gcc's post-RA scheduler to keep the
   generator's SU+GH order; how much that matters depends on the uarch's
   scheduling/rename behavior and on gcc's codegen for that target, both
   of which differ here. A single-vCPU KVM guest also raises the
   measurement floor.
2. **Not the methodology.** Body-dominated regime, warmup, min-of-11,
   median-of-3, pacing, and one-codelet-per-binary (the doc's clean
   config) were all matched. An earlier me=8 run was noisy; me=64/256
   removed that and the verdict held.
3. **Not the kind.** t1 (the doc's strongest-fence kind) reproduces the
   same null result as n1, so it is not an n1-only artifact.
4. **Not correctness.** All three M-modes are bit-identical per radix
   (they differ only in register pinning + scheduling fences, not in the
   arithmetic). Verified to ~1e-12 (see gate).

## Practical consequence

- The committed two-rule policy (pin: log3∧AVX512∧R≤32; fence: on except
  n1∧AVX2∧R∈{8,16}) is host-calibrated. On this Cascade Lake guest a
  "fence-off / M-off everywhere" choice would be as good or better for
  the n1/t1 AVX-512 sizes tested. Do NOT change the shipped policy based
  on a KVM-guest measurement; re-measure on the real target (EPYC,
  i9-14900KF) and decide there.
- Container timing is directional only (no PMU, L3≈DRAM, single vCPU).
  Binding numbers require metal.

---

# Correct fence/pin bench recipe

The reproducible procedure that produced the table above. Removes the
two methodology traps hit earlier (the me=8 call-overhead regime, and a
broken Parseval-only correctness check).

## 0. The three M-modes

```
M-on    : VFFT_PIN_FORCE=1   gen_radix.exe R [flags] --in-place --isa ISA --su --emit-c   (pin + fence)
M-off   : VFFT_NO_REGALLOC=1 gen_radix.exe R [flags] --in-place --isa ISA --su --emit-c   (neither)
M-fence : take M-on output, strip the pin clause, keep the fence:
          sed -E 's/ asm\("zmm[0-9]+"\)//g' mon.c > mfence.c
```
- n1 kind: no `--twiddled`. t1 kind: add `--twiddled`.
- Verify after generating: M-on has many `asm("zmm` (pins); M-fence has
  zero pins but the same count of `asm volatile` (fences); M-off has
  zero of both.
- All three share one signature per (R,kind): the 6-arg in-place rio
  form `(rio_re, rio_im, tw_re, tw_im, ios, me)`. One driver per (R,kind)
  drives all three.

## 1. Correctness gate FIRST (never time an unverified codelet)

Strict per-column reference DFT, natural output order (single-radix N
emits in natural order; confirmed empirically — the output permutation
came back identity). Layout: element j of transform at `rio[j*ios + k]`,
ios = me, me a multiple of the ISA width (8 for AVX-512, 4 for AVX2).

```c
// for each column k in [0,me): reference-DFT that column's N inputs,
// compare against rio[n*ios + k] in natural order n. PASS if max_abs < 1e-9.
```
All three M-modes must report the SAME error per radix (they are the
same arithmetic). If they differ, something is broken — stop.

Pitfalls that bit us:
- Do NOT use a Parseval/energy proxy as the gate; it passes broken
  permutations.
- Do NOT compare all columns against a single column-0 reference; each
  column has distinct input, so it needs its own reference.

## 2. Timing harness (body-dominated, low-noise)

```c
size_t me = 64 or 256;          // NOT 8 — me=8 is call-overhead-bound, pure noise
size_t ios = me;
long reps = 2000000 / me; if (reps < 200) reps = 200;   // auto-scale: body dominates
for (w=0; w<1000; w++) FN(...);                          // warmup 1000 calls
double best = inf;
for (tr=0; tr<11; tr++) {                                // min-of-11
    t0 = clock_gettime(MONOTONIC);
    for (r=0; r<reps; r++) FN(re,im,tw_re,tw_im,ios,me);
    t1 = clock_gettime(MONOTONIC);
    ns_per = (t1-t0)/reps/me;                            // ns PER TRANSFORM
    best = min(best, ns_per);
}
report best;
```
- One codelet per binary (the doc's clean config; avoids the i-cache /
  branch-predictor cross-talk the doc saw with 8-12 functions per binary).
- Whole binary run 3x; take the **median** of the 3 `best` values.
- **200 ms sleep between every run** (thermal/settle).
- t1 needs a populated twiddle table (`tw[g*me+k]`, g<R, k<me) with
  unit-modulus values; n1 barely touches tw. Timing is twiddle-value-
  independent (same work), but use sane values so it isn't a denormal
  path.

## 3. Compile flags

```
gcc-13 -O3 -march=native -ffp-contract=fast   (codelet + driver)
AVX-512: me multiple of 8.   AVX2: me multiple of 4.
```
me=8 on AVX-512 against a K=4-shaped buffer overruns (8-wide ops, 4
valid lanes) — heap corruption. Always me ≥ width and a multiple of it.

## 4. Report

Per cell: M-off / M-fence / M-on in ns/transform, plus
`fence_vs_off = (fence-off)/off` and `on_vs_off = (on-off)/off`, and the
min as `best`. Treat <5% as noise on a constrained host; only >10%
effects are directional. Binding verdict requires PMU + real metal.
