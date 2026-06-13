# 59 — r2c vs MKL: the layout/fold investigation, and the hc2c fusion target

Container: Cascade Lake-SP KVM, ~1 effective vCPU, no PMU, L3≈DRAM, oneMKL 2026
AVX-512 build, gcc-13. All timings T=1 (single thread), best-of-many or avg, with
the conservation check where phase-split. Numbers are DIRECTIONAL (this box is blind
to traffic/bandwidth wins; scratch lives in host LLC at cache speed). Ratios measured
in ONE process (shared thermal state) are the trustworthy comparisons; cross-process
absolute numbers drift with MKL's thermal/JIT nondeterminism and must not be compared.

This doc records both the findings and the wrong turns, because several conclusions
reversed under clean measurement and the corrections are the point.


## 0. Starting question

"How can MKL beat VFFT ~1.6x on r2c-256 while VFFT wins elsewhere?" The investigation
took the premise apart and found it was largely a comparison-methodology artifact,
with one genuine residual cause (an unfused Hermitian fold). The real, defensible
results are at the bottom; the path matters because it shows which intuitions held.


## 1. MKL is non-deterministic; only same-process ratios are trustworthy

MKL's DFTI runtime does JIT codelet generation, runtime kernel dispatch, and is subject
to AVX-512 thermal/frequency throttling and out-of-order load effects. Consequence:
its timing is state-dependent. Observed intra-run spread on an identical r2c-256 call:
43.9us to 189us (~330%), dominated by the JIT/warmup transient. Warm steady-state
(warmup excluded) is much tighter: CV ~2-4%.

Rules adopted:
- Never quote a single MKL draw. Average over 10+ warm reps; report mean+median+spread.
- Compare VFFT-vs-MKL only in ONE process (shared thermal state). Cross-process
  absolute comparisons are invalid — they conflate engine speed with thermal drift.
- This bit us twice: an early "MKL pays 2.2x for split r2c" and "VFFT beats MKL r2c
  1.6x split-for-split" were both cross-run/cross-config artifacts. WITHDRAWN. See §4.


## 2. Layout: the comparison-mode confound

VFFT is a SPLIT-complex engine (separate re[]/im[] arrays). MKL supports both split
(DFTI_REAL_REAL) and interleaved (default). Two benches in the tree compare against
DIFFERENT MKL configs, which made the same MKL look like it both won and lost:

| bench | MKL config | character |
|---|---|---|
| bench_pow2_vs_mkl, bench_oop_vs_mkl | split (REAL_REAL), often strided | MKL's slow path |
| bench_headline_3way (bh.c) | interleaved, contiguous, distance N | MKL's fast path |

Putting their numbers in one scoreboard was apples-to-oranges. The honest framing is
same-layout (both split) vs each-on-best-layout.

Layout-isolated measurement (one process, contiguous distance-N, only storage differs):
- MKL c2c: interleaved ~1.2-1.3x faster than split. A real but modest storage penalty.
- MKL c2c strided-split: ~2.4x (storage penalty + strided-access penalty stacked).


## 3. c2c: VFFT-split beats MKL (the solid win)

One-process, same machine, c2c-256 x256, T=1:

| engine | time | vs VFFT |
|---|---|---|
| VFFT (split, 16x16) | ~135 us | — |
| MKL interleaved (its best) | ~217 us | VFFT 1.6x faster |
| MKL split | ~258 us | VFFT 1.9x faster |

VFFT-split beats MKL even on MKL's preferred interleaved layout, ~1.6x. So VFFT's split
c2c kernels are excellent — good enough to overcome split's intrinsic disadvantage and
still win the layout MKL is optimized for. Odd / non-pow2 sizes: VFFT wins by even more
(2.4x-6.8x vs MKL split), MKL's mixed-radix coverage being thinner.

Note (correction): an earlier thread claimed "the OOP engine is the competitive one and
the stride engine loses to MKL." That was the config confound. Direct same-problem A/B
shows the STRIDE engine is ~3x FASTER than the OOP engine at c2c-128; the OOP "wins"
were against MKL's hobbled split-strided config. The stride engine is VFFT's fast path.


## 4. r2c: MKL wins split-for-split by ~1.3x — and why

The trustworthy r2c comparison (one process, both split, native distance H, shared
thermal): MKL r2c-256 ~159us vs VFFT r2c-256 ~208us -> MKL ~1.3x faster. VFFT loses
r2c even in its own native layout.

This is NOT a layout effect (MKL r2c split vs interleaved is a ~tie, ~1.05x — for r2c
the input is real, so the complex-side storage barely matters). It is a FUSION effect.
VFFT's r2c is: inner FFT producing Z in scratch, THEN a separate _r2c_postprocess pass
that re-reads Z[k] and Z[N/2-k] and folds. That separate pass is overhead MKL's fused
real kernel does not pay.

WITHDRAWN claims from this thread (cross-run/config artifacts, corrected by §1 rules):
- "MKL pays 2.2x for split r2c" — FALSE, it's ~1.05x (split≈interleaved for r2c).
- "VFFT beats MKL r2c 1.6x split-for-split" — FALSE, MKL wins ~1.3x in one process.


## 5. Where VFFT's r2c time goes (the phase breakdown)

Built-in VFFT_R2C_PROFILE splits r2c into pack / inner / fold. Single-thread,
conservation sum/wall ~1.00 (trustworthy). r2c-256, T=1, two valid factorizations:

| plan | 1st stage (pack) | inner | fold | total |
|---|---|---|---|---|
| (8,16) + log3 (r8 first) | ~53us (49%) | ~31us (28%) | ~25us (23%) | ~110us |
| (16,8) + log3 (r16 first) | ~58us (56%) | ~19us (18%) | ~27us (26%) | ~104us |

(Absolute totals here are ~104-110us in a faster thermal regime than §4's ~208us run;
the PROPORTIONS and the relative ordering are the stable finding, not the absolutes.)

Findings:
1. "pack" is NOT a memory shuffle. The path uses _r2c_fused_first_stage, so this phase
   is the radix butterfly of the first FFT stage with the real-input packing folded in.
   It is real FFT work over the entire N/2 x K dataset — the heaviest single pass.
2. It does NOT spill. The radix-8 first-stage codelet asm: 90 zmm ops, 0 ymm, 0 xmm,
   0 scalar, ZERO stack traffic, 21/32 registers. Fully SIMD. The generator provenance
   says "PRESSURE: SPILLS (peak_live 33 > budget 28)" but that is the generator's
   CONSERVATIVE model; gcc re-scheduled and fit it with no actual spills. Not the problem.
3. The fold is ~25-27us and is FACTORIZATION-INVARIANT (same in both rows). No
   factorization choice touches it — it operates on the final spectrum regardless.


## 6. Factorization + variant tuning (what helped, what is broken)

Correctness-gated (per-process, K=8, vs brute DFT-256), r2c-256:

| factorization | stage-2 variant | correctness | note |
|---|---|---|---|
| (8,16) | log3 | PASS 1.1e-12 | |
| (16,8) | log3 | PASS 1.2e-12 | BEST valid |
| (16,16) | log3 | FAIL err 34 | terminator bug, §7 |
| (8,32) | log3 | FAIL err 44 | terminator bug, §7 |
| (4,64) | log3 | FAIL err 33 | terminator bug, §7 |

Wins:
- log3 on the twiddled second stage: ~8% over flat (the planner does not currently
  select it; capture in the wisdom restructure).
- (16,8) beats (8,16) by ~7% (~104 vs ~110us). radix-16 FIRST is the better split: the
  bigger radix in stage 0 shrinks the inner (31->19us) more than it grows stage 0.
  Best VALID r2c-256 plan: (16,8) + log3.

The (4,64) trap (a process lesson): (4,64) full-r2c TIMED faster (~174 vs ~187us) and
was briefly reported as a win. It FAILS correctness (err ~33). The fast timer was the
broken terminator doing wrong/less work. Proven by the inner-only A/B: the (4,64) INNER
c2c-128 is ~97us vs (8,16) ~79us — radix-64 second stage SPILLS (128 reg) and is the
SLOWEST inner. A correct (4,64) would be slower, not faster. Lesson re-learned: gate
correctness BEFORE quoting any perf number.


## 7. Latent correctness bug: r2c only valid for radix<=16 two-stage shapes

The r2c terminator (_r2c_postprocess + _r2c_compute_perm) reads the inner FFT output via
a digit-reversal perm. It is correct only for a narrow set of factorization shapes —
(8,16) and (16,8) PASS; (16,16), (8,32), (4,64) produce SILENT WRONG OUTPUT (not a
crash — garbage values, err ~30-65). The c2c inner is fine for all of these (radix-64
works as a first stage, radix-32 works as a second stage); it is the terminator's
group-walk / perm interaction that breaks for these shapes.

This is a latent landmine: a planner could reasonably pick (16,16) or (8,32) for r2c and
get wrong answers with no error. ACTION: constrain the r2c planner to the verified shapes
until the terminator group-walk is generalized, OR fix the terminator (a shared root
cause across all three failing shapes — fixing it would unlock all three at once).
Fixing it for SPEED is not worth it (radix>=32 second stages overshoot the radix-16
register sweet spot); fixing it for CORRECTNESS/robustness is.


## 8. The fold is the gap — FFTW shows exactly how to remove it (hc2c)

The fold (~25-27us, factorization-invariant) is the one structural lever left. FFTW's
rdft source shows the canonical fix: FFTW has NO separate terminator. Its r2c is
cld (real-input FFT to halfcomplex) + cldw (hc2c codelet). The hc2c codelet IS the last
stage AND does the Hermitian recombination in one fused pass.

Decoded from FFTW's hc2cf_4 (genfft gen_hc2c output):

Signature: `hc2cf_4(Rp, Ip, Rm, Im, W, rs, mb, me, ms)` — Rp/Ip are +frequency, Rm/Im
are the MIRROR -frequency, and the loop walks Rp FORWARD while walking Rm BACKWARD
(Rp+=ms, Rm-=ms). Each (k, N/2-k) mirror pair is processed together.

Step 1 — twiddle that COMBINES +/- frequency inputs (Hermitian symmetry exploited
during the twiddle, not after):
```
a = W_re*Rp_leg + W_im*Rm_leg     (FMA)
b = W_re*Rm_leg - W_im*Rp_leg     (FNMS)
```
Step 2 — radix-r butterfly writing BOTH forward and mirror outputs:
```
out_plus  = sum + cross
out_minus = sum - cross      (conjugate-symmetric partner)
```

The efficiency trick (verified: hc2cf_4 is 5 FMA + 4 FNMS, only 4 plain muls): the r2c
recombination's /2 and -i constants are PRECOMPUTED INTO THE TWIDDLE TABLE W, so the
codelet is almost pure FMA — no explicit divide or multiply-by-i.

Why this removes our fold: FFTW reads each datum once, writes each output once. The
last-stage twiddle + radix butterfly + Hermitian fold are one codelet. No intermediate
Z is materialized; our scratch round-trip + re-read is the ~25-27us pass FFTW lacks.

FFTW ships hc2c for radices {2,4,6,8,10,12,16,20,32}, plus an hc2cf2_* family {4,8,16,
20,32}. IMPORTANT CORRECTION (an earlier draft of this doc, and the framing that led to
it, were wrong): hc2cf2 is NOT a register-blocked / two-pass variant. The codelet
generation headers settle it — hc2cf2_8 is generated by the SAME gen_hc2c.native with
exactly two extra flags: `-twiddle-log3 -precompute-twiddles`. It is a TWIDDLE-POLICY
variant (log3-compressed, precomputed twiddles), trading more arithmetic (hc2cf2_8: 74
add / 50 mul; hc2cf_8: 66 / 36) for fewer/structured twiddle loads. Confirmed: grepping
every hc2c codelet for scratch/buf/PASS/spill seams returns NOTHING — FFTW has NO
register-blocked hc2c at all. It caps the family at radix-32 and trusts `-variables 4`
plus the C compiler's register allocator. And `-twiddle-log3` is precisely the log3 from
§6: the entire content of the hc2cf2 family is "hc2c + log3," which a flag (or, on our
side, the same emitted form fed to gcc) composes.


## 9. Design spec: VFFT hc2c last-stage codelet family (= model-b, generalized)

This is exactly what model-b (dft_r2c_term_laststage) already prototyped (correct at
2.8e-14, parked opt-in for radix-8). FFTW validates the architecture and adds three
refinements:

1. Bake the fold constants (/2, -i) into the twiddle table so emission is FMA-dense.
2. Use the forward/backward mirror-pair pointer walk as the canonical access pattern.
3. Generate it as a distinct hc2c codelet FAMILY, not a one-off radix-8 fusion.

Targets, in order:
- hc2c_r8 first — the (16,8) winner ends in radix-8. radix-8 fused should fit registers.
- hc2c_r16 — for (8,16).
- log3 variant: generate the log3 form too (FFTW's hc2cf2 = hc2c + log3, and §6 measured
  log3 winning ~8% on the twiddled stage). On our side the C compiler composes it from the
  same emitted DAG, so it is a variant flag, not a separate codelet structure.
- Blocking: do NOT assume it is needed. FFTW does NOT block hc2c at all (family caps at
  r32, no scratch seam in any hc2c codelet — see §8 correction). r8/r16 fused should fit
  the register file; emit them flat first and check the asm for actual spill traffic (as
  §5 did for the radix-8 first stage, which the generator's model wrongly flagged as
  spilling but gcc fit in 21/32 registers). Only reach for the doc-58 PASS-1/PASS-2 seam
  if a specific large radix shows REAL stack traffic in the compiled asm — measured, not
  predicted by the generator's conservative liveness model.

Expected payoff: the ~25-27us fold pass goes to ~0 (fused into the last stage). On metal
this is also a deleted scratch round-trip (a traffic win this container cannot see).
Best valid r2c today is (16,8)+log3 ~104us; removing the fold targets ~80us, approaching
MKL's fused real kernel.


## 10. Net state

- c2c: VFFT-split BEATS MKL (1.6x vs interleaved, 1.9x vs split); odd sizes 2.4-6.8x. Solid.
- r2c: MKL beats VFFT ~1.3x split-for-split. Cause = unfused Hermitian fold, NOT layout,
  NOT kernel quality (the inner FFT is competitive; the fold is a separate pass).
- Best valid r2c plan: (16,8) + log3, ~104us (was quoting (8,16)+log3 ~110us; (16,8) is ~7% better).
- Latent bug: r2c silently wrong for (16,16),(8,32),(4,64) — constrain planner or fix terminator.
- The fix that closes the MKL r2c gap: generalize model-b into an FFTW-style hc2c codelet
  family (twiddle+butterfly+fold fused, constants baked into W), with a log3 variant
  (FFTW's hc2cf2 = hc2c + log3). Do NOT assume register-blocking is needed — FFTW does not
  block hc2c; emit flat and only block if the compiled asm shows real spill traffic.

All conclusions here rest on one-process / correctness-gated measurements. Absolute
microsecond values drift with thermal state and MKL nondeterminism; the ratios, the phase
proportions, and the correctness verdicts are the stable, trustworthy outputs.
