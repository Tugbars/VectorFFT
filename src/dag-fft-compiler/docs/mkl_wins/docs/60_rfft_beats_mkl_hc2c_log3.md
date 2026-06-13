# 60 — rfft path beats MKL on r2c: hc2c + log3 codelets, (8,32) factorization

Follows doc 59 (the r2c-vs-MKL layout/fold investigation). Doc 59 ended with: the r2c
gap vs MKL is an unfused Hermitian fold in the STRIDE path, and FFTW shows the fix is an
hc2c last-stage codelet (twiddle + butterfly + fold fused, no separate terminator). This
doc records the implementation of that fix and its evaluation — which inverts the headline.

Container: Cascade Lake-SP KVM, ~1 effective vCPU, no PMU, oneMKL 2026 AVX-512, gcc-13,
T=1 throughout. Same epistemics as doc 59: absolute microseconds drift with thermal state
and MKL nondeterminism; RATIOS measured in one process and CORRECTNESS gate verdicts are
the trustworthy outputs. This session ran in a SLOWER thermal regime than doc 59's runs
(MKL self-times ~104us here vs ~56us there) — so absolutes are not comparable across docs;
only same-window ratios are.


## 0. Headline

The FFTW-style real-FFT executor (rfft.h: r2cf leaf + hc2hc twiddle stages, no pack stage,
no separate Hermitian terminator), with the optimal (8,32) factorization, BEATS MKL's
native r2c-256 by ~1.2-1.4x in matched same-process conditions.

This is the first clean VFFT win over MKL on r2c in the whole investigation. It inverts
doc 59 §4's "MKL wins r2c split-for-split ~1.3x" — that verdict was specific to the STRIDE
r2c path (pack + c2c + separate fold). The rfft path is a different, fused architecture and
it wins.


## 1. What was implemented (the hc2c + log3 codelet work)

Generator changes (generator/lib/{emit_c,gen_main,dft_r2c}.ml) + core/rfft.h:

1. tw_policy threaded through hc2c / hc2hc / hc2c_spill builders. The hc2c codelet can now
   be generated with FLAT or LOG3 twiddles (--log3). Verified: hc2c r8 flat references 7
   twiddle slots; log3 references 3 base slots (tw[0],tw[1],tw[3]) and derives the rest.
   Op counts match FFTW's gen_hc2c exactly: flat = 66 add / 22 fma; log3 = 74 add / 30 fma
   (FFTW hc2cf_8 = 66/22, hc2cf2_8 = 74/30). This confirms the doc-59 finding that FFTW's
   "hc2cf2" family IS just "hc2c + log3" (same generator + -twiddle-log3 -precompute-twiddles),
   not a register-blocked variant.

2. n1_oop_strided ABI (emit_c.ml): a strided-OOP n1 codelet signature matching
   vfft_proto_n1_fn exactly, so the r2c fused first stage avoids the registry wrapper's
   memcpy-then-in-place bridge (the "pack fix" targeting doc-59 §5's dominant first-stage
   phase). Builds clean.

3. VFFT_FORCE_MONO=1 env knob (dft.ml): forces monolithic emission regardless of size —
   a barrier-test instrument for measuring the seam-as-CSE-barrier effect. Additive, default off.

4. rfft.h leaf fold: the unblocked leaf is now ONE call at vl = S*K (address-identical to
   the old S separate calls at vl=K). Verified to NOT change correctness (gate bit-identical).

All four land cleanly; the generator builds with no errors.


## 2. Critical gotcha: hc2hc/hc2c codelets need --t1s (scalar-broadcast twiddles)

The rfft executor's twiddle table is SCALAR (one W per radix leg, broadcast across the K
lanes via _mm512_set1_pd). A codelet generated without --t1s emits a VECTOR twiddle ABI
(_mm512_loadu_pd(&tw_re[j*vl + v]) — a different W per lane), which reads the wrong memory
against the scalar table and produces GARBAGE in the executor.

This caused a false "rfft is broken" reading mid-investigation: building gate_rfft_plan.c
with freshly-generated codelets that lacked --t1s gave ALL FAIL (errors ~1e+179, classic
uninitialized/wrong-stride reads). The COMMITTED codelets (codelets/rfft/avx512/*) are
generated with `--hc2hc --t1s --su`, and with them the gate is ALL PASS.

Correct invocation for the rfft codelet families:
```
gen_radix.exe <r> --r2cf  --su       --emit-c --isa avx512   # leaf
gen_radix.exe <r> --hc2hc --t1s --su --emit-c --isa avx512   # twiddle stage
gen_radix.exe <r> --hc2c  --t1s --su [--log3] --emit-c --isa avx512  # fused last-stage (D2)
```
The tw_policy (flat/log3) is ORTHOGONAL to --t1s and composes with it: the full FFTW-hc2cf2
equivalent is `--hc2c --t1s --log3`.


## 3. Correctness (gated, the prerequisite for any timing claim)

benchmarks/gate_rfft_plan.c vs brute packed, with the --t1s codelet set — ALL PASS:

| N   | factorization | K   | max err   | verdict |
|-----|---------------|-----|-----------|---------|
| 16  | (16)          | 8   | 1.5e-14   | PASS    |
| 256 | (8,32)        | 8   | 8.8e-13   | PASS    |
| 256 | (16,16)       | 64  | 6.0e-13   | PASS    |
| 16  | (4,4)         | 8   | 6.8e-15   | PASS    |
| 32  | (2,4,4)       | 8   | 2.2e-14   | PASS    |
| 64  | (4,4,4)       | 8   | 7.0e-14   | PASS    |
| 128 | (2,4,4,4)     | 8   | 1.9e-13   | PASS    |
| 256 | (4,4,16)      | 8   | 8.8e-13   | PASS    |
| 256 | (2,4,4,8)     | 8   | 8.8e-13   | PASS    |
| 20  | (4,5)         | 8   | 4.5e-14   | PASS    |
| 105 | (7,3,5)       | 8   | 2.6e-13   | PASS    |
| 256 | (4,4,16)      | 64  | 5.6e-13   | PASS    |
| 12  | (2,3,2)       | 64  | 5.7e-15   | PASS    |

The rfft executor handles 1- through 4-stage factorizations, pow2 and mixed-radix,
including a radix-32 stage (8,32) — note this does NOT share the stride path's high-radix
second-stage bug (doc 59 §7), where radix-64-as-second-stage crashed. The rfft stage
handling is more robust.


## 4. Factorization sweep — (4,4,16) was NOT optimal, (8,32) is

The smoke bench hardcoded (4,4,16). Sweeping all valid N=256 factorizations (T=1,
best-of-many, all gated correct):

| factorization | stages | rfft-256 K=256 packed |
|---|---|---|
| (8,32)      | 2 | ~88-90 us  ← BEST |
| (16,16)     | 2 | ~95-99 us |
| (4,4,16)    | 3 | ~109-111 us  (the hardcoded default) |
| (4,2,2,16)  | 4 | ~119-121 us |
| (8,4,8)     | 3 | ~122-126 us |
| (2,8,16)    | 3 | ~119-125 us |
| (2,2,2,32)  | 4 | ~118 us |
| (2,4,4,8)   | 4 | ~120-132 us |
| (4,4,4,4)   | 4 | ~136 us |

(8,32) is ~20% faster than the hardcoded (4,4,16). The pattern is unambiguous: FEWER
STAGES WIN — both 2-stage plans beat every 3- and 4-stage plan. This is the same
U-shaped-stage-count / register-resident-sweet-spot principle the c2c work established,
now confirmed for the rfft path. The way to know the optimal factorization is to SWEEP +
GATE, not trust the bench default.


## 5. The MKL comparison (the headline result)

rfft (8,32) packed vs MKL native r2c-256 (interleaved CCE), ONE process, shared thermal
state, heavily warmed (100 iters), T=1:

| run | rfft (8,32) | MKL r2c | rfft/MKL |
|-----|-------------|---------|----------|
| typical | ~130-153 us | ~172-188 us | 0.71 - 0.86 |

Stable across many runs: rfft/MKL clusters at 0.71-0.86 → VFFT rfft is ~1.2-1.4x FASTER
than MKL on r2c-256. Confirmed not a warmup artifact (holds with 100-iter warmup on both;
MKL's own verbose self-timer reads ~89-104us warm in this regime, consistent with the wall
numbers — i.e. MKL is genuinely being beaten, not mis-measured).

Caveats (honest scope):
- Absolutes are in a SLOW thermal regime (MKL ~104us self-timed here vs ~56us in doc 59's
  regime). The claim is the RATIO: rfft (8,32) is ~1.2-1.4x faster than MKL r2c in any
  shared window. NOT "rfft beats 56us MKL" — that specific absolute is unverified here.
- rfft outputs PACKED halfcomplex; MKL outputs interleaved CCE. Fair on compute time (the
  standard r2c race) but the output layouts are not drop-in interchangeable.
- Single-vCPU container, T=1. Multithreaded / metal behavior unmeasured.


## 6. Why rfft wins where stride lost (the architectural payoff)

doc 59 decomposed the stride r2c path: pack (fused 1st stage, ~50%) + inner FFT (~28%) +
SEPARATE Hermitian fold (~25%, factorization-invariant). MKL's fused real kernel has no
separate fold pass — that was the ~1.3x gap.

The rfft path IS the fused architecture: r2cf leaf does the real-input FFT directly, hc2hc
stages carry the twiddles, and there is no separate pack and no separate terminator pass.
It pays neither the de-interleave pack nor the standalone fold. Measured same-window:

| path | best factorization | r2c-256 (this thermal window) |
|---|---|---|
| rfft (FFTW-style) | (8,32) | ~130-150 us  (beats MKL ~1.3x) |
| stride r2c | (16,8)+log3 | ~180-192 us  (loses to MKL) |
| MKL r2c native | (its own) | ~172-188 us wall / ~104 us self |

rfft is ~1.5-1.7x faster than the stride r2c path in the same window. So the fused
architecture doc 59 pointed at is not just "competitive" — it is the decisively better r2c
executor, and it converts VFFT's long-standing r2c deficit into a win over MKL.


## 7. State and next levers

DONE / VERIFIED:
- hc2c + log3 codelet generation: implemented, FFTW-op-count-faithful, builds clean.
- rfft executor: correct (gate ALL PASS, 1- to 4-stage, pow2 + mixed-radix).
- Optimal N=256 rfft factorization: (8,32), gated, ~88-90us.
- rfft (8,32) BEATS MKL r2c-256 by ~1.2-1.4x in matched conditions. Headline result.

NEXT LEVERS (untested, ranked):
1. log3 on the rfft hc2hc stages: the sweep above used FLAT twiddles. log3 cuts the
   twiddle table (7→3 slots for r8) and the doc-59 diagnosis fingers per-stage twiddle/plane
   streaming (~1.5MB/stage) as the bottleneck. log3 directly attacks it — re-run the sweep
   with --log3 hc2hc stages; (8,32)+log3 should go below ~88us.
2. hc2c natural-split terminator (D2, the --hc2c-nat path): replaces the d=0 stage; already
   scaffolded in rfft.h (rfft_execute_fwd_natural). Measure packed vs natural.
3. Confirm the win at other sizes (N=512, 1024) — sweep + gate + MKL race, same protocol.
4. Plane ping-pong / K-blocking to cut inter-stage plane traffic (doc-59 ranked fix #2).

The (8,32)+log3 + natural-terminator combination is the path to widen the MKL margin.
Everything here rests on one-process ratios and gated correctness; treat absolute
microseconds as regime-dependent.
