# 62 — c2r build scope, anchored to the proven rfft forward design (doc 60)

The forward r2c win over MKL (~1.2-1.4x on r2c-256, doc 60) is the design c2r
must mirror in reverse. Key invariants carried from doc 60:

- WINNING ARCH: fused cascade, NO separate pack stage, NO separate Hermitian
  fold. r2cf leaf + hc2hc twiddle stages + optional hc2c D2 terminator.
- FACTORIZATION: fewest stages win. (8,32) beat every 3/4-stage plan ~20% at
  N=256. Selection rule is SWEEP + GATE, not a hardcoded default.
- --t1s GOTCHA (critical): the rfft executor twiddle table is SCALAR (one W per
  leg, broadcast via set1). Codelets MUST be generated with --t1s or they emit
  a VECTOR twiddle ABI that reads wrong memory -> garbage (~1e+179). The c2r
  backward codelets MUST also be --t1s.
- EPISTEMICS: ratios in one process only; gate correctness before any timing;
  absolutes are thermal noise (doc 59's 1.6x split-for-split was WITHDRAWN as a
  cross-run artifact).

## c2r piece-by-piece (what's free vs what's real construction)

| piece | source | status |
|---|---|---|
| backward hc2hc twiddle stage | `--hc2hc --bwd --t1s` | FREE: confirmed --bwd flows ~sign:`Bwd into the shared Dft.dft core; c2c bwd round-trips at 6.7e-14; body differs (not ignored) |
| backward hc2c terminator | `--hc2c --bwd --t1s` | FREE: same --bwd mechanism |
| r2cb leaf (hc2r: halfcomplex -> real) | NEEDS CONSTRUCTION | --r2cf --bwd is NOT it: sign flip keeps real INPUT, but the backward leaf needs halfcomplex INPUT -> real OUTPUT (different I/O type, no forward analog to flip). FFTW uses a separate gen_r2cb.ml. The math already exists inside dft_c2r_direct (Hermitian unfold); package it as a per-radix leaf. |
| c2r executor | mirror of rfft_execute, stages reversed, r2cb leaf last | NEW, templated on rfft.h |
| c2r registry | proven 6x pattern | TRIVIAL |
| c2r gate | differential vs dft_c2r_direct (monolithic oracle) | the end-to-end correctness check; also closes the "hc2hc-bwd isolated round-trip" gap |

## Verified this session
- --bwd accepted by hc2hc/hc2c construction, flows into shared DFT core,
  produces distinct real codelets (r8 hc2hc: fwd vs bwd bodies differ 292 lines,
  distinct _fwd_/_bwd_ symbols).
- Backward sign path numerically correct: c2c n1 fwd->bwd = N*x at 6.7e-14.
- r2cf --bwd is a sign-flipped real-INPUT leaf, NOT a usable hc2r backward leaf.
  r2cb leaf is the one genuine construction task.

## UPDATE: r2cb leaf BUILT and GATED (the one construction task)

dft_expand_r2cb added (generator/lib/dft_r2c.ml) + r2cb_signature emit block
(emit_c.ml) + --r2cb flag wired (gen_main.ml). Distinct halfcomplex-input
signature (in_re, in_im, out_re, is, os_re, vl) — NOT the r2cf real-input one.

Construction: reconstruct conj-symmetric spectrum from the packed half
(X[k>half] = conj(X[n-k])), backward DFT-n, real output. ONE bug found and
fixed by the gate: the backward DFT of the Hermitian spectrum comes out
TIME-REVERSED under this library's Bwd sign convention (r2cb[n] was
result[(n-k)%n]); fixed by mapping output index k <- result[(n-k) mod n]
(k=0, k=n/2 are reversal fixed points). 

Gate (brute forward rdft -> r2cb == N*x), ALL PASS:
  r2  0.0e+00 | r4 8.9e-16 | r5 3.7e-14 | r7 3.7e-14 | r8 2.0e-14 | r16 7.1e-14
Odd radices pass too, so the im_hi parity boundary is correct.

REMAINING for c2r: coverage quadrant (r2cb + --hc2hc --bwd --t1s + --hc2c
--bwd --t1s), c2r executor (mirror rfft_execute reversed), registry, end-to-end
gate vs dft_c2r_direct.

## FFTW execution structure (web-confirmed, FFTW source not in container)

Checked FFTW's actual hc2r execution against fftw3 master + the 3.3.10 docs.
Key confirmations for the c2r executor:

1. hc2r is the EXACT reverse of r2hc, unnormalized: r2hc -> hc2r = N*x. Our
   r2cb gate already matches this (round-trips to N*x). Convention correct.

2. DIT/DIF MIRROR (threads/hc2hc.c): forward R2HC uses apply_DIT; backward
   HC2R uses apply_DIF. So the c2r executor is the forward executor's mirror
   with:
     - leaf (r2cb) runs LAST (forward r2cf runs FIRST)
     - combine stages run d = 0 -> nf-2 (forward runs d = nf-2 -> 0)
     - twiddle stages are DIF + backward sign: --hc2hc --dif --bwd --t1s
       (NOT just --bwd on a DIT codelet). DIT = twiddle-then-DFT; DIF =
       DFT-then-twiddle, which is what the backward pass needs (input is
       already transformed). bytwiddle(... R sign) in hc2hc-generic.c confirms
       the twiddle stage is parameterized by sign.

3. Halfcomplex packing identical both directions: re in hc[k], im in hc[n-k],
   k=0 and (even n) k=n/2 have im=0 not stored. r2cb already expects this.

DESIGN DECISION for c2r coverage/executor:
  backward twiddle stage codelet = `--hc2hc --dif --bwd --t1s` (VERIFIED it
  generates + compiles: radix8_hc2hc_dif_bwd_avx512). The forward executor's
  populated-but-unused hc2hc_dif slots are exactly where these will live.
  The --t1s requirement (doc 60 scalar-twiddle gotcha) STILL applies.

So the c2r executor mirrors rfft_execute with DIF backward stages + r2cb leaf
last. End-to-end gate vs dft_c2r_direct remains the correctness arbiter.

## UPDATE 2: c2r executor foundation BUILT + GATED (nf=1 leaf-only)

core/c2r.h created. Mirrors rfft.h: reuses rfft_plan_t/rfft_plan_create for
factorization+planes+twiddle tables, re-points leaf to r2cb and stages to
hc2hc_dif_bwd. Added rfft_r2cb_fn typedef + r2cb / hc2hc_dif_bwd[_log3] slots
to rfft_codelets_t (r2cb is a DISTINCT 6-arg ABI: in_re,in_im,out_re,is,os_re,
vl — NOT r2cf's 7-arg).

nf=1 (leaf-only) execute path: c2r_execute_packed runs one r2cb leaf per group.
GATED vs N*x (brute forward halfcomplex -> c2r): N=8 -> 1.95e-14 PASS. The
plan/leaf/plane-layout scaffolding is proven.

REMAINING: the multi-stage cascade (stages d=0..nf-2, DIF backward hc2hc, leaf
last). This is where the twiddle-SIGN question resolves (forward table uses
tw_im=-sin; backward DIF codelet may need +sin or consume the conjugate). The
end-to-end gate vs dft_c2r_direct at nf=2 is the decisive hc2hc-DIF-bwd test.

## UPDATE 3: multi-stage cascade — FIRST ATTEMPT FAILS GATE (honest status)

Forward r2c uses DIT; backward c2r uses DIF — confirmed in OUR code: forward
rfft wires st->hc = hc2hc_log3/hc2hc (DIT slots), never reads hc2hc_dif. So the
DIT-fwd / DIF-bwd split mirrors FFTW exactly; it is a correct duality, not an
inconsistency. (Answer to "is the DIF trick used in our r2c?": no — r2c is DIT,
and that is correct; DIF is the right orientation only for the inverse.)

Multi-stage c2r_execute_packed written (stages d=0..nf-2, DIF backward hc2hc,
leaf last) but FAILS the nf=2 gate (N=16=(2,8): err 17, structurally wrong, not
close). Root causes identified, NOT yet fixed:
  1. The forward stage COMBINES child spectra into a parent halfcomplex; the
     backward must SPLIT parent -> children. That is a different memory access
     pattern, NOT merely reversed indices — my first cut reused forward index
     algebra with flipped in/out, which is insufficient.
  2. DC (k=0) and mid (k=m/2) column INVERSES are deferred/unimplemented in the
     first cut (noted in code). The forward k0 uses r2cf to combine; backward
     needs the r2cb-style split for the DC column.
  3. The twiddle SIGN (forward table tw_im=-sin) vs DIF-bwd codelet expectation
     is unverified — cannot isolate it until the split structure is right.

WHAT IS SOLID (gated, banked):
  - r2cb leaf: r2..r16 round-trip N*x at 1e-14 (UPDATE 1).
  - c2r nf=1 leaf-only executor: N=8 -> 1.95e-14 PASS (UPDATE 2).
  - FFTW execution structure + DIT/DIF duality confirmed.
  - rfft_r2cb_fn ABI + r2cb/hc2hc_dif_bwd slots in rfft_codelets_t.

NEXT (methodical, not guess-and-check): derive the backward stage's split
access pattern from the forward combine (it is the transpose of the forward
data movement, not the reverse of its loop indices), implement explicit DC/mid
inverse columns, then re-gate nf=2 vs dft_c2r_direct. The monolithic oracle is
the arbiter. Do NOT claim c2r works until nf>=2 gates pass.

## UPDATE 4: trace-driven inversion — tooling built, geometry gap remains

ROOT CAUSE of the cascade failures (finally pinned): the forward executor uses
MIXED im-packing conventions across op types, confirmed by instrumenting the
proven forward executor (core/rfft_trace.h, benchmarks/trace_rfft_fwd_moves.c):

Forward move map, N=16=(2,8), offsets in K-units:
  leaf g=0: x[0] is=2 -> planeA re[0]  im[16]  os=2   (im at +NK = row N+g)
  leaf g=1: x[1] is=2 -> planeA re[1]  im[17]  os=2
  k0:       planeA re[0]            -> out re[0] im[16] (im at +NK)
  hc k=1:   planeA re[2] im[14]     -> out re[1] im[7]  (OUT im at MIRROR row m-k)
  hc k=2:   planeA re[4] im[12]     -> out re[2] im[6]
  hc k=3:   planeA re[6] im[10]     -> out re[3] im[5]

So: leaf + k0 write im to the +NK region; interior hc writes im to the
in-plane MIRROR row (m-k). The FINAL out is single-plane halfcomplex (im in
upper rows 5,6,7); intermediate planeA uses +NK for leaf/k0 im. This MIXED
convention is what broke both hand-derivations.

REMAINING GAP: the stage reads planeA at rows {2,4,6}/{14,12,10} but the leaf
WROTE rows {0,1}/{16,17}. These don't line up in a single (q) view — the S-group
/ Q-fold interleaving across the S=2 leaf groups maps into the stage's r*k rows
in a way the single-move trace doesn't yet show. Need to trace ALL groups +
the q loop together to get the full leaf->stage row correspondence before the
backward replay is correct.

TOOLING BANKED (reusable): core/rfft_trace.h (VFFT_RFFT_TRACE dual-base move
recorder) + benchmarks/trace_rfft_fwd_moves.c. Next session: capture the
complete multi-group move map, build the inverse-replay backward executor from
it (correct BY CONSTRUCTION, not re-derivation), gate vs dft_c2r_direct.

STILL SOLID: r2cb leaf (gated r2-r16), c2r nf=1 (gated), DIT/DIF duality,
FFTW structure. The cascade remains the open item — honestly not done.

## UPDATE 5: cascade DONE — full gate matrix PASSES (the open item closes)

ROOT-CAUSE CORRECTION of update 4: there is NO mixed im-packing convention.
The forward layout is ONE uniform convention throughout — im streams descend
from one-past (+NK) bases via NEGATIVE strides (rfft.h says so at line 13:
"im stream reversed via os_im < 0 with base one-past the plane"). The
dual-base trace logged BASES, and with a negative stride base != first
written cell; rows {16,17} were never written, rows {14,12,10} were the
leaf's actual im writes. The "mystery" was a logging artifact. Both prior
hand-derivations died on it.

THE THREE FIXES (all that was actually needed):
1. r2cb ABI split: is -> is_re/is_im (emit_c.ml + rfft_r2cb_fn typedef).
   The DC-column and leaf inverses read re at +stride and im at -stride
   from a +NK base; a single shared `is` cannot express the sign split.
   This was the precise failure of attempt 1's stage_dc call.
2. Mid-column inverse implemented NUMERICALLY: the forward mid is an
   explicit r x r real map M (row t = mc[t,:] below np/2, ms[r-1-t,:]
   above); the plan Gauss-inverts it and scales by r (unnormalized
   convention). No trig re-derivation. Attempt 1 had no mid at all —
   rows {4,12} of the child plane were garbage at N=16, which alone
   explains err ~17.
3. Interior columns: attempt 1's mirror was ALREADY CORRECT (pointer
   pairs swapped, is/os swapped). Confirmed unchanged.

CONVENTIONS CONFIRMED BY THE GATE (pre-registered questions, answered):
- Twiddle table is SHARED with forward (tw_im = -sin); the --bwd codelet
  owns the conjugation. No conjugated copy needed.
- Every backward phase contributes factor r; composition = N*x.
- Plane scheme: backward stage d writes (d even ? planeA : planeB);
  planeB exists iff nf >= 3, matching forward's allocation.

GATE MATRIX (round-trip: random x -> rfft_execute_fwd_packed ->
c2r_execute_packed -> N*x; oracle chain ends at the MKL-gated forward):

  N=16  (16)      nf=1  4.4e-14 PASS   (regression, new ABI)
  N=16  (2,8)     nf=2  4.9e-14 PASS   (the old blocker: mid + DC)
  N=32  (2,16)    nf=2  1.0e-13 PASS   (kmax=7 + mid)
  N=24  (8,3)     nf=2  1.2e-13 PASS   (m odd: NO mid; r=8 stage)
  N=80  (5,16)    nf=2  5.6e-13 PASS   (odd radix stage + mid)
  N=64  (2,2,16)  nf=3  2.2e-13 PASS   (Q=2 fold path)
  N=64  (2,2,2,8) nf=4  2.1e-13 PASS   (ping-pong A,B,A)
  N=256 (4,4,16)  nf=3  8.5e-13 PASS   (the bench plan)
  N=256 (8,32)    nf=2  2.3e-12 PASS   (the MKL-beating fwd plan)
  N=256 (8,32) K=64     2.1e-12 PASS

FOOTGUN RECORDED: radix-32 plans need -DVFFT_RFFT_MAX_RADIX=32 (default 16,
#ifndef-guarded). The doc-60 benches did this; the first matrix run hit it.

FILES: core/c2r.h (rewritten: mirror cascade + mid_inv build/apply),
core/rfft.h (r2cb typedef), generator/lib/emit_c.ml (r2cb is_re/is_im),
benchmarks/gate_c2r_matrix.c (the comprehensive gate),
benchmarks/gate_c2r_nf1.c / gate_c2r_nf2.c (refreshed to new ABI as
roundtrip gates).

NEXT (per handoff, now unblocked): c2r coverage quadrant + auto-registry
(the proven 6x pattern), then the MKL c2r perf race (one-process ratios,
doc 60 epistemics). Performance work may also vectorize c2r_mid_inv_column
(scalar v1) and fold the backward leaf to one vl=S*K call (address-identical,
same fold the forward natural path uses).

## UPDATE 6: c2r coverage quadrant + auto-registry DONE (7th pattern application)

COVERAGE QUADRANT (generator/lib/coverage.ml): added c2r-avx2/c2r-avx512,
mirroring the rfft forward quadrant in reverse: r2cb leaf (radices 2,3,4,5,7,8,16
+ radix-32 big leaf for (8,32)) + DIF backward twiddle stages (flat
hc2hc_dif_bwd + log3). 22 codelets per ISA, all 22/22 compile.

AUTO-REGISTRY (generator/bin/emit_c2r_registry.ml): walks Coverage.files
"c2r-<isa>", classifies r2cb / hc2hc_dif_bwd / hc2hc_dif_bwd_log3 -> ABI-typed
slots of rfft_codelets_t. Two ABIs: r2cb 7-arg (is_re/is_im split), hc2hc 9-arg.
22 assignments, no double-assignment. Promoted to generated/c2r_registry_{avx2,
avx512}.h via dune (mode promote), same as the other 6 families.

GATE (benchmarks/gate_c2r_registry.c): the AUTO registry drives correct c2r.
c2r_register_all_avx512(&reg) fills the backward slots; round-trip vs N*x:
  (16) 8.3e-14 | (2,8) 8.5e-14 | (2,2,16) 4.2e-13 | (4,4,16) 1.8e-12
  | (8,32) 4.0e-12 | (8,32)K64 4.7e-12  -> ALL PASS

DESIGN NOTE surfaced by the gate: a complete c2r setup needs BOTH registrars on
one registry — rfft_register_all (forward codelets, for the shared geometry that
c2r_plan_create builds via rfft_plan_create) + c2r_register_all (backward
codelets, for execution). They fill DISJOINT slots, so they compose cleanly.
The c2r plan reuses the forward geometry/twiddle construction; only the leaf and
stage CODELETS differ.

FOOTGUN (again): the registry gate needs -DVFFT_RFFT_MAX_RADIX=32 for (8,32),
same as the matrix gate.

REGISTRY SCORECARD: all 7 generatable families now auto-emit ABI-typed
registries (c2c-inplace, c2c-OOP, rfft, trig, strided, AND c2r). The registry
is GENERATED from coverage; a wrong-ABI wire is a compile error.

NEXT: the MKL c2r perf race (one-process ratios, gate correctness first,
absolutes are thermal noise — doc 60 epistemics). Correctness is fully proven.

## UPDATE 7: MKL c2r perf race — directional baseline (doc-60 epistemics)

ENVIRONMENT CAVEAT: ~1-vCPU container, no PMU, rdtsc directional-only, MKL
non-deterministic. Numbers are RATIOS in one process, min-of-120, single-thread
(MKL_NUM_THREADS=1). Absolutes are thermal noise. Correctness gated FIRST.

METHODOLOGY (mirrors 09_compare_vs_mkl.c): opponent = MKL DFTI_REAL backward
(c2r), CONFIRMED unnormalized (matches N*x at 1.4e-14, same convention as ours)
and CONFIRMED correct in batched transform-major layout (1.2e-11). One bug
caught + fixed in the bench itself: MKL batches transform-major (transform t at
t*distance) while ours is lane-interleaved (bin*K+v) — the first cut fed MKL the
wrong layout and it computed garbage. Fixed; opponent now verified before timing.

BASELINE (the (8,32) MKL-beating forward plan, run backward):

  | N   | K  | nf | mkl correct | ours cyc | mkl cyc | mkl/ours | verdict   |
  | 256 | 8  | 2  | 1.2e-11     | 8092     | 5470    | 0.68     | mkl ~1.45x|
  | 256 | 64 | 2  | 1.4e-11     | 62786    | 43588   | 0.69     | mkl ~1.45x|

Ratio stable across 5 reruns (0.67-0.69, ~3% spread): the directional signal is
real. So our c2r is currently ~1.45x SLOWER than MKL on (8,32). Honest baseline,
not dressed up.

ROOT CAUSE (high confidence, not yet fixed): c2r_mid_inv_column is SCALAR v1 —
a triple-nested (vl x r x r) scalar-FMA loop, NO SIMD, and it IS on the (8,32)
hot path (r=8, m=32 even -> has mid). Everything around it is AVX-512. Our
FORWARD r2c beats MKL ~1.2-1.4x with the same codelets/cascade, so the backward
deficit is almost certainly this scalar mid path (+ the un-folded backward leaf),
not a structural problem.

NEXT (perf, separate work): (1) vectorize c2r_mid_inv_column (broadcast Minv,
AVX-512 over the vl lanes — the mc/ms accumulate pattern in rfft_mid_column is
the template). (2) fold the backward leaf to one vl=S*K call (address-identical,
the fold the forward natural path already uses). Re-race after each. Target:
close the 1.45x to parity-or-better, matching the forward win.

FILES: benchmarks/bench_c2r_vs_mkl.c (the race harness, opponent-verified).

## UPDATE 8: c2r perf — mid-inv SIMD + leaf fold (gated, measured)

Two named optimizations from update 7, both correctness-gated FIRST (matrix gate
all-pass, errors bit-identical to scalar -> math unchanged), then measured.

1. c2r_mid_inv_column VECTORIZED (was scalar v1): AVX-512 over the lane (v)
   dimension, hoist the r input row-vectors (reused across all j outputs),
   broadcast Minv coeffs, fmadd. AVX2 path + scalar tail included. Mirrors the
   forward rfft_mid_column lane-vectorization.
2. Backward LEAF FOLDED to one vl=S*K call (was per-group loop): address-
   identical (group g lane v0 -> global lane g*K+v0, same address), mirrors the
   forward natural path's single batched leaf. Applied to both nf=1 and
   multi-stage leaf calls. Kills S-1 call overheads.

MEASURED (256, (8,32) plan, mkl/ours ratio, one-process min-of-120, MKL=1 thr,
DIRECTIONAL — opponent verified correct 1.2e-11 each run):

  | stage                  | K=8 ratio | K=8 cyc | K=64 ratio | K=64 cyc |
  | baseline (u7)          | 0.68      | 8092    | 0.69       | 62786    |
  | + SIMD mid-inv         | 0.80      | 6966    | 0.77       | 56344    |
  | + folded leaf          | 0.94      | 5860    | 0.78       | 54946    |

K=8 went 1.45x-slower -> ESSENTIALLY PARITY (0.94, ~1.06x). K=8 cyc -28%.
The fold helps small-K most (per-call overhead amortizes at large K). K=64 still
0.78 (~1.28x slower): the bottleneck there is now the 15 interior
hc2hc_dif_bwd calls (kmax = m/2-1 = 15 at (8,32)), NOT the leaf or mid.

NEXT (feasibility CONFIRMED): RANGED DIF-backward interior. The codelet
generates+compiles (radix8_hc2hc_dif_rng_bwd_avx512, has kcount param). The
forward already has a ranged hcr path (one call walks kmax columns); the
backward mirror collapses 15 calls -> 1, targeting the K=64 gap directly.
Needs: ranged-bwd slot, executor wiring, backward cs_in/cs_out stride
derivation, gate. A construction step, not a quick edit.

## UPDATE 9: ranged interior — built, gated, integrated. K=8 -> PARITY.
##            Surprise: K=64 is MEMORY-bound, not overhead-bound.

RANGED DIF-BACKWARD interior: one call walks kmax columns (re-streams up,
im-streams down by cs per column: in_re+=cs_in/in_im-=cs_in, out_re+=cs_out/
out_im-=cs_out, tw+=r). Collapses the c2r interior per-k loop (15 calls at
(8,32)) to ONE call.

BUILT + WIRED:
- rfft_hc_rng_fn slot hc2hc_dif_rng_bwd added to rfft_codelets_t.
- c2r plan wires stage_hcr (NULL -> per-k fallback preserved).
- executor uses ranged when present. Stride derivation (the tricky part):
  bases at k=1 (in_re=src+Q*K, in_im=src+Q*(m-1)*K, out_re=dst+Q*r*K,
  out_im=dst+Q*r*(m-1)*K), is=QmK, os=QK, cs_in=Q*K, cs_out=Q*r*K, kcount=kmax.
  Derived from the per-k deltas; VERIFIED correct by the gate.
- coverage quadrant + auto-registry EXTENDED (29 codelets/ISA now, +7 ranged;
  registry 29 assignments, no double-wire). benchmarks/gate_c2r_ranged.c +
  the auto-registry gate both PASS all cases with ranged exercised end-to-end.

PERF (256 (8,32), mkl/ours, one-process directional, opponent verified):

  | stage                  | K=8 ratio | K=8 cyc | K=64 ratio | K=64 cyc |
  | baseline (u7)          | 0.68      | 8092    | 0.69       | 62786    |
  | + SIMD mid-inv (u8)    | 0.80      | 6966    | 0.77       | 56344    |
  | + folded leaf (u8)     | 0.94      | 5860    | 0.78       | 54946    |
  | + ranged interior (u9) | 0.99      | 5494    | 0.78       | 56178    |

K=8 reached PARITY (0.99, avg ~0.98 over reruns; 5494 vs mkl 5436). Total K=8
improvement baseline->now: 8092 -> 5494, -32%.

DIAGNOSIS CORRECTION (the honest finding): ranged helped K=8, did ~NOTHING for
K=64. My update-8 prediction (ranged closes K=64) was WRONG. Ranged removes
per-CALL overhead, which dominates only at small K. At K=64 there is 8x more
work per call, so 14 fewer call-overheads are negligible. K=64 is MEMORY-bound:
two N*K*8 = 128KB planes = 256KB working set borderline-exceeds L2, so the
cascade is bandwidth-limited and collapsing calls cannot help. Stable across
reruns (K=8: 0.97-1.00; K=64: 0.78-0.80).

NEXT for K=64 (different lever than I assumed): LANE-BLOCKING. The forward has
Kb (run the cascade per Kb-lane slab so both planes stay L2-resident across all
stages); c2r runs full-width Q-fold v1 (no blocking). Porting Kb to c2r is the
memory-bound fix. Separate, larger work. The ranged + mid + leaf wins stand:
small-to-mid K is now at parity with MKL.

## UPDATE 10: K=64 "memory-bound" claim WITHDRAWN (corrected by measurement)

The update-9 claim that c2r K=64 is memory-bandwidth-bound (256KB working set
exceeds L2, needs lane-blocking) is WRONG and withdrawn. Two measurements kill it:

1. The forward cascade HAS lane-blocking (Kb) and it is OFF by deliberate
   measurement: rfft.h section-65 note records the L2-slab heuristic as -22% at
   (4,4,16) and states "the cascade was never capacity-bound to begin with."
   Same planes, same hardware. So c2r is not capacity-bound either, and
   lane-blocking is NOT the fix (measured negative).

2. The FORWARD r2c shows the SAME K-dependent drop vs MKL, fully optimized
   (benchmarks/bench_fwd_vs_mkl.c, (8,32)):
       direction     K=8    K=64
       forward r2c   0.79   0.63
       backward c2r  0.99   0.78
   Both drop at large K; forward drops MORE. So the large-K gap is a SHARED
   cascade-vs-MKL batch-scaling property, not a c2r deficiency and not c2r's
   missing Kb. c2r at K=64 (0.78) actually beats forward at K=64 (0.63).

WHAT STANDS: the mid-SIMD / leaf-fold / ranged wins are real (small-K -> parity,
c2r large-K ratio now better than forward's). What does NOT stand: the
mechanism story for the residual large-K gap. It is shared with forward, the
forward team ruled out L2 capacity, and the true mechanism (likely MKL's
large-batch strategy) is open pending PMU hardware.

The corrected finding is written up in docs/63_c2r_K_dependent_bottleneck.md.
Lesson: a behavioral signature (X helped here not there) is consistent with
many mechanisms; check whether a comparable already-optimized path shows the
same gap before attributing it to a mechanism you cannot measure.
