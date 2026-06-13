# Step-2 design: fuse the forward r2c postprocess into the inner's last DIT stage

STATUS: design for review. Nothing implemented. This is the "MKL move" from the
v2 verdict — the only attackable phase (postprocess, ~50% of r2c-256) folded into
the inner FFT's final stage so the separate sweep and its scatter both vanish.

Decision context (settled this session):
- Rider 1: pack is FUSED stage-0 compute, not fallback, not a target.
- Rider 2: natural-order output is last-stage EMISSION surgery (the executor's
  last stage stores in-place at st->stride with no output-addressing parameter;
  rfft.h achieves natural order via a different codelet, not re-addressing). So
  the cheap intermediate (step 1) is NOT cheap → skip it, do full fusion here.

## What the fusion must reproduce (the r2c forward butterfly)

The inner is an (N/2)-point complex FFT leaving Z[0..N/2-1] in scratch. Postprocess
turns Z into the N-point real spectrum X[0..N/2]:

  X[0]   = Re(Z[0]) + Im(Z[0])                 (DC, real)
  X[N/2] = Re(Z[0]) - Im(Z[0])                 (Nyquist, real)
  for k = 1 .. N/2-1, paired (k, N/2-k):
    E = (Z[k] + conj(Z[N/2-k])) / 2
    O = (Z[k] - conj(Z[N/2-k])) / 2
    X[k]     = E + W_N^k     * (-i * O)
    X[N/2-k] = conj(E) + W_N^{N/2-k} * (-i * conj(O))   (the mirror, computed
                                                          from the SAME loads)

The twiddle is W_N^k with N = 2*halfN — a half-step power. This is the one real
difference from the hc family (whose combine twiddle is W_np^{jk}); the data-flow
skeleton is identical, the algebra in the middle is the r2c split, not the hc
combine.

## Why hc2c_nat is the right skeleton (structural match, confirmed in source)

rfft.h's hc2c_nat codelet signature:
    (in_re, in_im, Rp, Ip, Rm, Im, tw_re, tw_im, is, osp, osm, vl)
It already:
  - loads a residue pair once,
  - writes a "plus" output (Rp/Ip at +s*osp) and a CONJUGATED "minus" output
    (Rm/Im at +(r-1-s)*osm) from those shared loads,
  - hoists the twiddle, walks vl lanes.
That is exactly the (k, mirror) dual-output-from-shared-load shape postprocess
hand-rolls. We borrow the skeleton; we swap the hc combine algebra for the r2c
split algebra above.

## The design: a new fused codelet variant + an opt-in executor path

### Piece A — generator: a new emit variant `r2c_term` (renamed per review item 5)
(Was `r2cf_lastfuse`; renamed because the codelet's INPUT is complex last-stage
data, not real — it is a c2c-stage-plus-r2c-terminator. `r2c_term` keeps the
registry taxonomy honest for when the c2r `c2r_init` twin arrives.)

A codelet that IS the inner's last DIT stage AND emits the r2c butterfly on its
outputs, in natural frequency order, before they ever land in scratch.

The codelet processes a COLUMN PAIR (k, m−k) per call (see the layout proof
below): it consumes two columns of last-stage inputs (2r complex values) with
their two stage-twiddle sets, runs the radix-r butterfly on each, then folds the
mirror pairs and emits two constant-stride output streams.

Signature (dual-output, settling Q1 as leaned; now WITH the stage twiddle):
    void radix{R}_r2c_term_fwd_{isa}(
        const double *in_re,  const double *in_im,   /* last-stage inputs, 2 columns */
        double *Xp_re, double *Xp_im,                /* X[k]      natural-order output */
        double *Xm_re, double *Xm_im,                /* X[m-k]/mirror output (slot-reversed) */
        const double *stage_tw,                      /* W_{N'}^{j.k} per-leg stage twiddles
                                                        for BOTH columns (see Piece C packing) */
        const double *bw_re, const double *bw_im,    /* W_N^k half-step BUTTERFLY-fold twiddles */
        ptrdiff_t is, ptrdiff_t osp, ptrdiff_t osm, size_t vl);

WHY both twiddle inputs (review item 2): the codelet replaces the inner's last
stage, and in the DIT forward path that stage IS twiddled — its inputs get the
stage twiddle W_{N'}^{j.k} per leg BEFORE the butterfly. The bw twiddle (the
half-step W_N^f fold) multiplies O AFTER the butterfly. Different dataflow
positions → they cannot merge into one constant set. (Verified in source: the
DIT slice executor applies needs_tw[g] per-leg twiddles at each stage; the
"last stage has no twiddle" comment is the DIF path, not the DIT path the r2c
worker runs.)

Generation: VARIANT of the existing last-stage t1 codelet. DAG is (per-leg stage
twiddle → radix-r butterfly) THEN (E/O split + W_N^k fold), emitted as one
scheduled body so butterfly outputs are consumed from registers by the fold and
never round-trip to scratch. The bw table folds the 0.5 (bw' = W/2) — the
pre-flagged win, now gate-legal per item 1.

### Piece B — executor: an opt-in fused last-stage path in r2c.h (additive)
New worker branch (the OOP forward worker, where the loss was measured):
  - run the inner FFT stages 0 .. nf-2 as today (fused first stage + slice exec),
  - replace the final stage call + _r2c_postprocess with ONE call to the fused
    last-stage codelet, writing X directly to (out_re, out_im) in natural order.
Gated by a plan flag (e.g. d->use_lastfuse, set only when the registry has the
fused codelet for the last-stage radix). Default OFF → the existing
last-stage-then-postprocess path is textually unchanged (rider 4).

### Piece C — the twiddle tables (packed, per review item 2)
TWO twiddle sets at different dataflow positions:
  - stage_tw: the last DIT stage's per-leg W_{N'}^{j.k} (N' = halfN). Already
    built by the inner plan; the fused codelet needs the sets for BOTH columns
    of its pair per call.
  - bw_re/bw_im: the half-step W_N^f fold twiddles, N = 2*halfN. Same cos/sin
    table _r2c_init_twiddles builds (lines ~93-110), with the 0.5 folded in
    (bw' = W/2).
PACKING (the cheap path; avoids an IR Twiddle-namespace extension): the IR's
Twiddle(slot) is a single table, so the plan builds ONE packed per-column-pair
table — stage twiddles for columns k and m-k in low slots, bw twiddles above,
by a slot-partition convention stated here so the emitter and the plan agree.
A two-table codelet would otherwise need an IR extension; packing is cheaper.

## The layout proof (review item 3 — closes open question 3 in the hoped direction)

For the last DIT stage of N' = r*m: the butterfly at column k produces exactly
the frequency set { k + s*m : s = 0..r-1 }. The r2c mirror of f = k + s*m is
  N' - f = (m-k) + (r-1-s)*m.
So mirror pairs live in the COLUMN PAIR (k, m-k) with SLOT REVERSAL s <-> r-1-s
— literally the (r-1-s)*osm indexing already in the hc2c_nat typedef comment.
Same lemma rfft.h's D2 relies on (constant-boundary lemma,
docs/native_rfft_design.md) — cited, not re-derived.

Consequences baked into the design:
- The fused codelet takes TWO columns of last-stage inputs (2r complex values,
  two stage-twiddle sets) and emits TWO constant-stride output streams:
  Xp at base k*K stride m*K; Xm via the reversed-slot convention. Dual pointers
  (settles Q1). Both constant-stride → prefetcher-friendly, scatter gone by
  construction (no perm table).
- COLUMN TAXONOMY (same as rfft.h's): interior columns pair (k, m-k); k=0
  self-paired (slots s <-> r-s within the column; DC + Nyquist from its s=0
  output; self-conjugate Z[N'/2] at slot r/2 when r even); k=m/2 self-paired
  when m even. CODELET SET = interior-pair codelet + two small specials.
  v1: generate ONLY the interior-pair codelet; keep the two specials
  EXECUTOR-SIDE (O(r)-row analog of rfft.h's nat_k0 scatter, negligible).

STRATEGIC NOTE: after this lands, r2c.h forward = fused pack + c2c middles + an
rfft-style natural terminator. The Phase-1/Phase-2 merge stops being a question
and becomes a diff.

## What stays untouched (rider 4, provable-zero c2c regression)
- core/executor.h and the c2c stride path: textually unchanged. The fused codelet
  is r2c-only; the c2c last stage keeps its in-place n1_fwd store.
- _r2c_postprocess and the current worker branch: kept as the default/fallback
  (and for any radix lacking the fused codelet). The fused path is the opt-in.
- The in-place override (_r2c_execute_fwd): untouched; dct/dst shells unaffected.

## Gate (rider 3 reframed — bit-identity is the WRONG contract for fusion)

CORRECTION from review: the bit-identity gate was right for step 1 (natural-order
re-addressing: same arithmetic, different addresses) and was wrongly carried over.
Fusion is NOT store-only: the generator schedules the combined DAG (last-stage
butterfly → E/O split → W_N^k fold) as ONE body, and its algsimp/FMA passes will
not reproduce _r2c_postprocess's hand-rolled fmsub/fmadd association — nor should
they. The most obvious win, folding the 0.5 into the butterfly table
(bw' = W/2), changes the rounding sequence BY CONSTRUCTION. A bit-identity gate
would force us to either forbid the 0.5-fold (waste a win to satisfy a gate) or
misread expected reassociation as scope creep. So:

- PRIMARY: brute Hermitian reference, thresholds 8.1e-14 / 6.6e-13 (the existing
  gate_r2c numbers). This is correctness.
- CONSISTENCY: fused-vs-default agreement at a FEW-ULP RELATIVE threshold (not
  bit-identity, not epsilon). Pre-registered expected difference and its causes:
  (a) the 0.5-fold into bw (bw' = W/2) reorders the scaling multiply; (b) FMA
  reassociation from scheduling the combined DAG. An epsilon-scale mismatch here
  CONFIRMS the design (the fold + reassociation landed as intended); it is NOT a
  scope-creep signal.
- SCOPE-CREEP SIGNAL (replaces the old taxonomy): a mismatch ABOVE the few-ulp
  relative threshold but below the brute-Hermitian threshold means the math
  drifted further than the 0.5-fold + FMA reassociation explains — THAT is the
  reassess trigger. A failure of the brute-Hermitian PRIMARY is a correctness bug.
- PROFILE: VFFT_R2C_PROFILE must show postprocess GONE (folded into last-stage
  time), scatter eliminated.

NOTE: bit-identity language is retained ONLY if we deliberately constrain the
codelet to replicate postprocess's exact association — and we do NOT, because that
forbids the bw' = W/2 table fold. Bit-identity is off the table for this step.

## Pre-registered success criterion (rider 5, against the TOTAL not the share)
Measured baseline (N=256 K=256, container): total ~93,800 ns; pack ~44%, inner
~6%, postprocess ~50% (~46,900 ns); MKL ~56,300 ns → 0.60x.

Postprocess sequential-bandwidth floor (264 KB moved @ ~74.7 GB/s container
stream rate): ~3,600 ns. NOTE: current postprocess is ~13x that floor, so it is
scatter+compute-bound TODAY, not bandwidth-bound. Fusion removes the scatter and
the separate-pass traffic but the E/O+twiddle ARITHMETIC remains (now in
registers in the last stage). So the floor is the optimistic ceiling, not the
expectation.

PREDICTION (falsifiable):
  - Direction (high conf): total drops; the N=256-vs-N=64 ratio slide (0.60 vs
    0.86) largely reverses, because the scatter that caused the slide is gone.
  - Magnitude (low conf, container-directional): total drops by >= ~25,000 ns
    (roughly half the postprocess); MKL ratio recovers 0.60x -> at least ~0.80x;
    realistic landing 0.80-0.95x, optimistic ceiling ~1.11x (only if it hit the
    bandwidth floor, which it won't because the arithmetic stays).
  - FALSIFIER: if postprocess share shrinks but total does NOT drop ~that much
    (pack grew to fill the gap), the scatter theory is wrong — reassess.

## Resolved decisions (was "open questions"; review settled all four)
1. SIGNATURE: dual-output (Xp/Xm separate pointers). Settled by the layout proof —
   the two constant-stride streams (Xp base k*K stride m*K; Xm reversed-slot) are
   prefetcher-friendly and need separate bases. Confirmed against hc2c_nat.
2. RADICES: R=8 ONLY first (the losing (4,4,8) cell's last stage). Prove the full
   chain generator→gate→profile end-to-end on one radix, THEN spread to {4, 16}.
   FREE LEVER (review item 4): factor ordering is free at plan time — (4,4,8) vs
   (8,4,4) changes which radix is last. With only R=8 fused, the planner reorders
   factors to put a covered radix last. This makes single-radix coverage nearly
   universal for pow2 plans. Belongs in Piece B's flag:
     use_lastfuse = registry has the r2c_term codelet for SOME radix the planner
     can legally place last (then reorder to place it last).
3. LAYOUT (was the risky one): CLOSED by the proof above in the hoped direction.
   Mirror pairs live in column pair (k, m-k), slot-reversed s <-> r-1-s; natural
   order falls out with NO perm table. Same lemma as rfft.h's D2.
4. c2r SYMMETRY: defer the code, but FIX THE CONVENTION NOW — the same column-pair
   geometry applies to the backward preprocess (Z reconstructed from X pairs
   feeding the first DIF stage). Pin the bw-table sign/conjugation convention in
   this doc with the backward case written beside the forward, even though no
   backward code ships, so c2r_init reuses it cleanly.

## Refinements (review item 5)
- The conservative floor is MORE conservative than stated: fusion also deletes the
  last stage's scratch STORES and postprocess's scratch LOADS (the full halfN*B
  round trip), not only the scatter. So the >=25,000 ns total-drop floor
  under-promises — keep it as stated (under-promising is fine), but expect more.
- Naming: r2cf_lastfuse -> r2c_term (the input is complex last-stage data, not
  real; it's a c2c-stage + r2c-terminator). Keeps registry taxonomy honest for
  the c2r c2r_init twin.

## IMPLEMENTATION LOG

### Slice 1 (DONE, gated): the r2c_term fold math builder
lib/dft_r2c.ml: added dft_r2c_term_pair + dft_expand_r2c_term. This is the
terminator butterfly for one interior column pair (k, m=N/2-k), reading the
inner FFT's Z[k] (Input 0) and Z[m] (Input 1), emitting natural-order X[k]
(Output 0) and X[m] (Output 1). Fold algebra reuses dft_r2c_direct's verified
pair butterfly.

BUG CAUGHT BY THE ISOLATION GATE (the reason we gate math before wiring): the
first X[m] derivation used a "conj(E) + W^m*(-i*conj(O))" shortcut that was
WRONG (MAX ERR 3.0 at N=16). Correct form: run the SAME butterfly with loop
index = m and partner k — E_m,O_m from the swapped pairing (Z[m],Z[k]) with
twiddle theta_m. This shares the two loads but recomputes E/O with roles
swapped. After the fix, dft_r2c_term_pair matches brute r2c at:
  N=16: 7.1e-15 | N=32: 1.4e-14 | N=64: 1.2e-13 | N=256: 9.4e-13  ALL PASS.

v1 keeps the 0.5 explicit (bw'=W/2 fold deferred). Additive: 48/48 baseline
byte-identical, build clean.

### Slice 2 (NEXT, not started): executor wiring + emit/registry
- gen_main: --r2c-term flag + dispatch to dft_expand_r2c_term (needs the
  per-pair k; emission iterates the column taxonomy or emits a ranged variant).
- The stage-twiddle input (design Piece A/C): slice-1's builder reads Z directly
  (assumes the inner FFT already produced Z for the pair). Wiring must either
  (a) have the executor run stages 0..nf-2 then feed Z columns to the codelet
  (simplest, matches slice-1's input model), or (b) fold the last-stage radix
  butterfly INTO the codelet (the full design, needs stage_tw). Slice 1
  implements model (a)'s fold; decide (a) vs (b) at wiring time — (a) is a
  smaller first integration and still kills the postprocess pass + scatter.
- Executor: opt-in branch in _r2c_worker_fwd_oop replacing _r2c_postprocess
  with the fused codelet calls; the k=0 and k=m/2 specials stay executor-side.
- Gate: brute-Hermitian PRIMARY + few-ulp consistency (NOT bit-identity);
  VFFT_R2C_PROFILE must show postprocess gone.

### Slice 2 (DONE, emit half gated): the r2c_term codelet emits + verifies
Generator now emits the fused terminator codelet:
- lib/emit_c.ml: r2c_term_signature flag + emit branch (dual-output ABI:
  in_re,in_im,Xp_re,Xp_im,Xm_re,Xm_im,is,vl). Input addressing Input(0)->in[v],
  Input(1)->in[is+v]; output Output(0)->Xp, Output(1)->Xm; loop over vl lanes.
- lib/gen_main.ml: --r2c-term + --r2c-term-k flags, dispatch to
  dft_expand_r2c_term, name radix{N}_r2c_term_k{K}_{sgn}_{isa}, hoist-consts on.
VERIFIED: emitted/FMA-lifted/scheduled/compiled codelet vs brute r2c at N=256,
k=1/5/17/31/63 -> 1e-13 to 4e-13, ALL PASS (full pipeline preserves the math).
Durable gate: benchmarks/run_r2c_term_gate.sh + gate_r2c_term.c.
Additive: 48/48 baseline byte-identical.

### Slice 3 (NEXT, not started): executor wiring
Replace _r2c_postprocess's per-frequency butterfly with per-pair r2c_term calls
in _r2c_worker_fwd_oop, reading scratch rows in NATURAL order (kills the scatter).
- registry: register the r2c_term codelets for the plan's needed (N,k) set, OR
  emit a ranged variant walking kcount pairs per call (follow-up; v1 can call
  per-pair).
- k=0 and k=m/2 specials stay executor-side (DC/Nyquist + self-conjugate).
- gate: brute-Hermitian PRIMARY + few-ulp consistency (NOT bit-identity);
  VFFT_R2C_PROFILE must show postprocess folded away, scatter gone.
- pre-registered success: total drops >= ~25k ns, MKL 0.60x -> >= ~0.80x.
DESIGN NOTE for slice 3: slice-2's codelet takes Z[k],Z[m] as inputs (model a:
executor runs inner FFT fully, then the terminator folds — NOT the last-stage-
radix-fused model b). This is the smaller integration and still removes the
postprocess pass + scatter; model b (folding the last radix stage in too, needing
stage_tw) is a later optimization.

### Slice 3 KEY FINDING (verified): the scatter is block-local, not random
Empirically confirmed (N=256 inner=(4,4,8)): for the last radix r, column k's r
frequencies (f = k + s*m, s=0..r-1, m=half/r) map to a CONTIGUOUS physical block
in scratch (column 0 -> rows 0..7, column 1 -> rows 32..39, ...). The mirror
column (m-k) is another contiguous block, slot-reversed (s <-> r-1-s) -- exactly
the layout proof.

CONSEQUENCE: the current postprocess "scatter" (z_m = perm[mirror]*B) is an
artifact of iterating by FREQUENCY f while scratch is in column-block order. By
iterating per COLUMN BLOCK instead, BOTH the primary and mirror reads are
contiguous within their r-row blocks -- the natural-order benefit achieved by
loop reorganization, WITHOUT changing the inner FFT's output ordering (no
last-stage emission surgery needed). This makes model (a) deliver the scatter
kill after all, contradicting the earlier "step 1 needs surgery" conclusion: the
surgery was only needed if we insisted on frequency-order iteration. Column-block
iteration sidesteps it.

SLICE 3 EXECUTOR LOOP (the plan):
  for each column k in 0..m-1 (m = half/r):
    primary block = scratch rows [perm[k], perm[k]+r) ... actually the r rows
      perm[k + s*m] for s=0..r-1, which are contiguous (= perm[k] + s).
    mirror  block = rows for column (m-k), contiguous, slot-reversed.
    for each (primary slot s, mirror slot r-1-s) giving freq pair (f, half-f):
      call r2c_term with Z[f]=primary row, Z[half-f]=mirror row.
  specials: k=0 (DC+Nyquist+self-conj) and k=m/2 stay scalar/executor-side.
NOTE: this calls r2c_term per (f, mirror) PAIR; a ranged variant walking a whole
column's pairs per call is the follow-up optimization. v1 = per-pair calls,
contiguous reads.

### Slice 3 (IN PROGRESS): executor wiring — scaffold landed, ABI gap found
DONE this slice:
- core/r2c.h: stride_r2c_data_t gained term_fwd / term_r / term_m (default NULL/0,
  opt-in). _r2c_postprocess_fused added (interior frequency-pair loop calling the
  codelet, block-local reads). OOP worker has the opt-in branch with DC/Nyquist +
  self-paired (f=halfN/2) scalar specials; default path unchanged. Tree builds,
  48/48 baseline byte-identical, default r2c still 0.61x (unaffected).
- Coverage of the interior-pair loop verified (0 missing, 0 double; f=halfN/2 +
  DC/Nyquist handled as specials). Caught + fixed a column-vs-frequency pairing
  bug via the standalone coverage test.

ABI GAP FOUND (the real finding): the slice-1/2 codelet bakes W^f as CONSTANTS
(per-k: --r2c-term-k 5 emits cos/sin(5) as set1_pd literals). One codelet serves
ONE frequency. The executor loops over ALL f with a single term_fwd pointer, so a
fixed-k codelet cannot serve it. Slice 1/2's fixed-k codelet was correct for
VERIFYING THE MATH, but the executor-usable codelet needs RUNTIME twiddles
(load W^f and W^m from a tw pointer per call), not baked constants. This is
exactly the design's Piece C "loadable twiddle table" — I conflated "bake the
const" (fine for math verification) with "loadable twiddle" (needed for the loop).

CORRECTED NEXT STEP (slice 3b):
- Add a runtime-twiddle r2c_term builder: load W^f via Twiddle(0,*) and W^m via
  Twiddle(1,*) instead of Const(cos theta). One codelet for all frequencies.
- emit_c: add r2c_term twiddle addressing (tw_re[0..1] / tw_im[0..1] per call,
  the two per-pair twiddles), and add tw_re/tw_im pointers to the r2c_term
  signature.
- executor: term_fwd call passes &tw_re[f], &tw_re[mir] (or a packed per-pair
  twiddle). Then gate: brute-Hermitian PRIMARY + few-ulp consistency vs
  _r2c_postprocess; VFFT_R2C_PROFILE shows postprocess folded.
- iteration order: v1 loop is frequency-order (primary read perm[f] jumps).
  For the block-local locality win, reorder to physical-row order (sequential
  primary, block-local mirror) as a measured follow-up — correctness first.

### Review corrections (pre-3b) — re-pegging the prediction + the door the proof opened

ITEM 1 (the door): the block-local discovery RETROACTIVELY REVIVES ladder step 1.
Columns being contiguous in scratch means model (a) delivers the scatter kill via
LOOP ORDER ALONE — the cheap path existed after all, through iteration reordering
rather than the store-addressing surgery rider 2 ruled out. Rider 2's "skip step 1"
was correct given what we knew then (re-addressing genuinely was surgery); the
cheap path re-emerged from the layout proof, not from changing our minds. The
proof opened a door the code-read could not see.

ITEM 2 (RE-PEG THE PREDICTION — model (a) does NOT delete a pass):
  - Default:   stages -> scratch ; postprocess reads scratch -> writes out.
  - Model (a): stages -> scratch ; terminator  reads scratch -> writes out.
  IDENTICAL pass structure. Model (a)'s win = scatter elimination + codelet-quality
  arithmetic, NOT traffic removal. The >=25,000 ns / 0.80x prediction was
  calibrated for MODEL (b), which collapses the last stage's scratch round-trip
  into registers. So:
  - MODEL (a) PREDICTION (re-pegged): the recoverable part is the SCATTER's share
    of the postprocess's 13x-over-floor excess — i.e. the strided-access penalty,
    not the pass itself. A model-(a) result of e.g. -15k ns is a CONFIRMED scatter
    theory, not a failed prediction. Falsifier unchanged (total must drop; if it
    doesn't, scatter wasn't the cost).
  - MODEL (b) stays on the ladder as the rung that actually deletes traffic
    (fold the last radix stage into the terminator, needs stage_tw). Its >=25k /
    0.80x number belongs to (b), decided on (a)'s measurement with the traffic
    argument cleanly separated from the scatter argument.

ITEM 3 (the landed v1 loop is access-WORSE than baseline — do not time it as-is):
  v1 _r2c_postprocess_fused iterates by FREQUENCY (prow=perm[f]): BOTH streams
  jump. The original postprocess walks p sequentially (f=iperm[p]): sequential
  primary + one scattered mirror. So v1-as-landed trades one scattered stream for
  TWO strided ones — worse. The physical-row reorder (iterate p, f=iperm[p],
  block-local mirror) is the load-bearing perf piece and MUST land WITH 3b, not as
  a later follow-up. Correctness-first is right; just do not time the
  correctness-only build and call it the result. (The r,m params are kept for
  exactly this rewrite.)

ITEM 4 (3b is simpler: ONE twiddle slot, no packed table):
  Identity VERIFIED (max err 3.6e-16): W^{halfN-f} = (-tw_re[f], +tw_im[f]) in our
  convention (tw_re[k]=cos(-2pi k/N), tw_im[k]=sin(-2pi k/N), N=2*halfN). So the
  codelet loads Twiddle(0)=W^f only; the mirror twiddle is a Neg on the real part,
  free under FMA absorption. Signature gains just (tw_re, tw_im); executor passes
  tw_re+f, tw_im+f per call. Piece C's slot-partition packing is UNNECESSARY.
  This identity is also the c2r convention Q4 wanted pinned — derive the backward
  sign once while writing 3b.

ITEM 5 notes:
  (a) profile accumulators are non-atomic statics written from worker threads —
      fine for single-thread directional runs; a multi-thread profile would
      produce garbage silently. (Comment added at the accumulator decl.)
  (d) the IN-PLACE forward worker got timers but NOT the fused branch (correct
      prioritization). It still pays the old scatter — do not read its unchanged
      numbers as a refutation.
  (e) d->term_fwd NULL by calloc => rider 4 holds by construction. When
      registration lands, keep the "AND VFFT_R2C_FUSE" double-gate OR drop the env
      mention so flag and code agree. (Decide at registration.)
  (c) per-pair call granularity (~127 calls x ~26 flops x B) is fine at B>=128,
      marginal below; the ranged variant is queued — SAME lesson as rfft.h's
      item 2 (stage-0 column granularity). The two executors' fixes converge.

### Slice 3b RESULT (DONE): fused path correct, but BREAK-EVEN on container — and why

WIRED + VERIFIED:
- Runtime-twiddle r2c_term codelet (dft_r2c_term_pair_rt / dft_expand_r2c_term_rt;
  --r2c-term-rt). ONE codelet serves all interior frequencies via Twiddle(0); the
  mirror twiddle uses the verified identity W^{halfN-f}=(-W^f_re,+W^f_im) (Neg,
  FMA-absorbed). Isolation gate: 7e-15..9e-13 across N=16..256. End-to-end gate
  (one compiled codelet, all 63 interior freqs of N=256): 7.8e-13 PASS.
- emit_c: r2c_term_rt adds tw_re/tw_im to the signature, broadcasts Twiddle(0).
- executor: _r2c_postprocess_fused rewritten to PHYSICAL-ROW order (sequential
  primary p, block-local mirror perm[halfN-iperm[p]]) + per-pair twiddle pass.
  OOP worker opt-in branch with DC/Nyquist + self-paired specials.
- FUSED-vs-DEFAULT gate (full r2c-256, all freqs incl specials): 2.487e-14 PASS.
  The fused executor path produces the same answer as the trusted postprocess.

MEASUREMENT (clean, no instrumentation, 5 runs): default vs fused TOTAL runtime
is BREAK-EVEN: drops range -5.9% to +4.4%, centered at noise. NOT the predicted
win.

TWO honest findings from the discrepancy:
1. The PHASE PROFILER IS CONTAMINATED. Instrumented phase-sum (pack 70k + inner
   30k + post 238k = 338k ns) is 3.6x the clean total (~93k ns). The per-b0-block
   clock_gettime calls dominate, and postprocess does the most blocks so it
   absorbed the most overhead — manufacturing the "post = 50-70%" share. The
   "postprocess is 50%, inner is 6%" diagnosis (and the original r2c-loss
   decomposition, measured the same way) is therefore SUSPECT as to absolute
   shares. Within-run RATIOS may still be directional, but the magnitude that
   motivated "postprocess dominates" was inflated by the instrument.
2. THIS CONTAINER CANNOT MEASURE THE ACCESS-PATTERN WIN. The KVM box has L3~=DRAM
   and no real prefetcher/cache hierarchy. A scatter costs ~nothing here because
   there is no hierarchy to defeat. Block-local-vs-scatter is PRECISELY the
   difference this environment is blind to. At K=256 the data is small enough to
   be effectively flat-latency. So break-even-on-container does NOT mean
   break-even-on-silicon: the access-pattern win (if real) would show on i9/Zen4
   with a real L2/L3 + prefetcher, and is invisible here BY CONSTRUCTION.

NET VERDICT (honest): the fusion is CORRECT and the code is sound, but its value
is UNPROVEN — the container can't see the one effect it targets. This is a
genuine "needs metal" result, not a success and not a failure. Do NOT claim a
runtime win from these numbers. The model-(a) fusion neither helped nor hurt
total runtime here; whether it helps on real silicon is an open question only the
i9/Zen4 A/B can answer (default-vs-fused, same binary, the gate already exists).

STATUS: kept (opt-in, default off via term_fwd=NULL; rider-4 zero-regression
holds — 48/48 baseline byte-identical). NOT wired into plan creation (still
test-only via override_data). Model (b) (fold the last radix stage to delete the
pass) is the rung that could win on traffic alone regardless of cache hierarchy,
but given model (a) is break-even here, model (b) should be gated on a METAL
model-(a) measurement first: if (a) wins on silicon, (b) is the amplifier; if (a)
is break-even on silicon too, the whole r2c-fusion thread may not be worth the
carry and the native rfft.h path is the better bet.

### Slice 3b POST-MORTEM CORRECTED (the profiler bug was thread-seconds, not clock overhead)

My finding #1 had the right conclusion (phase shares were untrustworthy) with the
WRONG mechanism (I blamed clock_gettime overhead). The real bug, confirmed in code
+ ablation:

THE BUG: the phase accumulators (_r2c_prof_*) are summed from INSIDE the worker
functions, which are dispatched to T pool threads. Each thread adds its own
elapsed time to the shared globals, so Sigma(phases) = THREAD-seconds, while the
clean bench total is WALL-seconds. 338k/93k = 3.63 ~= T=4. A units error, not
instrument cost (clock_gettime is 28ns vDSO here; ~128 reads = ~3.6us, 3 orders
short of the 245k discrepancy). The non-atomic += races add run-to-run noise on
top. My "clock overhead" diagnosis is RETRACTED.

WHY THE SHAPE WAS WRONG TOO: thread-second summation overweights phases that scale
BADLY across threads. Compute-bound inner FFT scales ~linearly (no distortion);
memory-bound scattered postprocess contends, so its thread-seconds balloon. So the
contaminated "post=70%" was partly measuring contention, not single-stream cost.

THE FIX (review's protocol, now applied): profile at T=1 (stride_set_num_threads(1))
+ a CONSERVATION ASSERT Sigma(phases)/wall in [0.9,1.1]. At T=1 conservation passes
(cons=0.99/1.00). Two permanent protocol rules added (like the byte-gate):
  1. Profile single-threaded (or with per-thread accumulators).
  2. No phase table is admissible without its conservation check.

THE RECOVERED + ABLATION-VERIFIED PICTURE (T=1, N=256):
  - T=1 phase timer: post = 68.6% (cons=1.00) — but the timer is still somewhat
    granular, so the ZERO-INSTRUMENT ABLATION is the ground truth:
  - Ablation (stub _r2c_postprocess, no timers anywhere):
      full ~92,900 ns ; postprocess-stubbed ~60,300 ns
      => postprocess TRUE cost ~32,600 ns (~35% of runtime).
  - Fused total ~92,500 ns => the fused codelet costs the SAME ~32us as the
    hand-rolled postprocess. Fusion RELOCATED the work, did not reduce it.

FINAL HONEST VERDICT:
  - "postprocess is a large fraction" SURVIVES: ~35% (ablation-proven, not 6%,
    not the inflated 70%). The fusion was aimed at a real target.
  - The fusion is CORRECT (2.5e-14) and relocates the pass cleanly at equal cost.
  - Break-even-here is CONSISTENT WITH "scatter is free on this box" (L3~=DRAM,
    no prefetcher), NOT with "postprocess wasn't the target." The access-pattern
    win (block-local vs scatter) is the unmeasurable-here axis.
  - So: model (a) is correct + zero-regression + real-target, value pending metal.
    The i9/Zen4 A/B (default-vs-fused, gate exists) is now a SHARP test: postprocess
    is provably ~35%, so if the access pattern matters on silicon, the win is
    bounded by that 35%; if it's still break-even on silicon, the scatter was
    never the cost and model (b)'s traffic-deletion (or the native rfft path) is
    the only remaining lever.

CONDUCT NOTE (for the record): the contradiction was caught because the
conservation check was run and the result published against the prior verdict.
That check is now permanent protocol. Credit to the review for the thread-second
mechanism; I had the conclusion but the wrong cause.

### How to reduce postprocess cost — the decomposition + the answer (T=1, N=256 K=256)

Postprocess true cost ~32,600 ns (ablation). Floors: compute ~16,400 ns,
bandwidth ~14,000 ns (overlap => real floor ~16,400). So ~2x over floor.
But the per-pair MICROBENCH (codelet vs hand-rolled inline, same data, T=1) shows
codelet 20,473 ns vs hand-rolled 19,162 ns for one block of pairs — only 7%
apart, and ~19us is close to the ~16us floor. So:

LEVER 1 (codelet arithmetic quality): the rt codelet is 18 ops + 5 set1 broadcasts
  per pair, only 7% slower than hand-rolled. MINOR — codelet is competitive,
  not the problem. (Could shave the 7% but it's not where MKL wins.)

LEVER 2 (ranged codelet, kill per-pair call boundary + enable cross-pair ILP):
  worth ~the 7% above plus some latency hiding. MINOR-to-MODERATE, and it's the
  same fix as rfft.h's stage-0 granularity (converge the two).

LEVER 3 (model b — fuse the butterfly INTO the last FFT stage): THE STRUCTURAL
  WIN, and where MKL beats us. The postprocess READS 512KB of scratch that the
  last FFT stage just WROTE. Fusing the butterfly into the last stage (operate on
  the stage outputs while still in registers) DELETES:
    - the 512KB scratch read (~7us at 75GB/s), AND
    - the last stage's 512KB scratch write (~7us)
  ~14us of last-stage<->postprocess round-trip eliminated. MKL's real overhead
  ~19us vs our ~32us is consistent with MKL doing exactly this. This is the only
  lever that attacks TRAFFIC (not just access pattern), so it's the one that wins
  regardless of cache hierarchy — and the one this container CAN partly measure
  (it deletes real memory ops, not just reorders them).

VERDICT: the postprocess is ~32us = ~16us irreducible (compute+bw floor, codelet
already near it) + ~16us redundant round-trip traffic that model (b) deletes.
Levers 1+2 chase the 7% codelet gap (not where MKL wins). LEVER 3 (model b) is
the answer: fuse into the last DIT stage so the butterfly consumes last-stage
outputs from registers and writes X directly — no scratch round-trip. That needs
the stage_tw input (the design's original model b) + emitting the codelet AS the
last stage rather than as a separate terminator the executor calls after the FFT.

NEXT STEP if pursuing the r2c win: build model (b). It is the slice-1/2/3
machinery (the r2c_term fold is proven correct) PLUS folding the last-stage radix
butterfly into the same codelet body (stage_tw twiddle) so the DFT-r outputs feed
the fold in-register. The fold half is done and verified; the addition is
prepending the last-stage butterfly to the DAG. Gate: the existing FUSED-vs-
DEFAULT brute-Hermitian check; success = the scratch read/write pair disappears
from the executor (measurable: stub-ablation of the WHOLE last-stage+postprocess
vs model-b should show model-b ~14us cheaper than default).

### FULL GAP DECOMPOSITION (T=1 ablation, before model (b)) — review corrections 1+2

Ablation (stub each phase, measure total delta, zero timers, T=1, N=256 K=256):
  pack  (fused first stage):  ~44,375 ns  (48%)
  post  (terminator):         ~33,497 ns  (36%)
  inner (middle FFT stages):  ~14,716 ns  (16%)
  TOTAL ~92,588 ns ;  MKL ~56,300 ns  (1.64x)

CORRECTION 1 (the deletable round-trip spans two phases): model (b) deletes the
last-stage<->postprocess round-trip = ~7us READ (inside post) + ~7us WRITE
(currently booked under INNER, not post). So the re-profile shape is:
  post  -> ~18,600 ns (parity with MKL's ~19us terminal phase)
  inner -> shrinks ~7,000 ns
  total -> ~78,600 ns (~1.40x)
Pre-register THAT shape — expecting all 14us out of the post line would make a
correct model (b) look like it under-delivered. Container prediction is a RANGE:
~8-14us total drop (deleted traffic converts to time only where not overlapping
compute); the low end still confirms.

CORRECTION 2 (model (b) closes the POST gap, not THE gap): post-model-(b) ~1.40x,
NOT parity. The remaining ~22us lives in PACK (and a little inner). "That's where
MKL is winning" is true of the POST PHASE, false of the war. Model (b) is
NECESSARY, NOT SUFFICIENT.

THE BIGGER PRIZE IS PACK: pack is ~44us = 48% of runtime, ~3.2x its ~14us floor
(input read 512KB + scratch write 512KB), so ~30us of addressable headroom — MORE
than post's entire cost. If pack reached its floor, total ~48us = ~0.86x (BEATS
MKL) even before model (b). So the post-model-(b) roadmap is now NAMED + COSTED:
  1. model (b): -> ~1.40x  (post reaches MKL parity; build now, fold is proven)
  2. pack to floor: -> ~0.86x  (the larger lever; needs its own ablation-driven
     investigation — why is the fused first stage 3.2x its bandwidth floor?)
The whole MKL gap is now three costed pieces, not one fused mystery.

NEXT: build model (b) (agreed — the fold half is done + verified). THEN attack
pack with the same ablation discipline (its 3.2x-over-floor is the next unknown).

### Model (b) BUILD PLAN (agreed; the fold half is done, this is the structure)

GOAL: emit the codelet AS the last DIT stage — fold (last-stage radix butterfly +
stage twiddle) THEN (r2c terminator fold) into ONE DAG, so DFT-r outputs feed the
r2c fold in-register and the scratch round-trip never happens.

PIECES (in build order, each gated in isolation per the cadence):
1. MATH BUILDER (dft_r2c.ml): dft_expand_r2c_term_laststage. Compose:
   inner = Dft.dft r (the last-stage radix-r butterfly on the column's r inputs,
           with the per-leg stage twiddle W_{N'}^{j.k} applied) -> produces the r
           frequencies of column k AND column m-k;
   then the r2c terminator fold (the PROVEN dft_r2c_term_pair_rt algebra) on those
   in-register outputs.
   Gate: eval vs brute r2c (same harness as slice 1/3b), per column pair.
2. STAGE-TWIDDLE PACKING (rider): model (b) needs 2(r-1) stage twiddles (two
   columns x (r-1) per-leg) + 1 fold twiddle per column-pair call. The IR
   Twiddle(slot) is one table => pin a slot-partition convention:
     slots [0 .. r-2]      = column k stage twiddles
     slots [r-1 .. 2r-3]   = column m-k stage twiddles
     slot  [2r-2]          = fold twiddle W_N^f  (mirror via the Neg identity)
   State this in the doc before emitting; the plan packs the table to match.
3. EMIT (emit_c.ml): r2c_term_laststage signature = the column-pair inputs (2r
   complex) + the packed twiddle table + dual output (Xp/Xm) + strides. Reuses
   the r2c_term dual-output store; adds the r-input load addressing.
4. FORWARD _until EXECUTOR (stride_executor.h): _stride_execute_fwd_slice_until
   (twin of _stride_execute_bwd_slice_until) runs stages start..nf-2 (STOP before
   last). The r2c worker calls _until then the fused last-stage codelet per column
   pair, writing X directly to out. Deletes the last-stage scratch write AND the
   postprocess scratch read.
5. GATE: brute-Hermitian PRIMARY (the existing FUSED-vs-DEFAULT 2.5e-14 harness,
   extended to the model-b path). Few-ulp consistency, NOT bit-identity.
6. MEASURE (pre-registered, correction 1): post -> ~18.6us, inner -> -7us, total
   -> ~78.6us (~1.40x), container drop range ~8-14us. Verify via stub-ablation of
   the WHOLE last-stage+postprocess: model (b) should be ~14us cheaper than
   default's (last-stage-write + postprocess-read).

REGISTER BUDGET (rider): r=8 two-column working set (2r complex + twiddles) fits
32 zmm with normal spill. r=16 won't — R=8 FIRST (existing plan); r=16 spill is
codelet-level noise vs the deleted round-trip. Specials stay executor-side. Opt-in
stays rider-4 additive.

AFTER model (b): attack PACK (the bigger 48% phase, 3.2x over floor) with the same
ablation discipline — that is the lever that actually reaches MKL parity.

### Model (b) PIECE 1 (DONE, gated): the last-stage-fused math builder
lib/dft_r2c.ml: dft_r2c_term_laststage ~sign np r m. Composes, for a column pair
(k, m-k), TWO stage-twiddled DFT-r's (Dft.dft r with per-leg PRE-twiddle
W_{N'}^{j*k}, DIT fwd) + the PROVEN r2c terminator fold on the in-register DFT-r
outputs. Returns 2r outputs: [X[k+s*m], X[mirror]] for s=0..r-1.

Layout mapping verified: col k slot s pairs with col (m-k) slot (r-1-s) (= the
layout proof, re-confirmed numerically). Twiddle packing convention pinned:
  slots [0..r-1]   = col k stage tw ;  [r..2r-1] = col m-k stage tw ;
  slots [2r..3r-1] = fold tw W_N^{k+s*m}  (mirror via the Neg identity).

ISOLATION GATE (eval the DAG vs brute r2c, feeding sub-DFT_m outputs as the
last-stage inputs):
  r=4 m=4  (N=32):  4.6e-15
  r=8 m=16 (N=256): 4.4e-13
  r=8 m=8  (N=128): 1.4e-13   ALL PASS.
The full last-stage+fold DAG is correct. Additive: 48/48 baseline byte-identical.

REMAINING model (b) pieces (next):
2. emit (emit_c.ml): r2c_term_laststage signature — 2r complex inputs + packed
   twiddle table (3r slots) + dual output (Xp/Xm interleaved, 2r) + strides.
3. forward _until executor (stride_executor.h): _stride_execute_fwd_slice_until
   (run stages 0..nf-2), then call the fused last-stage codelet per column pair.
4. gate: brute-Hermitian FUSED-vs-DEFAULT (extend the existing harness).
5. measure (pre-registered, correction 1): post->~18.6us, inner->-7us,
   total->~78.6us (~1.40x); container drop range ~8-14us; verify via stub-ablation
   of last-stage-write + postprocess-read disappearing.

### Model (b) PIECE 2 (DONE, gated): emit the fused last-stage codelet
- lib/dft_r2c.ml: dft_expand_r2c_term_laststage wrapper (2r outputs, even slot
  -> Xp[s], odd -> Xm[s]).
- lib/emit_c.ml: r2c_term_laststage signature (ink/inm column bases, packed tw,
  Xp/Xm dual output, is_leg/osp/osm strides), input addressing (j<r -> ink leg j;
  r+j -> inm leg j), output addressing (even->Xp, odd->Xm), tw broadcast.
- lib/gen_main.ml: --r2c-term-ls + --r2c-term-ls-r, dispatch (half=N/2, m=half/r),
  name radix{N}_r2c_term_ls_r{R}_{sgn}_{isa}, hoist on.
VERIFIED end-to-end: emitted/compiled codelet vs brute r2c, N=256 all interior
column pairs -> 5.4e-13 PASS. Gate: benchmarks/run_r2c_term_ls_gate.sh.
Codelet: 16+16 leg loads, 48 tw broadcasts, 16+16 Xp/Xm stores. Additive: 48/48
baseline byte-identical.

PIECE 3 (NEXT): forward _until executor. _stride_execute_fwd_slice_until runs
stages 0..nf-2 (stop before last); the r2c worker then calls this codelet per
interior column pair, feeding the two columns' r legs from scratch and writing X
directly. This is where the last-stage scratch WRITE + postprocess scratch READ
are deleted (the ~14us round-trip). Then gate (brute-Hermitian) + the
pre-registered measurement.

### Model (b) PIECE 3 (IN PROGRESS): _until executor in; twiddle-convention finding

DONE:
- stride_executor.h: refactored _stride_execute_fwd_slice_from into a _range
  (start,stop) core + thin _from (stop=num_stages) and _until (explicit stop)
  wrappers. Default path TRANSPARENT (r2c still correct, xcheck passes). This is
  the partial-execution entry the rider named (fwd twin of bwd_slice_until).

FINDING (must resolve before wiring the codelet legs): probed the REAL inner plan
for half=128 (N=256 r2c). It is (4,4,8), 3 stages, last stage radix=8, 16 groups,
stride=256=B. Critically, last-stage needs_tw per group = 0111011101110111 — NOT
uniform. So:
  - the last stage DOES apply per-group twiddles, but via the plan's grp_tw_re[g]
    / cf_all tables with a specific per-group PATTERN (group 0 of each 4-block is
    no-twiddle), NOT the uniform W_{N'}^{j*k} formula my piece-1 builder assumed.
  - => model (b)'s stage-twiddle codelet inputs must be fed from the PLAN's actual
    grp_tw for the relevant groups, matching the executor's layout, not a formula.
  - the piece-1/2 math gate fed a formula-based stage twiddle and passed — that
    proved the FOLD algebra + DFT-r composition are correct, but the EXECUTOR
    wiring must supply the real per-group twiddles (and confirm the column<->group
    mapping: which group g corresponds to frequency-column k).

PIECE 3 REMAINING (next, precise):
  1. Map last-stage GROUP g -> frequency column k (the executor's group_base[g]
     layout vs the column index the codelet's twiddle packing expects).
  2. Feed the codelet's stage-twiddle slots from grp_tw_re[g]/grp_tw_im[g] for the
     two paired groups (NOT a W^{jk} formula). Confirm needs_tw[g]=0 groups get
     identity twiddles.
  3. Worker: replace _slice_from(...,1) with _until(...,1,nf-1) + per-column-pair
     fused codelet calls reading the last stage's leg layout (group_base + j*stride)
     and writing X direct.
  4. Gate brute-Hermitian FUSED-vs-DEFAULT; then the pre-registered measurement.

NOTE: this is why wiring is gated step-by-step — the formula-twiddle assumption
that passed the isolated math gate is NOT what the production plan uses. Caught
before producing silently-wrong output.

### Model (b) PIECE 3 — convention RESOLVED + group mapping verified

Step 1 (group->column map, verified): last stage = 16 groups, group_base[g]=g*2048,
leg stride = 256 = B (so group g leg j at scratch row g*8+j). Group g produces
column k where group g slot s -> frequency iperm[g*8+s] = 4g + s*16. So:
  - group g -> column k = 4g (columns step by 4 = first-stage radix).
  - group-pairing CONSISTENT: each group mirrors into exactly ONE partner group,
    slot s <-> partner slot r-1-s (the layout proof at group granularity). E.g.
    group 1 (k=4) <-> group 3 (k=12); self-paired groups 0 (DC/Nyq) and 2.

Step 2 (twiddle convention, RESOLVED empirically): probed the REAL inner FFT.
  - grp_tw is BROADCAST-equivalent: grp_tw_re[g][(j-1)*K + k] is CONSTANT across k
    (K redundant copies). So my codelet's set1_pd broadcast is correct; no
    per-element twiddle vector needed. The earlier "ABI mismatch" fear was wrong.
  - PRE vs POST: tested both against the real last-stage output Z. PRE-MULTIPLY
    MATCHES (stage tw applied BEFORE the DFT-r), POST does not. My piece-1/2
    builder uses pre-multiply => CORRECT. (The "post-multiply" code comment was
    about the t1 codelet internals, not the net DIT effect.)
  - Stage twiddle sources for the codelet: leg 0 = cf0_re[g]/cf0_im[g]; legs
    1..r-1 = grp_tw_re[g][(j-1)*K]; identity for needs_tw[g]=0 groups.

Also DONE: added _stride_execute_fwd_slice_until to BOTH stride_executor.h AND
proto_stride_compat.h (the compat header is the live path the bench/production
uses — its _from has its own copy, so _until needed adding there too). Default
path unchanged (r2c still correct, xcheck passes).

PIECE 3 REMAINING (next, now unblocked): the worker wiring —
  - iterate group pairs (g, partner_g) over the 16 last-stage groups;
  - per pair, build the codelet's stage-twiddle table from cf0+grp_tw of both
    groups (pre-multiply, broadcast) + the fold twiddle W_N^f;
  - feed the two groups' 8 legs (from the _until-processed scratch) to
    radix256_r2c_term_ls_r8, write X direct;
  - self-paired groups (DC/Nyquist + the k=hf/2 column) stay scalar;
  - gate brute-Hermitian FUSED-vs-DEFAULT, then the pre-registered measurement.

### Model (b) PIECE 3 — COMPLETE: correctness PASS, perf REGRESSION on container

DONE (the full wiring):
- core/r2c.h: stride_r2c_data_t gained ls_fwd fn-pointer (14-arg, NULL default).
  _r2c_laststage_fused() iterates last-stage GROUP PAIRS calling the fused codelet.
- OOP worker opt-in branch: when ls_fwd set, runs _until(stages 1..nf-2) + the
  fused codelet AS the last stage + scalar specials. Default (ls_fwd NULL)
  unchanged: 48/48 byte-identical maintained.

ITERATION LOGIC (debugged via the gate, 3 bugs caught + fixed):
- done[] array; per group g find partner pg = perm[half-kcol]/r; process the pair
  ONCE (mark both done). Verified coverage: 7 cross-pairs + group-0-internal +
  group-2-self-paired = 0 uncovered, 0 double.
- group 0 (kcol=0): DC/Nyquist + center column + group-0-INTERNAL interior pairs
  (freqs s*m) handled in the CALLER (scalar fold), last stage run once.
- self-paired GROUP (group 2, partner=itself): call the PROVEN codelet with
  ink=inm=this group's legs — the codelet's Xm[s] reads DFT slot r-1-s of inm =
  the mirror freq, so one call is correct (this REPLACED a hand-rolled fold that
  had a sign error — bug #3, caught by the gate showing exactly group-2's freqs
  {8,24,...,120} wrong).
- bug #1 was double-running group 0 (DC and "center group" were the SAME group 0);
  bug #2 was the partner-skip missing (120 double-covered). Each caught by the gate
  localizer printing which freqs were wrong, then fixed.

GATE (brute-Hermitian PRIMARY, few-ulp consistency): 
  # r2c N=256 MODEL-B vs DEFAULT: max abs diff 2.842e-14  PASS
Model (b) is NUMERICALLY CORRECT for the full transform (every frequency).
Gate: benchmarks/run_modelb_gate.sh.

PERF (T=1, conservation-checked phase profile) — the PRE-REGISTERED prediction
partially held but net is a REGRESSION on this container:
| phase            | default-ish | model-b | delta    |
|------------------|-------------|---------|----------|
| pack             | ~55,900     | ~44,100 | (noise, same code) |
| inner / until    | ~33,300     | ~16,700 | -16,600  (round-trip WRITE deleted ✓) |
| post / laststage | ~63,700     | ~90,300 | +26,600  (codelet path slower ✗) |
| TOTAL(T=1)       | ~113,000    |~130,000 | +17,700  (-15.7%, SLOWER) |

So: the round-trip WAS deleted (until shrank ~16us, matching correction-1's
prediction of inner -7us + the write). BUT _r2c_laststage_fused is ~26us MORE than
the old streamlined postprocess, wiping out the gain. Root cause (measured via the
phase split, not assumed): the per-group-pair codelet path does 8 separate calls/b0
each interleaving the DFT-8 with the fold and doing SCATTERED strided output writes
(Xp at kcol*K, Xm at mir0*K), versus the old path's tight _slice_from last-stage
loop (16 groups streamed) + one linear postprocess pass. On THIS container
(L3≈DRAM, no prefetch) the deleted round-trip's MEMORY saving is real and visible,
but the codelet's worse compute/write pattern costs more than the saving. The
balance is container-specific: real HW with working cache hides BOTH the round-trip
(smaller saving) AND the scattered writes / call overhead (smaller penalty)
differently. NEEDS METAL to adjudicate whether model (b) is a net win.

STATUS: model (b) is CORRECT and OPT-IN (ls_fwd NULL default = zero regression).
It is NOT enabled by default. The structural hypothesis (delete the round-trip)
was VALIDATED (until -16us) but the implementation's codelet path needs either
(a) metal confirmation that the scattered-write penalty vanishes, or (b) a
batched-call / contiguous-output redesign of _r2c_laststage_fused before it beats
the streamlined default. Recorded as a real result: necessary structure proven,
this implementation not yet sufficient on the container.

### Model (b) — the MONOLITHIC-emission tax, DIAGNOSED + FIXED (doc-58 seam)

ROOT CAUSE (the regression was NOT math, NOT model (b), NOT the wiring — it was the
EMISSION MODE): the emitted codelet's own provenance header said it outright:
  "Construction: MONOLITHIC (below blocking threshold...)"
  "PRESSURE: SPILLS (peak_live > budget)"
The fused DAG holds two stage-twiddled DFT-8s simultaneously (the fold needs both
columns' outputs) = 16 complex = 32 zmm of live data before any twiddle/temp. Over
budget by construction. And the r2c dispatch hardcoded (assigns, [], None) — no
spill markers, no ct — so the blocked recipe NEVER engaged; gcc handled the
overflow with a stack spill storm.

MEASURED (verified against the asm, not assumed):
  - 163 stack-relative movs vs 304 arithmetic ops per 8-lane loop iteration.
  - traffic: 163 x 64B x 32 iters x 8 calls ~= 2.7 MB stack traffic per block.
    We deleted a ~1 MB scratch round-trip and reintroduced ~2.7 MB through the
    stack frame (L1-resident, but ~42k extra vector mem instructions serialized
    into the dependency chains) = the +26.6us laststage tax.
  - same monolithic-vs-recipe gap the c2c side hit twice this session, now
    colliding with the first r2c DAG big enough to need the seam.

THE FIX (machinery already owned — the doc-58 PASS-1/PASS-2 seam):
  - dft_r2c.ml: dft_expand_r2c_term_laststage_spill captures the two columns'
    DFT-r outputs as spill markers (col k -> slots 0..r-1, col m-k -> r..2r-1),
    returns (assigns, markers, Some (2, r)). PASS-1 clusters split by slot/r (the
    two DFT-r's, register-resident each); PASS-2 = the fold reading slot pairs.
    Mirrors Dft.dft_expand_twiddled_spill's contract exactly (markers are
    topology-agnostic, as the newsplit prototype proved).
  - gen_main.ml: term-ls dispatch now calls the _spill variant (passes markers +
    ct through instead of [], None); recipe_applicable gets the term-ls case so
    spill+SU engage. The custom emit branch was confirmed to route through the
    STANDARD spill-aware body emitter (PASS-1/PASS-2 at the shared cluster_split
    path), so this was the hour-fix not the day-fix.

RESULT (the seam delivered, BEATING the prediction):
| phase            | MONOLITHIC | BLOCKED 2x8 | delta   |
|------------------|------------|-------------|---------|
| pack             | ~44,100    | ~39,000     | (noise) |
| until (rt delete)| ~16,700    | ~14,400     | (noise, still deleted) |
| laststage+spec   | ~90,300    | ~44,500     | -45,800 |
| TOTAL(T=1)       | ~151,000   | ~98,000     | -53,000 |
Codelet header flipped MONOLITHIC -> "BLOCKED two-pass CT 2x8; seam through L1
scratch by design (doc 58)". 16 spill slots declared. Correctness UNCHANGED
(2.842e-14, both gates). Full-path total regression went from ~-12% to ~-1.3%
(near break-even on the noisy container).

The laststage phase ~halved (90->44.5us), beating the predicted ~64us target. The
spill tax is essentially gone. The until phase's -16us round-trip deletion was
never wrong; removing the emission tax let it keep its money. STILL needs metal to
confirm the net win (container T=1 is noisy + L3≈DRAM), but the structural story is
complete: hypothesis validated (round-trip deleted), tax identified (monolithic
spill), tax removed (doc-58 seam), correctness preserved.

SECOND-ORDER (deferred, measure separately so effects don't conflate like pack/post
once did): the scattered strided output writes (Xp at kcol*K, Xm at mir0*K) + the
8-separate-call structure. Worth a batched-call / contiguous-output redesign AFTER
the recipe win is confirmed on metal.
