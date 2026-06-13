# Verdict (v2, measurement-backed): why we lose to MKL on r2c / c2r

Supersedes v1. v1 was right that the gap is glue not codelets, but wrong to call
the glue "inherent to the half-complex method." Direct phase profiling of the
executor that actually loses shows the dominant cost is an UNFUSED forward pass
that the file's own backward path already eliminates — a fixable asymmetry, not
an algorithmic floor.

## The discriminating measurement (the executor that lost the MKL number)

bench_headline_3way routes through stride_execute_r2c -> _r2c_worker_fwd_oop
(the r2c.h half-complex executor). Instrumented with gated VFFT_R2C_PROFILE phase
timers, N=256 K=256, container, stable across 3 runs:

| phase | share of r2c-256 runtime |
|---|---|
| pack (fused-first strided phase) | ~40-49% |
| inner complex FFT | ~6% |
| postprocess (Hermitian butterfly) | ~44-55% |

vs MKL: r2c-256 0.60x (lose), r2c-64 0.86x. The inner FFT — what we are good at —
is ~6% of the transform. ~94% is glue; the biggest single phase is postprocess.

## Which item owns the gap: item 3 (unfused forward postprocess)

- ITEM 3 (r2c.h unfused forward butterfly): CONFIRMED, DOMINANT. Postprocess is
  ~50% of the transform and ~9x the inner FFT. It is a separate full half-spectrum
  sweep at memory bandwidth, and its mirror reads (perm[mirror]*B) are scattered
  by the inner DIT permutation, defeating the prefetcher on half the pass. The gap
  grows N=64->256 (0.86->0.60) as postprocess grows (30%->55%): the scatter
  signature. The fix already exists in-file for the other direction —
  _r2c_fused_last_stage fuses the backward unpack into the final DIF stage. The
  forward fused the FIRST stage (pack) but left the butterfly standalone. Fusing
  the Hermitian butterfly into the last DIT stage removes the pass and the
  scatter. This is MKL's move.

- ITEMS 1-2 (rfft.h call granularity): CONFIRMED but on the OTHER executor.
  rfft.h native path (rfft_execute_fwd_packed, N=256 (4,4,16)) across K: leaf +
  stage0_cols ~60% at K=8, leaf fraction rises with K (27.8%->47.6%). Leaf-fold
  regression (item 1: 64 unfolded calls at vl=8 where one folded call at vl=512
  is legal; the stage loop folds, the leaf doesn't) and stage-0 column
  granularity (item 2: per-column vl=K, batched by the RANGED variants) are real
  and material to rfft.h. Not the MKL-loss measurement (that was r2c.h), but they
  gate whether the native path can win.

## Corrected conclusion

The r2c loss is NOT an algorithmic floor and NOT codelet quality:
  1. (r2c.h, dominant) unfused forward postprocess — ~50% of runtime, fixable by
     the fusion the backward path already has. Highest leverage; hits the largest
     phase and the scatter at once.
  2. (r2c.h, second) the pack pass (~45%), traffic-bound; harder (cross-stage
     register fusion) but attackable.
  3. (rfft.h, separate) leaf-fold regression (5-line fix mirroring the stage fold)
     + stage-0 RANGED granularity (already built, behind #ifdef). Decide whether
     the native path can win once forward fusion lands.

The native rfft.h path is the cleanest long-term answer (no pack, no separate
recombination) but is NOT the only fix and has its own unfixed regressions. The
fastest win on the losing executor is fusing the forward postprocess in r2c.h.

## Caveats
- Container timer is directional, but phase SHARES are within-run proportions, so
  "inner 6%, postprocess 50%" is robust to absolute-ns noise (stable across runs).
- The fair both-in-preferred-layout matrix still belongs to metal.
- c2r is symmetric: the forward's postprocess <-> backward's preprocess; backward
  already fused, forward is the open one. Verify c2r symmetry when fixing.

## Instrument added
core/r2c.h has gated VFFT_R2C_PROFILE phase timers (pack/inner/post) on both
forward workers. Zero effect on default builds; build any r2c bench with
-DVFFT_R2C_PROFILE to reproduce.

## ADDENDUM: the "pack" phase challenged and resolved (review point 0)

The 40-49% "pack" share was suspicious — a fused pack should not exist as a
phase. Probed: stage0.n1_fwd is a valid pointer, the FUSED path ran (not the
fallback copy-loop). So "pack" is NOT a fallback-memcpy regression (no free
registry-entry fix). It is the fused stage-0 FFT work wearing a coarse label.

Bandwidth sanity check: the pack timer brackets _r2c_fused_first_stage ALONE
(not the inner slice). It measures ~200-247us at N=256 K=256. Pure 512KB
read+write on this container ~14us. The 15x ratio is NOT anomalous: stage 0 is
the largest-radix butterfly stage applied to the FULL input, plus the even/odd
repack — it is real FFT arithmetic, not data movement. Comparing it to memcpy
was the wrong yardstick. Stage 0 is co-dominant because it works on the full
array while the remaining inner stages run B-blocked / cache-resident.

CONSEQUENCE FOR THE PLAN: pack is correctly fused and is NOT a target. The fusion
effort focuses entirely on the postprocess (item 3). The ladder:
  0. [DONE] fallback ruled out — fused path confirmed, pack is legit stage-0 work.
  1. NATURAL-ORDER OUTPUT (cheap half): the postprocess scatter is real (mirror
     reads via iperm/perm digit-reversal). rfft.h ALREADY has natural-order
     output (rfft_execute_fwd_natural / the D2 terminator). Routing r2c's inner
     through a natural-order plan turns the mirror read into two sequential
     streams (forward from f=1, backward from halfN-1). Kills the scatter, keeps
     the pass. Since the gap GROWS with the scatter (0.86->0.60 N=64->256), this
     alone may recover most of the N-growth. ~1/10th the effort of full fusion.
  2. FULL FUSION (the MKL move): fuse the Hermitian butterfly into the last DIT
     stage. Structurally the hc2c_nat skeleton (already walks k, m-k pairs); the
     one real difference is the r2c butterfly twiddle W_N^f at N=2*halfN (a
     half-step power the hc quadrant lacks) — a generator VARIANT, not an
     invention. Pass dead. Converges r2c.h toward the rfft.h design.
  3. Re-profile; expect stage-0/input to be the new dominant phase, at which
     point the honest question is whether post-fusion r2c.h and post-RANGED
     rfft.h are the same executor under two names (the Phase-1/Phase-2 merge).

c2r symmetric audit: backward already fuses its unpack; the PRE-process
(reconstructing Z from X via c2r_im_buf) is the same unfused-pass-plus-mirror
pattern on the input side. Same instrument, same questions.
