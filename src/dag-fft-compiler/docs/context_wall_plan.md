# Context wall: attack plan (section 67)

## The mechanism (model fitted to three measured regimes)
hc2hc_4 column: hot 0.23us | warm-plane looped 0.62 | full-pipeline
0.94-1.18. A column call touches ~4r short rows (2KB at K=256, one
4KB page each). The L2 streamer locks on after ~2 misses per row and
cannot prefetch across page boundaries, so each call pays roughly
2 misses x 4r rows of L3-class latency (~0.6us at r=4) before
streaming even starts. Residency (the warm regime) removes most of
it. The wall is COLD-START LATENCY PER SHORT ROW, not bandwidth,
capacity, spills, or twiddles (sections 64-66 falsified each).

## Levers, re-derived

E1. SOFTWARE PREFETCH (executor-side, all plans).
  Before issuing column k, prefetch column k+1's row starts: 2r reads
  (in_re ascending, in_im descending) + 2r write rows (RFO warm-up),
  line 0 + line 8 of each. 4r-8r prefetches amortized per ~1us call.
  Hides the streamer warm-up -> toward the warm regime.
  PRE-REGISTERED: (16,16) cols 41 -> [22, 32]us.

E2. RESTORE THE Q-FOLD where legal (Kb == K, single slab).
  The fold (vl = Q*K) makes Q>1 stages' slot streams CONTIGUOUS at
  Q*2KB instead of Q separate 2KB rows — 4x longer streams at
  (4,4,16)'s d=1. It was removed only because lane-blocking broke the
  (q,lane) fold; blocking is now off by default. Keep the per-q path
  for Kb < K.
  PRE-REGISTERED: (4,4,16) d=1 cols 33 -> [15, 25]us.

E3 (conditional on E1 data): leaf prefetch tuning. The leaf's 32KB-
  stride 16-stream pattern is the same disease; same medicine.

WITHDRAWN: "column scheduling for line reuse" (section 66 lever 2).
  Columns within a stage touch disjoint rows; there is no reuse to
  schedule. The read side is already two monotone sweeps.

QUEUED (behind E1/E2 results): cross-stage row blocking (consume leaf
  outputs while hot — design-heavy, software-pipelined schedule);
  per-pos twiddle policy for hc; mirror-paired spill-pass scheduling
  (ceiling 4-6us, section 66b).

## Acceptance for this attack
(16,16) clean total <= 70us with stretch <= 62 (the original P3
acceptance). Falsification: if E1+E2 land < 10us combined, the model
above is wrong and the next instrument is perf-counter-level (or
real-hardware) characterization rather than more executor surgery.

## OUTCOME (same session): the stop-clause FIRED

E1 (prefetch): NEGATIVE. A/B at (4,4,16): 109.5us off vs 113.2 on;
(16,16) 84.9 pre vs 90.1 with. Default flipped to opt-in
(VFFT_RFFT_PREFETCH); the mechanism note is in rfft.h.
E2 (Q-fold restore): ~+6us at (4,4,16); kept (also structurally
cleaner). Pre-registered bands: BOTH missed (ledger #25).
E1+E2 < 10us combined => per the falsification clause above, the
cold-start model is WRONG at the executor-schedule level on this
container. The residual warm-vs-pipeline gap is not addressable by
instruction-level scheduling here. Next instrument, as pre-committed:
real hardware (i9-14900KF / EPYC 9575F) with perf counters — which is
also where the global #1 queue item (stride-wisdom regen) already
lives. Container-side executor tuning for the native rfft is CLOSED
at (16,16) ~ 85us until then.
