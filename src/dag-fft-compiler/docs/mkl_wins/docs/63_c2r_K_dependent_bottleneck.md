# The c2r batch-width finding: per-call overhead is the fixable part; the large-K gap is shared with the forward path

## One-line finding

Optimizing VectorFFT's inverse real FFT (c2r) against MKL surfaced a clean
split at small batch width: the path was **per-call-overhead-bound**, and a
ranged codelet that collapses many calls into one took it to parity with MKL.
At large batch width the same optimization did essentially nothing -- but the
reason is NOT what the first draft of this document claimed. The large-K gap is
shared with the forward r2c path (which shows the same drop, already fully
optimized), and the forward team had already measured and rejected the
"capacity-bound, fix with lane-blocking" hypothesis. The honest finding is
two-part: (1) per-call overhead is a real, fixable, K-sensitive cost, and
ranged/fusion fixes it at small K; (2) the residual large-K gap is a shared
cascade-vs-MKL batch-scaling property, not a c2r-specific bottleneck, and the
mechanism is still open. **The instructive part is how easy it was to tell a
plausible memory-bound story that measurement then falsified.**

## The measurements (256-point c2r, (8,32) factorization)

Ratio is mkl/ours, single process, min-of-120, single thread, MKL pinned to one
thread. The container has ~1 vCPU and no PMU, so absolutes are thermal noise;
the RATIO in one process is the trustworthy signal and it is stable across
reruns (K=8: 0.97-1.00; K=64: 0.78-0.80). The MKL opponent was verified correct
(real, unnormalized, matching N*x to ~1e-11) before any timing was trusted.

| optimization applied            | K=8 ratio | K=8 cyc | K=64 ratio | K=64 cyc |
|---------------------------------|-----------|---------|------------|----------|
| baseline (scalar mid, per-call) | 0.68      | 8092    | 0.69       | 62786    |
| + SIMD mid-column inverse       | 0.80      | 6966    | 0.77       | 56344    |
| + folded leaf (one vl=S*K call) | 0.94      | 5860    | 0.78       | 54946    |
| + ranged interior (15 calls->1) | 0.99      | 5494    | 0.78       | 56178    |

ratio < 1.0 means MKL is faster. 1.0 is parity.

Read the last two rows: the ranged interior moved K=8 from 0.94 to 0.99 (to
parity) and moved K=64 from 0.78 to 0.78 (nothing, inside the noise band).

## Why the same change helps K=8 and not K=64

The ranged codelet's only effect is removing **per-call overhead**. The c2r
interior at (8,32) is `kmax = m/2 - 1 = 15` columns; the per-k path issues 15
separate codelet calls, the ranged path issues 1 (the codelet walks all 15
columns internally, advancing re-streams up and im-streams down by a fixed
column stride per step). So ranged deletes 14 call boundaries: 14 prologue/
epilogue sequences, 14 lost opportunities for the lane loop to run
uninterrupted, 14 chances for the scheduler to stall at a call edge.

That overhead is a **fixed cost per call**, independent of how much data each
call processes. The work per call scales with K (the batch / lane width):

- **K=8**: each call processes 8 lanes. The per-call overhead is a large
  fraction of the call's total time, so deleting 14 of them is a big relative
  win. The path is *overhead-bound*. Ranged closes the gap.

- **K=64**: each call processes 64 lanes, 8x the work. The same fixed overhead
  is now a small fraction of each call's time, so deleting it barely moves the
  total. The path is bound by something else.

## What K=64 is actually bound on: corrected by measurement

An earlier draft of this document claimed K=64 is memory-bandwidth-bound: the
two scratch planes (2 * 256 * 64 * 8 = 256 KB) exceed L2, so the cascade
streams from a slower level and call-collapsing cannot help. **That explanation
did not survive measurement and is withdrawn.** Two pieces of evidence kill it.

**Evidence 1: the forward path measured this exact hypothesis and rejected it.**
The forward r2c cascade has a lane-blocking mechanism (`Kb`): run the cascade
over a narrow `Kb`-lane slab so both planes for that slab stay cache-resident
across all stages. It is built and present. It is also **off by default, by
deliberate measurement.** The forward plan note records: the L2-slab heuristic
was NEGATIVE (-22% at this same (4,4,16)-class plan); narrow slabs cut each
stream burst to ~768 bytes and defeat the hardware prefetchers; and, decisively,
"the cascade was never capacity-bound to begin with." Same plane sizes, same
hardware, same architecture. If the forward cascade is not L2-capacity-bound at
these sizes, the c2r cascade is not either. The 256-KB-exceeds-L2 story was an
unmeasured inference, and the forward's own data argues directly against it.

**Evidence 2: the forward shows the SAME K-dependent drop, fully optimized.**
Racing the forward r2c (already folded, ranged, the whole kit) against MKL on
the same (8,32) plan:

| direction        | K=8 ratio | K=64 ratio |
|------------------|-----------|------------|
| forward r2c      | 0.79      | 0.63       |
| backward c2r     | 0.99      | 0.78       |

Both directions lose ground at K=64. The forward loses MORE (0.79 -> 0.63) than
c2r does (0.99 -> 0.78). So the large-K gap is **not a c2r deficiency and not
something c2r is missing relative to the forward** -- it is a property of the
shared cascade architecture versus MKL at large batch, present in both
directions and, if anything, milder in the direction we just optimized.

So what IS the K=64 gap? Honestly, this is still partly open, but the frame is
now "MKL scales better with batch width at this size" rather than "we are
capacity-bound." MKL likely uses a different large-batch strategy (its own
blocking or SIMD packing) that pulls ahead as K grows. Pinning the exact
mechanism would need a PMU this container does not have. What is certain: it is
shared between forward and backward, the forward team already ruled out L2
capacity as the cause, and lane-blocking is NOT the fix (it was measured
negative).

### What the c2r optimizations actually bought

The mid-column SIMD, leaf fold, and ranged interior took c2r from losing badly
at small K to parity, and left its large-K ratio (0.78) BETTER than the
forward's (0.63). They were real wins. The thing they did NOT do, and could not
do, is change the shared cascade-vs-MKL batch-scaling behavior, because that is
not what they target. That distinction is the actual lesson, below.

## Transferable lesson

The real lesson is epistemic, not architectural. It was easy and tempting to
write a clean memory-bound story: the working set is 256 KB, that is around L2,
large K spills it, therefore bandwidth-bound, therefore call-collapsing cannot
help. Every step sounds right and the conclusion even matched the timing (ranged
did not help K=64). But the story was never measured -- this container has no
PMU -- and two checks falsified it: the forward path had already measured
lane-blocking as NEGATIVE and concluded the cascade "was never capacity-bound to
begin with," and the forward path shows the SAME large-K drop while fully
optimized. A behavioral signature (X helped here, not there) is consistent with
many mechanisms; it does not pin one. The discipline that would have caught it
earlier: before attributing a gap to a mechanism you cannot measure, check
whether a comparable already-optimized path exhibits the same gap. If it does,
the cause is shared and is not the thing you were about to "fix."

What does survive: per-call overhead is a real, K-sensitive cost (large fraction
of a small-K call, negligible fraction of a large-K call), and collapsing calls
via a ranged codelet genuinely removes it -- which is why small-K reached parity.
That part is measured and solid. The large-K gap is real too, but it is shared
with the forward direction, the forward team already ruled out the obvious
capacity explanation, and the actual mechanism remains open pending hardware
with performance counters. Correctness gating came first throughout: every
speed number was taken only after the round-trip gate (c2r(fwd(x)) == N*x)
passed, because a faster wrong answer is not a win.

## UPDATE: Kb lane-blocking ported, swept, and independently re-verified

The lane-blocking port landed in core/c2r.h: an outer Kb-slab loop mirroring
the forward's section-65 mechanism (fold predicate `bw==K && bb==0 && Q>1`,
per-q at vl=bw when slabbed, leaf unfolds per slab since the one-call S*K fold
is address-contiguous only at full width). Default Kb=K reproduces the previous
full-width path byte-identically; override via the plan field or VFFT_C2R_KB.

INDEPENDENTLY RE-VERIFIED (fresh build from the pasted source, codelets
regenerated, gates rerun):

CORRECTNESS — bit-identical across Kb. The full gate matrix passes at Kb in
{default, 8, 16, 32}, and the (8,32) K=64 maxerr is EXACTLY 2.117e-12 at every
Kb (not "close" — identical). The slabbed path is address-exact to full width,
as the mirror argument predicts: same arithmetic, same rounding, only the visit
order changes. All 10 matrix rows pass at Kb=8 (most aggressive).

SWEEP (container, N=256 (8,32) K=64, one process, MKL 1 thread) — reproduced:

  | Kb        | this re-verify (mkl/ours) | ours cyc | first run |
  | 64 (full) | 0.800                     | 54458    | 0.656     |
  | 32        | 0.728                     | 59076    | 0.629     |
  | 16        | 0.684                     | 63724    | 0.684     |
  | 8         | 0.576                     | 74618    | 0.576     |

Direction CONFIRMED: monotonically worse as Kb narrows (ours 54458 -> 74618 cyc,
+37%, while MKL stays ~43000). The absolute full-width baseline differs run to
run (0.656 first vs 0.800 here; 5 reruns span 0.766-0.836, ~9% noise) — that is
the documented container timing noise. The IN-PROCESS sweep ordering is the
trustworthy part and it is unambiguous.

WHY THE BOX CANNOT TEST THE CAPACITY HYPOTHESIS — verified, with one correction:
- VERIFIED: this container's L2 is 1 MiB (lscpu + sysfs: L1d 32K, L2 1024K,
  L3 33M). The (8,32) K=64 working set is two 128 KB planes = 256 KB, which
  fits L2 with room to spare. It NEVER spills here. So the capacity hypothesis
  (working set > L2 -> bandwidth-bound) has no way to manifest on this box; only
  slabbing's COSTS are observable (short bursts defeating prefetch, lost Q-fold,
  S-1 extra leaf calls per slab), which is exactly why narrowing only hurts.
- CORRECTION to an earlier framing: the container was described as a
  "flat-memory KVM documented blind to locality." That phrasing is an
  overclaim — the journal documents cache-tier sensitivity (radix_memboundness.h
  with L1/L2/L3/DRAM inflation factors, and a prior "256 KB exceeds L2" cache
  effect) on the CALIBRATION host, a quiet metal box, NOT on this container. The
  container is documented as noisy/shared-vCPU, not flat-memory. The concrete,
  sufficient, verifiable fact is just L2 = 1 MiB here, which alone makes the
  256 KB working set non-spilling. The verdict is stronger stated that way and
  does not need the unverified "flat memory" mechanism.

DECISION PROCEDURE (must run on the dev/metal machine; one command per point):
  for kb in 64 32 16 8; do VFFT_C2R_KB=$kb ./bench_c2r 256 64; done
- If some Kb in {16,32} beats Kb=64 on metal: capacity hypothesis CONFIRMED for
  that machine (its L1=32KB is the boundary — 32 KB fits at K=8, 256 KB does
  not — so L1, not L2, is the relevant tier there). Tune Kb, set as the large-K
  plan default.
- If monotone-worse on metal too: capacity hypothesis DEAD on both machines; the
  large-K plateau is COMPULSORY plane traffic, and the deeper lever is STAGE
  FUSION (fuse last stage + leaf to delete one full plane round-trip: 4 plane
  passes -> 3 at nf=2). The Kb mechanism stays either way (default off, harmless,
  byte-identical, mirrors the forward's).

STATUS OF THE PREDICTION CHAIN: this is the second pre-registered prediction in
this document that the container falsified ("ranged fixes K=64" was the first).
Both falsifications were on a box that cannot host the experiment; neither
falsifies the hypothesis on hardware that can. The honest state is OPEN-pending-
metal, with the code correct, gated, and zero-cost-by-default in hand. That is
the right place to leave it: the mechanism is built and proven harmless, the
measurement that would decide it requires hardware this container is not.
