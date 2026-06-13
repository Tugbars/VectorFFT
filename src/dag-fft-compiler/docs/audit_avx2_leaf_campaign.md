# Full Audit: the avx2 Leaf Campaign

Consolidated record of everything done and found across the
investigation arc (lab notebook sections 22-33). Standalone; the
notebook holds the same content in session order with build recipes.

Starting point: VectorFFT beats MKL ~2.9x geomean on avx512 and beats
FFTW 1.21-1.80x at plan level on the primary targets. The one open
front: the symmetric avx2-vs-FFTW-avx2 race, where we lost 6 of 7
cells at 0.48-0.94, traced to leaf codelet quality. This audit covers
the hunt for why, what was fixed, what was falsified, and where the
true floor sits.

All spill numbers below use the corrected vec/gpr accounting unless
marked otherwise. Within-table numbers are generated and compiled
identically; cross-section drift of a few percent between object
vintages is noted where it occurs.

---

## Phase 1: The wiring investigation

**Did:** Compared generation flags and emitter machinery between the
in-place n1 family and the OOP leaf family, on the hypothesis that the
OOP path was missing optimizer wiring.

**Found, in order, each verified by reading or measurement:**

- The OOP emitter delegates the full algsimp cascade to the shared
  lib/pipeline.ml (zero pass drift vs gen_radix.ml, verified
  pass-by-pass). My initial claim that the cascade was missing was
  WRONG and is corrected in the notebook (section 23 correction
  header).
- Tier C inside codelet_oop already implements cluster-local SU with
  the GH auto-rule. The file's own line-769 "no SU scheduler" comment
  is stale documentation. Second wrong hypothesis, corrected.
- Fences: wired via prep.fence_enabled with save/restore. My redundant
  patch was reverted same day.
- Register allocator: gated to log3+avx512+R<=32 on the IN-PLACE path
  too. Not an OOP-specific absence.
- Blocking: symmetric, both paths block at n>=25 (an earlier
  "asymmetry" was a grep artifact).

**Landed (permanent):** per-ISA uarch selection in Tier C. The
hardcoded sapphire_rapids_avx512 profile (pressure_threshold=24) made
GH effectively inert on 16-register targets and fed SU wrong-ISA
latency tables; raptor_lake_avx2 (threshold=12) is selected when
vec_regs<=16. avx512 output byte-identical. avx2 n1_32 spills 548 to
531 (conflated counting). Both OOP sets regenerated, all plan gates
pass, FFTW-avx2 race unchanged within noise.

**Conclusion:** the gap is not a missing wire. Emitters at near-parity.

## Phase 2: The decomposition (op count vs scheduling)

**Did:** Pre-registered bets (Tugbars: op count; session: scheduling),
then compared genfft's stamped DAG op counts and FFTW's compiled
objects against ours, per transform (FFTW VL=2, ours 4 lanes).

**Found:** Op count FALSIFIED at the DAG level. Per-transform DAG
arithmetic: R8 13 vs 13.0, R16 36 vs 36.0, R32 97 vs 93, R64 238 vs
228, R128 582 vs 541.5. Identical to the digit at 8 and 16; the
cascade is genfft-grade.

Three-way compiled counts (conflated spill units of that session):

| R | in-place avx2 | OOP avx2 | FFTW avx2 |
|---|---|---|---|
| 8 | 52/6 | 52/48 | 46/0 |
| 16 | 144/113 | 144/329 | 65/15 |
| 32 | 386/289 | 386/531 | 171/65 |
| 64 | 952/817 | 952/881 | 425/179 |
| 128 | 2328/2141 | 2328/2055 | 1019/527 |

Three facts: (a) in-place and OOP arithmetic literally identical
(shared Pipeline confirmed at instruction level); (b) in-place nearly
as spill-heavy as OOP at R>=32, and WORSE at R=128: the spill problem
is GENERATOR-WIDE on avx2, not an OOP defect; (c) the true narrow
wiring gap is the OOP MONOLITHIC path (R<25), which orders by tag with
no SU while in-place monolithic gets SU as universal default. Small
leaves are exactly what the avx2 planner picks (4x16, 4x32 pairs), so
this became Tier 1 of the queue.

## Phase 3: Spill accounting correction (Tugbars's catch)

**Did:** Separated SIMD spills (vmov ymm to/from rsp) from GPR rsp
movs, which the original counter conflated.

**Found:**

| R | in-place SU | bisect final | OOP | FFTW |
|---|---|---|---|---|
| 8 | 6/0 | - | 22/33 | 0/0 |
| 16 | 90/27 | 109/32 | 226/111 | 15/0 |
| 32 | 179/125 | 292/146 | 272/265 | 65/0 |
| 64 | 561/308 | 855/312 | 765/117 | 179/0 |

(vec/gpr per object.) Corrected SIMD-only ratios vs FFTW: 6x / 2.8x /
3.1x at R=16/32/64. All directional conclusions survive. NEW LANE
DISCOVERED: FFTW carries ZERO GPR stack traffic at every size; we
carry up to 308 GPR movs at in-place R=64 (address arithmetic: k +
c*ios offsets spilling on 16 GPRs; FFTW's stride macros stay
register-resident). OOP's strided-edge addressing is cheaper (117 at
R=64), so the two families bleed in different lanes. Queued as an
independent, likely cheap work item. Convention since: vec/gpr pairs.

## Phase 4: The liveness autopsy

**Did:** Built a replay script that walks a codelet's emitted
statement order and counts peak simultaneously-live compute values
(constants excluded).

**Found, at R=16:** FFTW bisection order peaks at 8 live values; our
list-scheduler order at 35; tag order at 57. Peaks track the measured
spill columns. Convergence of three independent instruments: the
earlier MKL N=1024 assembly audit (we do less arithmetic than MKL,
which stays close anyway), the Phase-2 decomposition (arith parity,
multiple-x spill traffic), and this replay. The schedule, not the
math.

## Phase 5: The bisection campaign

**Did:** Audited --bisect: a complete Frigo port existed in
lib/schedule.ml and had NEVER been raced. Raced it, lost badly, then
repaired it against genfft/schedule.ml (on disk) as the spec.

**Repair sequence and spill cascade (conflated units, in-place avx2
n1):**

| R | as found | +components | +subset-relative | +reorder | +const-decoupling | SU recipe | FFTW |
|---|---|---|---|---|---|---|---|
| 16 | 245 | 176 | 152 | 131 | 137 | 113 | 15 |
| 32 | 1047 | 882 | 651 | 584 | 421 | 289 | 65 |
| 64 | 2997 | 2347 | 1665 | 1529 | 1119 | 817 | 179 |

The four fixes, all gated to the --bisect path, production untouched:

1. **connected_components recursion** at every level (genfft's
   schedule_alist decomposes before partitioning; ours bisected raw
   blobs).
2. **Subset-relative bisect** (genfft rebuilds the dag per sublist via
   makedag; our global preds/succs meant pure-compute subsets had no
   outputs, the BLUE wave never seeded, cuts degenerated, and the
   topological fallback front-loaded every load). After this fix the
   emission shows the genuine genfft cadence: load, load, sub, add.
3. **reorder port** (genfft annotate.ml's greedy overlap-maximizing
   ordering of sibling components; ours had concatenated in node-id
   order).
4. **Constant decoupling in components** (genfft constants are inline
   literals with no dag presence; our hash-consed NK_Const nodes
   welded every sub-DFT into one component, silently disabling the
   decomposition). First attempt attached shared constants to their
   min-id consumer, a LEGALITY BUG (use-before-def, caught by gcc);
   fixed by hoisting multi-consumer specials ahead of all components,
   structurally exempt from reorder.

**Verdict:** bisection improved 1.8-2.7x from the degenerate state and
STILL LOSES to the list scheduler at every size. The
wire-as-avx2-default gate was not met. No production wiring.

**Topology falsifications in the same campaign:**

- Split-radix (VFFT_SPLIT_RADIX=1): spills flat, liveness peaks worse
  at 16/32, arith UP 8-20 percent (fma_lift gated off for SR). Our SR
  construction materializes the same wide seams.
- Unbalanced CT (VFFT_CT_FACTOR 2,N/2 and reversed): worse than
  defaults under both schedulers. (First attempt used the wrong
  override syntax, 2x8 vs 2,8, silently measured the defaults; caught
  by identical arith counts.)

**THE FLOOR:** our CT(4,4) R=16 dag has a 32-real seam at the pass
boundary; any schedule completing pass 1 before pass 2 holds all of
it. The list scheduler's peak of 35 is already AT that floor.
FFTW's peak of 8 is a property of genfft's dag CONSTRUCTION
(split-radix whose even half streams through the combine layer),
not of its scheduler. Scheduling on our topologies is essentially
closed.

## Phase 6: Scheduler discovery ("SU is not SU")

**Did:** On Tugbars's hypothesis that our SU must be heavily modified
(else repaired Frigo would not lose to it), read lib/schedule.ml end
to end and inventoried the production scheduler. Then inventoried
genfft's shipped pipeline from magic.ml defaults plus the annotate.ml
driver.

**Found, ours (eight layers):** (1) lazy loads, never fired while
arithmetic is ready, demand-only in source order; (2) sink-first,
empty-user nodes fire immediately (documented R=17 t1_dif win, 176 to
~115 vmovapd); (3) cp_dist descending, latency-weighted critical path
from uarch tables; (4) Sethi-Ullman numbering, the only textbook
remnant, demoted to third tie-breaker; (5) stable tag order; (6)
Goodman-Hsu pressure mode with remaining_users and a uarch threshold;
(7) port-class balancing (P0/P1 vs P5, Ice Lake); (8) symbiosis with
the recipe's structural blocking.

**Found, theirs (shipped):** bisection + components, overlap reorder +
balanced linearize, buddy-store pairing (an interleaved-format need
our split format does not have), liveness analysis, -compact
-variables 4 scoping. schedule_for_pipeline, reorder_insns,
reorder_loads, reorder_stores ALL default false and absent from the
command lines stamped into the shipped codelets (verified, not
assumed). Their scheduler contains ZERO numbers: no latency tables, no
thresholds, no port models, no per-ISA anything.

**Verdict:** layers 1-2 of ours are greedy per-instruction
approximations of what bisection achieves structurally, independently
evolved. The bisect-vs-SU race was register-oblivious 1999 structure
vs a months-tuned multi-criteria machine; ours wins on equal topology.
There is nothing left to steal in FFTW's scheduler; their edge lives
entirely in fftgen's construction. Naming note: the production
scheduler is THE LIST SCHEDULER; "SU" survives only in flag names.

## Phase 7: The knob-space sweep

**Did:** On the decision that the construction philosophy stays,
swept the existing knob space for spill reductions: factorization
(VFFT_CT_FACTOR) x scheduler (list/bisect/bb, BB's first run ever) x
regalloc force (VFFT_PIN_FORCE), avx2 n1, vec/gpr counted, winners
raced same-binary and checked bit-exact.

**Found:**

- R=16: def 83/35 = bb; bisect 109; (2,8) 87; (8,2) 93.
- R=32: def 196/138 = (4,8); bb-5s 200; (2,16) 218; (16,2) 525.
- R=64: def 584/322 = (4,16); bb-5s 996; (16,4) 1179.

(1) The hand-tuned factorization table is AT the sweep optimum;
big-first splits are catastrophic. (2) BB engages on the spill path
but never wins; its 5s incumbent at R=64 is 1.7x worse than the list
scheduler; 60s budgets exceed practical codegen time here. Parked.
(3) THE FIND: PIN_FORCE. The linear-scan regalloc forced onto avx2 n1
(production gate: log3+avx512+R<=32, written when the allocator was
new, never re-tried) cuts vec spills 196 to 166 (R=32) and 584 to 491
(R=64), gpr down too, output BIT-EXACT. Same-binary micro-race on the
container host (Cascade Lake VM, not the i9 target): +3.2 percent at
R=32, noise at R=64. Lesson recorded: L1-resident spill traffic is
cheap under OOO; counts rank candidates, races decide.

**Action item:** A/B the widened regalloc gate (avx2 n1 R>=32) on the
i9 during the Phase 6 hardware audit; flip the gate if confirmed.

## Phase 8: The construction audit

**Did:** Mapped dft.ml's dag-building layer (pick_algorithm +
override; Direct primes via conjugate-pair factoring; recursive
dft_ct; the doc-58 blocked recipes with total marker capture; twiddle
policy; SR path) and examined five improvement candidates.

**Found:**

- A. Marker totality (all pass-1 bins round-trip scratch): analyzed,
  near-wash. Pass-2 cluster k2 consumes one bin from EVERY pass-1
  cluster; holding any cluster register-resident trades reload savings
  for equal long-range pressure. No action.
- B. Marker placement / twiddle side (THE LIVE CANDIDATE): markers
  capture pre-twiddle values, parking the internal-twiddle cmul on the
  post-reload side where fma_lift cannot fuse it into pass-1 tails.
  Implemented post-twiddle capture env-gated; default path proven
  byte-identical; the variant GENERATED ILLEGAL CODE (use-before-def
  in pass 2): heterogeneous marker depths violate an emit_c
  classify_passes assumption. REVERTED with byte-identical
  restoration. Candidate stands with a known blocker: classifier
  support for mixed-depth markers, ~half a day, payoff =
  cross-boundary FMA fusion + pass-pressure rebalancing.
- C. Store placement: fear disproven by measurement; output stores
  begin at line 388/666 (R=32) and interleave per pass-2 cluster.
  FFTW's earlier stores are their streaming topology, not our defect.
- D. Multi-level blocking for R>=128: real, low priority (opt-in
  family).
- E. SR-blocked path: skip (SR loses regardless, Phase 5).

**Verdict:** construction vindicated by its own audit; small concrete
in-family room (B blocked, D low priority); the CT seam is a
deliberate architectural trade, not an oversight.

---

## Code changes: landed vs reverted

**Landed (all regression-gated):**
1. codelet_oop.ml: per-ISA uarch in Tier C. avx512 byte-identical;
   production OOP sets regenerated and re-gated.
2. schedule.ml: connected_components_of + recursion in schedule_nodes.
3. schedule.ml: subset-relative bisect (member table,
   preds_in/succs_in).
4. schedule.ml: reorder_components (overlap chaining).
5. schedule.ml: constant decoupling + structural hoisting of shared
   specials.
   (2-5 affect ONLY the Bisection arm; su_schedule_subset and all
   production recipes untouched, stated and verified.)

**Reverted, with byte-identical restoration proofs:**
- Fence-policy patch in codelet_oop (redundant; prep.fence_enabled
  already wired).
- VFFT_TW_PRESPILL marker variant in dft.ml (classifier blocker).

## Instruments built (reusable)

- vec/gpr spill counters (objdump-based, separated species).
- Liveness replay script (peak simultaneously-live values from
  emitted statement order; the dag-floor acceptance test).
- Knob-sweep harness (factorization x scheduler x toggles, identical
  build flags).
- Same-binary micro-race harness with objcopy symbol renaming and
  bit-delta verification.

## Falsification ledger (honesty record)

- Op count as the gap mechanism (Tugbars's bet): falsified, DAG parity.
- Missing algsimp cascade in OOP (mine): falsified, Pipeline shared.
- Missing SU/GH in OOP (mine): falsified, Tier C exists.
- Missing fences (mine): falsified, prep-wired.
- Missing regalloc as OOP-specific (mine): falsified, gated both paths.
- Blocking asymmetry (mine): grep artifact.
- schedule_for_pipeline as FFTW's secret (mine, briefly): parked
  correctly, later verified off by default.
- Split-radix as the topology fix: falsified by measurement.
- Unbalanced CT streaming: falsified by measurement.
- Store batching at end (mine): disproven, stores interleave.
- Spill counts as runtime proxy: demoted; counts rank, races decide.

## Standing queue (priority order)

1. Phase-6 i9/EPYC hardware audit, now carrying the PIN_FORCE gate A/B
   (the deciding measurement for the one sweep find).
2. Tier 1: SU over the OOP monolithic node set (R<25 leaves; the
   planner's hot pairs). Pre-registered: n1_16 toward in-place's
   level.
3. GPR addressing lane (per-group base pointers / strength-reduced
   offsets; FFTW reference = zero).
4. Candidate B: mixed-depth marker support in classify_passes, then
   post-twiddle capture (~half day).
5. Parked: BB with offline-scale budgets; tiny-scope (-variables
   style) emission A/B; streaming dag construction (only if the avx2
   front ever justifies revisiting the construction-stays decision);
   D multi-level blocking for R>=128.

## Bottom line

The avx2 leaf gap is real, measured, and located: equal arithmetic,
2.8-6x SIMD spill traffic plus an addressing lane FFTW doesn't pay,
bounded below by the CT seam that our construction deliberately
accepts and our schedulers already sit against. The architecture
survived every audit it was subjected to, two genuine configuration
bugs were fixed (per-ISA uarch; the PIN gate pending i9), the
bisection scheduler went from broken to faithful, and the remaining
work is a short, costed queue rather than a mystery.
