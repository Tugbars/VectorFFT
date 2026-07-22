# 65: Selective duplication (un-CSE) — breaking the mandatory-sharing wall on prime codelets

> One-sentence version: cloning a handful of long-lived, leaf-fed values
> immediately before their last use — with a one-value asm barrier so gcc's
> FRE cannot merge the clone back — cuts realized stack spills by 28–60% on
> monolithic prime codelets (R = 11/13/17/19), beating the schedule
> annealer's converged optimum at R=13 under plain strict-SU order, while
> regressing on R=23 and on all blocked pow2 codelets, where the pass
> structure already performs the range-splitting this transform provides.

Environment for every number below: Linux container, gcc 13.3,
`-O3 -mavx2 -mfma -march=raptorlake -w`, main-loop-only artifacts
(`VFFT_NO_ANYK_TAIL=1`), strict-SU base schedule, objective = realized asm
(insns / stack-spill movs) with the FMA-invariance gate, per the phase-0
methodology. Probe: `tools/dup_probe.py`. Companion results:
`docs/performance/schedule_search_phase2_container_results.md`; the
annealer's own consolidation and its post-duplication role: doc 66.

## 1. The blind spot this attacks

The hash-consed Algsimp IR makes common-subexpression sharing MANDATORY.
A value with two uses 100 slots apart must be held — in a register or on
the stack — for the whole span. Every scheduler in the pipeline (SU, GH,
bisection, the annealer) can only PERMUTE that DAG: it can move the span's
endpoints around, it can never cut the span. Sharing is treated as free;
on a 16-register file it is not.

The tell that this was the binding wall came from the schedule-search
campaigns: the R=64 annealer — a direct search on the realized assembly,
the strongest order-optimizer we have — converged with spills pinned at
~585 and only ~2% of instructions recovered. Permutation was exhausted.
Whatever lever remained had to change the DAG, not the order. Duplication
is the smallest DAG change that reduces liveness: the same expression,
computed twice, so the original's live range ends at its second-to-last
use.

## 2. The double-CSE discovery

The first probe cloned candidates as plain C:

```c
const __m256d t21_d = _mm256_sub_pd(t20, t8);   /* same rhs as t21 */
```

gcc emitted BYTE-IDENTICAL assembly at every setting. GIMPLE-level FRE /
value numbering recognizes a pure expression over the same SSA inputs
dominated by an equal definition and merges the clone straight back.

This is worth stating as a finding in its own right: **CSE is enforced
twice — once by our hash-consing, once by gcc.** No duplication pass can
be smuggled in as ordinary code; the clone must be opaque to value
numbering. The working form is a per-clone, one-value barrier:

```c
__m256d t21_d = _mm256_sub_pd(t20, t8);
__asm__ volatile("" : "+x"(t21_d));
```

This is surgically different from the historical global value fences,
which defeated gcc's operand folding broadly and cost +9% (doc 28 /
M-project record). Here the folding damage is confined to the ≤16 cloned
values; the R=64 single-clone case below shows what one barrier's local
damage looks like when the clone itself buys nothing.

## 3. Selection rule (the validated heuristic)

A candidate is a temp that satisfies ALL of:

* defined by a single `add/sub/mul` (never FMA — keeps the FMA count, and
  therefore the binding-port profile, invariant by construction);
* both operands are DIRECT leaves — input loads or `set1` constants — so
  recomputation costs one port slot and extends no other live range: gcc
  rematerializes broadcasts and refolds loads freely, so re-touching the
  operands at the clone site is close to free;
* at least 2 uses;
* span (last use − def, in emitted lines ≈ schedule slots) ≥ S, with
  S = 30 the working setting.

Transform: insert the barriered clone immediately before the LAST use and
re-point that use (and only that use) to the clone. All earlier uses keep
the original, whose live range now ends at the second-to-last use. Cap
the clone count (8–16 works; dose-response is real, see R=23).

Correctness: identical expressions over identical inputs produce
bit-identical doubles, and the barrier does not alter the value —
verified end-to-end at R=13 with the standard harness (`cmp` of output
buffers vs the un-dup'd baseline).

## 4. Results

| codelet | su (insns/spills) | dup best | clones | Δinsns | Δspills |
|---|---|---|---|---:|---:|
| R=11 | 317 / 42 | **269 / 17** | 7 | −48 (−15%) | **−25 (−60%)** |
| R=13 | 446 / 70 | **377 / 31** | 10 | −69 (−15%) | **−39 (−56%)** |
| R=17 | 756 / 175 | **705 / 118** | 16 | −51 (−7%) | −57 (−33%) |
| R=19 | 876 / 192 | **849 / 139** | 16 | −27 (−3%) | −53 (−28%) |
| R=23 | 1275 / 294 | regresses at every dose (best 1283/301) | 2–16 | — | — |
| R=64 blocked (wisdom floor 2490/585) | — | regresses in every variant | 1–64 | — | — |

Two comparisons put the R=13 number in context:

* **10 clones under plain strict-SU order (377/31) beat the annealer's
  800-iteration converged optimum (392/51).** A search over the full
  permutation space lost to a ten-line transform, because the transform
  was outside the search space. This is the cleanest possible
  demonstration that the wall was the DAG, not the order.
* **Stacking order matters.** Applying the probe to the annealed winner
  gives 391/42 — worse than dup-on-strict. The annealed order was tuned
  around the pre-dup liveness structure; once the structure changes, its
  advantages are stale. Pipeline consequence: duplicate FIRST, then
  anneal the transformed DAG.

## 5. Why BLOCKED emission resists — pinpointed (2026-07-02)

The earlier framing ("blocked pow2 resists") conflated two factors.
The factorial separates them (gcc 13.3, raptorlake, main loop,
insns/spills; "pass" is the OCaml pass at default S=30; probe doses
shown for the guard-excluded blocked rows):

| R | structure | baseline | pass S=30 | probe S=30 | probe S=10 | verdict |
|---|---|---|---|---|---|---|
| 4 | mono pow2 | 55/0 | 0 clones | dups=0 | 55/0 | no material (max span 17), no pressure |
| 8 | mono pow2 | 132/6 | **127/6** | 127/6 (exact) | 129/6 | **WIN, bit-exact** |
| 16 | blocked pow2 | 391/59 | skipped (markers) | dups=0 | 386/54 | see nuance below |
| 25 | blocked prime-power | 1000/171 | skipped | dups=0 | 1011/168 | resists |
| 32 | blocked pow2 | 1031/183 | skipped | 1049/202 | 1036/192 | resists |
| 64 | blocked pow2 | 2543/594 | skipped | 2561/601 | 2567/617 | resists |

CONCLUSION: the resisting factor is BLOCKED EMISSION, not pow2 DAG
shape. Two decisive cells: (a) mono pow2 R=8 WINS (127/6, pass equals
probe, bit-exact) — pow2-ness alone does not resist; (b) blocked
prime-power R=25 resists exactly like blocked pow2 — blocked-ness
resists even off pow2.

MECHANISM: blocking already converts long-lived values into explicit
spill-slot store/reload pairs; the register temps that survive are
short-span BY CONSTRUCTION. At S=30 there are literally zero
candidates in the blocked files (probe dups=0 at R=16/25). Forcing
S=10 selects medium-span cluster residents whose duplication re-pays
compute the explicit spilling already paid for, while the barriers
constrain gcc inside an already tight multi-cluster schedule —
R=25/32/64 regress at every dose.

NUANCE, recorded not pursued: blocked R=16 at S=10 shows a small
probe-only win (386/54, −1.3%/−8.5%). Harvesting it would require
teaching the pass the spill-marker interplay its guard exists to
avoid; not worth −5 insns.

COMPLETED FACTORIAL (2026-07-02, same day): the missing cell — pow2
WITH pressure in MONOLITHIC form — exists via the tool's own levers:
`--no-recipe --su` (auto spill-recipe off, SU explicitly on; no code
change, zero markers confirmed). Mono prime-power added too:

| cell | R | mono base | dup best | vs blocked |
|---|---|---|---|---|
| pow2 mono, no pressure | 4 | 55/0 | = (0 cand) | n/a |
| pow2 mono, low pressure | 8 | 132/6 | **127/6** (S<=30) | n/a (8 is mono-native) |
| pow2 mono, pressure | 16 | 402/78 | **399/75** (S=20) | blocked 391/59 still wins |
| prime-power mono, pressure | 25 | 1124/307 | = (**0 candidates, any dose**) | blocked 1000/171 wins big |

TWO INDEPENDENT SUPPRESSORS, now separated:
(1) CT-FACTORIZATION SHAPE: mono R=25 (5x5 two-stage) has ZERO
    candidates even fully unblocked — stage-1 outputs are consumed
    immediately by local stage-2 butterflies; no long-span dense-sum
    material exists to duplicate. The material is a property of the
    DIRECT-DFT dense coefficient structure (primes 11-23), with pow2
    butterfly networks intermediate (R=8: spans 65/46/44, 2
    candidates; R=16 mono: a few, small win).
(2) BLOCKED EMISSION: removes whatever residual material the shape
    provides and pre-pays the pressure (previous table).

DEPLOYMENT UNCHANGED by this cell: mono-16+dup (399/75) still loses
to blocked-16 (391/59), so R=16 stays blocked, dup off. R=8 remains
the only pow2 deployment.

Pass-vs-probe divergence note (first observed here): at mono-16 the
OCaml pass (400/77 at S=30) BEATS the probe (407/79) on the same
input — schedule-space selection outperforming the probe's raw-line
spans. The probe was scaffolding; the product now exceeds it in at
least one cell.

## 6. Relation to the standing charges

* **C1** (construction treats liveness as the scheduler's problem): this
  is the codelet-scale proof and remedy in one. The liveness wall is a
  DAG property; changing the DAG moved it 56%, weeks of ordering work
  moved it ~0. C1's thesis is confirmed from the opposite direction.
* **S1/A10** (ILP objections): untouched — the FMA count and therefore
  the binding-port profile are invariant by construction.
* **A3** (spills as IR nodes): complementary, not competing. Duplication
  REMOVES ranges; A3 schedules the placement of whatever spill residue
  survives.
* **The annealer** (phase-2 doc): demoted to second stage of the codelet
  pipeline — run it on the dup'd DAG, where its leaf-placement expertise
  still applies to a smaller problem.
* **A4** (generate-both, count, keep-winner): the deployment mechanism.
  Duplication is a raced dimension per codelet with the choice stamped in
  provenance: on for R ∈ {11,13,17,19}, off for R=23 and all blocked.

## 7. Productization plan (the OCaml pass)

The C-level probe de-risked the selection heuristic and the payoff; the
real pass lives post-algsimp, pre-schedule:

1. Run SU once to obtain slot positions; compute true slot spans (the
   probe's line-number spans were a stand-in that works only because
   emission is one definition per line).
2. Select by the validated rule (add/sub/mul over leaves, ≥2 uses,
   span ≥ ~30 slots, cap 8–16).
3. Clone with a fresh tag (bypassing the hash-cons intern), re-point the
   LAST consumer, rebuild that consumer's ancestors (immutability
   cascade).
4. Fence ONLY the clone, through the existing value-fence machinery —
   never the global fences.
5. Re-schedule the transformed DAG; then, optionally, anneal it.
6. Gates: FMA-count invariance, bit-exact numeric harness, spills AND
   insns strictly below the un-dup'd sibling, per-codelet race with the
   decision stamped in provenance. Wisdom interaction: the dup'd DAG has
   a new `#dagsig`, so schedule-wisdom entries regenerate — dumps first,
   anneal after, exactly the normal flow.
7. Env: `VFFT_DUP=<span>:<cap>`, default off; the locked 48-codelet
   baseline must regenerate byte-identically with it unset.

## 8. Follow-up: monolithic pow2 and the chained variant (v4)

Question raised in review: primes are monolithic — does the win come
from the CONSTRUCTION (monolithic, long ranges) rather than the
ALGORITHM family? Test: force R=16 monolithic (`VFFT_N1_BLOCK_MIN=64`)
and probe it.

Baselines first: blocked R=16 = 391/59, forced-monolithic R=16 =
402/78 — blocking earns its threshold. Then every duplication variant
on the monolithic build FAILED: v2 (leaf-fed) finds only 2–3
candidates and is flat; v3 (any-operand) regresses to 86–101 spills;
v4 (below) regresses to 86. Anatomy explains it: R=16 mono HAS eight
span-89–110 multi-use values, but they are DEEP — mid-cascade, with
chains of >=7 nodes threaded through FMAs. So the true divide is not
monolithic-vs-blocked but **shallow long spans vs deep long spans**:
prime algorithms create a leaf-fed pre-addition fold layer whose
outputs live across the whole body (shallow, cheap to re-derive);
pow2 CT cascades create long-lived values only deep in the lattice
(expensive to re-derive). Blocked R=16 remains the right construction,
unbeaten.

The anatomy also predicted a cure for the R=23 mystery. **v4 chained
duplication** (`tools/dup_probe4.py`): re-derive the whole operand
chain at the last use from RE-LOADED leaves, barriering ONLY the
leaves — the recomputed interior over barriered leaves is
automatically opaque to FRE (one barrier per leaf instead of per
node). Chains restricted to add/sub/mul + leaves, <=8 nodes.

| codelet | base | v4 best | note |
|---|---|---|---|
| R=23 | 1275/294 | **1274/268 (cap=8)** | the "unexplained" negative, cured: its long values are deep; −9% spills at EQUAL insns; **bit-exact verified**; dose curve saturates hard above cap=8 (cap>=12: 306–329) |
| R=16 mono | 402/78 | 419/86 | deep chains too long/FMA-blocked |
| R=17 / R=19 (v4 alone) | — | 797/204, 936/225 | badly worse than v2 — deep chains are the wrong first move where the shallow layer dominates |
| v2 THEN v4 (R=13/17/19) | v2 bests | 443/66, 771/167, 916/197 | stacking regresses everywhere: after v2 relieves pressure, chains only add cost |

Deployment table, final: v2 for R in {11,13,17,19}; v4 cap=8 for R=23;
NOTHING for pow2 (mono or blocked) and for stacking. One variant per
codelet, chosen by the A4 race, decision stamped in provenance.

### 8b. The depth axis, exhausted (v5)

Can v2 go deeper? The principled generalization
(`tools/dup_probe5.py`): allow operands at ANY depth, gated on
availability at the clone point — an operand is free iff it is a leaf
OR its own last use is at/after the clone point (alive there anyway:
zero lifetime extension BY CONSTRUCTION); operands that miss coverage
but are leaf-fed may be re-derived at +1 def each, cost-capped.

Result: v5 strictly subsumes v2 and reproduces its optimum EXACTLY on
every winning prime (R=13 377/31, R=17 705/118, R=19 849/139) while
finding 25–56% more eligible candidates — all of which are flat. On
R=23 it mirrors v2's regression (v4 chained stays that codelet's only
winner); on pow2 (R=16 mono, R=64 blocked) it finds only the same few
flat candidates.

Why depth is barren: a deep value's operands are liveness-covered
precisely BECAUSE they are themselves long-lived — the allocator is
already paying for them, so cloning the root renames a range without
shrinking the live mass. The economics work only when operands are
free to re-touch, which means leaves (depth-1, v2) or a chain re-rooted
on fresh leaf reloads (v4, profitable only where deep spans dominate
and chains are short: R=23). Consequence for the OCaml pass: implement
v5's coverage rule as THE selector — it is the clean superset that
degrades gracefully to v2 and can never extend a lifetime — plus the
v4 chained mode as a raced alternative; expect v2-equal numbers on the
shallow primes.

## 9. Honest limits

* All magnitudes are gcc 13.3 / container numbers. The mechanism —
  cutting live ranges the allocator must otherwise spill — is
  allocator-independent in direction, but the barrier's folding cost is
  compiler-specific; re-score under gcc 15.2 mingw before banking, and
  run the paired i9 bench (static objective here is the phase-0 ρ≈0.94
  proxy, not runtime).
* Selection is greedy last-use-only. Mid-span splitting, multiple clones
  per value, and chained duplication (non-leaf operands with recursive
  re-living) are unexplored; the naive any-operand widening regressed at
  R=64 and was not tried on primes.
* The asm barrier is a hammer. A compiler-agnostic opaque form was not
  found; if gcc's FRE ever learns to see through `"+x"` barriers, the
  pass needs a new cloak.
* AVX-512 untested; the 32-register file changes the saturation point in
  both directions (more slack for clones, less spilling to save).
* R=23 is now explained and partially recovered by v4 (section 8); its
  residual 268 spills and the sharp cap saturation remain uncharacterized.
* v4's leaf-reload cost model is implicit; on AVX-512's 32-register file
  the shallow/deep economics shift in both directions and are untested.

## §8 IMPLEMENTATION STATUS (2026-07-02, second session) — CONVERGED

`Algsimp.duplicate_uncse` is probe-equivalent and BIT-EXACT (verify13
and verify17 binary-identical vs baseline — clones compute identical
values, so unlike butterfly_share_mul this transform keeps the
bit-reproducibility guarantee; no README disclosure needed). Env
`VFFT_DUP=1` (+`VFFT_DUP_S/CAP/COST`, `VFFT_DUP_ONLY`,
`VFFT_DUP_TRACE`); default off, byte-identical off, 10-radix gate +
reproduce.sh green.

RESULTS (gcc 13.3, raptorlake, main loop, insns/spills):

| R | baseline | dup (OCaml) | probe ref | dup + affinity | best config |
|---|---|---|---|---|---|
| 8 | 132/6 | **127/6** exact | 127/6 | 130/8 | dup only (S≤30 plateau; bit-exact verified) |
| 11 | 317/42 | **269/17** exact | 269/17 | **267/16** | dup+affinity — ALL-TIME RECORD |
| 13 | 446/70 | **377/31** exact | 377/31 | 377/31 | dup (beats annealed 392/50) |
| 17 | 756/175 | 713/118 (S=40: **711/115**, beats probe spills) | 705/118 | **693/108** | dup+affinity — ALL-TIME RECORD |
| 19 | 876/192 | **849/139** exact | 849/139 | 854/145 | dup only |
| 23 | 1275/294 | regresses | (probe-negative) | — | off; chain mode (v4) is the follow-up |

Deployment: per-codelet build-time env flags (a DAG transform cannot
ride the wisdom-file mechanism); primes only; pow2/blocked auto-skip
via the spill-marker guard.

THE FIVE ROOT CAUSES the port had to find (all measured, all
documented at their fix sites): (1) single-use clones were INLINED —
no declaration, no barrier, gcc re-CSEd them to zero effect; (2)
spans measured in topo/tag space select wrong values (schedule space
is mandatory); (3) redirect rebuilds re-tag ancestor cones and SR's
tag tie-breaks scramble globally — the schedule must be PINNED
(Kahn-priority over sched0 pre-images; naive tag-chasing yields
duplicate/incomplete orders because hashcons can MERGE rebuilt
kinds); (4) su_schedule's returned list re-appends sinks as
(Some ref) pairs — every sink appears TWICE, and last-occurrence
ranking inflated sink ranks by ~+44, making sink-consumers spuriously
win the last-use argmax (the R=17/19 regression cause); (5) clones
must pin to their consumer's DECLARED ANCHOR slot, not the raw inner
chain node's slot — SU interleaves sibling fma chains, so inner slots
sit ~18 positions before the line they inline into.

FOLLOW-UPS: (a) anneal-AFTER-dup needs dump/inject moved to post-dup
space (dup's own order pin currently occupies the injection channel);
the probe header's prior stands: dup-on-strict 377/31 already beat
dup-on-annealed 391/42. (b) v4 chain mode for R=23 needs a load-clone
declaration path in emit (loads currently print as lane names — no
decl, no barrier possible). (c) R=17's last 6 insns vs the probe are
one candidate (t80 vs t97, span-unit boundary); S=40 already beats
the probe on spills.
