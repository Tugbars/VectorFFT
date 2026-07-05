# 69: The scheduler question, resolved — SU certified at the model floor

> One-sentence version: challenged to replace SU with a principled
> successor ("I did SU because I only knew SU"), three literature-grounded
> redesigns — Belady-in-the-loop greedy, MRIS lineage sequencing
> (Govindarajan et al., IEEE ToC 2003), and beam-Belady lookahead — all
> LOST to SU in the compiler-free model (Belady traffic 110 / 98-class /
> 90 vs SU's 30 at R=13), while 200,000 random legal orders bottom out at
> 87 and 4,000 hill-climb moves from SU cannot improve it: SU's
> cp-descent is a certified (local-, and empirically global-class)
> optimum of the correct objective, and "improving the scheduler" was the
> wrong quest — the DAG (doc 65) and the compiler residue (doc 68 F4)
> were the actual gaps.

## 1. The three raced-and-lost designs (tools/, kept as negatives)

* `lineage_sched.py` v1 — Belady-simulated regfile in the priority
  (traffic, −kills, heir, height). R=13: gcc 80 spills, MODEL traffic
  110 vs SU 30. Root cause identified: **traffic is a cliff function**
  — zero gradient until the file overflows, so the phase that commits
  the damage (which values are born and stranded) is decided by the
  secondary keys, which shred cone structure.
* v2 — lineage-continuation primary (MRIS-faithful): worse (gcc 98).
* `beam_sched.py` v3 — beam width 4–64 over ready choices, full Belady
  evaluator, (traffic, Σlive) score: best model traffic 90; the
  one-step candidate pruner filters out SU-like continuations before
  the beam can score them. Lookahead does not fix a poisoned key.

## 2. The floors

| quantity (R=13, N=245, R_budget=16) | Belady traffic |
|---|---:|
| best of 200,000 random legal orders | 87 |
| best of the three designed successors | 90 |
| gcc-optimal orders (annealed / minimax) | 71 |
| **SU** | **30** |
| 4,000-move hill climb seeded from SU | 30 (stuck) |

SU is ~3× below the sampled floor and a strict local optimum. The
gcc-search orders (70→50 realized spills) have HIGHER model traffic —
their gains live in gcc-specific rematerialization/operand-folding the
model (correctly) does not contain: that 20-spill delta is gcc's
private residue, not schedule quality.

## 3. What this resolves

* **σ\*(R-finite, W-large) = SU.** Doc 68 argued the pressure-first
  family is the operating point; this document certifies the specific
  incumbent within the family. The scheduler is not to be replaced; it
  is to be UNDERSTOOD: cp_dist-descent performs an implicit depth-first
  cone traversal in which values die near birth — the structural reason
  three sophisticated priority functions could not reproduce it.
* **"Improve the scheduler" now means exactly one thing:** the algsimp
  duplication pass (doc 65: −28…−60% spills on primes, compiler-free
  mechanism, v5 coverage selector), which improves the DAG the
  certified scheduler orders. Everything else on the ordering axis is
  either compiler residue (report as tax, per doc 68 §5 — never chase
  with per-compiler schedulers) or in-order territory (doc 67's B&B,
  parked until such a target exists).
* The search apparatus (doc 66) stands as certifier: it produced the
  floors above and the residue attribution; it is not the scheduler.

## 4. Open

Exact traffic-B&B for small radices (turn "local optimum, 3× below
200k samples" into "global optimum, proved"); traffic floors for
R=17/19/23 and the blocked cluster subsets; a Belady+remat model to
close the 30-vs-realized-50/70 accounting; the cone-traversal
formalization of WHY cp-descent minimizes traffic on butterfly DAGs —
the theorem-shaped item this certification points at.


## CORRECTION AND ATTRIBUTION (same day, design-owner review)

The design owner's correction: "My SU is heavily modified — it uses SU
just to find the reach; the rest is bespoke." Verified by ablation
(`tools/ablate.py`, a Python replica of the production picker that
reproduces the emitted order EXACTLY at every radix tested, then knocks
out one component at a time; `tools/traffic.py` scores the model). This
document's earlier framing ("the 1970 tree heuristic won") is WRONG in
attribution: Sethi–Ullman numbers sit at rank three of four in the
comparator and, per the ablation, contribute approximately nothing.
What was certified at the model floor is the BESPOKE design.

Component ledger (R=13; keys S=sink-first, D=cp_dist, U=su_num; load
laws: src=lazy+source-order, any=lazy, arith=loads compete):

| variant | model traffic | gcc insns/spills |
|---|---:|---|
| SDU src (production) | 30 | 446/70 |
| S src (sink-first + lazy ONLY) | **28** | **431/68** |
| — src (tag-only + lazy) | 43 | 431/68 |
| SDU any (source order dropped) | 30 | 446/70 |
| SDU **arith** (load law removed) | **230** | 502/132 |

* **The load law IS the algorithm**: defer loads until arithmetic
  starvation. Removing it collapses every key combination to
  traffic 230–272 regardless; keeping it, even tag-only ordering
  scores 43. Worth ~8×. This rule is the design owner's invention.
* **Sink-first is the second load-bearing component** (43 → 28) —
  also bespoke (section 30's DIF-cmul insight).
* **cp_dist DESC is near-neutral with radix-dependent sign** (helps
  R≥17 by ≤2 units, hurts R≤13 by ≤2); su_num ~0 on traffic;
  source order among loads is a no-op given laziness (any ≡ src).
* Cross-family: S-only vs production trades within ±4 units at every
  R ∈ {11,13,17,19,23}. Production stays as-is (no churn for noise),
  but the algorithm's NAME and analysis must center the two bespoke
  rules. The cone-traversal theorem target in §4 is therefore restated:
  why does STARVATION-GATED LOAD ADMISSION plus EAGER SINK RETIREMENT
  minimize Belady traffic on butterfly DAGs? The cp-descent framing of
  the earlier text was a red herring — ablated to ±2.

Naming resolved: **Starve–Retire (SR) list scheduling** — STARVE =
starvation-gated load admission (late birth), RETIRE = eager sink
retirement (early death). Historical flag --su retained. Full
algorithm description now lives in schedule.ml above `su_schedule`.
Conjecture, restated with the name: SR is the scheduling-side dual of
Belady MIN eviction on butterfly DAGs.
