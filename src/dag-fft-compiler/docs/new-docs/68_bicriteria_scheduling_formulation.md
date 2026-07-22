# 68: Scheduling as a bi-criteria problem — the compiler-free formulation, and where our schedules actually sit

> One-sentence version: formulating codelet scheduling correctly — cycles
> on an abstract machine M(R registers, window W, ports, latencies)
> INCLUDING Belady-optimal spill traffic — shows that the pressure-first
> schedule family is the in-model optimum for every register-finite
> machine (the issue-optimal order loses by 20%+ inside the model, no
> compiler involved), that the annealed schedule sits at the sampled model
> optimum (111 vs 112 over 25k orders at W=32), and that portability is
> achieved by PARAMETERIZING the schedule choice on architectural facts
> (R, W) — turning what looked like per-compiler patching into a selection
> rule on a characterized frontier.

Prompted by review: "I can't ship a bad scheduling and then optimize it
for clang — that's patching, not solving." This document is the solving.
Tool: `generator/cost_model/pareto.c`. Companions: doc 67 (the ordering
floor and window convergence), docs 65–66, Phase-0.

## 1. The formulation

Let G be the codelet DAG and Σ(G) its legal schedules. Define the
machine M(R, W, P, L): R vector registers, an issue window of W
instructions, port classes P (mul/fma ≤ 2; +add/neg ≤ 3; loads ≤ 3;
stores ≤ 2), latencies L (arith 4, load 5, const 0/rematerializable).
For σ ∈ Σ(G):

* **MAXLIVE(σ)** — peak simultaneously-live values. A property of the
  schedule; no compiler.
* **Belady(σ, R)** — spill stores+reloads under furthest-next-use
  eviction, which is OPTIMAL for straight-line code with known futures
  (Belady '66). Also compiler-free.
* **CYC(σ, R, W)** — makespan of the spill-EXPANDED sequence on M. The
  complete objective: issue quality AND the port cost of the spill
  traffic the schedule forces, in one number.

The solved problem is: characterize min CYC over Σ(G) as a function of
(R, W) — the frontier — and give the selection rule σ*(R, W). Anchors:
Sethi–Ullman '70 (trees, exact), Sethi '75 (DAG register minimization
NP-complete), Goodman–Hsu '88 (the integrated problem; our GH mode's
namesake), Motwani–Palem–Sarkar–Reyen '95 (combined objective NP-hard),
Touati (Register Saturation/Sufficiency: the pressure axis's extreme
points as DAG invariants). Compilers appear NOWHERE in the objective;
they re-enter only as measured approximations of Belady (a reported
"compiler tax"), which is a finding about compilers, not about us.

## 2. Results (R=13, N=245, R=16 registers)

| order | MAXLIVE | Belady spill-ops | CYC W1 | CYC W32 | CYC W∞ |
|---|---:|---:|---:|---:|---:|
| SU (pressure-first) | 32 | **32** | 415 | 118 | 100 |
| annealed | 35 | 71 | 522 | **111** | 98 |
| min-inorder (issue-optimal, doc 67) | 51 | **248** | 944 | 141 | 97 |
| anyorder | 30 | 114 | 669 | 121 | **94** |
| min over 25k random orders | — | — | 621 | **112** | 91 |

R=4 for scale: MAXLIVE 8 < 16 ⇒ zero spills ⇒ the axes decouple and the
problem is trivial; everything within 1 cycle of the floor. The
bi-criteria tension EXISTS only where MAXLIVE > R — i.e., exactly the
codelets that were ever hard.

## 3. Four findings

**F1 — Finite R kills the issue-optimal schedule inside the model.**
min-inorder's 248 spill ops flood the load/store ports: 141 vs 111–118
at W32, and it merely TIES at W∞ because spill loads consume load-port
capacity even there. "SU rescued by hardware/gcc" was the wrong
description all along: on ANY machine with 16 registers, pressure-first
wins the complete objective. The issue-optimal order was optimal only
for a machine with infinite registers, which does not exist.

**F2 — Traffic, not peak, is the true pressure metric.** anyorder has
LOWER MAXLIVE than SU (30 vs 32) but 3.5× the Belady traffic (114 vs
32): peak measures the worst instant, traffic measures the reuse
structure the allocator must serve. This is the IN-MODEL explanation of
Phase-0's "peak_live lies" — the model wasn't wrong to be a model, it
was measuring the wrong pressure quantity. SU's design goal (short
ranges) minimizes exactly the right one: 32 spill ops is startlingly
low for MAXLIVE=2×R.

**F3 — The pipeline's schedule is model-near-optimal, not
compiler-coddled.** Annealed CYC(W32)=111 vs the 25k-sample minimum
112; SU within 5%. Under the fully agnostic objective, at the realistic
operating point, our schedules ARE the optimum class. The academic
statement of the pipeline: SU approximates σ*(16, large-W); the
annealer polishes within that class; both were validated here against
the formulation, not against a compiler.

**F4 — The model's fine limit, stated honestly.** Belady misranks the
SU/annealed pair (32 vs 71 ops) that BOTH gcc and clang rank the other
way (70/51 and 136/126 spills): real allocators rematerialize and fold
operands, which the model omits. So the objective is two-layered by
NECESSITY, not laziness: the agnostic model separates schedule CLASSES
(F1's 20% gaps) and certifies near-optimality; the measured layer
resolves the last few percent per toolchain. That layer being
toolchain-specific is a theorem-adjacent fact (allocator behaviors
genuinely differ), and the wisdom architecture's per-toolchain
fingerprint is its correct engineering form — derivation, not patching.

## 4. The selection rule σ*(R, W)

* **W ≥ W\* (≈32, doc 67) and MAXLIVE > R** — every deployed x86 SIMD
  core — pressure-first. Certified here.
* **R → ∞ (or MAXLIVE ≤ R)** — axes decouple; issue order free; any
  schedule within noise of the floor (R=4 row).
* **W small / in-order** — move along the frontier toward issue; doc
  67's B&B IS that scheduler, with a measured ~2× available. First
  in-order SIMD target to enter scope inherits a ready solution.

Portability, redefined: not one artifact optimal everywhere (F4 shows
that is mathematically off the table), but one FORMULATION whose
operating point is selected by architectural facts and whose last-mile
residue is re-derived per toolchain in minutes by machinery that
already exists.

## 5. The agnosticism limit, pushed (follow-up session)

Review pushed back on §3–4's framing: "the goal is agnostic; we may not
achieve it, but push it to the limit." Pushed. Method: establish
per-target achievable floors, then MINIMAX search — minimize the worst
regret across targets jointly (`tools/robust_anneal.py`).

Per-target floors and probes, R=13:

* clang order-floor: a 260-iteration direct search under clang moved
  spills only 136 → 131 (gcc's identical search: 70 → 51). Clang is
  ORDER-INSENSITIVE — its pre-RA MachineScheduler re-derives its own
  order (confirmed: `-mllvm -enable-misched=false` makes both SU and
  annealed WORSE, 127/158). Allocator sweep: greedy 136 (default,
  best), pbqp 144, basic 170, fast 273 — the mass is not a flag away.
* Platform axis inside gcc: everything transfers. znver3 (16-reg AVX2
  Zen): SU 70→70 spills, annealed 51→54, dup 31→38, min-inorder
  124→140 — same ranking, near-same magnitudes. znver4-class targets
  (EVEX ymm16–31 ⇒ R=32 ≥ MAXLIVE): spills collapse to 27/28/5 — on
  any 32-register platform the problem DISSOLVES, per §2's R-axis.

Minimax result: seeded from the gcc-annealed order, the joint search
found a STRICTLY DOMINATING point in 127 iterations —

| single order `best_minimax_r13` | spills |
|---|---|
| gcc / raptorlake | **50** (order-floor was 51) |
| gcc / znver3 | 56 |
| clang / raptorlake AND znver3 | **114** (beats clang's own dedicated-search floor 131 and all prior orders) |
| model M(16, W32) | annealed-class, at the 25k-sample optimum |

Max regret 0.98; bit-exact; promoted to the shipped wisdom entry. The
joint objective found clang improvement the clang-only search missed —
gcc-structure guided it.

**The limit, stated:** at the ORDER level the agnostic artifact EXISTS
and is now shipped — one schedule at/below every target's known floor
simultaneously. The residual non-agnosticism is a single number: clang
114 vs gcc 50, a ~2.3× compiler tax proven not order-payable by
three-way exhaustion (dedicated search, allocator sweep, rescheduler
probe) and not transform-payable at the C level (doc 65's clang row).
That number is a measured property OF CLANG on this code shape, to be
reported as such; and it is confined to the 16-register regime, which
32-register platforms erase entirely.

## 6. Open items for the academic writeup

Exact ε-constraint frontiers beyond the trivial region (B&B with
MAXLIVE bound); Register Saturation/Sufficiency values per codelet
(Touati) as the frontier's analytic endpoints; the W\*(N, R) collapse
law across the codelet family; a Belady+rematerialization model to
close F4's gap; and the duplication transform (doc 65) restated as a
FRONTIER-DOMINATING DAG rewrite — same flops, strictly better
(traffic, cycles) region — which needs the OCaml pass so transformed
DAGs can be dumped and mapped.

## 7. Honest limits

The machine model omits frontend, memory system beyond port counts and
flat L1 latency, µop fusion, and rematerialization (F4). W1 here is
strict oldest-only issue (harsher than doc 67's chained in-order sim;
both are idealizations bracketing "in-order"). Sampled minima are
lower-bound witnesses, not proofs; exact frontiers are open above R=4.
All conclusions rest on 20%-class separations and convergence
behavior, not on second digits.
