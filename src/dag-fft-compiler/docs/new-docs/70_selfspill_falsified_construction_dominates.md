# 70: Self-spilling falsified — construction-level spilling dominates optimal instruction-level spilling

> One-sentence version: the strongest compiler-agnosticism proposal —
> generator-owned Belady-optimal explicit spilling on monolithic pow2,
> demoting compilers to instruction selectors — was prototyped
> (tools/spill_inject.py), made bit-exact, and FALSIFIED on two
> independent grounds: (1) even the optimal PLAN's scattered traffic at
> implementable budgets (76–88 ops @R=16, 262–299 @R=32) exceeds the
> blocked construction's total realized memory ops (52–59 / 150–183),
> and (2) no compiler executes the plan faithfully without
> compiler-specific defenses (gcc SRA-deletes undefended scratch;
> pointer-laundering makes gcc's allocator panic, +127–160 stack
> spills; clang needs the per-reload barriers gcc chokes on) — so the
> mechanism reintroduces per-compiler fragility by its nature. The
> positive discovery: the doc-58 blocked construction is NOT a gcc
> workaround; it is a spilling ALGORITHM that beats Belady-optimal
> scattered allocation on the same DAG, because bulk, sequential,
> algorithm-placed seam traffic is cheaper than optimally-placed
> per-value traffic.

## 1. The proposal and why it was worth trying

Review direction: pow2 (R=16/32) is what matters; make it agnostic
through DAG/construction, not per-compiler scheduling. Self-spilling
was the maximal version: if emitted code never exceeds the register
file, allocators have nothing to decide. Ingredients all existed (SR
order certified at the traffic floor, Belady expansion validated,
spill-recipe precedent in the blocked path).

## 2. The numbers (insns / memory-ops class)

| R=16 | gcc | clang |
|---|---|---|
| blocked (production) | 391 / 59 | 441 / **52** |
| mono baseline | 402 / 78 | 470 / 82 |
| mono + self-spill, best defended | 622–642 / 76-plan + 151–160 gcc-added | 564–573 / 76–88-plan + 6–22 |
| **plan floor (Belady @B=15–16)** | **76–88 ops** | — loses to blocked's 52 before any compiler touches it |

| R=32 | gcc | clang |
|---|---|---|
| blocked (production) | 1031 / 183 | 1048 / **150** |
| mono + self-spill plan @B=15–16 | **262–299 ops** | — loses to blocked's 150 |

Bit-exactness of the honored mechanism verified (clang, R=16).
Model-vs-plan note: DAG traffic at the FULL 16-register file is 36/143,
but a compiler needs headroom; at implementable budgets the curve is
steep (~25 ops per register at R=16), and the seam alternative was
always cheaper than where the curve lands.

## 3. Why blocked wins — the structural argument

Belady is optimal among SCATTERED allocations of a fixed DAG+order. The
blocked construction changes the game: it REWRITES the DAG so the
overflow crosses one algorithm-chosen boundary as bulk, sequential,
streaming traffic through a private L1 seam (doc 58) — fewer total
ops, better access pattern, and each half then fits the file with
margin so compilers allocate the remainder near-perfectly (gcc 59,
clang 52 — clang actually BETTER, see §4). Structured beats optimal:
spilling is a construction decision, not an allocation decision. This
also retro-explains doc 65's pow2 negative — duplication had nothing
to fix because blocking already restructured the traffic.

## 4. Agnosticism status for the codelets that matter

Measured this session: clang ≤ gcc on blocked pow2 (52 vs 59; 150 vs
183) — the "compiler tax" was prime-shaped, and primes are explicitly
secondary. The blocked construction is already the agnostic artifact
for R=16/32. Untested and commercially relevant: MSVC (the Windows
driver product) — the one remaining allocator to baseline.

## 5. Surviving roadmap for pow2 performance (priority order)

1. Belady-account the BLOCKED construction itself (per-cluster floors
   + seam ops): is 52–59 near the blocked-structural floor, or is
   there a mapped gap? The doc-68 machinery pointed at the
   construction — the right next measurement before touching anything.
2. Blocked-path SR ablation (subset comparator has port-balance/GH
   keys never ablated; the monolithic ablation found a simplification
   that ties production — doc 69 CORRECTION).
3. Cluster-level minimax wisdom (robust_anneal extended to subsets;
   optional garnish per doc 66, gcc floor 168 known at R=32).
4. Memory side (A9 padding, seam placement, i9 items) — where the
   MKL-gap giveback actually lives per the phase-3 port-slot analysis;
   register-side pow2 is near closed.

Primes inherit nothing further from this line (they already carry SR +
duplication + minimax wisdom); self-spilling for primes remains
possible in-plan (R=13 traffic 30 vs gcc 70) but inherits §2's
mechanism fragility and is deprioritized with the primes themselves.

## 6. Artifacts

`tools/spill_inject.py` (recorded negative; barrier and laundering
variants inline), ss*/ssL*/ssNB* probe outputs regenerable; baselines
and sweeps in this doc. All monolithic probes bit-exact-gated.


## CORRECTION (same week, doc 71): shaping, not spilling

Doc 71's decomposition shows the seam never survives compilation —
both compilers scalarize spill_re/im entirely and re-decide all
spilling. §3's mechanism is therefore refined: blocking dominates by
SHAPING the allocation problem (order-floor 70→21 / 249→53, doc 71
§2), not by performing structured spills; the emitted seam is dataflow
routing that constrains the order, and the memory it names is fiction.
The falsification of self-spilling stands unchanged and gains a
unifying explanation: allocators erase explicit spill mechanisms
whether ours or the construction's.
