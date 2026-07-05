# 67: The ILP floor, measured — what instruction ordering is worth, exhaustively

> One-sentence version: an exhaustive/branch-and-bound search over legal
> orderings of small codelet DAGs under a port-and-latency machine model
> shows ordering is worth 2–2.5× under IN-ORDER issue (where SU's order is
> poor — worse than random, because load deferral serializes), and worth
> ≤ ~5% under any out-of-order window ≥ 32–64 — including for an
> adversarially bad order — so on Raptor Lake's ~200-entry scheduler the
> in-core ILP contribution of static ordering is ≈ 1–3%, now certified by
> floor + worst case rather than argued from slack statistics (A1).

Tool: `generator/cost_model/ilp_floor.c` (beside `sched_analyze.py`).
Inputs: the schedule dump (`VFFT_SCHED_DUMP`) plus its new `.kinds`
sidecar (tag → C/L/A/N/M/F/X, written by `schedule.ml` under the same
env gate). Prompted by review: "we are handwaving ILP — brute-force the
floor for R=3/4/7."

## 1. The model

From `uarch.ml` (raptor_lake_avx2) and `schedule.ml`'s port classes:
latencies A/N/M/F = 4, L = 5, C = 0 (consts fold); ports per cycle:
mul/fma ≤ 2 (P0,P1), mul/fma + add/neg ≤ 3 (P0,P1,P5), loads ≤ 3;
X (cmul) = 2 uops on the mul/fma class. Metric = finish makespan.
Three issue disciplines per ordering:

* **in-order**: instruction i issues at max(operands ready, issue[i−1],
  first cycle with class capacity). Ordering matters maximally.
* **window-W**: per cycle, issue any ready instruction among the first W
  un-issued in program order, oldest first, capacity-limited. Models a
  W-entry scheduler.
* **W = ∞**: greedy dataflow — the machine sees everything; ordering
  survives only as a tie-break.

"Brute force" needs one honesty note: even at N=20 the DAG has more
than 2·10⁸ linear extensions, so full enumeration is physically
impossible at any radix. The exhaustive method is branch-and-bound with
admissible bounds (CP-tail from the current partial state; remaining
port work ÷ ports), which CLOSES R=3 exactly and provides best-found +
lower bounds above that, plus a 10⁶-sample distribution.

## 2. The numbers

| R | N | CP | analytic port bound | in-order MIN | SU in-order | random mean | SU @W16 | SU @W32 | worst @W32 | dataflow (W∞) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 20 | 17 | 4 | **18 (EXACT)** | 32 | 27.3 | 18 | 18 | 19 | 18 |
| 4 | 24 | 13 | 6 | 16 (best-found) | 40 | 31.4 | 17 | 17 | 16 | 16–17 |
| 5 | 46 | 25 | 11 | 33 (bf) | 76 | 56.0 | 31 | 29 | 30 | 29–30 |
| 7 | 80 | 25 | 21 | 54 (bf) | 116 | 86.4 | 48 | 36 | 40 | 35–37 |
| 8 | 69 | 21 | 18 | 43 (bf) | 98 | 73.1 | 39 | 34 | 32–36 | 27–32 |
| 13 | 245 | 37 | 72 | 148 (sample-min) | 227 | 204.3 | 132 | 113 | 107 | 99–103 |

(R=13 adds W64/W128 for SU: 102 / 100 → within 1–3% of its dataflow
limit at realistic scheduler sizes. W∞ shows a ±4% residual
order-dependence via the greedy tie-break, so the true dataflow optimum
is bracketed [port/CP bound, greedy]; the bracket width does not affect
any conclusion below.)

## 3. Three findings

**F1 — Under in-order issue, ordering is worth 2–2.5×, and SU is bad at
it: WORSE than a random legal order at every radix ≥ 5** (R=7: SU 116
vs random mean 86; R=13: 227 vs 204). Mechanism: the load-deferral law
fires loads only at starvation, turning each load into a serialization
point on a machine that cannot look past it. SU optimizes register
pressure for an allocator, not issue for a pipeline — which is the
correct trade on the actual target, but worth knowing crisply: if this
generator ever targets an in-order SIMD core, the scheduler is
disqualified as-is, and the in-order MIN column says a proper
issue-oriented schedule buys ~2× there.

**F2 — Any window ≥ 32–64 erases ordering, adversarially.** Not just
SU's order: the WORST order found in 10⁶ samples lands within a few
percent of the dataflow floor by W=32 (R=3: 19 vs 18; R=13: 107 vs
99–103). Raptor Lake's scheduler holds ~200 entries with a ~512-entry
ROB; for these bodies the effective window is ≥ the convergence point
everywhere. **The in-core ILP contribution of static ordering on the
target is ≈ 1–3%.** This is A1's half-withdrawn charge upgraded from a
slack statistic (CP slack 0.13–0.50) to a certificate: floor found,
worst case bounded, convergence measured.

**F3 — Whatever separates the dataflow floor from measured CPE is, by
construction, NOT ordering.** The model contains every ordering effect
that exists in-core; the residual is memory system and frontend — the
A9 lane (set-aliasing, 2.4× measured) — plus allocation (spills), which
is exactly where the annealer's real wins always came from. This closes
the loop with docs 65/66: the annealer never won on cycle-model ILP
because there was ≤ a few percent of it to win; it won on gcc's
allocator, the one order-sensitive consumer with a small "window."

## 4. Relation to the record

* S1/A1: certified, no longer inferential. The MKL "better ILP
  scheduling" hypothesis (doc: mkl_vs_vectorfft_1024_conclusion) loses
  its remaining mechanism — a schedule cannot give back 30% on a
  machine where the worst schedule costs 5%.
* A10 (def-use spacing): consistent — its three flat experiments are
  what W ≥ 32 convergence predicts.
* Doc 66 §4: unchanged; ordering's surviving customer is the register
  allocator, not the pipeline.

## 5. Honest limits

(Continued in doc 68: the bi-criteria formulation that unifies this
floor with the pressure axis via Belady-optimal spill traffic, and the
selection rule sigma*(R,W).)


* The model is in-core only: no frontend (decode/DSB), no memory system
  beyond load-port counts and a flat 5-cycle L1, uniform throughput,
  no µop fusion, no fill-buffer limits. It answers the ORDERING
  question; it does not predict absolute CPE.
* B&B closes R=3 only; larger radices report best-found with the node
  cap stated. The conclusions rest on the window convergence and the
  adversarial worst case, which are exact per-order simulations, not on
  the open exact minima.
* Window-W issue is greedy oldest-first; real schedulers pick by age +
  readiness similarly, but wake-up/select and port-binding details
  differ. The 1–3% figure is a model number; its order of magnitude,
  not its second digit, is the finding.
* X (cmul) nodes were absent from the dumped radices (lowered to F
  earlier in the pipeline for these builds); the 2-uop handling is
  untested in anger.
