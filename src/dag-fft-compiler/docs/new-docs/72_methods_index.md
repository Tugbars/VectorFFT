# The four schedule-improvement methods — what they are, where they live

Companion index to docs 65–71. One table row per method, then a
per-method section with the full name, the mechanism in two
sentences, the measured results, and **every file that implements,
proves, or ships it**. All paths are repo-relative. Base scheduler
underneath everything: **Starve–Retire (SR)**, the certified
production list scheduler in
`src/dag-fft-compiler/generator/lib/schedule.ml` (naming, algorithm
block, experimental record: header comments above `su_schedule`;
certification: `docs/69_su_certified.md`). Delivery mechanism for all
promoted orders: the **wisdom directory**
(`src/dag-fft-compiler/generator/sched_wisdom/`, consumed via
`VFFT_SCHED_WISDOM`, `#dagsig`-verified, provenance-stamped into the
emitted C).

| Method | Kind | Cost | Where it wins |
|---|---|---|---|
| Annealer (autotuner) | searcher | minutes/codelet | sole working lever on blocked pow2 (R=32/64) |
| Minimax | searcher | minutes/codelet | portable cross-compiler orders (R=13) |
| Sink-cone affinity | policy | zero, deterministic | mid-size monolithic DAGs (R=11–19) |
| Selective duplication | transformation | zero once the OCaml pass lands | primes, from plain SR order |

---

## 1. Annealer ("autotuner")

**Full name:** iterative-compilation autotuning via simulated
annealing over legal topological orders, scored on realized assembly.

**Mechanism:** seeds from SR's order; move operators (reinsert /
block-move / segment-reverse) preserve topological legality, so every
candidate is bit-exact by construction; each candidate is generated,
compiled with production flags, and objdump/asm-counted — the search
optimizes the *measured* output, never a model. Objective: total
instructions down, gated on spills not increasing and FMA count
invariant (the Phase-2 noise-robust win rule).

**Results:** R=13 70→51 spills (later superseded by minimax); blocked
R=32 1031/183→**996/168**; blocked R=64 2543/594→**2490/585** (spills
pinned = permutation exhausted, which motivated duplication).
Post-duplication role: retired on primes, sole ordering lever on
blocked pow2, permanent ceiling instrument and toolchain-transfer
tool.

**Files:**
- `src/dag-fft-compiler/tools/anneal_linux.py` — monolithic-codelet
  annealer (container form; `VFFT_WARM=` warm-start supported).
- `src/dag-fft-compiler/tools/blocked_anneal.py` — per-cluster
  annealer for the blocked path; one order file per pass/subset,
  scored on the whole codelet.
- `src/dag-fft-compiler/tools/anneal.py` — original Windows/WSL-shaped
  driver (historical record).
- Injection plumbing it rides on: `VFFT_SCHED_DUMP` /
  `VFFT_SCHED_ORDER` in `generator/lib/schedule.ml` (both the
  monolithic and the keyed per-subset variants).
- Shipped artifacts: `generator/sched_wisdom/radix32_n1_fwd_avx2_*.txt`
  (×12) and `radix64_n1_fwd_avx2_*.txt` (×20); raw campaign winners in
  `experiments/sched_search/best_blocked_r{32,64}_s*/` and
  `best_r13_s1.txt`.
- Write-ups: `docs/66_schedule_annealer_architecture_and_role.md`
  (component doc), `docs/performance/schedule_search_phase2_container_results.md`
  (campaign numbers; repo-root docs).

## 2. Minimax (cross-compiler robust search)

**Full name:** cross-compiler minimax schedule search — the annealer
engine with the objective replaced by worst-case deficit across
per-compiler floors (gcc and clang scored every candidate; minimize
max shortfall).

**Mechanism:** same legal-move annealing, but each candidate is
compiled under BOTH toolchains and scored against each one's
best-known floor; the search returns a single order that no target
regrets. Found that clang is order-insensitive (its own dedicated
search bottomed at 131 spills; the minimax order reaches 114) and
that the residual clang-vs-gcc gap is allocator tax, confined to
16-register files.

**Results:** one R=13 order at **392/50 (gcc)** and **540/114
(clang)** — at or below every per-target floor simultaneously;
transfers to znver3; dissolves on 32-register EVEX targets.

**Files:**
- `src/dag-fft-compiler/tools/robust_anneal.py` — the searcher.
- Shipped artifact: `generator/sched_wisdom/radix13_n1_fwd_avx2.txt`
  (the production R=13 wisdom entry IS the minimax order).
- Write-up: `docs/68_bicriteria_scheduling_formulation.md` §5 (with
  the clang order-insensitivity evidence: misched/allocator sweeps).

## 3. Sink-cone affinity

**Full name:** sink-cone affinity — a dynamic tie-break that
maximizes transitive-fanout overlap with the last-issued instruction
(among cp-tied ready nodes, argmax |TFO(n) ∩ TFO(last)|, TFO = set of
reachable output sinks). Our coinage; nearest relatives are lineage
scheduling (input-side, single-heir — raced and lost) and static
cone clustering from logic synthesis.

**Mechanism:** replaces the SU-number tie-break slot. Rationale is
the input-side-blindness result: the slot fires only at cp-ties,
cp-tied ready nodes on butterfly DAGs are isomorphic siblings with
congruent pred-cones, so every input-side label (classic SU, both
DAG-corrected SUs, kills) is provably schedule-identical; only
output-side measures discriminate, and affinity is the minimal
dynamic one — "stay in the cone you already opened," SR's own
philosophy pointed at the future.

**Results (gcc insns/spills):** R=11 317/42→**306/41**, R=13
446/70→**417/59**, R=17 756/175→**741/169** (first-ever R=17
improvement), R=19 876/192→868/189 (PROVISIONAL — Belady traffic and
llvm-mca mildly oppose; i9 arbitrates), R=23 spill-gate fail, R=4/8
inert-to-harmful, blocked R=25/64 lose, blocked R=32 small win
(subsumed by the annealed entry). Operating window: mid-size
monolithic DAGs. llvm-mca audit: −1.7% cycles at R=13, −2.8% at R=17.

**Files:**
- Implementation: `generator/lib/schedule.ml` — env knob
  `VFFT_SU_TIEBREAK={cone,affinity}` in BOTH `su_schedule`
  (monolithic; sink masks + `last_mask`) and `su_schedule_subset`
  (blocked; subset-relative masks, per-cluster reset). Default =
  classic, byte-identical (10-radix identity gate).
- Theory + full 10-radix race table + operating window + runtime-proxy
  audit: the SU NUMBER header comment above `compute_su_number` in
  `schedule.ml`; condensed as item 11 of the EXPERIMENTAL RECORD
  above `su_schedule`.
- Race harness: `src/dag-fft-compiler/tools/ablate2.py` (pluggable
  tie-break keys: classic / first-owner DAG-SU / shared-as-1 / kills /
  cone / affinity; exact SR replica).
- Shipped artifacts: `generator/sched_wisdom/radix{11,17,19}_n1_fwd_avx2.txt`
  (provenance-commented; R=19 carries the PROVISIONAL flag inside the
  file).

## 4. Selective duplication (un-CSE)

**Full name:** selective un-CSE duplication of long-span leaf-fed
values — cloning a cheap, leaf-computed value at each distant use so
its live range never crosses the body, trading a recompute for a
spill-store/reload pair.

**Mechanism:** CSE creates values whose uses span hundreds of
instructions; on 16-register targets each such value is a guaranteed
spill. The v5 selector picks clone targets by liveness coverage of
long spans; v4 adds chained duplication (clones fed from barriered
leaf reloads) for the deepest cases. Wins are DAG-level: they land
from PLAIN SR order, no search needed, and beat the annealer's
converged results on primes.

**Results (probe-level, bit-exact):** R=11 317/42→**269/17**, R=13
446/70→**377/31** (beats the annealed 51 from plain order), R=17
→**705/118**, R=19 →**849/139**, R=23 →1274/268 (v4). Pow2 negative
in all variants (shallow-long-span vs deep-cascade divide). Status:
probes are C-level post-processing; the production OCaml algsimp pass
(spec: doc 65 §8 — v5 selector + v4 chained mode, `VFFT_DUP` env,
byte-identity gate, raced per codelet A4-style) is **the one
remaining code item**. Open stacking cell: dup targets were selected
on classic-order spans; dup-on-affinity (R=13 from 417/59) is
unmeasured.

**Files:**
- `src/dag-fft-compiler/tools/dup_probe.py` — v2 (leaf-fed, span≥S,
  per-clone asm barrier).
- `src/dag-fft-compiler/tools/dup_probe4.py` — v4 chained (the R=23
  variant).
- `src/dag-fft-compiler/tools/dup_probe5.py` — v5 coverage selector
  (the recommended formulation for the OCaml pass).
- Write-up + pass spec: `docs/65_selective_duplication_uncse.md`.
- Future home of the pass: `generator/lib/algsimp.ml` (not yet
  written).

---

## Cross-cutting instruments (used to prove all of the above)

- `tools/ablate.py` — exact SR-replica ablation (established the
  lazy-load ~8× / sink-first weights).
- `tools/traffic.py` — dump-convention Belady traffic scorer.
- `tools/lineage_sched.py`, `tools/beam_sched.py` — literature rivals,
  kept as recorded negatives (doc 69).
- `tools/spill_inject.py` — generator-owned explicit spilling,
  recorded negative (doc 70); MSVC-reusable counting harness.
- `generator/cost_model/ilp_floor.c` — exhaustive/B&B ordering floors
  (doc 67).
- `generator/cost_model/pareto.c` — bicriteria MAXLIVE/Belady/cycles
  scorer (doc 68).
- `experiments/sched_search/verify{13,17,32,64}.c` + your
  `bench_codelet.c` — bit-exactness and the i9 paired-A/B gate.

## How they compose

Searchers and the policy all emit ORDERS; orders compete on measured
assembly; winners become dagsig'd wisdom entries; `schedule.ml`
injects them with provenance and refuses stale ones. Duplication is
the odd one out — it changes the DAG itself, so it lives upstream of
all ordering and will ship as a generator pass, not a wisdom file.
Current leaderboard: minimax holds R=13, affinity holds R=11/17/19
(19 provisional), the annealer holds R=32/64. Standing gate before
any of this is trusted as runtime performance: the i9 paired A/B
(`bench_codelet.c` methodology, MDE 0.1–0.3%), per
`integration/INTEGRATION.md`.
