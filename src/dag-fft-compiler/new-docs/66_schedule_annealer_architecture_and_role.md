# 66: The schedule annealer — architecture, record, and post-duplication role

> One-sentence version: the annealer is iterative compilation over legal
> instruction orders — stochastic superoptimization with bit-exactness by
> construction, scored on the realized assembly — which won on every
> codelet class it touched, was then BEATEN on primes by a ten-line
> deterministic transform its own forensics motivated, and settles into
> three permanent jobs: sole optimizer of blocked pow2 schedules,
> ceiling instrument for every future deterministic idea, and
> toolchain-transfer tool for schedule wisdom.

Lineage: `docs/roadmap/schedule_search_plan.md` (the plan and kill
criteria), `docs/performance/schedule_search_phase0_results.md` (the
objective validation: asm spills ↔ runtime Spearman ρ≈0.94; peak_live is
a liar; `--su` is the incumbent to beat), `tools/anneal.py` (first win,
R=13 −7 spills), and `docs/performance/
schedule_search_phase2_container_results.md` (the container campaigns
this document consolidates). Companion finding: doc 65 (selective
duplication), which this tool's exhaustion at R=64 motivated.

## 1. What it is, precisely

A search over the LEGAL REORDERINGS of a fixed codelet DAG, scored by
compiling each candidate and counting the artifact. Three properties
define it:

* **Legality by construction.** Every move operator preserves a valid
  topological order, so every candidate computes bit-identical results —
  correctness is not tested per candidate, it is guaranteed by the move
  set. (The bit-exact harness is still run once on each final winner.)
* **The objective is the artifact, never a model.** Phase 0 killed the
  model route: `bb.ml` provably minimizes `peak_live` and emitted 2× the
  realized spills. The annealer scores gen → `gcc -S` → parse: total
  instructions primary, hard-gated on FMA-count invariance (port profile
  untouched) and spills never above the su baseline.
* **The search space is permutation only.** It cannot duplicate,
  re-associate, select instructions, or cross cluster boundaries. Doc 65
  is the record of what lives outside that space.

## 2. Architecture

Two drivers, one scoring path, both in `tools/`:

**`anneal_linux.py`** — monolithic codelets (primes, forced-mono pow2).
Simulated annealing, moves: slack-window single reinsert (a node moves
only within [after last pred, before first succ]), block move (2–6
contiguous nodes, block-level slack window), antichain segment reversal.
Memoized (order-hash → score; 141/800 evals saved at R=13), adaptive
reheat on 120 stalled iterations. ~0.39 s/candidate.

**`blocked_anneal.py`** — blocked pow2 (R≥25 recipe codelets, plus R≥16
AVX2). The subset-keyed injector in `schedule.ml` is the interface.
Design decision that matters: clusters are independent for LEGALITY, but
gcc allocates over the whole function, so cluster-local optima need not
compose — therefore **moves are per-cluster, scoring is always global**
(every candidate compiles the full codelet). Coordinate descent over
clusters, largest first. **Isomorphic transfer**: same-size clusters are
instances of the same sub-DFT; a winner's su-rank permutation is tested
once on each sibling (fired measurably on both R=32 and R=64, and the
seeded sibling improved further within 2 iterations). **Warm start**
(`VFFT_WARM=<dir>`): resume from a saved incumbent; the true su
reference is still reported.

**Scoring path.** Native gen (0.04 s injected) → `gcc -S -O3 -mavx2
-mfma -march=raptorlake -w` → parse the `.s` (no link, no objdump, no
dlopen). gcc is ~96% of eval cost; this is why the driver language is
irrelevant and why the search lives OUTSIDE the OCaml pipeline — the
generator stays a deterministic pure function, the annealer is a wisdom
FACTORY beside it (see the schedule-wisdom machinery: `#dagsig`
verification, `VFFT_SCHED_WISDOM` consumption, provenance trailer).

**Order fidelity.** Measured: `-fno-schedule-insns2` changes nothing on
these kernels and pre-RA scheduling is off by default on x86 — the
emitted IR order reaches gcc's allocator intact. Whatever the search
finds, silicon receives; nothing is laundered.

## 3. The record

| codelet | su baseline | annealed | Δ | notes |
|---|---|---|---|---|
| R=13 prime, monolithic | 446 / 70 | 392 / 51 | −12% insns, −27% spills | 659 evals, ~5 min; bit-exact |
| R=32 pow2, blocked | 1031 / 183 | 996 / 168 | −3.4% / −8.2% | 12 clusters; transfer fired |
| R=64 pow2, blocked | 2543 / 594 | 2490 / 585 | −2.1% / −1.5% | 2 rounds (warm-started); spills PINNED |

The two findings its forensics produced outrank the numbers:

* **The leaf blind spot.** Decomposing the R=13 winner: 40/41 leaves
  moved (loads earlier, first-use distance 1→8, source order broken
  88/325; constants later, 55→21; arithmetic median shift −2). The
  entire prime win was leaf placement — decisions the deterministic
  scheduler makes by fiat. Three greedy rule families built to capture
  it ALL regressed (phase-2 doc, follow-up §1): the pattern is a
  property of a jointly-optimized solution, not a decision rule. The
  headroom is search-only.
* **The R=64 exhaustion → duplication.** Spills pinned at ~585 across
  two rounds said "permutation is done; the remaining lever must change
  the DAG." That sentence is doc 65's origin. The resulting transform
  then beat the annealer itself on primes: v2 from plain strict-SU
  order (377/31) over the converged search (392/51).

## 4. Post-duplication role (the decision this doc records)

| role | status | evidence |
|---|---|---|
| Primary optimizer, primes | **RETIRED** | dup v2 on strict SU beats the converged search; deterministic, one rule |
| Only working lever, blocked pow2 | **IRREPLACEABLE** | duplication negative in all variants there; greedy leaf rules negative; nothing else has ever moved R=32/64 schedules |
| Second stage after the dup pass | **OPEN CELL** | dup-on-annealed measured (391/42, stale order loses); anneal-AFTER-dup needs the OCaml pass so the search sees the transformed DAG. Pre-registered prediction: residual ≤ 4 spills at R=13, since the search's prime win was placement of exactly the values duplication deletes |
| Ceiling instrument | **PERMANENT** | both of this cycle's findings came from its forensics/exhaustion; a search that converges at zero gain is the certificate that a construction is at its floor |
| Toolchain transfer | **KEEP** | wisdom is gcc-fingerprinted by design; warm-started re-anneal converges the compiler-specific residue in a fraction of a cold run |

Pipeline, post-decision: construction (dup as a raced dimension: v5
coverage selector + v4 chained mode) → strict-SU schedule → annealer
where headroom remains (blocked pow2; optional prime polish once the
OCaml dup pass exists) → wisdom (dagsig'd order files) → gcc.

## 5. Operating manual

```
# monolithic:
python3 tools/anneal_linux.py R [iters] [seed]
# blocked:
python3 tools/blocked_anneal.py R [iters/cluster] [seed]
VFFT_WARM=<saved incumbent dir> ...        # resume
VFFT_GENDIR / VFFT_WORK                     # relocation
```

Both drivers read/write `#dagsig`, so every output drops into a wisdom
directory as-is and every injected candidate is verified — a
driver/dump desync fails loudly mid-search. Gotchas: score main-loop
artifacts only (`VFFT_NO_ANYK_TAIL=1`; the tail cascade is
once-per-call); the monolithic injector takes an exact file path, the
subset injector a prefix; magnitudes are toolchain-specific — re-score
under the production gcc before banking, warm-start from the shipped
order.

## 6. Retirement conditions

If A6/A7 streaming construction lands and removes the blocked spill
floor constructionally, the annealer's last production territory
evaporates and it shrinks to instrumentation + toolchain transfer. The
certificate of that retirement is, characteristically, the annealer
itself: run it on the streaming-emitted codelets; convergence at zero
gain IS the proof the construction sits at the floor. A search whose
highest use is proving it has nothing left to find is the falsification
apparatus the program docs keep asking for — cheap to keep, expensive
to be without, and the one component that tells the truth about the
others.

## 7. Honest limits

* Campaigns were single-seed (seed 1; R=64 round 2 used seed 2); the
  multi-seed spread is uncharacterized. Budgets were modest (80–120
  iters/cluster); the big-cluster trajectories were still descending at
  budget on R=64 round 1.
* The blocked search space EXCLUDES inter-cluster interleaving by
  emission contract (cluster contiguity) — fuse-M-style moves are
  invisible to it; a schedule property involving them would need an
  emission change first.
* Objective is the static phase-0 proxy (ρ≈0.94), not runtime; the i9
  paired bench remains the shipping gate.
* All magnitudes are gcc 13.3 / raptorlake-march / container; the
  mechanism transfers, the numbers do not.
