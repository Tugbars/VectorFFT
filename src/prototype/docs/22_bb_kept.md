# 22. Branch-and-Bound Cluster Scheduler — Kept as Opt-In Alternative

## Summary

Implemented a time-budgeted branch-and-bound cluster-local scheduler with a multi-objective lexicographic cost. On Raptor Lake the schedules it produces are **structurally different from SU+GH but perform roughly the same on average**, with a real K-regime crossover at R=64 AVX2. **Kept as an opt-in `--bb` flag** because (a) the schedules genuinely differ, (b) other µarchs may favor BB-lex for reasons we can't predict from a single test platform, (c) the code footprint is small (~270 lines), and (d) it serves as a useful research probe.

Default remains SU+GH. The flag is opt-in.

## Cost function: lexicographic (saturated_peak ASC, -progress ASC)

```
saturated_peak = max(peak_live, uarch.vec_regs)
progress = sum_i cp_dist[node_i] × (N - 1 - i)
```

The saturation makes peak counts that already fit in the architectural register file equivalent — only over-budget peaks (real spills) count. The progress term rewards scheduling high-cp_dist nodes early, matching SU's primary key.

This was the second iteration. The first iteration used peak-live-only as the cost; B&B beat SU+GH on peak (R=32 AVX2: 12 → 9 across all 8 clusters) but lost at runtime by ~3% — pure peak-live minimization extended dependency chains and reduced ILP. The lexicographic cost stops fighting for peak reductions that don't matter (below the register file) and uses progress to pick latency-friendly schedules among ties.

## Cost function diagnostic results

The lex cost says SU+GH is already optimal (or near-optimal) under this metric:

- **R=32 AVX2 (CT 4×8)**: B&B finds same peak (12) and same progress as SU+GH across all 8 clusters
- **R=64 AVX-512 (CT 8×8)**: B&B finds same peak (21-25 per cluster) and same progress across all 8 clusters
- **R=64 AVX2 (CT 8×8)**: 4/8 clusters tied; 4/8 with tiny progress gains (0.1-0.3%); 1/8 with peak improvement (18→17)

So under the cost function we picked, SU+GH is at the practical optimum for clusters at our scale. Reaching the same point from B&B exploration is the empirical confirmation.

## Bench results (Raptor Lake)

R=32 AVX2 — median ratio of B&B-lex to GH (3 runs per K):

| K | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|----|----|----|----|----|----|----|
| L/GH | 0.994 | 1.005 | 0.998 | 1.001 | 0.982 | 1.001 | 1.004 |

All within ±2%. Mean ≈ 1.000. Effectively tied across all K.

R=64 AVX2 — median ratio of B&B-lex to GH:

| K | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|----|----|----|----|----|----|----|
| L/GH | 1.025 | 1.014 | 1.012 | **0.942** | **0.962** | 0.989 | 0.997 |

Real K-regime crossover: GH wins at small K (1-3%); BB wins at K=512-1024 (3.8-5.8%); tied at large K. The K=512 win replicated cleanly across all three runs (0.933, 0.955, 0.942) — not noise.

## Structural difference

Even when (peak, progress) are tied, the schedules are not the same code. At R=32 AVX2:

```
diff GH BB-lex: 413 lines differ out of 886 total lines (~47%)
```

DFS lands on a different one of the equivalent picks at each step. Same DAG, same arithmetic ops, same peak live count, same total cp_progress, but different ordering. The runtime variance per K reflects this — BB-lex explores a different point in the cost-tied-equivalence-class.

## Why keep it

1. **The schedules genuinely differ.** Same cost function value, structurally different code. They sample different points in the schedule space, which means the runtime distribution differs even when means are close.
2. **µarch portfolio.** All measurements were on Raptor Lake. Zen5, Sapphire Rapids, Skylake-X, and Tiger Lake have different port-pressure profiles, OOO-window depths, and prefetcher behaviors. The BB-lex orderings might be the better fit on a µarch we haven't measured. Keeping the flag costs us nothing if we don't enable it by default.
3. **K-regime crossover at R=64 AVX2.** There's a measurable BB win at K=512-1024 (~4-6%). Workloads concentrating in that K range may want `--bb` on by default.
4. **Small footprint, fully opt-in.** ~270 lines for `bb.ml`, the diagnostic, and the dispatch wiring. Default off (`--bb` to enable). The default codegen path is unchanged from before.
5. **Useful research probe.** `bb_diagnostic.exe` reports cluster sizes and peak/progress per cluster, which is independently useful for understanding the structure of any new radix or factorization.

## Why not auto-enable

- On Raptor Lake the average runtime is essentially tied with GH; no reason to pay codegen-time cost (~1s/cluster × ~8 clusters per codelet) by default.
- The K-regime crossover at R=64 AVX2 is real but localized — auto-enabling would slightly hurt small-K performance for the sake of mid-K.
- Without per-µarch measurements, we don't know which side of the crossover dominates on other targets.

## Usage

```bash
# Default: SU+GH (recipe + Goodman-Hsu mode switch where applicable)
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place 32 --isa avx2 --uarch raptor_lake_avx2

# Opt in to B&B-lex
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --bb 32 --isa avx2 --uarch raptor_lake_avx2

# B&B with custom budget (default 1.0s/cluster)
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --bb --bb-budget 5.0 64

# Diagnostic: see per-cluster peak/progress before/after B&B
dune exec bin/bb_diagnostic.exe -- 64 avx2 raptor_lake_avx2 1.0
```

## Future direction (not implemented)

**Per-(R,K,ISA,µarch) autotuning.** Generate both `--gh` and `--bb` variants, micro-bench at install time on the target hardware, cache the winner per cell. This is the FFTW-wisdom pattern, and it'd convert the either-or knob into "always pick the best option for this exact configuration." Requires a benchmarking harness baked into the build pipeline, which is a non-trivial lift.

## What this rules in / out

- **Rules out**: B&B as a strict improvement over SU+GH on Raptor Lake. The cost function is right; SU+GH already optimizes it.
- **Rules in**: B&B as a portfolio member. Different schedules with similar cost-function value can perform differently on different µarchs; having a second variant is cheap insurance.

## Implementation status

- `lib/bb.ml`: B&B scheduler with lex cost (~270 lines)
- `bin/bb_diagnostic.ml`: per-cluster diagnostic tool
- `lib/emit_c.ml`: `?bb_budget` parameter on `emit_codelet`, dispatched at all 3 cluster scheduling sites
- `bin/gen_radix.ml`: `--bb` and `--bb-budget T` CLI flags
- `lib/dune` adds `bb` module + `unix` library dependency
- `bin/dune` adds `bb_diagnostic` executable

Default behavior unchanged: SU+GH everywhere it was before.
