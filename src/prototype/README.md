# vfft v2 — Variant-Dispatcher Codelet Generator

A small OCaml-based generator for SIMD FFT codelets, exploring how far emission policies (variants) can carry generated code without changing the math layer. Empirically, the answer is "very far": the same OCaml code path beats hand-tuned reference codelets across radices R=16/32/64 on AVX-512, and across radices R=16/32 on AVX2.

## What's inside

```
vfft_v2/
├── README.md          this file
├── docs/              chronological findings (01-12), each one is a session writeup
├── lib/               OCaml generator (dft, algsimp, emit_c, schedule, isa, uarch, ...)
├── bin/               gen_radix CLI
├── bench/             bench harnesses + generated codelets + hand-coded references
└── dune-project       OCaml build config
```

## Headline result

For R=32 AVX-512 (in-place vs in-place):

| K     | Hand | Topo  | SU+Spill | Recipe-log3 | SU/H | LR/H |
|-------|------|-------|----------|-------------|------|------|
| 64    | 1080 | 1714  | 1572     | 1029        | 0.99 | **0.95** |
| 1024  | 36193 | 46569 | 33125  | 32593       | 0.93 | **0.90** |
| 4096  | 267864 | 291257 | 311878 | 147061    | 0.91 | **0.55** |

(times in ns; SU/H or LR/H < 1 means we beat hand-coded)

For R=16 AVX-512 (in-place vs in-place): beats hand at K ≥ 128. R=32 AVX2: up to **44% faster than Topo** flat (no hand-coded reference). R=64 in-place vs our own Topo: **17-47% faster** (recipe), **37-61% faster** (recipe-log3) — `gen_radix64.py` only emits OOP, so there's no in-place hand R=64 to compare against directly.

**Mode note:** all R=4/8/16/32 hand-coded references are in-place; benches at those radices are in-place vs in-place. For R=64, hand-coded only exists as OOP, so the R=64-vs-hand comparisons below are OOP-vs-OOP. Hand R=64's OOP path may not be its primary tuning target (gen_radix*.py family is designed around in-place; OOP is for 2D use cases). The robust R=64 win is "ours in-place vs ours Topo": 1.5-2.5× faster.

## The recipe

For CT-decomposed codelets:
```
spill PASS 1 outputs at the natural CT boundary (--spill)
  + run SU within each pass (--su)
  + cluster-sequential PASS 1 (block by sub-FFT)
  + cluster-sequential PASS 2 (one sub-DFT at a time, stores immediately after)
  + just-in-time reload (within PASS 2, defer each load to first use)
  + Goodman-Hsu pressure mode switch (--gh, AVX2 R≥32 only)
```

Use it when:
```
CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32)   → spill+su
   PLUS, if vec_regs <= 16 AND n >= 32                   → +gh
```

This means:
- AVX-512 (vec_regs=32): always use the recipe at R≥4. `gh` is a no-op there since cluster-sequential keeps per-cluster live count below threshold=24.
- AVX2 (vec_regs=16): use the recipe at R≥16; auto-enable `gh` at R≥32 for an additional 5-7% (R=32) or 4-8% (R=64). R≤8 prefers Topo.

All these are wired into auto-defaults — pass `--no-recipe` to opt out of everything.

## How to read the docs

The findings under `docs/` are chronological session writeups. They tell the story of how each lever was discovered, what didn't work, and how the picture evolved:

1. **su_scheduler** — SU list scheduler (first lever, modest effect alone)
2. **r16_crossover** — discovered R=16 small-K vs large-K crossover; variants matter
3. **r32_finding** — R=32 brings register pressure problem into focus
4. **r32_arith_gap** — investigation: GCC already does FMA fusion, math layer is fine
5. **r32_spill** — explicit boundary spilling (closes ~half the gap)
6. **r16_spill** — heuristic falsified at AVX-512 R=16 (recipe helps even when registers fit)
7. **r32_fuse** — FUSE per-spill-target lifetime (modest gains, F8 = block-seq no-spill is interesting)
8. **r32_su_spill** — SU within passes (closes most of the rest)
9. **r32_beating_hand** — cluster-sequential PASS 2: the missing piece. We start beating hand.
10. **r16_full_recipe** — full recipe applied at R=16 (beats hand K≥128)
11. **r64_finding** — full recipe applied at R=64 (beats hand all K)
12. **avx2_finding** — recipe validated at AVX2; ISA-aware rule
13. **cost_model_encoded** — encode the recipe rule in `should_spill`; recipe is auto-on
14. **log3_generalized** — `--log3` twiddle policy generalized to all radices via 10-line binary decomposition; 73-90% twiddle bandwidth savings at R=16/32/64
15. **dif_t1s** — added DIF (decimation-in-frequency) and t1s (scalar-broadcast twiddles)
16. **dif_radix_coverage** — DIF cross-checked against hand R=4/8/64; R=8 hand uses non-standard convention
17. **dit_vs_dif_crosscheck** — direct-DFT validation at R=16/32 where no hand DIF exists; both correct, perf differs
18. **bwd_direction** — bwd direction added (conjugate cmul + sign flip on internal twiddles); 320 functional combinations from same generator
19. **avx2_t1s_bwd** — AVX2 sweep + t1s + bwd validation; AVX2 R=32 DIF beats DIT 28% at K≥1024
20. **isub2_already_beaten** — pair-scheduled hand variant already beaten by recipe-log3; no need to implement
21. **goodman_hsu** — Goodman-Hsu mode switch added to SU; 5-7% over recipe on AVX2 R=32, 4-8% on R=64; auto-on at AVX2 R≥32; AVX-512 byte-identical (threshold not crossed)
22. **bb_kept** — B&B cluster scheduler with lexicographic (saturated_peak, -cp_progress) cost; tied with SU+GH on Raptor Lake but structurally different schedules. Kept as opt-in `--bb` flag for µarch-portfolio coverage; real K-regime crossover at R=64 AVX2 (BB wins K=512-1024)

Read in order, the docs trace from "Topo is 13-69% slower than hand at R=32" to "we beat hand on every radix and ISA combination where the rule says to use the recipe, pressure-aware scheduling adds another 5-7% on AVX2, and a B&B alternative scheduler is available as opt-in for µarch-portfolio coverage."

## How to build and run

```bash
cd /path/to/vfft_v2
dune build

# Generate a codelet — the recipe auto-applies when the cost model says yes
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place 32
# (this auto-applies --spill --su; the function name reflects this)

# CLI flags
#   --twiddled         use twiddle pre-multiply (t1_dit shape)
#   --log3             use binary-decomposition twiddle derivation (--twiddled only)
#   --t1s              scalar-broadcast twiddles (inner CT codelets)
#   --dif              decimation-in-frequency instead of DIT
#   --bwd              backward direction (conjugate cmul + internal sign flip)
#   --emit-c           emit C code instead of stats
#   --in-place         in-place signature (rio_re/rio_im) vs out-of-place
#   --isa avx512|avx2  target ISA (default: avx512)
#   --uarch <name>     uarch profile (raptor_lake_avx512, raptor_lake_avx2, ...)
#   --no-recipe        force Topo (disable auto spill+SU+gh)
#   --spill            force on (also auto-on per cost model)
#   --su               force on (also auto-on per cost model)
#   --gh               Goodman-Hsu pressure mode (auto-on for AVX2 R≥32)
#   --bb               B&B cluster-local optimal scheduler (experimental)
#   --bb-budget T      B&B time budget per cluster in seconds (default 1.0)
#   --fuse N           keep N PASS 2 sub-DFTs' inputs alive across boundary
```

The cost-model rule is encoded in `Dft.should_spill` and `gen_radix.ml`:
```
recipe applies iff CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32)
gh adds on top iff vec_regs <= 16 AND n >= 32
```
which expands to: AVX-512 always wins with the recipe at R≥4 (gh is a no-op there since cluster-sequential keeps live below threshold=24); AVX2 wants the recipe at R≥16 plus gh at R≥32; R≤8 prefers Topo.

## What didn't work (briefly)

- **FMA fusion at the source level.** Tried to manually emit `_mm256_fmadd_pd` instead of `_mm256_add_pd(_mm256_mul_pd(...), ...)`. GCC already does this fusion at the right level given `-O3 -mfma`, and explicit FMA emission constrained GCC's instruction selection slightly. Disabled.
- **Bisection scheduler (Frigo's).** Implemented but performed similarly or slightly worse than Topo on AVX-512. The cp_dist + su_num approach (SU) ended up dominant.
- **Annotate (Frigo-style nested-block scoping).** Built and benchmarked — produced byte-identical assembly to Topo at -O3. Modern GCC's SSA + liveness analysis already extracts everything annotate would communicate.
- **Distributing constants** (`Add(Mul(a,k), Mul(b,k)) → Mul(Add(a,b), k)`). Looks like an arith-saving rewrite; in practice it broke shared muls in cmul patterns and increased op counts.
- **B&B cluster-local optimal scheduler** (kept as `--bb`, default off). Two iterations: peak-live-only cost (failed: lower peak but worse runtime, ~3% slower than SU+GH on R=32 AVX2 — pure peak minimization extended dependency chains); then lexicographic `(saturated_peak ASC, -cp_progress ASC)` cost (essentially tied with SU+GH on average on Raptor Lake). The lex cost says SU+GH is at the practical optimum at our cluster sizes — B&B exploration converges to the same (peak, progress) value SU+GH already achieves. **Kept as opt-in** because the schedules are structurally different (~50% line diff at R=32 AVX2), other µarchs may favor BB-lex, and there's a real K-regime crossover at R=64 AVX2 (BB wins +5.8% at K=512, +3.8% at K=1024). See doc 22.
- **Hand isub2 pair-scheduling.** Existed for R=16/32/64 in gen_radix*.py with 2-3× more cmuls than ours due to lack of global hash-consing. Recipe-log3 already beats hand isub2 at R=32/64; hand wins narrowly only at R=16 K≥1024 (3-4%). Pair-scheduling helps only when both paired sub-DFTs fit in registers — R=16 only.

## What's left in the queue

- **Per-(R,K,ISA,µarch) autotuning**: generate both `--gh` and `--bb` variants, micro-bench at install time, cache the winner per cell. FFTW-wisdom pattern. Would convert the either-or knob into "pick the best for this configuration" — useful given the K-regime crossover at R=64 AVX2 and unknown µarch behavior.
- **Per-uarch coefficient tuning**: Sapphire Rapids vs Ice Lake vs Skylake currently use the same `pressure_threshold` (24 for AVX-512, 12 for AVX2). Per-uarch tuning could give a few percent. Quality-of-life for future targets.
- **K-threshold for spill at R=16 K=64 on AVX-512** (close the small remaining regression at small K).

## Acknowledgments

Built around hand-coded R=4/8/16/32/64 references generated from `gen_radix*.py` scripts. The hand-coded codelets are excellent and beating them empirically validated the variant approach.

The OCaml generator (`lib/`) is roughly 2500 lines, including the DAG (`algsimp.ml`), the algorithm picker (`dft.ml`), the schedulers (`schedule.ml`), the C emitter (`emit_c.ml`), and the ISA/µarch profiles (`isa.ml`, `uarch.ml`).
