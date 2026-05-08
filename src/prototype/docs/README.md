# Findings Index

These are session writeups from building the vfft_v2 generator. Each one captures a hypothesis tested, result, and what changed in the design as a result. Read in order, they trace the path from "Topo is 13-69% slower than hand at R=32" to "we beat hand on every radix and ISA combination where the rule says use the recipe."

## Chronological order

1. **[01_su_scheduler.md](01_su_scheduler.md)** — SU list scheduler (cp_dist + su_num priority). First lever: small wins on its own (~2-5%). Foundation for what comes later.

2. **[02_r16_crossover.md](02_r16_crossover.md)** — R=16 surprised us with a small-K vs large-K crossover between Topo and Hand. First sign that variant choice depends on K.

3. **[03_r32_finding.md](03_r32_finding.md)** — R=32 enters: SU scales nicely (6-24% over Topo), but Topo is **13-69% slower than hand**. The register-pressure problem is structural at R=32.

4. **[04_r32_arith_gap.md](04_r32_arith_gap.md)** — Investigation: distributing constants made arithmetic worse. GCC already does FMA fusion. The arithmetic gap turns out to be ~5 ops, not 22.

5. **[05_r32_spill.md](05_r32_spill.md)** — Explicit boundary spilling (the **first big lever**). Closes ~half the gap. Key bug: block-sequential PASS 1 ordering matters; tag-order interleaves sub-FFTs and defeats spilling.

6. **[06_r16_spill.md](06_r16_spill.md)** — `should_spill` heuristic falsified at AVX-512. The recipe helps R=16 even though peak live (22) fits in 32 registers. (Later: this is GCC-specific.)

7. **[07_r32_fuse.md](07_r32_fuse.md)** — FUSE per-spill-target: keep first-consumed PASS 2 inputs alive across boundary. Modest gains. F8 (block-seq no-spill) emerges as a separate variant.

8. **[08_r32_su_spill.md](08_r32_su_spill.md)** — **SU within passes** (the **second big lever**). Closes most of the rest at R=32 (8-37 percentage point improvement).

9. **[09_r32_beating_hand.md](09_r32_beating_hand.md)** — **Cluster-sequential PASS 2** (the **third big lever**). The structural ceiling was 148 extra memory ops from GCC re-spilling, not arithmetic. Cluster-sequential drops to 81 stack ops (matches hand). **We start beating hand at every K.**

10. **[10_r16_full_recipe.md](10_r16_full_recipe.md)** — Full recipe applied at R=16. Beats hand at K ≥ 128 by 2-9%.

11. **[11_r64_finding.md](11_r64_finding.md)** — One-line addition to algorithm picker (`| 64 -> Cooley_Tukey (8, 8)`). Beats hand at every K (1-7%). Recipe scales to 16× DAG-size growth without tuning.

12. **[12_avx2_finding.md](12_avx2_finding.md)** — AVX2 validation. R=8 mildly regresses (peak live fits in 16 YMM, spill is overhead); R=16 wins 12-20%; R=32 wins **19-44%** — the biggest recipe wins we've measured. Cost-model rule becomes ISA-aware.

## The three levers, ranked by contribution

| Lever | Mechanism | R=32 contribution |
|-------|-----------|-------------------|
| Spill + block-sequential PASS 1 | Explicit boundary memory + sub-FFT clustering | 4-19% over Topo |
| **SU within passes** | ILP/cp_dist scheduling per pass | **8-37% over Spill** |
| **Cluster-sequential PASS 2** | One sub-DFT at a time, stores immediately | **5-15% (closes the rest)** |

Total: from 13-69% behind hand to 1-9% ahead.

## What didn't work

- **FMA fusion at source level.** GCC already fuses `mul + add` → `fmadd` with `-O3 -mfma`. Our explicit `_mm512_fmadd_pd` emission slightly constrained GCC's instruction selection.
- **Distributing constants** (`Add(Mul(a,k), Mul(b,k)) → Mul(Add(a,b), k)`). Looked like an arith-saving rewrite; in practice broke shared muls in cmul patterns and increased op counts.
- **Frigo's bisection scheduler.** Implemented but tied or slightly worse than Topo on AVX-512.
- **Annotate (Frigo-style nested-block scoping).** Built and benchmarked — produced byte-identical assembly to Topo. Modern GCC's SSA + liveness analysis already captures everything annotate would communicate.

## Final cost-model rule

```
if CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32):
    use full recipe (--spill --su)
else:
    use Topo
```

Two lines. No quantitative cost model needed.
