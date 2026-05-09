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

13. **[13_cost_model_encoded.md](13_cost_model_encoded.md)** — Cost-model rule encoded in `Dft.should_spill` and the CLI now auto-applies `--spill --su` per the rule. R=64 AVX2 validated: **18-47% faster than Topo**, the biggest single win we've measured. Generator is now self-tuning across (R, ISA).

14. **[14_log3_generalized.md](14_log3_generalized.md)** — `TP_Log3` generalized from R=4/R=8 hardcoded cases to all radices via 10 lines of binary-decomposition recursion. Recipe-log3 beats hand at R=32 (5-45%) and R=64 (6-32%) by trading arith for **73-90% reduction in twiddle bandwidth**. Composes orthogonally with the spill+SU recipe.

15. **[15_dif_t1s.md](15_dif_t1s.md)** — Three variant additions: in-place R=64 hand reference (corrects an OOP-only earlier comparison), `--t1s` for scalar-broadcast twiddles in inner codelets, `--dif` for output post-multiply (output side twiddles). All four (`t1`/`t1s` × `dit`/`dif`) generate correctly. Recipe applies through all of them. Two classifier bugs surfaced and fixed by DIF testing.

16. **[16_dif_radix_coverage.md](16_dif_radix_coverage.md)** — DIF validated across all radices with hand references (R=4, R=8, R=64). R=8 hand DIF uses a nonstandard convention that drops `W^4 = -1`; our DIF is mathematically correct (verified via direct-DFT reference) but doesn't byte-match hand R=8. R=64 DIF + log3 wins 21-28% over hand DIF. Classifier extended to a fixpoint backward pass to handle log3 cmul derivations whose consumers are exclusively Pass2.

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

17. **[17_dit_vs_dif_crosscheck.md](17_dit_vs_dif_crosscheck.md)** — R=16 and R=32 have no hand DIF reference, so DIT vs DIF compared directly via direct-DFT validation. Both correct to FP precision. R=16: DIF 3-10% faster than DIT at most K. R=32: DIF wins K=64 (10%), DIT wins K=128-1024 (4-15%), tied at 2048+. Pattern explanation: DIF's `butterfly→cmul→store` has shorter critical path at small K; R=32's 31 post-multiply cmuls saturate store buffer at mid-K.

18. **[18_bwd_direction.md](18_bwd_direction.md)** — Bwd (inverse FFT) added as orthogonal axis to DIT/DIF. Three changes: internal CT twiddle θ sign flip, external cmul becomes conjugate cmul, log3 derivations unchanged. `--bwd` CLI flag. Caller passes the same forward-convention W array; codelet conjugates internally. Recipe + log3 still wins: 9-29% faster than hand bwd at R=64. All 320 functional combinations now in the variant matrix.

19. **[19_avx2_t1s_bwd.md](19_avx2_t1s_bwd.md)** — AVX2 sweep across DIF and bwd axes; t1s + bwd combinations. All 8 R={16,32} × {DIT,DIF} × {fwd,bwd} AVX2 codelets validated. Big finding: AVX2 R=32 DIF is **28% faster than DIT** at K≥1024 (vs ~tied on AVX-512). R=64 t1s bwd recipe beats hand 8-18%. Register-pressure explanation: AVX2's 16 registers force DIT into heavy spilling because pre-multiplied inputs live across the butterfly; DIF's cmul→store pattern keeps peak live at 4 values.

20. **[20_isub2_already_beaten.md](20_isub2_already_beaten.md)** — Hand `ct_t1_dit_log3_isub2` (the pair-scheduled log3 variant from gen_radix*.py) compared against our recipe-log3. Result: **our recipe-log3 already wins at R=32 (7-15%) and R=64 (5%), ties at R=16 small K, loses narrowly at R=16 K≥1024 (3-4%)**. Hand has 2.3-2.8x more cmuls than ours because we hash-cons derived twiddles globally; hand recomputes per pair. Pair-scheduling helps only when paired sub-DFTs fit in registers — only at R=16. No need to implement isub2 in our generator.

21. **[21_goodman_hsu.md](21_goodman_hsu.md)** — Goodman-Hsu mode switch added to `su_schedule_subset`. Base SU picks by `(cp_dist DESC, su_num ASC)`; pressure-mode comparator activates when live-count exceeds a uarch-specific threshold (24 for AVX-512, 12 for AVX2) and ranks ready nodes by `delta = births - kills` (most-negative first). Empirically delivers **4-8% on AVX2 R={32,64}** on top of the recipe; byte-identical to baseline on AVX-512 (cluster-sequential keeps per-cluster live below threshold).

22. **[22_bb_kept.md](22_bb_kept.md)** — Branch-and-Bound cluster scheduler implemented as opt-in `--bb` flag with multi-objective lexicographic cost (saturated_peak ASC, -progress ASC) and a time-budget. On Raptor Lake the schedules are structurally different from SU+GH but perform roughly the same on average, with a real K-regime crossover at R=64 AVX2. Default stays SU+GH; BB is kept because (a) the schedules genuinely differ, (b) other µarchs may favor BB-lex, (c) ~270 lines is small, (d) it's a useful research probe.

23. **[23_conjugate_pair.md](23_conjugate_pair.md)** — Built `dft_direct_conjugate_pair` for odd-prime N, plus sign-aware FMA chain construction (`make_sum_with_init`) and a `share_subsums` skip for direct primes. R=11 n1 dropped **300 → 190** ops (matching hand). t1_dit dropped **336 → 230** ops. Bench: G/H ratios for R=11 went from 1.6-1.8× down to 1.0-1.4× across K. The earlier "Rader-Winograd is required" claim was wrong — hand-coded primes are just direct DFT with conjugate-pair sum/difference factoring, standard for any prime.

24. **[24_single_use_inlining.md](24_single_use_inlining.md)** — Single-use inlining in the SU emit path (values with one consumer get inlined into the consumer's expression instead of named via `const __m512d t<N>`). Closes the nested-intrinsic gap to hand: R=13 t1_dif 24 → 102 nested patterns (hand: 120). DIT primes win: R=13 t1_dit −20 movapd, R=17 t1_dit −19 movapd, bench beats hand at K≥512 (R=13 K=1024 hits **0.83 G/H**, 17% faster than hand). DIF primes still trail 5-20% (raw outputs have 2 uses, excluded from inlining; needs destructive-update emission). Same session: split output stores by Pass classification in the spill-emit path, fixing a pre-existing compile-failure bug in R=32/64 t1_dit/t1_dif composite codelets.
