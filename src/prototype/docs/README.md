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

25. **[25_llvm_mca_headroom.md](25_llvm_mca_headroom.md)** — Static IPC measurement via `llvm-mca` to bound how much performance is on the table from scheduling improvements. **R=64 t1_dit on SKX is at 86% of port-saturation IPC**; R=32 t1_dit at 82%. Composites are MORE port-saturated than primes (R=17 at 76.8%) because CT decomposition exposes more independent FMA chains. The scheduling rabbit hole is mostly closed on SKX (5-10% best case from beam search + port-aware tie-breakers). **R=32 on Sapphire Rapids is the outlier at 65.6%** — port 0 saturated while ports 1, 5 idle. SPR's 3-way FMA dispatch is unused, suggesting ~30% available on SPR specifically (gated on VTune confirmation). Decision: pivot to composite emission improvements (10-30% from spill-reload locality and cross-boundary inlining, NOT from scheduling); defer per-μarch tuning to SPR-specific work after hardware confirmation; drop beam search.

26. **[26_composite_inlining_negative.md](26_composite_inlining_negative.md)** — Extended single-use inlining from the SU non-spill path (where it gave R=13/17 t1_dit −20/−19 movapd) to the spill path used by composites. Result: **8-9% reduction in source-level SSA names (R=32: 653→605, R=64: 1530→1390), but ZERO change in generated asm or llvm-mca cycle counts on SKX/SPR/Zen 4 with both GCC and Clang.** Two diagnostic results made this predictable: (1) `--fuse=0..8` testing showed both GCC and Clang's RA already optimize our explicit spill traffic — fuse=0 and fuse=8 produce byte-identical asm because both compilers promote the spill array to virtual registers; (2) composite DAGs don't have prime-style SSA-name explosion. The 10-30% prediction from doc 25 was wrong on the inlining side. Cross-compiler validation: GCC and Clang within ~3% on median codelet but differ case-by-case (DIF primes 12-17% slower on Clang; R=64 t1_dif 14% faster on Clang due to GCC RA over-preserving via 1071 vmovapd reg-copies vs Clang's 83). Change kept for cleanup/consistency value (faster compile, unifies two emission paths). Re-ranked composite leverage: alternate CT decompositions, hand-vs-ours benchmarking, cluster-aware scheduler priority. Side observation: shipping Clang-compiled R=64 t1_dif gives 14% perf for free, no code changes.

27. **[27_hand_vs_ours_composite.md](27_hand_vs_ours_composite.md)** — Hand-vs-ours on composites was already established by `docs/09_r32_beating_hand.md` and `docs/11_r64_finding.md`: bare-metal Skylake-X virt with the `bench/bench_r32_spill_su.c` harness shows **SU+Spill beats hand 1-9% on R=32 and R=64 across K=64-4096**. This document records two failed attempts to re-establish that result during this session: (v1) llvm-mca SPR static analysis claimed 5-22% wins but overstated the magnitude by ~15pp because the static port-saturation model can't predict OoO real-hardware cycles for asymmetric instruction counts; (v2) a runtime bench inside this heavily virtualized container showed 17-30% LOSS, but the avg/best ratio (1.20× ours vs 1.01× hand) reveals that's container-virtualization noise amplifying our larger instruction-count footprint, not real-silicon signal. Reproducing doc 09's bench in this container shows the gap closes from 1.27× at K=64 to 0.98× at K=4096 — exactly the pattern of "container noise dominates at small K, codelet quality dominates at large K." **Bare-metal docs 09/11 stand**; the composite work IS validated. Loose ends: `--no-recipe --spill` variant has a compile regression after the doc 26 cross-pass inlining work; bench harnesses have stale `../radix32_handcoded.h` paths (now at `bench/references/`).

28. **[28_composite_regression_fma_lift.md](28_composite_regression_fma_lift.md)** — Tugbars hypothesized: "something we did for primes/odd codelets leaked into pow2 CSE and codegen." **Confirmed and FIXED.** v0 of this doc claimed share_subsums was the culprit but that was wrong — forcing share_subsums on actually makes performance much WORSE (910 → 1300 total FP ops). The real culprit is `Vfft_v2.Algsimp.fma_lift` — added unconditionally to the pipeline at some point post-doc-11, it explicitly emits `_mm512_fmadd_pd(a, b, c)` which constrains GCC's RA more than letting GCC auto-fuse mul+add. With the fix (gating fma_lift behind aggressive flag — primes only): R=32 total FP 910 → **717** (essentially matching hand's 709), vmovapd 288 → 101. llvm-mca cycles R=32 SKX 312 → **226** (vs hand 338). R=64 SKX 784 → **459** (vs hand 821). Runtime EMR container R=32 K=4096 SU/Hand recovered from 0.98 to 1.01 (≈doc 09's 0.91 range), R=64 K=4096 from 1.36 to **0.77** (significantly EXCEEDING doc 11's 0.93). Prime correctness 56/56 PASS, primes still get the small ~1-2% fma_lift benefit. Fix is one conditional in `bin/gen_radix.ml`. Separate `--no-recipe --spill` compile regression remains; unrelated to fma_lift.
