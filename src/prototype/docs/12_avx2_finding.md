# AVX2 Validation — Recipe Wins on R≥16, Regresses on R=8

## Summary

Validated the full recipe (Spill + SU + cluster-sequential PASS 2) on AVX2 at R=8, R=16, R=32. Result: **massive wins at R≥16, mild regression at R=8**. The regression at R=8 is consistent with the original `should_spill` heuristic — AVX2 has only 16 YMM registers, and R=8 peak live (~10) fits comfortably without spilling.

This narrows the cost-model rule: AVX-512 wants the recipe everywhere CT applies; AVX2 wants it only when `n + 6 > vec_regs`.

## Bench results (AVX2, 3 runs per K, median SU/T ratio)

(SU/T < 1 means recipe is faster than Topo)

| Radix | K=64 | K=128 | K=256 | K=512 | K=1024 | K=2048 | K=4096 |
|-------|------|-------|-------|-------|--------|--------|--------|
| R=8 | 1.06 | 1.07 | 0.99 | 0.99 | 0.96 | 1.01 | 1.00 |
| R=16 | **0.86** | **0.86** | **0.88** | **0.80** | **0.83** | **0.83** | **0.84** |
| R=32 | **0.69** | **0.56** | **0.67** | **0.75** | **0.77** | **0.80** | **0.81** |

**R=16 AVX2:** Recipe is 12-20% faster. Win across all K.
**R=32 AVX2:** Recipe is 19-44% faster. The biggest wins we've measured anywhere.
**R=8 AVX2:** Recipe is 3-7% slower at small K, tied at large K. Mild regression.

## Why R=32 AVX2 wins MORE than AVX-512

| | AVX-512 R=32 | AVX2 R=32 |
|---|--------------|-----------|
| vec_regs | 32 | 16 |
| Peak live (full no-spill) | ~38 | ~38 |
| Excess over budget | +6 | **+22** |
| SU/T median (K=128) | 0.61 | **0.56** |

On AVX2, peak live exceeds vec_regs by 22, so Topo's GCC produces extreme re-spilling. The recipe avoids this entirely by spilling only at the natural CT boundary. Larger excess → bigger recipe win.

## Why R=8 AVX2 regresses

For AVX2 R=8: peak live ≈ 10 < vec_regs = 16. **Registers fit; spill is gratuitous overhead.**

Specifically:
- 8 spill stores + 8 spill loads = 16 extra memory ops per iteration
- These don't fix any register pressure problem
- Just pure waste

The same situation at AVX-512 (R=8 peak live ~10 < 32) DIDN'T regress — earlier benches showed SU+Spill was 2-10% faster than Topo on AVX-512 R=8. Why the difference?

Hypothesis: AVX-512 has wider vectors (8 doubles vs 4), so each spill store/load is more expensive in absolute time but cheaper relative to the work done. AVX2 has narrower vectors and proportionally more loop overhead, so per-iteration spill cost matters more.

This is consistent with the bench data: R=8 AVX-512 win was small (~2-10%); the same win on AVX2 becomes a small loss (~3-7%) because the spill overhead crosses zero.

## Updated cost-model rule

The rule is now ISA-aware:

```
if CT-decomposed:
    if vec_regs >= 32 (AVX-512):
        use full recipe   # always wins or ties at R ≥ 4
    elif vec_regs == 16 (AVX2):
        if n + 6 > vec_regs:  # R ≥ 16
            use full recipe
        else:                  # R ≤ 8
            use Topo
else:
    use Topo
```

The `n + 6 > vec_regs` heuristic from earlier was wrong for AVX-512 (too conservative) but correct for AVX2.

A cleaner unified form:

```
use_recipe = CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32)
```

The `vec_regs >= 32` clause captures AVX-512's tolerance for the recipe even when it's not strictly needed. AVX2 only wants the recipe when it's strictly needed.

## What this reveals about the levers

**Cluster-sequential PASS 2 is the dominant lever at AVX2.** With only 16 YMM registers, the peak-live management we get from cluster-sequential PASS 2 is essential for any radix where peak live exceeds 16. Without it, GCC re-spills aggressively and Topo is dramatically slower.

**SU within passes still helps regardless of ISA.** The scheduling improvement is independent of register-pressure savings.

**Spill alone (without cluster-sequential PASS 2) would likely regress more on AVX2.** We didn't measure this directly, but at AVX2 R=32 the gap between Topo and full-recipe (44%) is much larger than at AVX-512 (37%). The AVX2 case has more headroom for memory savings.

## Status across all (radix, ISA) combinations

| | AVX-512 (vec_regs=32) | AVX2 (vec_regs=16) |
|---|-----------------------|---------------------|
| R=4 | tied/noise | (not tested) |
| R=8 | recipe 2-10% better | **recipe 3-7% worse at small K** |
| R=16 | recipe beats Hand K≥128 | recipe 12-20% better than Topo |
| R=32 | recipe beats Hand all K | recipe 19-44% better than Topo |
| R=64 | recipe beats Hand all K | (not tested; R=64 AVX2 may need more spill slots) |

## What's left

- ✓ AVX2 validated at R=8/16/32
- ✓ Cost model rule updated with ISA awareness
- → R=4 AVX2 (likely fine, but worth checking)
- → R=64 AVX2 (CT(8,8) at AVX2 — 64 PASS 1 outputs, 16 YMM regs, should be very spill-favorable)
- → Encode the rule in dft.ml's `should_spill`?
- → Cost model document with empirical justification across (R, ISA, K)

## What we learned

1. The recipe is ISA-aware, but not in the way I expected. Earlier I thought cluster-sequential PASS 2 was the recipe's main virtue. AVX2 confirms that — at 16 YMM, cluster-sequential is essential. AVX-512 just has enough headroom that it's a smaller relative win.

2. The original `should_spill` heuristic was right for AVX2 and wrong for AVX-512. The asymmetry comes from GCC's different scheduling behavior with 32 vs 16 architectural registers — at 32, GCC has slack to scratch around our DAG; at 16, it has to spill aggressively without our help.

3. R=8 is now the lower-radix boundary where the recipe stops being universally beneficial. R=4 is below that; R=8 is borderline (AVX-512 wins, AVX2 loses); R≥16 always wins.
