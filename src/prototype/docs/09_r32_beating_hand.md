# R=32 Beating Hand: Cluster-Sequential PASS 2

## Summary

By processing PASS 2 sub-DFTs **sequentially** (load 4 inputs → compute one sub-DFT-4 → store 4 outputs → repeat) instead of mixing all 8 sub-DFTs through a global SU schedule, we cut stack memory ops from 230 to 81 — matching Hand. Combined with all earlier work (Spill + SU within passes + deferred reload + block-sequential PASS 1), we now **beat Hand at every K we tested** by 1-9%.

This is the first time we've crossed below Hand-coded performance on R=32.

## Bench results (R=32 AVX-512, 3 runs per K, median)

| K | Hand | Topo | Spill | **SU+Spill** | SU/Hand |
|---|------|------|-------|--------------|---------|
| 64 | 1600 | 2442 | 2569 | 1572 | **0.98** |
| 128 | 3560 | 6588 | 5361 | 3481 | **0.99** |
| 256 | 9173 | 14228 | 12203 | 9066 | **0.99** |
| 512 | 23097 | 32001 | 28803 | 22565 | **0.97** |
| 1024 | 58729 | 75853 | 69995 | 56359 | **0.93** |
| 2048 | 166167 | 199958 | 183927 | 150476 | **0.91** |
| 4096 | 338886 | 398469 | 381908 | 311878 | **0.91** |

All times in ns. SU+Spill is the cluster-sequential variant.

## The diagnosis

Earlier the gap between SU+Spill and Hand was 4-22%. The decomposition I gave was:
- 22 extra arith ops → ~8%
- FMA variant selection → ~3-5%
- Reg-reg copy count → unclear

That was wrong. Detailed assembly analysis showed:

| Category | Hand | SU+Spill (no clustering) | Δ |
|----------|------|--------------------------|---|
| Arith total | 539 | 544 | +5 |
| Reg-reg moves | 20 | 17 | -3 |
| **Stack memory ops** | **82** | **230** | **+148** |

**The arith gap was negligible (5 ops). The real gap was 148 extra stack memory ops** — GCC re-spilling registers because peak live in PASS 2 exceeded 32.

Why? Our PASS 2 emission ran SU globally over all 8 sub-DFT-4s, mixing them. With deferred reload, peak live was still ~30-40 (32 spill values still loaded throughout PASS 2 + working set). Hand processes one sub-DFT-4 at a time — peak live ≈ 16.

## The fix

Cluster PASS 2 by sub-DFT, run SU within each cluster, emit each cluster's stores immediately after its computation:

```
for k2 in 0..N2-1:
    [SU schedule of cluster k2's nodes]
    [reloads emitted just-in-time within cluster]
    [stores for cluster k2's outputs]
```

This matches Hand's structure. After this fix:

| Category | Hand | SU+Spill (clustered) | Δ |
|----------|------|----------------------|---|
| Arith total | 539 | 544 | +5 (negligible) |
| Reg-reg moves | 20 | **13** | **-7 (we win)** |
| **Stack stores** | **43** | **43** | **0 (matched)** |
| **Stack loads** | **39** | **38** | **-1 (matched)** |

Stack ops 230 → 81. 65% reduction.

## How we identify clusters

For CT(N1, N2): PASS 2 sub-DFT #k2 consumes spill slots `{n1 * N2 + k2 : n1 in 0..N1-1}`. So a PASS 2 node "belongs to" cluster k2 if its minimum-reachable spill slot mod N2 equals k2.

Implementation: walk PASS 2 nodes in topo order, propagate minimum-input-slot through predecessors (Hashtbl). Cluster = `min_input_slot mod N2`.

For non-CT codelets (Direct DFT cases), no clustering — falls back to flat SU on all PASS 2 nodes.

## What this revealed about the cost model

Earlier I said the variant decision was a 2-line rule:
```
if N >= 16 and CT-decomposed:
    use Spill + SU + Block-seq
```

Now it's a 3-line rule:
```
if N >= 16 and CT-decomposed:
    use Spill + SU + Block-seq + cluster-sequential PASS 2
```

The cluster-sequential PASS 2 isn't optional — it's the difference between matching Hand and beating Hand. It costs nothing at runtime (ordering decision only); the only complexity is in the generator.

## What we know now vs three sessions ago

**Three sessions ago:** Topo was 13-69% slower than Hand at R=32. We had no path to closing the gap.

**Two sessions ago:** Spill + block-sequential ordering closed it to 13-67% behind. Variants matter.

**One session ago:** SU within passes closed it to 4-22% behind. Scheduler is the dominant lever.

**Today:** Cluster-sequential PASS 2 closes the rest, and we're 1-9% AHEAD of Hand across K=64-4096.

## What the wins look like by lever

| Lever | Mechanism | R=32 Contribution |
|-------|-----------|-------------------|
| Block-sequential PASS 1 | Sub-FFTs as units | 4-23% |
| Explicit boundary spilling | GCC structural hint | 4-19% |
| SU within passes | ILP/cp_dist scheduling | 8-37% |
| Deferred reload | Just-in-time PASS 2 loads | 1-5% |
| **Cluster-sequential PASS 2** | **Sub-DFTs as units in PASS 2** | **~5-15% (closes the rest)** |

Total: from 13-69% behind to 1-9% ahead.

## What's left

- **R=16 with the full recipe**: should now beat Hand too. Quick validation.
- **R=64**: bigger DAG, even more register pressure to manage. Likely the next major target.
- **Cost model document**: the 3-line rule plus its empirical justification, ready to encode.
- **AVX2 (R=8 or R=16)**: vec_regs=16 changes the calculus; worth validating the recipe.

The math layer hasn't changed at all this session. All wins came from emission policy: ordering decisions, when to spill, when to load, when to store. The variant approach worked.

## Status

- ✓ R=32 beats Hand 1-9% across K=64-4096
- ✓ Stack memory ops match Hand (81 vs 82)
- ✓ Cluster-sequential PASS 2 implemented and validated
- ✓ FMA fusion experiment: GCC already does it, our explicit FMA constrains scheduling — disabled
- → R=16 with full recipe (quick win expected)
- → R=64 as next radix target
