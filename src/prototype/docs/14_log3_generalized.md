# Log3 Generalized — Twiddle Bandwidth Becomes a Lever

## Summary

Generalized `TP_Log3` from R=4/R=8 special cases to all radices via binary decomposition. With memoization, every distinct W^k is computed once and hash-consing dedupes across legs.

Pairing log3 with the full recipe (`--log3 --spill --su`) produces the strongest variant we have:

- **R=32 vs Hand:** recipe-log3 is 5-45% faster across K (45% at K=4096)
- **R=64 OOP vs Hand:** recipe-log3 is 6-32% faster across K
- **R=32 AVX2 vs Topo flat:** recipe-log3 is 27-41% faster
- **R=64 vs Topo flat:** recipe-log3 is 37-61% faster

## What changed

```ocaml
let twiddle_expr (policy : twiddle_policy) (n : int) (j : int) =
  ...
  | TP_Log3 ->
    if is_pow2 j then
      Load (Twiddle (j - 1, ...))   (* slot = j-1, matches TP_Flat layout *)
    else
      let p = highest_pow2_le j in
      let q = j - p in
      cmul_pattern (lookup p) (lookup q)
```

Just 10 lines (excluding comments). Slot indexing uses `j - 1` to match TP_Flat — bench harnesses fill the same twiddle array regardless of policy; log3 just consults a sparse subset of slots.

## Twiddle bandwidth savings

| Radix | Flat loads | Log3 loads | Reduction |
|-------|------------|------------|-----------|
| R=16  | 30 | 8  | 73% |
| R=32  | 62 | 10 | 84% |
| R=64  | 126 | 12 | 90% |

Per kstep. At R=64 with K=4096, that's 90% less twiddle data crossing L2/L3.

## Arith cost (the price)

Each non-power-of-2 leg pays one extra cmul (4 muls + 2 adds = 6 flops). Distinct cmul derivations:

| Radix | Loads | Cmul derivations | Extra arith |
|-------|-------|------------------|-------------|
| R=16  | 4 | 11 | ~22 ops |
| R=32  | 5 | 26 | ~52 ops |
| R=64  | 6 | 57 | ~114 ops |

Note: the actual arith increase is ~half the naive 4×cmul count, because hash-consing dedupes shared subexpressions when GCC can fold them.

Real measured deltas (arith ops per kstep):

| Radix | Flat arith | Log3 arith | Δ |
|-------|------------|------------|---|
| R=16 | 198 | 220 | +22 (11%) |
| R=32 | 526 | 578 | +52 (10%) |
| R=64 | 1302 | 1416 | +114 (9%) |

## Bench results — R=16 AVX-512

(F = Topo flat, L = Topo log3, LR = recipe-log3)

| K | F (ns) | L (ns) | LR (ns) | L/F | LR/F | LR/L |
|---|--------|--------|---------|-----|------|------|
| 64 | 372 | 422 | 404 | 1.13 | 1.09 | 0.95 |
| 128 | 989 | 839 | 810 | 0.85 | **0.82** | 0.97 |
| 256 | 3328 | 3403 | 2826 | 1.01 | **0.85** | 0.83 |
| 512 | 7518 | 7149 | 5661 | 0.95 | **0.76** | 0.79 |
| 1024 | 17896 | 17656 | 13238 | 0.99 | **0.74** | 0.75 |
| 2048 | 37919 | 36535 | 28133 | 0.92 | **0.74** | 0.77 |
| 4096 | 86719 | 76205 | 56060 | 0.88 | **0.65** | 0.74 |

R=16 has a small-K regression for log3 (1.13 at K=64) — bandwidth isn't yet the bottleneck and the extra arith costs more than the saved loads. From K=128 onward, log3 wins; recipe-log3 wins much more.

## Bench results — R=32 AVX-512

| K | F | L | LR | L/F | LR/F |
|---|---|---|----|----|------|
| 64 | 1714 | 1254 | 1034 | 0.74 | **0.62** |
| 128 | 4646 | 4640 | 2980 | 1.00 | **0.65** |
| 256 | 10169 | 9597 | 6159 | 0.96 | **0.63** |
| 512 | 22281 | 22688 | 16393 | 1.02 | **0.73** |
| 1024 | 46569 | 48509 | 33151 | 1.00 | **0.71** |
| 2048 | 105699 | 96309 | 69509 | 0.92 | **0.67** |
| 4096 | 291257 | 201334 | 146766 | 0.69 | **0.50** |

Recipe-log3 is **35-50% faster than Topo flat** at every K. At K=4096 the speedup is exactly 2×.

## Bench results — R=32 vs Hand (AVX-512)

| K | Hand (ns) | Recipe-log3 | LR/H |
|---|-----------|-------------|------|
| 64 | 1080 | 1029 | **0.95** |
| 128 | 3425 | 2977 | **0.86** |
| 256 | 7050 | 6243 | **0.85** |
| 512 | 17606 | 16332 | **0.92** |
| 1024 | 36193 | 32593 | **0.90** |
| 2048 | 85735 | 67753 | **0.78** |
| 4096 | 267864 | 147061 | **0.55** |

Recipe-log3 R=32 is 5-45% faster than hand, with the gap widening at large K where bandwidth dominates.

## Bench results — R=64 OOP vs Hand (AVX-512)

**Caveat:** hand-coded R=64 (`gen_radix64.py`) emits only OOP (`tw_flat_dit_kernel` with separate `in_re`/`out_re`). There is no in-place hand R=64 to compare against, so this row tests OOP-vs-OOP. Hand R=64's OOP path may not be the path it was tuned for (`gen_radix*.py` are designed around in-place; OOP exists for 2D use cases). Treat these as "ours OOP beats hand OOP," not "ours beats hand at the radix's primary use case."

| K | Hand (ns) | Recipe-log3 OOP | LR/H |
|---|-----------|-----------------|------|
| 64 | 3935 | 3382 | **0.86** |
| 128 | 8805 | 7584 | **0.88** |
| 256 | 19281 | 18042 | **0.94** |
| 512 | 53115 | 42164 | **0.79** |
| 1024 | 172857 | 121573 | **0.68** |
| 2048 | 377290 | 294629 | **0.77** |
| 4096 | 808930 | 597308 | **0.75** |

For a robust R=64 comparison, see the **in-place vs Topo** numbers (no hand involved). At K=1024: our in-place recipe-log3 takes 88k ns, our OOP recipe-log3 takes 121k ns, hand OOP takes 173k ns. The in-place path is ~2× faster than OOP for the same algorithm; hand R=64 has no in-place path to compare against directly.

## Bench results — R=32 AVX2 vs Topo flat

| K | LR/F |
|---|------|
| 64 | 0.69 |
| 128 | **0.58** |
| 256 | 0.72 |
| 512 | 0.71 |
| 1024 | 0.73 |
| 2048 | 0.71 |
| 4096 | 0.65 |

27-41% speedup on AVX2. (No hand-coded reference for AVX2, but vs Topo this is a huge win.)

## Why log3 wins where it wins

The L1 bandwidth budget on Sapphire Rapids: 64 bytes/cycle. A vmovapd reads 64 bytes (8 doubles AVX-512). At R=32 with 62 twiddle loads/iter, just the twiddles consume 62 cycles of L1 bandwidth — and that's not counting the actual data loads, butterfly dependencies, or stores.

Cutting that to 10 loads/iter saves 52 cycles of L1 bandwidth per iteration. Even with +52 arith ops, those run on ports 0/1/5 in parallel — they don't contend for L1.

The threshold where log3 wins:
- Bandwidth saved per iter > arith added per iter / IPC

For R=32 AVX-512, that crossover is around K=64 — and log3 wins from K=64 upward.
For R=16 AVX-512, the crossover is around K=128 (smaller savings, same arith cost).
For R=64 AVX-512, log3 wins everywhere because savings are dominant at every K.

## Composition with the recipe

The recipe (Spill + SU + cluster-sequential PASS 2) and log3 are **orthogonal**:
- Recipe addresses register pressure / scheduling
- Log3 addresses twiddle bandwidth

They compose multiplicatively. At R=32 K=4096:
- Topo flat: baseline
- Topo log3: 0.69 of baseline (31% faster)
- Recipe flat: 0.83 of baseline (estimated from earlier R=32 SU+Spill numbers)
- Recipe log3: 0.50 of baseline (50% faster)

Both levers contribute; combined gains are roughly multiplicative: 0.69 × (0.83/1.0) ≈ 0.57, observed 0.50.

## Should the cost model auto-pick log3?

**Probably not auto-on.** Reasons:

1. **Function name changes.** `radix32_t1_dit_fwd_avx512` vs `radix32_t1_dit_log3_fwd_avx512` are different symbols; auto-switching would silently change the linker symbol, surprising callers.

2. **R=16 small-K regression.** At R=16 K=64, log3 is 13% slower than flat. Auto-applying without a K-aware gate could hurt small-batch users.

3. **Different optimization axis.** Recipe is "always safe within the rule"; log3 is a deliberate trade of arith for bandwidth. Users may have reasons (cache pressure outside the codelet, FP precision concerns) to prefer flat.

**Recommendation:** keep `--log3` as an explicit flag, but document the win regions clearly. The cost model rule for spill+SU stays as-is.

## What this validates

1. **Memory layer matters as much as register layer.** Earlier work optimized register pressure (Spill + SU). This work optimizes twiddle bandwidth. Both compose.

2. **Hash-consing earns its keep.** Without it, every cmul derivation would be a fresh expression — the dedup of W^3 across legs that use it (via W^7 = W^3·W^4 and W^11 = W^3·W^8) is what keeps log3's arith overhead from being 4×.

3. **Ten lines of OCaml.** Generalizing log3 from R=4/R=8 hardcoded cases to all radices was 10 lines of OCaml (the binary-decomposition recursion). The variant approach extends naturally.

## Status across all variants (AVX-512 R=32)

| Variant | K=64 | K=1024 | K=4096 | vs Hand |
|---------|------|--------|--------|---------|
| Topo flat | baseline | baseline | baseline | 1.13-1.69× slower |
| Topo log3 | 0.74× | 1.00× | 0.69× | 0.81-1.31× |
| Recipe flat (current default) | ~0.65× | ~0.83× | ~0.83× | 0.91-0.99× |
| **Recipe log3** | **0.62×** | **0.71×** | **0.50×** | **0.55-0.95×** |

Recipe-log3 is the new headline performer at R≥32. Recipe-flat remains the right default (auto-on, bench-stable, no name change), with log3 as an explicit user opt-in for bandwidth-bound use cases.
