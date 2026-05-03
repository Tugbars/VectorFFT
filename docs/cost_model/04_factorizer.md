# 04 — Factorizer (the cost formula)

How `stride_score_factorization` turns a candidate factorization
`{R_0, R_1, ..., R_{nf-1}}` into a single unitless number.

## The big picture

```
score(factorization) = sum_over_stages( data_cost_s + tw_cost_s )
```

Lower is better. Numbers are unitless — what matters is **ranking**
candidates against each other, not the absolute value.

The recursive search `stride_factorize_scored` enumerates every ordered
factorization of `N` (up to `FACT_MAX_STAGES = 9` factors), scores each,
and returns the lowest-scoring one. For typical `N ≤ 16384` the search
visits dozens to hundreds of candidates — sub-millisecond total.

## Per-stage decomposition

For each stage `s ∈ [0, nf)`:

```
R         = factors[s]
groups    = N / R
stride    = K * Π factors[s+1 .. nf-1]   /* in doubles */
ws_bytes  = R * stride * 16              /* working set per group */
bf_cost   = _radix_butterfly_cost(R, s, K, stride, isa)
```

`bf_cost` is the per-butterfly cycle cost from `radix_cpe.h`, with
variant selection (n1 vs t1 vs t1s vs log3) handled inside that
function (see [05_variant_selection.md](05_variant_selection.md)).

`stride` measures the byte distance between butterfly legs. It's
important because the access pattern for outer stages is non-sequential —
each butterfly touches `R` cache lines that may span megabytes of
address space.

## Cache-fit penalty (`cache_factor`)

A 3-tier multiplier on `data_cost_s`:

| Working set | `cache_factor` | Interpretation |
|-------------|----------------|----------------|
| `ws_bytes ≤ L1` (~48 KB) | 1.0 | Hot in L1. Each load is ~1 cycle. |
| `ws_bytes ≤ L2` (~2 MB) | 3.0 | L1 evictions; L2 latency ~12 cycles. |
| `ws_bytes > L2` | 10.0 | DRAM/L3 misses. Expensive. |

Step-function, not gradient. A 50 KB working set and a 1.5 MB working
set both score `3.0` even though latency differs 5×. This is a known
limit; see "Limits" below.

L1/L2 sizes are detected at runtime via CPUID leaf 4 (Windows) or
sysconf (Linux) in `stride_detect_cpu()`. Defaults to 48 KB / 2 MB if
detection fails.

## Per-stage data cost

```c
data_cost = groups * K * bf_cost * cache_factor
```

Reading this:

- `groups * K` = total butterfly calls in this stage. Each butterfly
  processes `R` complex elements; total elements processed per stage =
  `groups * K * R = N * K`, regardless of `R`. So all stages do the
  same total element-work, but split it differently.
- `bf_cost` = per-butterfly cycles, captured by the CPE table. Already
  reflects per-radix bottlenecks (decoder, dependency chains, TLB).
- `cache_factor` = the only stride-aware part. For a given `(N, K)` and
  factorization, outer stages have larger strides → larger working
  sets → higher penalties.

## Twiddle penalty (`tw_cost`)

Stages 1+ also pay for twiddle-table memory traffic (stage 0 has no
twiddles):

```c
if (s > 0) {
    tw_bytes = (R - 1) * accumulated_K * 16;
    if (tw_bytes > L1)
        tw_cost = (R - 1) * accumulated_K * 4.0;   /* miss the L1 */
    else
        tw_cost = (R - 1) * accumulated_K;          /* fits L1 */
}
```

`accumulated_K = K × Π factors[0 .. s-1]` is the effective batch count
at stage `s`. The codelet reads `(R - 1)` complex twiddles per
butterfly indexed by `accumulated_K`, so the table is `(R - 1) ×
accumulated_K × 16 bytes`. If that exceeds L1, every twiddle access
likely misses → 4× penalty per twiddle load.

This penalty captures the "twiddle-table cache pressure" effect that
makes log3 win at large radixes — log3 derives twiddles from a small
base set and never builds the full table. The cost model picks up that
benefit when it consults `cyc_log3` (see
[05_variant_selection.md](05_variant_selection.md)), but the penalty
itself is variant-agnostic.

## Worked example: N=64, K=1024

For factorization `[4, 4, 4]` on a 48 KB / 2 MB CPU:

| Stage | R | groups | stride | ws | cache | bf_cost | data_cost |
|-------|---|--------|--------|------|-------|---------|-----------|
| 0 | 4 | 16 | 16384 | 1 MB | 3.0 | cyc_n1[4] = 1.17 | 16·1024·1.17·3 = 57,532 |
| 1 | 4 | 16 | 4096 | 256 KB | 3.0 | cyc_t1[4] = 1.53 | 16·1024·1.53·3 = 75,202 |
| 2 | 4 | 16 | 1024 | 64 KB | 3.0 | cyc_t1[4] = 1.53 | 16·1024·1.53·3 = 75,202 |

Twiddle penalties:

| Stage | accumulated_K | tw_bytes | fits L1? | tw_cost |
|-------|---------------|----------|----------|---------|
| 1 | 1024·4 = 4096 | 3·4096·16 = 192 KB | no | 3·4096·4 = 49,152 |
| 2 | 4096·4 = 16384 | 3·16384·16 = 768 KB | no | 3·16384·4 = 196,608 |

```
total_score = 57,532 + 75,202 + 75,202 + 49,152 + 196,608
            = 453,696
```

Compared against `[8, 8]`:

| Stage | R | groups | stride | ws | cache | bf_cost | data_cost |
|-------|---|--------|--------|------|-------|---------|-----------|
| 0 | 8 | 8 | 8192 | 1 MB | 3.0 | cyc_n1[8] = 27.14 | 8·1024·27.14·3 = 666,973 |
| 1 | 8 | 8 | 1024 | 128 KB | 3.0 | cyc_t1[8] = 23.68 | 8·1024·23.68·3 = 581,959 |

Plus tw_cost stage 1 = `7·8192·4 = 229,376`.

```
total_score = 666,973 + 581,959 + 229,376 = 1,478,308
```

So `[4,4,4]` (453K) beats `[8,8]` (1.48M) by ~3.3× in the model. Real
bench shows `[4,4,4]` ≈ 40 µs vs `[8,8]` larger — agreement within the
expected band.

## What the cost model captures well

| Phenomenon | How the model sees it |
|------------|----------------------|
| Per-radix codelet efficiency | `cyc_n1`/`cyc_t1`/`cyc_t1s`/`cyc_log3` from CPE table |
| Per-stage memory pass cost | Implicit in `bf_cost` (load/store ops are part of the cycle count) |
| Cache-pressure of large strides | `cache_factor` step function |
| Twiddle-table cache pressure | `tw_cost` step function |
| Variant choice (t1 vs t1s vs log3) | Mirrors plan-build via wisdom_bridge predicates |
| Many vs few stages | Sum over stages; fewer stages means fewer terms |
| Bigger codelet vs deeper plan | Captured implicitly via per-radix CPE numbers |

## Limits

| Effect | Status |
|--------|--------|
| L1/L2 step function (no gradient) | Coarse but workable; no L3 tier |
| TLB pressure independent of cache | Not modeled separately; partly captured in CPE numbers |
| Stride effects on prefetcher | Not modeled |
| Calibration-K assumption (K=256) | Numbers may understate large-K stride penalties |
| DIF orientation | Not modeled (estimate is DIT-only) |
| K-blocked executor | Not modeled (estimate uses standard executor) |

The bench data (see [06_validation.md](06_validation.md)) shows the
remaining gap between estimate and wisdom is mostly variance from these
limits, not from cost-model bugs. They're known v1.x improvements.

## Source

Two functions in `src/core/factorizer.h`:

- [`_radix_butterfly_cost`](../../src/core/factorizer.h) — picks a CPE
  number for `(R, stage_idx, me, ios, isa)`. Mirrors plan-build's
  variant selection (see [05_variant_selection.md](05_variant_selection.md)).
- [`stride_score_factorization`](../../src/core/factorizer.h) — the
  per-stage loop above. Returns the unitless score.

The recursive search:

- [`stride_factorize_scored`](../../src/core/factorizer.h) — entry
  point, returns the lowest-scoring factorization in the supplied
  output struct. Falls back to `stride_auto_plan` (which handles
  Bluestein/Rader for prime-heavy `N`) if `N` can't decompose into
  available radixes.

## See also

- [03_dynamic_cpe.md](03_dynamic_cpe.md) — where `cyc_*` numbers come from
- [05_variant_selection.md](05_variant_selection.md) — how the variant choice happens inside `_radix_butterfly_cost`
- [06_validation.md](06_validation.md) — what the resulting picks look like across the bench
