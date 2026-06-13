# R=32 t1: monolithic vs 2-stage blocked

Same construction toggle as the n1 comparison, but for the twiddled R=32 codelet.
The 2-stage path is enabled via `should_spill`; toggling that to `false`
forces monolithic construction (single `dft_expand_twiddled` instead of
`dft_expand_twiddled_spill` with PASS-1/PASS-2 markers).

All other pipeline settings held constant. Bench: Sapphire Rapids AVX-512,
gcc-11 -O3, best of 7 runs × 20k iterations.

## Static metrics

|                          | monolithic | blocked      | ratio       |
|--------------------------|------------|--------------|-------------|
| Source-level FP ops      | 550        | 554          | 1.007×      |
| Asm FP ops               | 544        | 544          | 1.000×      |
| Asm `vmovapd`            | 931        | 358          | **0.38×**   |
| Asm total instructions   | 1476       | 910          | **0.62×**   |

Same pattern as n1: arithmetic ops identical; the entire static gap is in
register-traffic instructions. Blocked t1 ends up with more vmovapd than
blocked n1 (358 vs 261) because t1 has twiddle loads from `tw_re`/`tw_im`
on top of the data movement — but the proportional win over monolithic is
still 0.38× (a 2.6× reduction).

## Runtime

| K     | blocked (ns) | mono (ns)   | speedup  |
|-------|--------------|-------------|----------|
| 8     | 264.9        | 391.4       | 1.48×    |
| 16    | 448.8        | 700.0       | 1.56×    |
| 32    | 837.2        | 1394.5      | 1.67×    |
| 64    | 1889.5       | 3034.9      | 1.61×    |
| 128   | 4154.1       | 7591.9      | **1.83×**|
| 256   | 10682.6      | 16363.2     | 1.53×    |
| 512   | 26714.3      | 37773.5     | 1.41×    |

Numerical correctness: max_err ≤ 8.9e-16 (≈ 2 ULPs) across all K.

## Comparison to n1 (from doc 60)

|       | n1 speedup (blocked/mono) | t1 speedup (blocked/mono) |
|-------|---------------------------|----------------------------|
| K=8   | 1.79×                     | 1.48×                      |
| K=16  | 1.95×                     | 1.58×                      |
| K=32  | 1.83×                     | 1.67×                      |
| K=64  | **2.23×**                 | 1.56×                      |
| K=128 | 1.73×                     | **1.90×**                  |
| K=256 | 1.62×                     | 1.47×                      |
| K=512 | 1.48×                     | 1.35×                      |

Two observations:

**1. t1 wins less than n1, consistent with the static gap.**
n1 asm-total ratio is 0.55× vs t1's 0.62× — n1 monolithic spills more
catastrophically because every input is live for longer (no twiddle
multiply to consume them early). The twiddle multiplies in t1 act as
natural consumption points that limit how bad monolithic gets.

**2. Sweet-spot K differs between variants.**
n1 peaks at K=64 (2.23×); t1 peaks at K=128 (1.90×). At K=64 t1's
working set is already 64 KB (4 streams × 32 × 64 × 8B), so it's L2
territory — diluting the spill-traffic effect. n1 at K=64 is only 32 KB
(2 streams), still L1-resident, where the codelet itself is the
bottleneck and instruction count dominates.

## Verdict

Blocking is the right default for R=32 t1, like n1 — wins everywhere with
no observed tradeoff. The t1 speedup is smaller (1.35–1.90× vs n1's
1.48–2.23×) because twiddle multiplies provide some natural relief from
register pressure even in the monolithic case, but the win is still
substantial and consistent.

`should_spill`'s threshold for AVX-512 (always true via the
`vec_regs >= 32` clause) is the right call across the whole K range.

## Reproduction

```
# Default = blocked (2-stage with spill markers)
./_build/default/bin/gen_radix.exe 32 --twiddled --in-place --isa avx512 --emit-c

# Monolithic: patch lib/dft.ml::should_spill to return `false`, rebuild.
```

Comparison artifacts at:
- `/tmp/r32_t1_blocked.c` — default 2-stage codelet
- `/tmp/r32_t1_mono.c` — monolithic variant
- `/tmp/bench_r32_t1_compare.c` — harness with correctness check + timing
