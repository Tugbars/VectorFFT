# R=32 n1: monolithic vs 2-stage blocked (doc 58)

Direct head-to-head comparison of the two construction strategies for the
no-twiddle R=32 codelet on AVX-512. Both compiled with `gcc-11 -O3 -mavx512f
-mavx512dq -mfma` from the same generator; only `should_block_n1` for n=32
toggled to switch construction.

Bench platform: Sapphire Rapids AVX-512.

## Static metrics

|                          | monolithic | blocked (doc 58) | ratio       |
|--------------------------|------------|-------------------|-------------|
| Source-level FP ops      | 426        | 430               | 1.009×      |
| Asm FP ops (vfma/vmul/…) | 420        | 420               | 1.000×      |
| Asm `vmovapd`            | 842        | 261               | **0.31×**   |
| Asm total instructions   | 1263       | 689               | **0.55×**   |

At the source level, the two are essentially equivalent (the blocked variant
has 4 more sub-DFT-induced ops, but the asm normalizer collapses both to
exactly 420 FP ops). The entire static difference is in **register-traffic
instructions**: monolithic peak-live exceeds 32 ZMM, so gcc spills
aggressively. Blocked PASS 1 / PASS 2 each fit in registers (peak-live
bounded per pass).

## Runtime — best of 7 runs × 20k iterations

| K     | blocked (ns) | mono (ns)   | speedup  |
|-------|--------------|-------------|----------|
| 8     | 148.0        | 274.9       | 1.86×    |
| 16    | 273.7        | 522.3       | 1.91×    |
| 32    | 518.1        | 969.8       | 1.87×    |
| 64    | 1115.4       | 2344.7      | **2.10×**|
| 128   | 3857.6       | 6492.5      | 1.68×    |
| 256   | 8925.9       | 14532.0     | 1.63×    |
| 512   | 20919.7      | 31305.5     | 1.50×    |

Numerical correctness: max_err ≤ 1.3e-15 (≈ 3 ULPs) across all K — the two
agree to FMA-tolerance rounding.

## Reading

- **Sweet spot at K = 64** (2.10×). This corresponds to ~16 KB working set
  (32 × 64 × 16 B = 32 KB for both re and im interleaved buffers — within
  L1). Blocked structure keeps both passes register-resident; monolithic
  is spill-bound throughout.
- **Win shrinks for K ≥ 128** as working set exceeds L1 (32 KB at K=128).
  Both versions become DRAM-bound for higher K and the spill overhead
  matters relatively less. Still consistent ≥ 1.50× win even at K=512.
- **Sub-K=64 wins are also strong (1.86–1.91×)** because the codelet
  itself is the bottleneck at small K and that's exactly where vmovapd
  count dominates throughput.
- The ratio is essentially `(asm total of mono) / (asm total of blocked)
  = 1.83×` for the small-K regime — the runtime gap matches the
  instruction-count gap almost exactly, suggesting gcc's monolithic
  output is ALU-and-port-bound on the spill traffic.

## Verdict

Doc 58's n1 blocking is unambiguously the right call at R = 32. The win
holds across the entire K range and is largest in the L1-resident regime
which is the production hot path. No tradeoff identified — blocked is
strictly better here.

This is consistent with the doc 58 transcript's measurements at R=32 K=128
(+24.8% then; the comparison here is sharper because we hold all other
pipeline settings constant).

## Reproduction

```
# Default = blocked
./_build/default/bin/gen_radix.exe 32 --in-place --isa avx512 --emit-c

# Monolithic: patch lib/dft.ml::should_block_n1 Cooley_Tukey arm to `false`
# (one-line change), rebuild, re-emit.
```

Both codelets and the comparison harness are at /tmp on the host:
`/tmp/r32_blocked.c`, `/tmp/r32_mono.c`, `/tmp/bench_r32_compare.c`.
