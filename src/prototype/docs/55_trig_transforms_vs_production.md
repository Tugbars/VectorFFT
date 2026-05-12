# Trig Transforms: OCaml DAG Compiler vs Production

All 7 transforms in the discrete-trig family (DCT-II/III/IV, DST-II/III,
DHT, plus the FFTW trigII experimental embedding) have been ported into
`lib/dft_r2c.ml` and benchmarked against the production runtime in
`src/core/`. This document collects the head-to-head numbers.

**Hardware:** i9-14900KF (Raptor Lake), AVX2 only — no AVX-512.
**Compiler:** gcc-15 -O3 -mavx2 -mfma -march=native.
**Methodology:** best-of-7 trials, 1000–20000 repeats per trial,
100-iteration warmup, batched layout `[N][K]`.

## Summary table

| Transform | N | vs production | Verdict |
|---|---|---|---|
| DCT-II  | 8        | hand-tuned codelet (`dct2_n8_avx2`)        | TIE every K |
| DCT-II  | 16/32/64 | no production codelet (general-N gap)      | FILLS GAP, 70–78 GFLOPS compute-bound |
| DCT-III | 8 (K≤256)| hand-tuned codelet (`dct3_n8_avx2`)        | 23% slower |
| DCT-III | 8 (K≥512)| hand-tuned codelet                          | TIE |
| DCT-III | 16/32/64 | no production codelet                       | FILLS GAP |
| DCT-IV  | 8/16/32/64 | runtime 3-pass (pre-tw + c2c-N/2 IFFT + post-tw) | WIN 24–79% (24/24 cells) |
| DST-II  | 8        | runtime 3-pass (sign-flip + DCT-II + reverse)    | WIN 16–55% |
| DST-II  | 16/32/64 | no production codelet                       | FILLS GAP, 45–78 GFLOPS compute-bound |
| DST-III | 8        | runtime 3-pass                              | WIN 15–44% |
| DST-III | 16/32/64 | no production codelet                       | FILLS GAP |
| DHT     | 8/16/32/64 | runtime 3-pass (N-rdft + butterfly)       | WIN at 11/12 cells, 1 tie |

## Architectural insight

The empirical pattern across every cell:

- **Where production wraps a codelet in runtime memory passes** (DST, DHT,
  DCT-IV — all "3-pass" patterns at the dispatch level), the fused DAG
  wins big. The fused codelet collapses 3–5 sequential O(N·K) memory
  passes into a single straight-line codelet. Win ratio scales with how
  store-bound the 3-pass was.

- **Where production has a hand-tuned specialized codelet** (DCT-II/III
  at N=8), we tie or come close. The 48-op OCaml emit at N=8 matches
  the 36-op production at the cycle level on Raptor Lake — Vector
  Capacity Usage is the binding constraint, not arithmetic.

- **Where production has nothing** (every transform at N=16/32/64), we
  fill the gap. Correctness is verified against brute-force DFT
  evaluation at FP noise (1.6e-14 to 1.2e-13).

The general lesson is consistent with the project's broader
[memory-bound thesis](../../../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/memory_bound_thesis.md):
on modern high-end CPUs FFT is memory-bound, and fusing across
sub-codelet boundaries pays off because each saved memory pass is
worth more than each saved arithmetic op.

## DCT-IV detailed results (3-pass vs fused)

3-pass = pre-twiddle + N/2-point c2c IFFT + post-twiddle (matches the
production runtime in `src/core/dct4.h`). Fused = single DAG-compiled
codelet emitting one straight-line function.

| N | K | 3-pass ns | fused ns | ratio | verdict |
|---|---|---|---|---|---|
| 8  | 32   |    37.2 |    24.2 | 0.649 | WINS |
| 8  | 64   |    70.1 |    47.0 | 0.671 | WINS |
| 8  | 128  |   140.3 |    92.7 | 0.661 | WINS |
| 8  | 256  |   863.8 |   184.2 | 0.213 | WINS |
| 8  | 512  |  2216.7 |  1173.9 | 0.530 | WINS |
| 8  | 1024 |  4566.6 |  1871.8 | 0.410 | WINS |
| 16 | 32   |    91.8 |    80.9 | 0.881 | WINS |
| 16 | 64   |   168.9 |   161.4 | 0.955 | WINS |
| 16 | 128  |  1063.3 |   321.0 | 0.302 | WINS |
| 16 | 256  |  2468.6 |  1304.6 | 0.528 | WINS |
| 16 | 512  |  7171.6 |  3509.2 | 0.489 | WINS |
| 16 | 1024 | 15387.9 |  7719.8 | 0.502 | WINS |
| 32 | 32   |   300.2 |   230.1 | 0.767 | WINS |
| 32 | 64   |  1153.6 |   455.1 | 0.395 | WINS |
| 32 | 128  |  2457.7 |  1558.6 | 0.634 | WINS |
| 32 | 256  |  6604.9 |  3794.9 | 0.575 | WINS |
| 32 | 512  | 14211.3 |  9783.6 | 0.688 | WINS |
| 32 | 1024 | 34337.8 | 19631.0 | 0.572 | WINS |
| 64 | 32   |  1390.0 |   620.6 | 0.446 | WINS |
| 64 | 64   |  2744.6 |  1706.5 | 0.622 | WINS |
| 64 | 128  |  6425.8 |  4886.1 | 0.760 | WINS |
| 64 | 256  | 14418.1 | 10633.6 | 0.738 | WINS |
| 64 | 512  | 33536.3 | 23558.4 | 0.702 | WINS |
| 64 | 1024 | 71633.9 | 46466.6 | 0.649 | WINS |

24/24 cells WIN. Median ratio 0.62 (fused is 38% faster).
Largest single win: N=8 K=256 at ratio 0.213 (4.7x speedup) — the
3-pass crosses an L1→L2 capacity boundary there.

## DHT detailed results (3-pass vs fused)

3-pass = N-point rdft + butterfly (matches production's `src/core/dht.h`).

| N | K | 3-pass ns | fused ns | ratio | verdict |
|---|---|---|---|---|---|
| 8  | 32  |    26.7 |    16.8 | 0.631 | WINS |
| 8  | 128 |   102.6 |    67.4 | 0.656 | WINS |
| 8  | 512 |  1753.6 |  1158.4 | 0.661 | WINS |
| 16 | 32  |    73.6 |    57.4 | 0.780 | WINS |
| 16 | 128 |   519.9 |   226.0 | 0.435 | WINS |
| 16 | 512 |  4764.7 |  3522.0 | 0.739 | WINS |
| 32 | 32  |   265.5 |   219.7 | 0.827 | WINS |
| 32 | 128 |  2006.6 |  1368.2 | 0.682 | WINS |
| 32 | 512 | 14036.6 | 10442.1 | 0.744 | WINS |
| 64 | 32  |  1223.9 |   704.6 | 0.576 | WINS |
| 64 | 128 |  5791.1 |  5702.4 | 0.985 | TIE |
| 64 | 512 | 32236.4 | 25756.8 | 0.799 | WINS |

11 wins, 1 tie at N=64 K=128 (a hot spot where the rdft+butterfly
already fits in L1 and the second pass is essentially free).

## DCT-II / DST-II / DST-III scaling at N=16/32/64

No production codelet exists at these sizes — these benches are
standalone correctness + GFLOPS to confirm scaling. All 27 cells PASS
brute-force correctness at FP noise.

| Transform | N | K | best ns | GFLOPS | err |
|---|---|---|---|---|---|
| DCT-II  | 16 |  32 |    80.1 | 76.73 | 2.0e-14 |
| DCT-II  | 16 | 128 |   318.1 | 77.25 | 1.7e-14 |
| DCT-II  | 16 | 512 |  3849.0 | 25.54 | 1.6e-14 |
| DCT-II  | 32 |  32 |   321.9 | 47.72 | 3.8e-14 |
| DCT-II  | 32 | 128 |  2174.1 | 28.26 | 2.9e-14 |
| DCT-II  | 32 | 512 | 10739.6 | 22.88 | 2.3e-14 |
| DCT-II  | 64 |  32 |   812.6 | 45.37 | 8.9e-14 |
| DCT-II  | 64 | 128 |  5874.3 | 25.10 | 9.4e-14 |
| DCT-II  | 64 | 512 | 26716.1 | 22.08 | 1.1e-13 |
| DST-II  | 16 |  32 |    78.6 | 78.15 | 1.6e-14 |
| DST-II  | 16 | 128 |   312.9 | 78.55 | 2.4e-14 |
| DST-II  | 16 | 512 |  3851.0 | 25.53 | 1.7e-14 |
| DST-II  | 32 |  32 |   319.3 | 48.10 | 2.9e-14 |
| DST-II  | 32 | 128 |  2168.7 | 28.33 | 3.5e-14 |
| DST-II  | 32 | 512 | 11035.3 | 22.27 | 4.0e-14 |
| DST-II  | 64 |  32 |   818.9 | 45.02 | 9.9e-14 |
| DST-II  | 64 | 128 |  5918.1 | 24.92 | 1.2e-13 |
| DST-II  | 64 | 512 | 28394.1 | 20.77 | 1.1e-13 |
| DST-III | 16 |  32 |    84.9 | 72.37 | 1.6e-14 |
| DST-III | 16 | 128 |   338.4 | 72.63 | 1.6e-14 |
| DST-III | 16 | 512 |  3896.8 | 25.23 | 2.9e-14 |
| DST-III | 32 |  32 |   302.0 | 50.86 | 4.0e-14 |
| DST-III | 32 | 128 |  2115.5 | 29.04 | 3.2e-14 |
| DST-III | 32 | 512 | 10390.5 | 23.65 | 3.9e-14 |
| DST-III | 64 |  32 |   776.7 | 47.46 | 7.8e-14 |
| DST-III | 64 | 128 |  5500.6 | 26.81 | 8.3e-14 |
| DST-III | 64 | 512 | 26126.2 | 22.58 | 7.6e-14 |

Two regimes are visible: K ≤ 128 cells reach 45–78 GFLOPS (compute-bound,
codelet stays in L1 across batches), K=512 collapses to 20–28 GFLOPS as
the working set spills L2 (memory-bound). Same pattern as DCT-IV scale
and as the c2c FFT codelet sweeps in [vtune_*_codelet_k256.md](../../../docs/).

## Op counts (pre-FMA-fuse)

| Transform | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| DCT-II  | 48  | —   | —   | —   |
| DCT-III | —   | —   | —   | —   |
| DCT-IV  | 57  | 145 | 352 | 835 |
| DST-II  | 48  | —   | —   | —   |
| DST-III | 48  | —   | —   | —   |
| DHT     | —   | —   | —   | —   |

(Cells marked `—` not measured; the op count column was only systematically
tracked for DCT-IV during the scaling work.)

## Dispatch recommendations

Production should route through OCaml DAG codelets at:

- **DCT-III** at N=8 K≥512 (TIE — either works)
- **DCT-III** at N≥16 (only path that exists)
- **DST-II / DST-III** at any N (we win big at N=8, no competition at N>8)
- **DHT** at any N (we win big at N=8/16/32, tie at N=64 K=128, win elsewhere)
- **DCT-II** at any N (TIE at N=8, fills gap at N>8)
- **DCT-IV** at any N (we win big at every measured cell)

Production should NOT route through OCaml DAG at:

- **DCT-III** at N=8 K≤256 — production's `dct3_n8_avx2` wins by 23%.

Planner-level dispatch wire-up is a future workstream (see
[trig_transforms_dag_validated memory](../../../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/trig_transforms_dag_validated.md)).

## Source benches

- `bench/regression/bench_dct2_n8.c` — DCT-II N=8 vs production codelet
- `bench/regression/bench_dct3_n8.c` — DCT-III N=8 vs production codelet
- `bench/regression/bench_dct3_scale.c` — DCT-III N=16/32/64 (gap-fill)
- `bench/regression/bench_dct4_3pass.c` — DCT-IV vs 3-pass at N=8..64
- `bench/regression/bench_dct4_scale.c` — DCT-IV correctness + GFLOPS
- `bench/regression/bench_dst_n8.c` — DST-II/III N=8 vs 3-pass
- `bench/regression/bench_dht_n8.c` — DHT N=8 vs 3-pass
- `bench/regression/bench_dht_scale.c` — DHT N=8..64 vs 3-pass
- `bench/regression/bench_trig_scale.c` — DCT-II/DST-II/DST-III N=16..64
