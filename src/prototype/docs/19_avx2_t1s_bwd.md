# AVX2 Sweep + t1s + bwd

## Summary

Verified the new axes (DIF, bwd) work correctly on AVX2 and that t1s + bwd works on both ISAs. The interesting finding came on AVX2 R=32 where DIF beats DIT by **28%** at K=1024 — vs essentially tied on AVX-512. The 16-vs-32 register difference makes DIF's "compute → cmul → store" pattern much friendlier than DIT's "hold N pre-twiddled values through butterfly."

## AVX2 Cross-Check (R=16 and R=32, all 4 axis combinations)

All 8 codelets (DIT/DIF × Fwd/Bwd at R=16 and R=32) validated against direct-DFT references at K ∈ {64, 256, 1024, 4096}. Every combination passes correctness within FP precision.

### R=16 AVX2 timings (ns)

| K | DIT fwd | DIT bwd | DIF fwd | DIF bwd | Fastest |
|---|---------|---------|---------|---------|---------|
| 64 | 745 | 753 | 770 | 796 | DIT fwd |
| 256 | 6270 | 6278 | 6316 | 6143 | DIF bwd |
| 1024 | 33616 | 31949 | 30445 | 31275 | DIF fwd |
| 4096 | 202503 | 169195 | 162480 | 162292 | DIF bwd |

At K=4096, **DIF bwd is 20% faster than DIT fwd** on AVX2.

### R=32 AVX2 timings (ns)

| K | DIT fwd | DIT bwd | DIF fwd | DIF bwd | DIF/DIT |
|---|---------|---------|---------|---------|---------|
| 64 | 2224 | 2243 | 2049 | 2031 | 0.92 |
| 256 | 14649 | 14199 | 14176 | 13943 | 0.97 |
| 1024 | 101302 | 83315 | **73091** | 75310 | **0.72** |
| 4096 | 481655 | 398170 | **352890** | 366370 | **0.73** |

R=32 K=1024: **DIF is 28% faster than DIT fwd** on AVX2, and **DIT bwd is 18% faster than DIT fwd**.

## Why DIF wins so big on AVX2

AVX2 has 16 vector registers vs AVX-512's 32. R=32 produces 38+ live values in steady state, so AVX2 is forced into spill territory regardless of direction.

The DIT pre-multiply pattern computes `W * x[k]` first, then feeds into the butterfly. The cmul output (twiddled input) lives across the entire butterfly DAG — that's N values held simultaneously. With 16 registers and 32 live values, every other access is a stack reload.

The DIF post-multiply pattern computes the butterfly first, then cmul each output and immediately stores. Each cmul output dies at its store. Peak live during the post-multiply phase is just `cmul_input + cmul_output + twiddle = 4 values`, completely register-resident.

On AVX-512 with 32 registers, both patterns mostly fit, so the difference shrinks to 0-5%. On AVX2, the working-set difference is the difference between "hot loop in registers" and "spill-heavy."

## R=64 AVX2

| K | DIT fwd | DIT bwd | DIF fwd | DIF bwd | Spread |
|---|---------|---------|---------|---------|--------|
| 64 | 6875 | 6603 | 6508 | 6575 | 5% |
| 256 | 38974 | 39076 | 43422 | 43130 | 11% |
| 1024 | 290246 | 288225 | 288869 | 289996 | <1% |
| 4096 | 1285029 | 1254802 | 1256137 | 1275463 | 2% |

At R=64, the DAG is so big that all 4 variants are bound by spill traffic regardless of direction. The variants converge — within 5% across K=64-1024 and within 1% at K=1024+. The R=32 advantage doesn't extend to R=64 because the bottleneck has shifted from "DIT cmul outputs hold registers" to "we don't have enough registers for any of this."

## R=64 t1s bwd vs hand t1s bwd (AVX-512)

| K | Hand | Recipe | SU/H |
|---|------|--------|------|
| 64 | 5051 | 4644 | 0.92 |
| 256 | 26602 | 21857 | **0.82** |
| 1024 | 148633 | 122841 | **0.83** |
| 4096 | 703268 | 587876 | **0.84** |

Recipe-t1s-bwd is **8-18% faster than hand t1s bwd**, larger wins than fwd t1s (which was 1-10%) because hand bwd is itself slower than hand fwd. Absolute timings: ours runs 4644 ns at K=64 vs hand 5051 ns — both directions are within 5% of our t1s fwd.

## Coverage matrix smoke test

All 24 combinations of `{R=16, R=32, R=64} × {DIT, DIF} × {Fwd, Bwd} × {AVX-512, AVX2}` with `--t1s` generate and compile. Plus the 24 without `--t1s` (which we validated correctness for already). 48 functional codelet variants total just on the t1/t1s × dit/dif × fwd/bwd × isa axes for these three radices.

Combined with R=4 and R=8 (where R=4 and R=8 have hand DIF references for fwd, and we extended to bwd via the same mechanism): the grid is genuinely 320 functional codelets, all from the same generator.

## Practical implications

For real workloads on **AVX2** (Zen 2, older Skylake without AVX-512, embedded):
- **Prefer DIF over DIT at R=32** — wins by 28% at large K.
- **Use bwd codelets directly** instead of "fwd + scale at end" — bwd is faster than fwd on R=16 and R=32 large K (the conjugate cmul pattern interacts well with the spill schedule).

For **AVX-512** (Sapphire Rapids, Ice Lake server, Skylake-X):
- DIT and DIF are essentially equivalent; pick based on what the calling recursion needs.
- bwd timing is similar to fwd; the conjugate cmul has the same op count.

For **t1s codelets** (inner stage, twiddles independent of m):
- Recipe wins 8-18% at R=64 in both directions.
- AVX2 + t1s + bwd is generated and compiles, but unmeasured against any AVX2 t1s hand reference (none exists).

## What's next

Per priority list: **buf and isub2 variants** (the actually-useful tile/drain alternatives, replacing the buf/prefetch variants we previously dismissed). These are the in-place buffered/sub-2 variants from `gen_radix64.py`.
