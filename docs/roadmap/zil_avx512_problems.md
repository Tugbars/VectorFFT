# AVX-512 open problems

Problems found while wiring and measuring the AVX-512 interleaved (zil) library, with
the evidence for each. All numbers are from the Claude Code cloud VM (Emerald Rapids,
4-vCPU KVM guest, gcc 13) and carry its noise: MKL alone moved 10-15% between runs.
They are leads, not verdicts; confirm on owned hardware before acting.

## 1. Odd-N 1D (the flat DIT and chain3) does not gain from AVX-512 as it should

Gauntlet, 1D C2C, K = 1, natural order, out of place, one thread, cells raced cold
with `--calibrate` (runs and reports:
`zil_avx512_prototypes/harness/runtime/gauntlet_odd/`):

| N | route | AVX2 ns (this VM) | AVX-512 ns (2 runs) | AVX-512 gain | vs MKL (AVX-512, 2 runs) | i9 AVX2 record vs MKL |
|---|---|---|---|---|---|---|
| 1125 = 3^2 5^3 | chain3 | 2969 | 2042 / 2063 | 1.45x | 0.74 / 0.95 | 1.18-1.23 (flat) |
| 1215 = 3^5 5 | chain3 | 2662 | 2755 / 2768 | **0.96x** | 0.99 / 0.74 | 1.18-1.22 |
| 1575 = 3^2 5^2 7 | chain3 | 3786 | 3310 / 3259 | 1.15x | 0.62 / 0.59 | 1.12-1.15 (flat) |
| 2025 = 3^4 5^2 | chain3 | 5868 | 3452 / 3751 | 1.6x | 1.42 / 1.23 | 1.07-1.18 |
| 2187 = 3^7 | chain3 | 6407 | 4209 / 4196 | 1.5x | 1.01 / 1.30 | 1.09-1.17 |
| 2401 = 7^4 | flat | 8028 | 6824 / 7815 | **1.03-1.18x** | 0.71 / 0.63 | 0.89 |
| 2835 = 3^4 5 7 | chain3 | 6608 | 6495 / 6544 | **1.0x** | 0.98 / 1.18 | 1.10 |

(The ratio columns are MKL time / ours, the worse of the two engine orders. The i9
record, `gauntlet/results/gauntlet_2026-09-20`, is AVX2 against MKL-AVX2; here MKL runs
its AVX-512 code.)

- The gain from AVX-512 is uneven: 1.45-1.6x at 1125, 2025 and 2187, nothing at 1215
  and 2835, and little at 2401, the one cell the race gave to the flat DIT.
- The race picked chain3 at 1125 and 1575, where the i9 picked the flat DIT: at
  AVX-512 the flat DIT lost those races.
- Correctness is not the issue: the flat DIT engine sweep
  (`runtime/flatdit_sweep.c`, 72,207 checks) and the gauntlet roundtrips are clean.

### A lead: the odd-composite radices only exist in the direct form for the flat kinds

The factored odd-composite construction (`--cil-oddct`: 9 -> 3x3, 15 -> 3x5, 21,
25 -> 5x5, 27 -> 3x9) is emitted only for the pair / chain3 kinds (t2, n1t, t2t and
the backward n1). The flat DIT's kinds (t2cp, t2cs, t2csg, t2csgn, msz) and the n1c
leaf, and the 2D column kinds, exist only in the direct conjugate-pair form, which
spills heavily at the large odd radices. Stack traffic as a share of all instructions
in the compiled kernel (whole file, every tail arm):

| radix | t2cp AVX2 | t2cp AVX-512 | n1 AVX2 | n1 AVX-512 | t2 AVX2 | t2 AVX-512 |
|---|---|---|---|---|---|---|
| 9 | 22.5% | 17.5% | 9.5% | 11.3% | 8.7% | 10.9% |
| 15 | 29.6% | 19.3% | 20.8% | 6.8% | 20.2% | 9.2% |
| 25 | 56.1% | 22.9% | 51.8% | 14.4% | 53.5% | 16.6% |
| 27 | 56.6% | 26.5% | 55.7% | 22.4% | 53.0% | 19.7% |

This is the same at both ISAs, so it is not an AVX-512 generation fault: every one of
the 716 AVX-512 zil codelets was checked against its AVX2 twin and carries exactly the
same generator flags and environment (apart from `--isa` / `--uarch`). It is a missing
variant. The pair race already shows what the factored form is worth there
(il2p.h: R = 25 / 27 win it by about 2.5x).

Next: emit `--cil-oddct` for the flat kinds (if the generator admits it), race it per
stage like the pair's variant 5, and re-run these cells.

## 2. 2D 256x256 (IL C2C) gains almost nothing from AVX-512

2D interleaved C2C, natural, out of place, one thread (`runtime/il2d_vs_mkl.c`):
AVX-512 over AVX2 on the same VM is 1.37x at 64x64 and 1.19x at 1024x1024 but 1.03x at
256x256 (route chain, the plane L2-resident); against MKL-AVX-512 it is 0.76-0.78x
there. Undiagnosed: time the row and column passes separately on both builds.

## 3. ZTURN-T with more than one thread (known, scheduled after 3D)

`ztt_mt.h` still splits columns in 4-column groups and steps twiddles by the AVX2
record: wrong at AVX-512 at T > 1. Everything measured above ran ZTT single-threaded.

## 4. The tail study's batched-K rows are not reachable through the front door

`docs/design/avx512_tail_handling.md` measured the tail arms with count = K (batched
lane-major columns). For interleaved data the front door serves howmany = K as a
wrapper over K = 1 plans and refuses lane-major, so K never reaches a kernel as its
column count. The tail arms run on counts inside one transform (odd 2D plane
dimensions, odd radices, the flat DIT's blocks). The L10 decision stands; the batched-K
table describes the kernels, not a user path.
