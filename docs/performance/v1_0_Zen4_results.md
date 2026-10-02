# VectorFFT v1.0 — performance results, **Zen 4 host**

Comparator: **FFTW 3.3.10 AVX2** (built from source, `FFTW_MEASURE`, bound at
runtime). Single-threaded, K=1, interleaved, natural order, out of place.
Ratio > 1 = VectorFFT wins. Figures are best-of-both engine orders.

MKL is absent here and dispatches to SSE2 on AMD anyway, so FFTW AVX2 vs our
AVX2 is the fair local yardstick. The i9-14900KF/MKL record is
[v1_0_results.md](v1_0_results.md) — **not comparable with this file**.

```
 CPU         AMD Ryzen 5 PRO 8640HS (Zen 4 "Phoenix", non-hybrid)
 cores       6 physical / 12 logical (SMT 2)
 caches      L1d 32 KB 8-way · L2 1 MB per core · L3 16 MB shared
 toolchain   MSYS2 UCRT64 gcc 16.2.0, -march=native → znver4
 ISA         AVX2 for every result below (build.py's default clamp)
 host tag    amd-f25m117
```

---

## Routes

The `route` column names the engine the planner raced and committed for that
cell. Five appear in this file:

- **`mono`** — one whole-N kernel, no decomposition; the cheapest option while N
  still fits in registers and L1.
- **`2p`** — a Bailey pair: N splits into two factors and the data gets two
  kernel passes with a twiddle between them.
- **`chain3`** — the same idea at three stages, which is what mixed radices need
  when no clean two-factor split exists.
- **`ztt`** — ZTURN-T, a run-contiguous DIT that walks contiguous runs so each
  pass's working set stays inside L1, tiled to a raced width.
- **`fs`** — four-step: N is handed to the 2D tier as N1×N2, turning a huge 1D
  transform into two batched passes plus transposes.

The trailing digits are the committed factorization — `2p 16.64` is the pair,
`chain3 27.9.3` the three stages, `ztt 8.5.8.8` the DIT chain.

---

## 1. 1D C2C — power of two, 2 … 2²³

`gauntlet/results/zen4_pow2_fftw_2026-09-28/` · every cell re-raced on this host

```
 N          route    ours ns     FFTW ns   ratio
─────────────────────────────────────────────────
 2          mono          10           8   0.80×
 4          mono          13           9   0.69×
 8          mono          17          13   0.76×
 16         ztt           23          21   0.91×
 32         ztt           35          39   1.11×
 64         mono          53          68   1.28×
 128        2p           123         153   1.24×
 256        2p           332         265   0.80×
 512        2p           579         811   1.40×
 1024       2p          1246        1343   1.08×
 2048       ztt         2768        3668   1.33×
 4096       ztt         6326        8528   1.35×
 8192       ztt        13182       18745   1.42×
 16384      ztt        28980       39674   1.37×
 32768      ztt        70703      116318   1.65×
 65536      ztt       137033      504837   3.68×
 131072     ztt       297533      511900   1.72×
 262144     ztt       668388     1196850   1.79×
 524288     fs       2612925     6008413   2.30×
 1048576    fs       7570138    15164713   2.00×
 2097152    fs      22154850    50079650   2.26×
 4194304    fs      51288713    85846575   1.67×
 8388608    fs     135686063   168701250   1.24×
```

Roundtrip error 0 … 2.8e-15 across every cell. The sub-32 losses are call
overhead, not the transform. **65536 is an FFTW anomaly** — it reads
504,837 ns against 511,900 ns at twice the size, so its plan is bad rather than
ours being good; the summary below reports with and without it.

```
 set                        cells   median   gmean     min     max    wins
──────────────────────────────────────────────────────────────────────────
 all                           23    1.35×   1.36×   0.69×   3.68×   18/23
 N≥32                          19    1.40×   1.52×   0.80×   3.68×   18/19
 N≥32, less 65536              18    1.39×   1.45×   0.80×   2.30×   17/18
 N≥32, worse-of-flips          19    1.33×       —   0.71×       —       —
```

Which engine the planner picked, and how it did. Crossovers are measured, not
configured: `mono` at 2–8 and 64, `ztt` at 16–32, `2p` at 128–1024, `ztt` again
at 2048–262144, `fs` from 524288 up.

```
 route   cells   median     min     max   band
───────────────────────────────────────────────────────
 fs          5    2.00×   1.24×   2.30×  524288 … 2²³
 ztt         9    1.42×   1.11×   3.68×  16–32, 2048 … 262144
 mono        1    1.28×       —       —  64  (plus 2–8, below)
 2p          4    1.16×   0.80×   1.40×  128 … 1024
```

---

## 2. 1D C2C — mixed radix, 3ⁿ, 5ⁿ

`build_tuned/results/gauntlet_zen4_fftw/` · 2026-09-21 · primes excluded by
decision

```
 N      route             ours ns   FFTW ns   ratio
────────────────────────────────────────────────────
 243    2p 9.27               255       520   2.04×
 625    chain3 25.5.5        1172      1267   1.08×
 729    chain3 27.9.3        1069      2288   2.14×
 1024   2p 16.64             1108      1253   1.13×
 1200   chain3 16.15.5       1383      1385   1.00×
 1536   chain3 8.12.16       1835      1864   1.02×
 2187   chain3 27.9.9        4135      7661   1.85×
 2560   ztt 8.5.8.8          2630      3603   1.37×
 3125   chain3 25.25.5       5009      8532   1.70×
 4096   ztt 8.4.4.8.4        4540      7740   1.70×
────────────────────────────────────────────────────
 median                                       1.54×
 gmean                                        1.45×
 wins                                         10/10
```

Pure powers of 3 are the strongest family on this host, served by `chain3` with
27/9/3 kernels. Mixed-radix cells nearest a clean power of two are flattest.

**A low ratio on pow2 is headroom, not a weakness.** Powers of two are the most
heavily optimised path in every FFT library, so margins compress there for
everyone. Measured on the 14900KF with the comparator swapped on identical cells,
FFTW takes **0.98×** MKL's time on pow2 — level — but **1.64×** on odd N. A
family's ratio here therefore tracks how much room the opponent has left as much
as it tracks our own engine.

```
 family                   cells   median          range
────────────────────────────────────────────────────────
 3ⁿ  243 · 729 · 2187         3    2.04×   1.85 .. 2.14×
 pow2  1024 · 4096            2    1.42×   1.13 .. 1.70×
 5ⁿ  625 · 3125               2    1.39×   1.08 .. 1.70×
 mixed  1200 · 1536 · 2560    3    1.02×   1.00 .. 1.37×
```

---

## 3. r2c / 2D / 3D — race verdicts

Race-time ns only — no FFTW comparison was captured for these cells, so these
are **not** head-to-head numbers.

```
 transform   cell          served plan                      race ns
────────────────────────────────────────────────────────────────────
 r2c         4096          zr2c route=child_oop_il             3367
 2D c2c      512×512       chain=4.4.4.8 wl=16 tf=1          725467
 3D c2c      16×16×16      chain=4.4 chain1=16 s=2 wl=8        2300
 3D c2c      16×16×4096    chain=4.4 chain1=4.4 s=2 wl=16   1159300
```

---

## 4. Dense sweep — every N from 2 to 2048

### 4.1 Served from the 14900KF store (no host calibration)

The run store was seeded from `src/wisdom` — stamped
`@meta host=intel-f6m183 isa=avx2 l1d=49152` — and run **without**
`--calibrate`, so every covered cell replays the **Intel** verdict on this host.
297 cells the Intel campaign never banked are store misses, which race by law,
so they are reported apart. Source: `gauntlet/results/zen4_2_2048_A_intel/`.

```
 population                cells   median   gmean     min     max    wins
──────────────────────────────────────────────────────────────────────────
 replayed (Intel plans)     1750    1.55×   1.82×   0.60×   7.35×   89.9%
 raced here (store gaps)     297    1.30×   1.29×   0.69×   2.79×   83.5%
```

9.8% of the replayed cells fall below 1.0×. The two populations are not
comparable: the 297 are all N > 1024 and are the awkward factorizations the
Intel campaign never banked, so they are a harder set, not evidence about
calibration.

```
 route    cells   median        band          cells   median
─────────────────────────       ──────────────────────────────
 2p         256    3.33×        2..63            62    1.07×
 flat       240    3.25×        64..255         192    1.73×
 chain3     346    1.71×        256..511        256    1.63×
 prime      883    1.42×        512..1023       512    1.44×
 ztt          3    1.02×        1024..2048      728    1.59×
 mono        22    0.86×
```

`mono` is the only losing route and holds only the smallest cells, where the
call overhead dominates. `prime` carries half the range.

What the Intel store **selects** here — `ztt` is all but absent, so this sweep
says nothing about it either way; its band starts at 2048 (§1):

```
 band            2p  chain3   flat   mono  prime   ztt
──────────────────────────────────────────────────────
 2..255         118      18     23     22     72     1
 256..1023      113     125    109      0    421     0
 1024..2048      25     203    108      0    687     2
```

Run noise: 44 control readings at N=4096, median 6052 ns, 31 of them within 10%
of the fastest — occasional outliers rather than drift.

### 4.2 Calibrated on this host

The same 2047 cells with `--calibrate`, every one re-raced here at
`VFFT_PATIENT`. Source: `gauntlet/results/zen4_2_2048_B_recal/`.

```
 population                cells   median   gmean     min     max    wins
──────────────────────────────────────────────────────────────────────────
 all raced here             2047    1.56×   1.81×   0.60×   7.01×   89.5%
```

10.3% of cells fall below 1.0×.

```
 route    cells   median        band          cells   median
─────────────────────────       ──────────────────────────────
 2p         292    3.93×        2..63            62    1.15×
 flat       120    3.85×        64..255         192    1.78×
 chain3     435    3.35×        256..511        256    1.69×
 prime     1172    1.43×        512..1023       512    1.57×
 ztt          4    1.14×        1024..2048     1025    1.52×
 mono        24    0.90×
```

Powers of two inside this range, both phases, read from N ≥ 16. This is
the family with the least headroom (see the note in §2), and the range stops
before `ztt` and `fs` take over above 2048 where pow2 actually wins (§1):

```
 pow2 subset        cells   A median   A gmean   B median   B gmean
───────────────────────────────────────────────────────────────────
 N ≥ 2 (all)           11      1.02×     1.01×      1.08×     1.03×
 N ≥ 16                 8      1.04×     1.04×      1.09×     1.07×
 N ≥ 32                 7      1.06×     1.05×      1.09×     1.08×
```

Calibration's pow2 moves are `2p → ztt` at 32 (+15%) and 64 (+5%) and a gain at
2048, against `ztt → 2p` at 1024 which lost 5% — eight cells, so directional
only.

What this host selects for itself — against §4.1, `flat` halves (240 → 120) and
`chain3` takes most of it (346 → 435):

```
 band            2p  chain3   flat   mono  prime   ztt
──────────────────────────────────────────────────────
 2..255         140       7     14     24     66     3
 256..1023      133     169     45      0    421     0
 1024..2048      19     259     61      0    685     1
```

### 4.3 What calibration is worth

Median vs FFTW moves **1.55× → 1.56×** over all cells (1.55× → 1.64× on the
1750 cells §4.1 could replay). But the raw A→B delta is not the gain: the 297
cells that raced in **both** phases are a null control, and they move by the
same amount, so ~3% of it is a run-to-run systematic that affects our engine and
not FFTW (whose own readings match to 0.4% across the two runs).

Null-corrected, over the 1750 replayable cells:

```
 population                      cells   A/B median   corrected
───────────────────────────────────────────────────────────────
 NULL -- raced in both phases      297      1.031         —
 route UNCHANGED by calibration   1535      1.023      −0.8%
 route CHANGED by calibration      215      1.176     +14.0%
```

**Calibration changes the plan on 12.3% of cells and is worth ~14% on those;
on the rest it is worth nothing.** Where the change goes:

```
 transition        cells   median gain
──────────────────────────────────────
 flat → chain3       114       1.19×
 flat → 2p            34       1.31×
 chain3 → 2p          25       1.18×
 2p → flat            16       1.06×
 chain3 → flat         7       0.94×
 2p → chain3           7       0.98×
 prime → flat          5       1.89×
 other                 7          —
```

Moving **off `flat`** is 148 of the 215 changes — consistent with those flat
verdicts having been raced against a 48 KB L1. The `prime → flat` cells are the
largest single wins (N = 82, 86, 94 = 2×41, 2×43, 2×47): the 14900KF store sent
them to Bluestein, this host found flat DITs on the radix-41/43/47 kernels.

Caveats: 15.3% of cells read >2% slower after calibration, which at this noise
level cannot be separated into real regressions and measurement scatter; and the
null correction assumes the 297 gap cells represent the systematic, which is
reasonable but not airtight since they are all N > 1024. Per-cell join:
`gauntlet/results/ab_calibration_2026-10-02.csv`.

---

## 5. AVX-512 — pow2 ladder, 2 … 2²³

Zen 4 has AVX-512 (`avx512f`/`vl`/`bw`/`dq` all present) on double-pumped
256-bit datapaths, so any gain comes from the ISA — masking, 32 registers — not
from width. Built with `VFFT_ISA=avx512` (which lifts build.py's
`-mno-avx512f` clamp), 1299 codelets against avx2's 1588. Comparator is FFTW's
own **AVX-512** build, `fftw/avx512/bin/libfftw3-3.dll`. Full recalibration.
Source: `gauntlet/results/zen4_pow2_avx512_2026-10-02/`.

**Reported from N ≥ 128.** Everything below is excluded as not measuring the
ISA: 2–8 are call-overhead dominated, and 16/32 are blocked by a registry
gap (§5.1). N=64 is excluded with them for a clean floor, though it is in fact a
win (48 ns vs FFTW's 60, 1.25×), so the exclusion is conservative.

```
 N         route       ours ns      FFTW ns       x
────────────────────────────────────────────────────
 128       ztt             102          101   0.99×
 256       ztt             204          206   1.01×
 512       ztt             440          449   1.02×
 1024      ztt             973         1044   1.07×
 2048      ztt            2191         2597   1.19×
 4096      ztt            4867         6774   1.39×
 8192      ztt           10208        14765   1.45×
 16384     ztt           31593        44022   1.39×
 32768     ztt           49759        74077   1.49×
 65536     ztt          112753       213830   1.90×
 131072    ztt          241313       470973   1.95×
 262144    ztt          526012      1070837   2.04×
 524288    fs          2083675      5756725   2.76×
 1048576   fs          6871437     15720312   2.29×
 2097152   fs         13668137     31419387   2.30×
 4194304   fs         28703375     66331162   2.31×
 8388608   fs        120599437    160482738   1.33×
────────────────────────────────────────────────────
 17 cells    median 1.45×   gmean 1.56×   16/17 win
             range 0.99× … 2.76×
```

Level at 128–512, then the margin climbs monotonically to 2.0–2.8× from 65536
up. The single non-win is 128 at 0.99×. The 8388608 fall-off to 1.33× is the
only break in the trend — that cell is DRAM-bound at 128 MiB of working set, so
neither engine's ISA matters much there.

Our own gain from the ISA, same contract and protocol, both benched 2026-10-02
(the avx2 column is §4.2's run):

```
 N         avx2 ns   avx512 ns   speedup   avx2 route   avx512 route
─────────────────────────────────────────────────────────────────────
 128           120         102     1.18×      2p            ztt
 256           234         204     1.15×      2p            ztt
 512           521         440     1.18×      2p            ztt
 1024         1303         973     1.34×      2p            ztt
 2048         2797        2191     1.28×      ztt           ztt
─────────────────────────────────────────────────────────────────────
 5 cells    median 1.18×   gmean 1.22×
```

The avx2 reference only reaches 2048 (§4.2's range), so this is the 128–2048
overlap. **~15–34%**, and the route column is the structural part:
AVX-512 pushes `ztt` down from 2048 to 128, displacing `2p`. The wider registers
make ZTURN-T's L1-resident tiling viable at sizes where it was not.

### 5.1 N=16 and N=32 are a registry gap, not a result

Both cells regress under AVX-512 — and reproducibly, across three independent
re-calibrations with a fresh store each time:

```
 N    avx2 verdict        avx512 verdict (3/3 reps)    delta
──────────────────────────────────────────────────────────────
 16   ztt 4.4   19.3 ns   2p     24–25 ns              −25%
 32   ztt 8.4   30.4 ns   mono   52–53 ns              −42%
```

The cause is that the plans the avx2 race chose **do not exist** as AVX-512
entry points, so the race never saw them:

```
 ztt_16_4_4    avx2 present    avx512 ABSENT
 ztt_32_8_4    avx2 present    avx512 ABSENT

 ztt registry covers N =
   avx2       16 32 64 128 … 262144
   avx512           64 128 … 262144      ← starts at 64
```

It traces to four missing radix-4 AVX-512 ztt codelets (radix8 is fully
covered):

```
 radix4_z_t0tp      radix4_z_t0tp_bwd
 radix4_z_tld       radix4_z_tld_bwd
```

So these two rows are not AVX-512-vs-AVX-512 comparisons; they are AVX-512
minus two kernels. **The fix is code generation, not tuning** — emit those four
from `generator/bin/gen_radix.ml` / `emit_ztt_drivers.ml` at `--isa avx512`,
then regenerate `ztt_registry_avx512.h` via `emit_ztt_registry.ml`, rebuild and
re-race 16 and 32. It needs the OCaml/dune toolchain, which is **not installed
on this host**, and the generated headers are marked do-not-edit-by-hand, so it
is left open here.

---

## 6. Wisdom store

Stores never mix — each calibration host gets its own folder, and 14900KF
wisdom is never loaded here. `src/wisdom/Zen4/` holds this host's verdicts, all
five shards stamped `@meta host=amd-f25m117`:

```
 shard               rows   source
──────────────────────────────────────────────────────────────────
 wisdom2_oop.txt     4105   §4.2 recalibration (2026-10-02)
 wisdom2_prime.txt   1509   §4.2 recalibration (2026-10-02)
 wisdom2_scr.txt        4   2026-09-03
 wisdom2_2d.txt         1   2026-09-03 (512×512)
 wisdom2_real.txt       1   2026-09-03 (r2c 4096)
```

Merging uses `gauntlet/merge_to_host.py`, never the gauntlet's own `--merge` —
that one targets `src/wisdom/` root and would mix hosts. It lifts only
`src=race` rows carrying the run's date, backs up each shard, and takes `@meta`
from a sibling shard in the destination (taking it from the source would stamp
this host's verdicts with the seed store's host — it did, once, and is fixed).

```
python gauntlet/merge_to_host.py --name <run> --host Zen4 --dry-run
python gauntlet/merge_to_host.py --name <run> --host Zen4
```

⚠️ Every shard records `l1d=49152` even here, because `VFFT_L1D_DISCOVER`
defaults to 0 and pins `l1d_used` to 48 KB on any host. The stamp therefore
does not record this machine's 32 KB L1, and must be left as the library writes
it — correcting it by hand would make `vw2_open` report a spurious
`HOST MISMATCH` on every run.

---

## See also

- [v1_0_results.md](v1_0_results.md) — i9-14900KF / MKL record.
- [../design/cpu_discovery.md](../design/cpu_discovery.md) — discovery layer.
- `gauntlet/sibling_guard.h` — MONITORX/MWAITX sibling guard (AMD path).
