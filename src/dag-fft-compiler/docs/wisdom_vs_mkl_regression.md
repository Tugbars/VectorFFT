# Wisdom-plan regression vs MKL (pow2 AVX-512, non-pow2 AVX-512, pow2 AVX2)

This note records a regression check of the in-place stride codelets against
Intel MKL. For every cell in the v5 stride wisdom file, we build the plan from
the wisdom's factorization and per-stage variant codes **verbatim** (no search),
execute it through the current core, and time it head-to-head with MKL on the
same machine. Three sweeps were run: pow2 sizes on AVX-512, non-pow2
(odd-radix) sizes on AVX-512, and pow2 sizes on a true AVX2 build.

The short version: the in-place codelets clear MKL on every cell that runs.
Geomean is 2.78x on pow2 AVX-512, 3.44x on non-pow2 AVX-512, and 3.20x on pow2
AVX2 (against MKL's AVX-512 path). Two things bound how strong that is. First,
this is **perf only**: output correctness was not verified, so a fast-but-wrong
plan would look identical to a good one here. Second, K=4 (and any batch below
the SIMD width) cannot run at all on the current codelets, so a third of the
pow2 grid is a crash rather than a number. The ratios are also a *floor*: the
plans are the old wisdom's choices, which are not necessarily optimal on the
current codelets.

## 1. Environment

All numbers come from the dev container, not a deployment target.

- CPU: Intel Xeon, family 6 / model 85 / stepping 7 = **Cascade Lake-SP**
  (Skylake-server line, confirmed by `avx512_vnni` present with no `bf16`/`fp16`/`amx`),
  2.8 GHz, exposed as a generic masked model name under KVM. Single vCPU visible,
  33 MB host L3, ~3.6 GB RAM available.
- This is native full-width 512-bit AVX-512 (real 512-bit FMA datapaths, not
  the double-pumped 256-bit of a Zen 4 laptop), so the AVX-512 codelets ran at
  genuine full width.
- Compiler: gcc-13 at `-O3`. Note production uses gcc-11, whose AVX-512
  register allocator differs, so absolute numbers will not match a gcc-11 build.
- MKL 2026, ILP64 + sequential, forced to a single thread (`mkl_set_num_threads(1)`)
  to stay apples-to-apples with the single-threaded core. On this CPU MKL
  auto-dispatches to its SKX AVX-512 kernels.

**Timing trust.** The container is a single virtualized vCPU with roughly plus
or minus 5 percent run-to-run noise plus occasional ~2x deschedule spikes.
Absolute ns and GFLOP/s are therefore not trustworthy. Only the **same-machine,
same-run vfft/MKL ratio** is meaningful, and even that is taken as a min-of-5 to
reject spikes. Cross-run comparisons are not valid (see section 6).

## 2. Method

- **Plans:** parsed straight from `wisdom_v198.txt`. For each cell we pass the
  wisdom's `factors[]` and per-stage `variant_codes[]` (0=FLAT, 1=LOG3, 2=T1S)
  directly to `vfft_proto_plan_create(N, K, factors, variants, nf, reg)`. No
  variant search.
- **Layout:** split-complex, stride = K (each of the K batched transforms is a
  column). MKL is configured to match: `DFTI_REAL_REAL`, `INPLACE`,
  `NUMBER_OF_TRANSFORMS=K`, `INPUT/OUTPUT_DISTANCE=1`, `INPUT/OUTPUT_STRIDES={0,K}`.
  Both engines get an identical copy of the same random input.
- **Timing:** warmup 10, then best (min) of 5 trials, each trial an adaptive rep
  count (`2e6 / N*K`, clamped to 8..100000). A 32 MB junk-buffer walk busts cache
  between the prototype and MKL within each cell. Cells are paced 250 ms apart.
- **Ratio:** `ratio = mkl_ns / vfft_ns`, so ratio > 1 means the prototype is
  faster. CSV columns: `N,K,factors,variants,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl`.
- **Not done:** FTZ/DAZ were not set in the driver. If anything that penalizes
  the prototype (denormals), so the ratios are conservative on that axis. MKL
  sets its own denormal handling internally.

## 3. What ran and what did not

The wisdom has 44 pow2 cells across K in {4, 32, 256}.

- **K=4 cells (15) do not run.** The AVX-512 codelets vectorize the batch
  dimension K in 8-double lanes; K=4 is below the SIMD width and overruns the
  N*K buffer, aborting with heap corruption. This is not a driver bug: the
  existing `demo_vs_mkl_one_cell` also aborts at K=4 but runs fine at K=8 and
  K=32. The wisdom carries K=4 timings, so the old path handled sub-width K and
  the current in-place codelets do not. The driver now skips `K % 8 != 0` with a
  printed note rather than crashing.
- **Memory cap.** Cells with `N*K > 16.77M` elements are skipped to stay under
  container memory (~805 MB per six working buffers). This only affects four
  large non-pow2 K=256/K=32 cells.

So: 29 of 44 pow2 cells benched (all K=32 and K=256). For non-pow2, 98 cells
benched and 56 skipped (the K=4 cells plus the four over the memory cap).

## 4. Results

Ratio is vfft/MKL on the same machine and run; higher is faster. "geomean
excl N<=16" drops the tiny L1-resident sizes where MKL's per-call overhead
dominates and the ratio is not representative of throughput.

### 4.1 pow2, AVX-512 (`benchmarks/pow2_vs_mkl.csv`)

| set | cells | geomean | median | min | max |
|---|---|---|---|---|---|
| all benched | 29 | 2.78x | 2.38x | 1.52x | 12.76x |
| excl N<=16 | 27 | 2.39x | ~2.4x | 1.52x | 3.59x |
| K=32 | 15 | 2.64x | | 1.66x | 6.59x |
| K=256 | 14 | 2.93x | | 1.52x | 12.76x |

Softest cells: 4096 K256 (1.52x), 32768 K32 (1.66x). Strongest non-trivial:
16384 K256 (3.59x), 32768 K256 (3.15x).

### 4.2 non-pow2 / odd-radix, AVX-512 (`benchmarks/odd_vs_mkl.csv`)

| set | cells | geomean | median | min | max |
|---|---|---|---|---|---|
| all benched | 98 | 3.44x | 3.43x | 2.00x | 7.52x |
| K=32 | 51 | 3.75x | | 2.26x | 7.52x |
| K=64 | 2 | 3.98x | | 3.50x | 4.52x |
| K=128 | 2 | 5.55x | | 4.42x | 6.98x |
| K=256 | 43 | 3.02x | | 2.00x | 5.67x |

The odd-radix margin being larger than pow2 is the expected shape: MKL's prime
and mixed-radix paths (Rader/Bluestein and less-tuned kernels) leave more on the
table than its heavily optimized pow2 path. Prime-power sizes do well, e.g.
14641 = 11^4 at K256 (3.83x), 28561 = 13^4 at K256 (4.22x), 1331 = 11^3 at K32
(3.75x). Softest cells are the large composite K=256 ones, e.g. 4000 K256 (2.00x).

### 4.3 pow2, true AVX2 (`benchmarks/pow2_avx2_vs_mkl.csv`)

Compiled with `-mavx2 -mfma` and no avx512f. objdump confirmed zero zmm /
mask-register instructions, so the runtime took the AVX2 registry and AVX2
fast-paths with no AVX-512 anywhere. MKL still runs its AVX-512 kernels, so this
is vfft-AVX2 versus MKL-AVX512.

| set | cells | geomean | median | min | max |
|---|---|---|---|---|---|
| all benched | 29 | 3.20x | 2.62x | 1.88x | 12.64x |
| K=32 | 15 | 3.01x | | | |
| K=256 | 14 | 3.42x | | | |

vfft's half-width AVX2 path still beats MKL's full-width AVX-512 path on every
benched cell.

## 5. The real AVX2 vs AVX-512 cost

Comparing the prototype's own time across the two builds (vfft_ns only, MKL
removed): AVX2 is **geomean 1.36x slower** than AVX-512, range 0.86x to 2.16x.
The sub-2x slowdown is expected because many of these cells are memory-bound,
where halving the SIMD width buys less than 2x. The few cells showing AVX2
nominally faster (the 0.86x floor) are container noise, not a real effect. This
cross-run delta carries run-to-run variance and should be read as a rough figure.

## 6. Caveat: do not compare ratios across runs

The AVX2 sweep's geomean (3.20x) is numerically higher than the AVX-512 sweep's
(2.78x). This does **not** mean AVX2 wins harder. MKL is the denominator, and its
measured time drifted between the two runs (e.g. 8 K32 measured MKL at 602 ns in
the AVX-512 run and 1091 ns in the AVX2 run, pure variance). Within a single run
the ratio is valid; across runs only the prototype's own time is comparable, and
by that measure AVX2 is slower (section 5), as it must be. Treat each CSV's
ratios as internally consistent and do not cross-compare them.

## 7. What these numbers are not

- Not absolute performance. ns and GFLOP/s are container noise; only the ratios
  carry signal.
- Not verified correct. This is a perf-only harness. The cheapest next step is
  an L2/max-error column comparing the prototype output to MKL per cell; until
  that is green, "Nx faster" means "Nx faster at producing arrays not yet
  confirmed to be correct transforms". This matters more for the odd-radix set,
  where prime Winograd kernels and deep mixed-radix chains have more ways to be
  subtly wrong than clean pow2 butterflies.
- Not the codelet optimum. Plans are the old wisdom's factor and variant choices
  run verbatim. On the current codelets these may be suboptimal, so the ratios
  are a floor. A per-cell variant re-sweep would likely lift them.
- Not a deployment-target number. Cascade Lake (2019) is two generations behind
  Sapphire Rapids and unrelated to the EPYC 9575F (Zen 5) target. On a 2-FMA
  Sapphire Rapids or Zen 5 part with faster memory, both libraries shift and the
  margin tends to compress under sustained full-width load. MKL was pinned to one
  thread; a multi-threaded MKL comparison is a separate exercise.
- K below the SIMD width is unsupported, not slow. See section 3.

## 8. Reproduction

Drivers live in `benchmarks/`; generated AVX2 codelet sources are in
`benchmarks/avx2/`. The codelets themselves are build artifacts regenerated from
the generator in `generator/`.

Registry note: the registry init references every radix in
`emit_registry_h.ml`'s `standard_radixes`, so for a pow2-only build that list is
trimmed to `[2;4;8;16;32;64]` (fewer codelets to link); it is restored to the
full 18-radix production set afterward. The shipped tree is in the full state.

AVX-512 build (used for pow2 and non-pow2 sweeps), per ISA codelets compiled and
linked, driver compiled with `-mavx512f -mavx512dq -mfma`:

```
# generate the in-place codelet matrix per radix (n1 + t1/t1s x dit/dif x log3 x fwd/bwd),
# compile to .o, then:
gcc-13 -O3 -mavx512f -mavx512dq -mfma -DMKL_ILP64 -D_GNU_SOURCE -I$MKLROOT/include \
  benchmarks/bench_pow2_vs_mkl.c <codelets>/*.o \
  -L$MKLROOT/lib -l:libmkl_intel_ilp64.so.3 -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 \
  -lpthread -lm -ldl -o bench_pow2_vs_mkl
./bench_pow2_vs_mkl wisdom_v198.txt pow2_vs_mkl.csv 250   # args: wisdom, csv, pace_ms
./bench_odd_vs_mkl  wisdom_v198.txt odd_vs_mkl.csv  250
```

AVX2 build (true AVX2, no avx512f), AVX2 codelets compiled with `-mavx2 -mfma`:

```
gcc-13 -O3 -mavx2 -mfma -DMKL_ILP64 -D_GNU_SOURCE -I$MKLROOT/include \
  benchmarks/bench_pow2_vs_mkl.c benchmarks/avx2/*.o \
  -L$MKLROOT/lib -l:libmkl_intel_ilp64.so.3 -l:libmkl_sequential.so.3 -l:libmkl_core.so.3 \
  -lpthread -lm -ldl -o bench_pow2_avx2
./bench_pow2_avx2 wisdom_v198.txt pow2_avx2_vs_mkl.csv 250
```

For a same-ISA AVX2-vs-AVX2 comparison instead of AVX2-vs-AVX512, run the AVX2
binary with `MKL_ENABLE_INSTRUCTIONS=AVX2`.

## 9. Files

- `benchmarks/bench_pow2_vs_mkl.c` - pow2 driver (filters pow2 N, skips K%8!=0).
- `benchmarks/bench_odd_vs_mkl.c` - non-pow2 driver (filters non-pow2 N).
- `benchmarks/avx2/` - the 108 generated AVX2 pow2 codelet sources.
- `benchmarks/pow2_vs_mkl.csv` - pow2 AVX-512 results (29 rows).
- `benchmarks/odd_vs_mkl.csv` - non-pow2 AVX-512 results (98 rows).
- `benchmarks/pow2_avx2_vs_mkl.csv` - pow2 AVX2 results (29 rows).

## 10. Post-rewiring spot check (section 39 verification)

After the coverage/gen_set rewiring and tree restamp, 4 pow2 cells
(1024/4096 x K32/K256, including the softest, 4096 K256) were re-run
on both ISAs with codelets compiled 100% fresh from the regenerated
tree (648 files, zero errors; avx2 binary zmm-free). Result: PASS.
vfft_ns reproduced or beat the reference on all 8 cells (range -25%
to +7%); every ratio stayed comfortably above 1. Ratio deltas vs the
tables above are MKL-side cross-run drift (MKL ran 20-46% faster than
in the reference run, consistently across both legs), which is
exactly the cross-run variance section 1 warns about.

BUILD RECIPE CORRECTION to section 8: an AVX-512 build must now link
the AVX2 codelet objects as well. plan_executors.h carries both-ISA
specializations (only the avx512 side is __AVX512F__-guarded), and
executor entries referencing avx2 codelets were added with the wisdom
reconstruction, after the original sweep. The avx2-only build is
unaffected (avx512 executors compile out).
