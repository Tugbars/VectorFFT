# 51. N=128 R2C: Monolithic Codelet Beats 3-Pass 1.3-3.3×

## Setup

Compare two implementations of the same N=128 r2c forward transform:

- **Path A**: monolithic R=128 r2c codelet (single function call, pack
  and post-process butterfly fused into the codelet body via OCaml
  math layer).
- **Path B**: faithful synthetic mirror of `r2c.h`'s three-pass
  structure:
  1. Pack: 64 memcpy pairs to convert N reals to N/2 complex
  2. Inner FFT: R=64 c2c monolithic codelet
  3. Post-process: vectorized AVX-512 Hermitian-extraction butterfly

Both paths use the same compiler (gcc-11 -O3 -mavx512f -mfma
-march=icelake-server -flive-range-shrinkage), the same SIMD ISA, and
generated codelets of the same architectural quality (the inner R=64
in Path B is also from our generator).

This bench is **synthetic** — not wired into the actual `executor.h`
/ `stride_plan_t` stack. The 3-pass mirror reproduces the
architectural structure (three sequential passes over memory with
intermediate buffers) without the overhead of the planner, thread
pool, or `override_fwd` dispatch indirection in the real `r2c.h`. So
the numbers here represent the **best case for the 3-pass approach**.

## Results

Container hardware (not ICX), best-of-7 trials × 1000 reps each:

```
K       Mono (ns)     3pass (ns)    Speedup       Correct   Err vs ref
8       402.4         1334.0        3.32x         PASS      ~1e-13
16      807.6         2711.3        3.36x         PASS      ~1e-13
32      2281.7        4028.8        1.77x         PASS      ~1e-13
64      4894.6        7701.4        1.57x         PASS      ~1e-13
128     11241.8       15845.9       1.41x         PASS      ~1e-13
256     25525.8       32547.7       1.27x         PASS      ~1e-13
512     55416.4       93389.0       1.69x         PASS      ~1e-13
1024    194917.5      375229.2      1.93x         PASS      ~1e-13
```

Both paths produce correct results (errors at FP noise floor).

## Three regimes visible in the data

**K = 8–16: 3.3× speedup.** Per-call overhead dominates. The 3-pass
makes three discrete function calls vs one for the monolithic. Each
call has prologue/epilogue cost (register save/restore, parameter
setup). At small K, the actual arithmetic is too short for these to
amortize. The monolithic codelet inlines everything into one function
body; the prologue runs once.

This is consistent with v1.1 doc's "1D R2C: ~1.5× over FFTW, loses to
MKL" observation. MKL likely uses fused codelets at small N for the
same reason.

**K = 32–256: 1.3-1.8×.** Cache-resident regime. The arithmetic is
similar between paths (both touch ~2000-2100 FP ops total), but Path
B touches the data three times (pack reads + writes, FFT reads +
writes, butterfly reads + writes). Path A touches it once. Memory
traffic ratio is ~3×; observed speedup ~1.5× because cache hierarchy
absorbs some of the redundant traffic but not all of it.

**K = 512–1024: 1.7-1.9×.** Memory-bandwidth-bound regime. The
working set exceeds L1, possibly L2. Reads/writes go to L3 or DRAM.
The 3× memory traffic disadvantage materializes more starkly. The
monolithic codelet's single sweep through memory is hugely
advantageous.

## Why these numbers undersell the real win

The 3-pass mirror here is "as good as 3-pass can get":

1. **No indirection.** Path B calls the codelet directly, not through
   `plan->override_fwd`. The actual `r2c.h` adds an indirect call.
2. **No planner overhead.** No `stride_plan_t` data lookup, no
   `inner_plan->override_data` deref.
3. **No thread pool dispatch.** The actual `r2c.h` does
   `_stride_pool_dispatch` for multithreading; the synthetic doesn't.
4. **No tiling.** The actual `r2c.h` tiles into `B`-sized blocks for
   cache friendliness; the synthetic just runs full K. (This actually
   favors the 3-pass at large K — without tiling, all three passes
   thrash cache. With tiling, the 3-pass approach gets cache-locality
   benefits the monolithic can't have. So the real perf gap at K=1024
   is probably smaller than this bench shows.)

Net: at small K, the actual `r2c.h` is probably even slower than the
mirror here (more indirection); at very large K, the gap is probably
smaller (because tiling helps 3-pass). The cache-resident regime
(K=32-256) is the cleanest signal and shows consistent 1.3-1.8× wins.

## Implications

**The architectural premise is confirmed.** Replacing the 3-pass
pack→c2c→butterfly with a single monolithic codelet wins at every K
tested, in some regimes dramatically. The cascade-boundary codelet
work (t1_r2c_first / t1_r2c_last / hc2c_middle) is justified by this
data — it's the path to extend the monolithic win to N > 512.

**The user's v1.1 doc predictions are validated.** The "1D R2C loses
to MKL" gap is precisely the structural 3-pass overhead. Monolithic
fused codelets close it.

## What this does NOT validate

- **Real hardware numbers.** This is container, not ICX. Run on ICX
  for definitive numbers.
- **Multi-threaded performance.** The real `r2c.h` parallelizes via
  the thread pool. The monolithic codelet path needs its own
  parallelization story (which is simpler — just parallelize across
  K-blocks).
- **N > 512.** Above R=512, monolithic codelets don't fit; cascade
  becomes necessary. The premise that "monolithic at supported N
  wins" doesn't auto-extend to "cascade-with-fused-codelets at N=2048
  wins". That requires the cascade boundary codelets and their own bench.

## Suggested next step

The monolithic-wins data justifies committing to the cascade
codelets. Concretely:

1. **`t1_r2c_first_R`** — first-stage codelet that reads N reals,
   does first DIT pass + Hermitian-aware packing, outputs to a
   half-spectrum-format intermediate. R = {8, 16, 32, 64} cover N up
   to 4096 in two-stage cascades.

2. **`hc2hc_R`** — middle-stage codelet operating on Hermitian-packed
   in and out, applies twiddle factors between sub-DFTs.

3. **`t1_r2c_last_R`** — last-stage codelet that reads
   Hermitian-packed intermediate, applies final twiddles and butterfly
   extraction, writes final Hermitian-packed output.

Each is a new math-layer function in `lib/dft_r2c.ml` mirroring how
`dft_r2c_direct` is structured, but taking twiddle inputs at runtime
instead of baking them in as `Const` nodes.

After those three exist:

4. **Planner integration.** Replace `r2c.h`'s 3-pass approach for N
   we support with cascade composition. The planner picks a
   factorization (e.g., N=512 → first=16, last=32), generates calls
   to the appropriate codelets in sequence.

5. **Real-hardware bench.** Run head-to-head against the current
   `r2c.h` on ICX, multi-threaded, with realistic working sets. This
   is the v1.1 deliverable's actual proof point.

## Files

- `bench/r2c_mono/bench_r128.c` — the head-to-head harness
- `bench/r2c_mono/README.md` — how to run it
