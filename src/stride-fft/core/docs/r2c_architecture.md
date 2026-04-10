# R2C/C2R Architecture

## Algorithm: Pair-Packing

N-point real FFT → N/2-point complex FFT + post-process butterfly.

**Forward (R2C):**
1. Pack pairs: `z[n] = x[2n] + i·x[2n+1]`
2. N/2-point complex FFT on z
3. Post-process: extract X[k] from Z[k] via Hermitian butterfly

**Backward (C2R):**
1. Pre-process: reconstruct Z from X (reverse butterfly)
2. N/2-point complex IFFT on Z
3. Unpack: `x[2n] = 2·Re(z[n])`, `x[2n+1] = 2·Im(z[n])`

## Fused Pack (Novel)

The pack step `re[2n·K] → scratch[n·B]` was 20-29% of total R2C time. We eliminate it entirely.

**Key insight:** DIT stage 0 is always twiddle-free (`cf0 = 1`, `needs_tw = 0` for all groups). So all first-stage groups can use `n1_fwd(is=2K, os=B)` — the existing n1 codelet with different input/output strides. The butterfly reads directly from the R2C input at stride 2K and writes to scratch at stride B. One pass instead of two.

```
Before:  input[2n·K] ──copy──> scratch[n·B] ──n1──> scratch  ──t1...──> scratch
After:   input[2n·K] ──────── n1(is=2K,os=B) ──────> scratch  ──t1...──> scratch
```

Remaining stages (1+) run on dense scratch via `_stride_execute_fwd_slice_from(plan, sr, si, B, B, start_stage=1)`.

**Benchmark (N=1000, K=256):** R2C is 2.0x over complex FFT — at the theoretical ceiling.

## Permuted-Index Post-Process

DIT forward produces digit-reversed output. The post-process reads Z at permuted indices instead of doing a separate O(N·K) permutation pass.

Iterates sequentially through scratch (`iperm[p]` maps scratch position → natural bin). Processes (f, mirror) pairs when `f ≤ mirror`, skipping already-processed entries. Mirror twiddle broadcasts hoisted outside k-loop.

**Tradeoff:** sequential primary reads (prefetcher-friendly) + one scattered mirror read per pair. Each Z element loaded once (paired processing).

## Pre-Process (Backward)

Paired processing: iterates `f=1..halfN/2`, writes both `Z[perm[f]]` and `Z[perm[mirror]]` from the same E/D values. Each X element loaded exactly once. Mirror twiddle hoisted.

## Data Layout

Split-complex, batched:
- Real input: `real[n·K + k]` for n=0..N-1, k=0..K-1
- Complex output: `re[f·K + k]`, `im[f·K + k]` for f=0..N/2, k=0..K-1

N must be even. Block-walked with block size B for cache efficiency.

## t1_oop Codelets

Out-of-place twiddle codelets generated for all 18 radixes (2-64). Signature:
```c
void radixR_t1_oop_dit_fwd(
    const double *in_re, const double *in_im,   // separate input
    double *out_re, double *out_im,              // separate output
    const double *W_re, const double *W_im,      // twiddle table
    size_t is, size_t os, size_t me);            // input stride, output stride, batch count
```

Reads `in_re[m + j·is]`, applies twiddle, butterflies, writes `out_re[m + j·os]`. Same twiddle layout as t1: `W[(j-1)·me + m]`.

Not needed for R2C (stage 0 is twiddle-free), but built as infrastructure for 2D FFT strided executor where middle stages need strided I/O.

No other FFT library has this codelet type. FFTW separates copy from compute. VkFFT fuses via GPU shared memory. We fuse on CPU via parameterized strides in the codelet itself.

## Files

- `core/r2c.h` — R2C/C2R plan, execute, post/pre-process, fused first stage
- `core/executor.h` — `_stride_execute_fwd_slice_from()`, `stride_t1_oop_fn` typedef
- `core/registry.h` — `_REG_T1_OOP(R)`, t1_oop slots in registry struct
- `core/planner.h` — t1_oop populated from registry into plan stages
- `generators/gen_radix*.py` — `ct_t1_oop_dit` variant, `t1_oop` addr_mode
- `codelets/{avx2,avx512,scalar}/*_ct_t1_oop_dit.h` — generated t1_oop codelets (all ISAs)

## C2R Fused Unpack

Symmetric to the forward fused pack. DIF backward runs stage `num_stages-1 .. 1` on scratch, then stage 0 (last, twiddle-free) writes directly to output at stride 2K with ×2 normalization baked in.

Uses `n1_scaled_bwd(is=scratch_stride, os=output_stride, scale=2.0)` — a dedicated codelet that applies `vmulpd(vscale, result)` before each store. Benchmarked at 1.4-1.85x over the separate unpack approach. No impact on the regular C2C hot path.

```
Before:  IFFT stages → scratch  ──×2 copy──>  output[2n·K]
After:   IFFT stages 1+ → scratch  ──n1_scaled_bwd(os=2K, scale=2.0)──>  output[2n·K]
```

## TODO

- Wire t1_oop into 2D FFT strided executor for non-contiguous axis
