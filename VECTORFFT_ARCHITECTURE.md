# VectorFFT Architecture Guide

## Overview

VectorFFT is a high-performance split-radix FFT library for double-precision complex data (split real/imaginary arrays). It uses a Cooley-Tukey multi-radix factorization with hand-optimized SIMD codelets (AVX-512, AVX2, scalar) and a zero-table twiddle walker for large transforms.

**Key design choices:**
- Split-complex layout: separate `double *re` and `double *im` arrays (not interleaved)
- Forward = DIT (Decimation-In-Time): input permutation → stages inner→outer → natural output
- Backward = DIF (Decimation-In-Frequency): natural input → stages outer→inner → output permutation
- DIT fwd + DIF bwd = zero-permutation roundtrip (forward output feeds backward directly)

---

## Project Tree

```
highSpeedFFT/src/
├── radix2/
│   ├── scalar/
│   │   ├── fft_radix2_scalar.h              # notw scalar
│   │   └── fft_radix2_scalar_dif_tw.h       # DIF tw scalar
│   ├── avx2/
│   │   ├── fft_radix2_avx2.h                # notw AVX2
│   │   └── fft_radix2_avx2_dif_tw.h         # DIF tw AVX2
│   ├── avx512/
│   │   ├── fft_radix2_avx512.h              # notw AVX-512
│   │   └── fft_radix2_avx512_dif_tw.h       # DIF tw AVX-512
│   ├── fft_radix2_dispatch.h                # DIT cross-ISA dispatch
│   └── fft_radix2_dif_dispatch.h            # DIF cross-ISA dispatch
│
├── radix3/                                  # same structure as radix2
├── radix4/
├── radix5/
├── radix7/
├── radix8/
│
├── radix16/
│   ├── scalar/
│   │   ├── fft_radix16_scalar_n1_gen.h      # notw scalar (generated)
│   │   └── fft_radix16_scalar_tw.h          # DIT + DIF tw scalar (4 functions)
│   ├── avx2/
│   │   ├── fft_radix16_avx2_n1_gen.h        # notw AVX2 (generated)
│   │   ├── fft_radix16_avx2_tw.h            # DIT tw AVX2 (generated)
│   │   ├── fft_radix16_avx2_dif_tw.h        # DIF tw AVX2 (generated)
│   │   └── fft_radix16_avx2_tw_pack_walk.h  # pack/unpack + walk + packed drivers
│   ├── avx512/
│   │   ├── fft_radix16_avx512_n1_gen.h      # notw AVX-512 (generated, 4×4)
│   │   ├── fft_radix16_avx512_tw.h          # DIT tw AVX-512 (generated, 4×4)
│   │   ├── fft_radix16_avx512_dif_tw.h      # DIF tw AVX-512 (generated)
│   │   └── fft_radix16_avx512_tw_pack_walk.h # pack/unpack + walk + packed drivers
│   ├── fft_radix16_dispatch.h               # DIT dispatch (notw + strided + packed + walk)
│   ├── fft_radix16_dif_dispatch.h           # DIF dispatch
│   └── fft_radix16_tw_packed.h              # packed layout helpers
│
├── radix32/                                 # same structure as radix16
│   ├── scalar/
│   ├── avx2/
│   ├── avx512/
│   ├── fft_radix32_dispatch.h
│   ├── fft_radix32_dif_dispatch.h
│   └── fft_radix32_tw_packed.h
│
├── radix64/                                 # N1-only (always innermost, K=1)
│   └── ...
├── radix128/                                # N1-only
│   └── ...
│
├── radix11/ ... radix23/                    # genfft primes (notw only, no tw codelets)
│
└── 3-execute/
    ├── vfft_planner.h                       # planner, executor, factorizer
    ├── vfft_register_codelets.h             # registry wiring (guard-based)
    └── tests/
```

---

## Include Order (Critical)

In any `.c` file that uses the planner, includes must follow this order:

```c
/* 1. Radix DIT dispatch headers — define kernel functions, set guards */
#include "fft_radix2_dispatch.h"       // → #define FFT_RADIX2_DISPATCH_H
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#include "fft_radix7_dispatch.h"
#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"
#include "fft_radix32_dispatch.h"

/* 2. Radix DIF dispatch headers — must come AFTER DIT dispatch */
/*    (SIMD DIF wrappers for R=2-8 are guarded on the notw header guard) */
#include "fft_radix2_dif_dispatch.h"   // → #define FFT_RADIX2_DIF_DISPATCH_H
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix16_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"

/* 3. Planner — radix-agnostic, works through function pointers */
#include "vfft_planner.h"

/* 4. Registry — checks #ifdef guards from step 1-2, registers what's available */
#include "vfft_register_codelets.h"
```

**Why this order matters:** The registry uses `#ifdef FFT_RADIX8_DISPATCH_H` etc. to conditionally compile dispatch wrappers and register them. If a dispatch header isn't included, that radix falls back to the naive O(N²) DFT codelet. No compile error — just slower.

---

## Planner Architecture

### Codelet Function Signatures

```c
// notw: no twiddle, just butterfly at stride K
typedef void (*vfft_codelet_fn)(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t K);

// tw: fused twiddle × butterfly in one memory pass
typedef void (*vfft_tw_codelet_fn)(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K);
```

### Codelet Registry

```c
typedef struct {
    vfft_codelet_fn     fwd[256];        // notw forward
    vfft_codelet_fn     bwd[256];        // notw backward
    vfft_tw_codelet_fn  tw_fwd[256];     // DIT: twiddle BEFORE butterfly
    vfft_tw_codelet_fn  tw_bwd[256];
    vfft_tw_codelet_fn  tw_dif_fwd[256]; // DIF: twiddle AFTER butterfly
    vfft_tw_codelet_fn  tw_dif_bwd[256];
} vfft_codelet_registry;
```

Indexed by radix. `NULL` = not available (falls back to naive or separate twiddle).

### Registry Status

| Radix | notw | DIT tw (fused) | DIF tw (fused) | Type |
|-------|------|----------------|----------------|------|
| 2 | ✓ scalar+avx2+avx512 | ✓ | ✓ | hand-optimized |
| 3 | ✓ scalar+avx2+avx512 | ✓ | ✓ | hand-optimized |
| 4 | ✓ scalar+avx2+avx512 | ✓ | ✓ | hand-optimized |
| 5 | ✓ scalar+avx2+avx512 | ✓ | ✓ | hand-optimized |
| 7 | ✓ scalar+avx2+avx512 | ✓ | ✓ | hand-optimized |
| 8 | ✓ scalar+avx2+avx512 | ✓ | ✓ | hand-optimized |
| 16 | ✓ scalar+avx2+avx512 | ✓ | ✓ | generated (4×4 CT) |
| 32 | ✓ scalar+avx2+avx512 | ✓ | ✓ | generated (8×4 CT) |
| 64 | ✓ scalar+avx512 | — (N1 only) | — | generated (8×8 CT) |
| 128 | ✓ scalar+avx512 | — (N1 only) | — | generated (16×8 CT) |
| 11,13,17,19,23 | ✓ genfft | — | — | FFTW-style genfft |

### Factorizer

Decomposes N into supported radixes. Preference order:
```
128, 64, 32, 16, 8, 4, 2,    ← greedy, largest first
9, 10, 6,                     ← composites
23, 19, 17, 13, 11, 7, 5, 3  ← primes
```

**SIMD-aware reordering** after extraction:
- Power-of-2 radixes go **innermost** (stages[0], stages[1], ...)
- Non-pow2 radixes go **outermost**
- Within each group: largest first (descending)

This ensures K is SIMD-aligned (multiple of 8) after the first pow2 stage:
```
N=1000: extracted {8,5,5,5} → reordered {8,5,5,5}
  K values: 1, 8, 40, 200  ← all divisible by 8 → AVX-512 for all stages

N=1000 (without reorder): {5,5,5,8}
  K values: 1, 5, 25, 125  ← all scalar fallback
```

### Plan Structure

```c
typedef struct {
    size_t radix;
    size_t K;                    // stride = product of all inner radixes
    vfft_codelet_fn fwd, bwd;   // notw codelet (always non-NULL)
    vfft_tw_codelet_fn tw_fwd, tw_bwd;       // DIT fused (may be NULL)
    vfft_tw_codelet_fn tw_dif_fwd, tw_dif_bwd; // DIF fused (may be NULL)
    double *tw_re, *tw_im;      // twiddle table (NULL if K=1)
} vfft_stage;

typedef struct {
    size_t N, nstages;
    vfft_stage stages[32];
    size_t *perm;       // DIT input permutation (digit reversal)
    size_t *inv_perm;   // DIF output permutation (inverse of perm)
    double *buf_a_re, *buf_a_im;  // ping-pong buffers
    double *buf_b_re, *buf_b_im;
} vfft_plan;
```

### Twiddle Table Layout (flat)

For a stage with radix R at stride K:
```
tw_re[(n-1)*K + k]  for n=1..R-1, k=0..K-1
tw_im[(n-1)*K + k]

where W = exp(-2πi·n·k / (R·K))  [forward sign convention]
```

Total: `(R-1) × K` doubles per component. Allocated at plan creation.

---

## Execution Model

### Forward (DIT)

```
Input → digit-reversal permutation (gather) → buf_a
for s = 0 to S-1:              // inner to outer (K grows)
    for each group g:
        if (K > 1 && tw_fwd):  fused DIT tw codelet (one pass)
        elif (K > 1 && tw_re): apply_twiddles(input) → notw codelet (two passes)
        else:                   notw codelet only (K=1, innermost)
    pointer swap (src ↔ dst)
Last stage writes directly to caller's output (zero-copy)
Output → natural order
```

### Backward (DIF)

```
Input → natural order (no permutation — takes DIT forward output directly)
for s = S-1 down to 0:        // outer to inner (K shrinks)
    for each group g:
        if (K > 1 && tw_dif_bwd): fused DIF tw codelet (one pass)
        elif (K > 1 && tw_re):     notw codelet → apply_twiddles_conj(output) (two passes)
        else:                       notw codelet only (K=1, innermost)
    pointer swap (src ↔ dst)
Output → inverse digit-reversal permutation (gather) → caller's output
```

### Auto-select

`vfft_execute_bwd()` checks if all stages have backward codelets:
- **Yes** → uses DIF backward (preferred, zero-perm roundtrip)
- **No** → falls back to `conj(DFT(conj(x)))` via forward executor

### Three-Tier Codelet Dispatch (per stage)

```
Priority 1: Fused tw codelet (tw_fwd/tw_bwd or tw_dif_fwd/tw_dif_bwd)
  → single memory pass, twiddle × butterfly fused
  → available for R=2,3,4,5,7,8,16,32

Priority 2: notw codelet + separate twiddle application
  → two memory passes
  → used for R=11,13,17,19,23 (no fused tw codelets)

Priority 3: notw codelet only
  → innermost stage (K=1), no twiddles needed
  → R=64,128 always land here
```

---

## Codelet Naming Conventions

### notw (twiddle-less)

```
radixN_notw_dit_kernel_{fwd,bwd}_{scalar,avx2,avx512}   // R=2,3,4,5,7,8
radixN_n1_dit_kernel_{fwd,bwd}_{scalar,avx2,avx512}     // R=16,32,64,128 (N1-gen)
```

### DIT tw (twiddle before butterfly)

```
radixN_tw_dit_kernel_{fwd,bwd}_{scalar,avx2,avx512}      // R=2,3,4,5,7,8
radixN_tw_flat_dit_kernel_{fwd,bwd}_{scalar,avx2,avx512}  // R=16,32 (flat twiddle table)
```

### DIF tw (twiddle after butterfly)

```
radixN_tw_dif_kernel_{fwd,bwd}_{scalar,avx2,avx512}      // R=2,3,4,5,7,8
radixN_tw_flat_dif_kernel_{fwd,bwd}_{scalar,avx2,avx512}  // R=16,32
```

### Dispatch (ISA routing)

Each radix has two dispatch headers at the root of its folder:

```
fft_radixN_dispatch.h       // DIT: notw + tw, ISA auto-detect
fft_radixN_dif_dispatch.h   // DIF: tw only, ISA auto-detect
```

The DIT dispatch sets a header guard (`FFT_RADIXN_DISPATCH_H`) that the registry checks.

---

## ISA Routing Rules

```
AVX-512:  K >= 8 && K % 8 == 0   (8-wide __m512d)
AVX2:     K >= 4 && K % 4 == 0   (4-wide __m256d)
Scalar:   any K >= 1              (always available)
```

The SIMD-aware factorizer ensures power-of-2 radixes are innermost, so K reaches SIMD alignment after the first stage. Non-pow2 outer stages still get SIMD because K is already aligned.

---

## Pack+Walk (Radix 16 and 32)

For large K (>512), the twiddle table exceeds L2 cache. The walker eliminates it:

### Radix-16 Walk (4 bases)
- Bases: W^1, W^2, W^4, W^8
- Derive 15 twiddles per block via binary tree: 10 cmuls
- Walk step: 4 cmuls per block
- Block size: T=8 (AVX-512) or T=4 (AVX2)
- Data must be in packed layout (block-contiguous)

### Radix-32 Walk (5 bases)
- Bases: W^1, W^2, W^4, W^8, W^16
- Derive 31 twiddles: 26 cmuls
- Walk step: 5 cmuls per block

### Packed Layout

```
Strided:  data[n*K + k]           n=0..R-1, k=0..K-1
Packed:   data[b*R*T + n*T + j]   b=block, n=0..R-1, j=0..T-1

Pack:   radix16_pack_input_avx512(strided → packed, K)
Unpack: radix16_unpack_output_avx512(packed → strided, K)
```

---

## How to Write a Benchmark

```c
#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#include "fft_radix7_dispatch.h"
#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"
#include "fft_radix32_dispatch.h"

#include "fft_radix2_dif_dispatch.h"
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix16_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"

#include "vfft_planner.h"
#include "vfft_register_codelets.h"

#include <fftw3.h>

int main(void) {
    vfft_codelet_registry reg;
    vfft_register_all(&reg);
    vfft_print_registry(&reg);

    size_t N = 4096;
    vfft_plan *plan = vfft_plan_create(N, &reg);
    vfft_plan_print(plan);

    double *ir = ..., *ii = ..., *or = ..., *oi = ...;

    // Forward (DIT)
    vfft_execute_fwd(plan, ir, ii, or, oi);

    // Backward (DIF) — takes forward output directly, no reordering
    double *br = ..., *bi = ...;
    vfft_execute_bwd(plan, or, oi, br, bi);

    // Normalize: br[i] /= N, bi[i] /= N  → recovers ir[i], ii[i]

    vfft_plan_destroy(plan);
}
```

### Good Benchmark Sizes

Sizes that decompose into optimized radixes R={2,3,4,5,7,8,16,32}:

```
Pure pow2:     256, 512, 1024, 2048, 4096, 8192, 16384, 32768
With R=5:      200, 400, 1000, 2000, 5000, 10000, 20000
With R=3:      192, 384, 768, 1536, 3072, 6144
Mixed 3×5:     120, 240, 480, 960, 1920, 4800
With R=7:      224, 448, 896, 1792, 3584
```

**Avoid** single-stage sizes like N=64 or N=128 — those just call one N1 codelet, not the multi-stage planner.

---

## File Modification Rules

- **Never modify existing comments or code** in LINAK code unless explicitly requested
- Generators (Python) produce headers — regenerate, don't hand-edit
- The planner (`vfft_planner.h`) and registry (`vfft_register_codelets.h`) are the only files that need editing when adding new features
- Dispatch headers are standalone per-radix — adding a new radix doesn't touch existing ones
