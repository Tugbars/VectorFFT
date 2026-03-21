# VectorFFT Radix-5 Codelet Package

## What changed

Replaced serial log3 twiddle derivation (1 load → W²→W³→W⁴, 15-cycle chain)
with flat twiddle loads (4 independent loads, zero latency) for K ≤ 2048.
Falls back to log3 for K > 2048 (saves 3 cache misses when twiddle table >> L1).

Measured improvement (AVX2, K=4..2048): **10–30% faster per codelet call**.

## Directory structure

```
radix5/
├── generators/
│   ├── gen_r5_scalar.py    # Scalar generator (flat only, no dual dispatch)
│   ├── gen_r5_avx2.py      # AVX2 generator (flat+log3 dual, 16 YMM)
│   └── gen_r5_avx512.py    # AVX-512 generator (flat+log3 dual, 32 ZMM)
├── scalar/
│   ├── fft_radix5_scalar.h           # notw fwd/bwd + DIT tw fwd/bwd
│   ├── fft_radix5_scalar_dif_tw.h    # DIF tw fwd/bwd
│   ├── fft_radix5_scalar_il.h        # IL notw + DIT tw
│   └── fft_radix5_scalar_il_dif_tw.h # IL DIF tw
├── avx2/
│   ├── fft_radix5_avx2.h             # notw + DIT tw (dual: flat+log3)
│   ├── fft_radix5_avx2_dif_tw.h      # DIF tw (dual: flat+log3)
│   ├── fft_radix5_avx2_il.h          # IL notw + DIT tw (dual)
│   ├── fft_radix5_avx2_il_dif_tw.h   # IL DIF tw (dual)
│   └── fft_radix5_avx2_n1_mono_il.h  # N1 IL (genfft DAG, UNCHANGED)
├── avx512/
│   ├── fft_radix5_avx512.h           # notw + DIT tw (dual: flat+log3)
│   ├── fft_radix5_avx512_dif_tw.h    # DIF tw (dual: flat+log3)
│   ├── fft_radix5_avx512_il.h        # IL notw + DIT tw (dual)
│   ├── fft_radix5_avx512_il_dif_tw.h # IL DIF tw (dual)
│   └── fft_radix5_avx512_n1_mono_il.h# N1 IL (genfft DAG, UNCHANGED)
├── fft_radix5_dispatch.h             # ISA dispatch (UNCHANGED)
├── fft_radix5_dif_dispatch.h         # DIF dispatch (UNCHANGED)
└── bench_radix5_flat.c               # A/B bench: old vs new
```

## Regenerating

```bash
python generators/gen_r5_scalar.py  scalar/
python generators/gen_r5_avx2.py    avx2/
python generators/gen_r5_avx512.py  avx512/
```

## ISA-specific strategies

| | Scalar | AVX2 (16 YMM) | AVX-512 (32 ZMM) |
|---|---|---|---|
| Flat path | direct loads | load-apply-free per row | all 4 tw upfront |
| Peak regs | N/A | ~14 YMM (zero spills) | ~22 ZMM (10 spare) |
| Log3 path | serial chain | serial chain | serial chain |
| Threshold | flat only | K ≤ 2048 → flat | K ≤ 2048 → flat |
| `#define` | — | `R5_FLAT_THRESHOLD_AVX2` | `R5_FLAT_THRESHOLD_AVX512` |

## Deployment

1. Copy scalar/, avx2/, avx512/ contents into src/radix5/{scalar,avx2,avx512}/
2. dispatch.h and dif_dispatch.h need **zero changes** — function names are identical
3. N1 IL files (n1_mono_il.h) are UNCHANGED genfft DAGs
4. Run bench_radix5_flat.c on your hardware to verify crossover point
5. If crossover differs from 2048, adjust the `#define` in the generated headers

## Also deployed: vfft_planner.h

R=25 unlocked at SIMD-aligned K — reduces R=5 exposure at high K:
- N=800:  32×25 (2 stages instead of 3)
- N=4000: 32×25×5 (3 stages instead of 4)
- N=5000: 25×10×10×2 (4 stages instead of 5)
