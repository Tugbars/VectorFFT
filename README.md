<img width="1800" height="400" alt="banner" src="https://github.com/user-attachments/assets/218d4f28-d5c5-4630-82dc-cab06acc6783" />


<p align="center">
  A from-scratch mixed-radix FFT library in C99 with hand-tuned AVX2/AVX-512 codelets.<br>
  Beats FFTW on 23 of 39 tested sizes. No external dependencies.
</p>

---

## Benchmark Results

<img width="1800" height="1040" alt="benchmark_chart" src="https://github.com/user-attachments/assets/8e628b89-85ea-44f8-a3ee-ac0bf8fa9704" />

> **Platform:** Intel Core i9-14900KF · 48 KB L1d · DDR5 · AVX2 · FFTW 3.3.10 (FFTW_ESTIMATE)

---

## Accuracy

<img width="1720" height="1040" alt="accuracy" src="https://github.com/user-attachments/assets/5c1b7c32-8f26-49a5-a3a9-c62d2a8fec08" />

---

## Getting Started

```bash
git clone https://github.com/yourusername/VectorFFT.git
cd VectorFFT && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

```c
#include "vfft_planner.h"
#include "vfft_register_codelets.h"

vfft_codelet_registry reg;
vfft_register_all(&reg);

vfft_plan *plan = vfft_plan_create(N, &reg);
vfft_execute_forward(plan, re, im, out_re, out_im);
vfft_plan_destroy(plan);
```

---

## Performance Tuning

VectorFFT uses **measurement, not heuristics**. Three calibration benchmarks determine every performance-critical threshold on your specific hardware:

```bash
./bench_walk          # Walk thresholds per radix    → vfft_calibration.txt
./bench_il            # IL crossovers per radix      → vfft_calibration.txt
./bench_factorize     # Optimal factorizations       → vfft_wisdom.txt
```

Both files are read automatically at plan creation. Without them, the planner uses conservative defaults — calibration typically improves performance by 15–40%.

**Known limitations:**
- DIF backward codelets are ~10–15% slower than DIT forward — primary source of roundtrip losses
- Small N (≤128) — per-stage overhead dominates; FFTW's monolithic codelets win here
- R=5 at K>2048 falls back to log3 derivation; planner minimizes this via R=10/R=25 fusion

---

## Acknowledgments

[FFTW](http://www.fftw.org/) by Matteo Frigo and Steven G. Johnson — the gold standard. VectorFFT's prime codelets (R=17, 19, 23) are translated from FFTW's genfft output.

<p align="center"><sub>VectorFFT — Because every nanosecond counts.</sub></p>
