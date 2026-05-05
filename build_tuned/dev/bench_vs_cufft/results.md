# VectorFFT (T=8 CPU) vs cuFFT (RTX 5080) — latency comparison

**Hardware:** Intel i9-14900KF (8 P-cores, AVX2) vs NVIDIA RTX 5080 (84 SMs, CC 12.0, 16 GB GDDR7)
**Power plan:** High Performance (set + restored by `run.bat`)
**Method:** min-of-21 reps after 5 warmups, FP64 complex-to-complex forward FFT
**Grid:** N ∈ {64, 256, 1K, 4K, 16K, 64K, 256K} × K ∈ {8, 128, 256}

## Numbers (all wall time, nanoseconds)

### K=8 — small batch (latency-sensitive workload)

| N | VFFT T=8 | cuFFT compute | cuFFT +D→H | cuFFT roundtrip | crossover (compute) | crossover (roundtrip) |
|--:|--------:|--------------:|-----------:|----------------:|:-------------------:|:----------------------:|
| 64 | 1,000 | 7,328 | 50,300 | 56,800 | CPU 7× | CPU 57× |
| 256 | 2,200 | 15,360 | 62,600 | 64,200 | CPU 7× | CPU 29× |
| 1,024 | 8,000 | 19,616 | 58,700 | 103,000 | CPU 2.5× | CPU 13× |
| 4,096 | 24,400 | 45,280 | 117,200 | 189,100 | CPU 1.9× | CPU 7.7× |
| 16,384 | 83,300 | **34,848** | 207,100 | 382,900 | **GPU 2.4×** | CPU 4.6× |
| 65,536 | 427,200 | **114,208** | 702,900 | 1,263,900 | **GPU 3.7×** | CPU 3.0× |
| 262,144 | 2,231,100 | **467,168** | **2,664,500** | 4,686,500 | **GPU 4.8×** | CPU 2.1× |

**K=8 takeaway:** GPU wins compute-only from N=16K, but **CPU wins end-to-end (with H→D + D→H) at every tested N**. For latency-sensitive workloads with low batch (per-frame audio, real-time DSP, single-stream pipelines), VectorFFT-CPU is the right answer up to N=256K.

### K=128 — medium batch

| N | VFFT T=8 | cuFFT compute | cuFFT +D→H | cuFFT roundtrip | crossover (compute) | crossover (roundtrip) |
|--:|--------:|--------------:|-----------:|----------------:|:-------------------:|:----------------------:|
| 64 | 5,600 | 7,456 | 64,700 | 91,100 | CPU 1.3× | CPU 16× |
| 256 | 14,200 | 15,520 | 87,500 | 157,400 | CPU 1.1× | CPU 11× |
| 1,024 | 47,700 | **21,824** | 195,500 | 363,300 | **GPU 2.2×** | CPU 7.6× |
| 4,096 | 227,100 | **83,488** | 674,800 | 1,199,100 | **GPU 2.7×** | CPU 5.3× |
| 16,384 | 1,341,700 | **313,152** | 2,508,100 | 4,557,600 | **GPU 4.3×** | CPU 3.4× |
| 65,536 | 27,632,000 | **1,298,336** | **9,898,800** | **18,590,000** | **GPU 21×** | **GPU 1.5×** |
| 262,144 | 146,612,700 | **6,404,640** | **78,222,100** | **143,531,200** | **GPU 23×** | **GPU 1.02×** (tie) |

**K=128 takeaway:** GPU compute-only crossover at N=1024; +D→H crossover at N=65,536; full roundtrip crossover ~N=65,536–262,144.

### K=256 — large batch

| N | VFFT T=8 | cuFFT compute | cuFFT +D→H | cuFFT roundtrip | crossover (compute) | crossover (roundtrip) |
|--:|--------:|--------------:|-----------:|----------------:|:-------------------:|:----------------------:|
| 64 | 6,700 | 8,032 | 64,600 | 119,100 | CPU 1.2× | CPU 18× |
| 256 | 20,800 | **16,672** | 126,700 | 236,700 | **GPU 1.25×** | CPU 11× |
| 1,024 | 100,600 | **38,304** | 411,800 | 1,187,800 | **GPU 2.6×** | CPU 12× |
| 4,096 | 584,700 | **156,160** | 1,319,700 | 4,443,000 | **GPU 3.7×** | CPU 7.6× |
| 16,384 | 12,585,300 | **609,120** | **9,110,000** | 17,905,200 | **GPU 21×** | CPU 1.4× |
| 65,536 | 73,808,600 | **2,567,552** | **21,826,900** | **72,030,500** | **GPU 29×** | **GPU 1.02×** (tie) |
| 262,144 | 294,055,500 | **12,752,736** | **153,667,900** | **291,213,500** | **GPU 23×** | **GPU 1.01×** (tie) |

**K=256 takeaway:** GPU compute-only crossover at N=256; +D→H crossover at N=16K; full roundtrip basically a wash at N≥65K (within 1%).

## Crossover map — where to use which

| Workload type | Recommendation |
|---------------|----------------|
| Real-time per-frame audio (K=8, N≤4K) | **VFFT-CPU** wins by 7–57× |
| Embedded DSP loops (K=8, N≤256K) | **VFFT-CPU** wins on full roundtrip every tested cell |
| Image processing batch (K=128, N≥4K) compute-only | **GPU** if data already resident; **CPU** if data is on host |
| Convolutional pipelines (K=256, N≥16K) | **GPU** if data resident; tie at N≥65K with transfer |
| Throughput-bound batch jobs (huge N, K, data already on GPU) | **GPU** by 20–30× — no contest |
| Mixed CPU/GPU pipelines | **per-stage decision** — depends entirely on data residency |

## Key floor numbers

- **GPU latency floor at K=8 N=64:** ~50 µs full round-trip, 7 µs compute-only. Kernel launch + sync overhead dominates.
- **CPU at K=8 N=64:** 1 µs total. VectorFFT has no launch overhead; it's just a function call into AVX2 codelets.
- **D→H transfer alone for 1 KB (K=8 N=64):** ~50 µs. PCIe roundtrip latency floor, regardless of payload size.
- **Compute-only "ideal" GPU advantage:** caps at ~23–29× for the largest N×K combinations. This is the upper bound assuming data is already on device.

## Implications for "GPU port + unified wisdom"

The data says **a GPU port would be a real win, but a unified wisdom file is the wrong abstraction.** Here's why:

### 1. The decision isn't (N, K) alone — it's (N, K, residency)

Same (N, K) cell can want CPU or GPU depending on where the data starts. Wisdom file would need a third dimension (data residency hint), and that would need to come from the caller at plan time. That's a real API change — not a minor tweak.

### 2. The crossover lines depend on K in ways that can't be summarized cleanly

- K=8: CPU wins everywhere for round-trip
- K=128: crossover around N=65K
- K=256: crossover around N=64K (round-trip) but N=256 (compute-only)

A flat 2D table can't capture this without effectively storing every (N, K, residency) tuple.

### 3. Latency floor mismatch is structural

GPU has a ~50µs floor. CPU has no floor. **Anything below ~5K total elements always goes CPU**, regardless of what the wisdom says about the larger cells. That's not a per-cell wisdom decision — it's an architectural rule.

### 4. The good news: a sister library (`vectorfft-gpu`) targeting K≥128, N≥4K, GPU-resident workloads could ship as a focused project

It wouldn't share wisdom format with the CPU library (different codegen, different cost model). But it could share:
- Public API surface (same `vfft_plan_*` shape, with a backend tag)
- Codelet generation philosophy (generate per-shape)
- Wisdom infrastructure (load/save/forget, calibration), separately per backend

That's the cleaner architectural play: **two libraries, one API contract, no shared wisdom files**.

### 5. The realistic v1.x positioning

VectorFFT v1.0 = **CPU FFT for latency-sensitive and small/medium-batch workloads**. That's a defensible niche — cuFFT/VkFFT can't touch the K=8 single-frame audio case, and FFTW/MKL can't touch the AVX2-tuned codelet performance.

A future GPU sibling project (`vectorfft-cuda` or similar) would target **the GPU-resident large-batch regime** — different niche, complementary not competing. That's a separate v2.0 / standalone-project decision, not a v1.0 closeout item.

## Reproducing

```cmd
build_tuned\dev\bench_vs_cufft\run.bat
```

Requires:
- CUDA Toolkit 13.x (nvcc + cuFFT)
- MSVC 2022 (vcvars64.bat)
- Intel oneAPI 2025.x (libircmt.lib for ICX-built vfft.lib)
- Pre-built `vfft.lib` at `build/lib/vfft.lib` (run cmake build at project root first)
- Optional: admin rights for `powercfg /setactive` (script falls through with a warning otherwise)
