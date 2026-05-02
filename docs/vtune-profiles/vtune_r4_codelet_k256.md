# VTune Profile: R=4 Isolated Codelet (t1_dit_fwd, K=256)

**Date:** 2026-04-10  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.71 GHz turbo  
**Binary:** `vfft_bench_r4.exe 256 5000 t1_dit_fwd`  
**Codelet:** `radix4_t1_dit_fwd_avx2` — in-place DIT twiddle + butterfly, AVX2  
**Data size:** R=4, K=256, total=1024 elements = 16 KB (re+im), fits L1  
**Analysis:** Microarchitecture Exploration (uarch-exploration)

---

## Top-Level Breakdown

| Category | % Pipeline Slots |
|----------|-----------------|
| **Retiring** | **85.9%** |
| Front-End Bound | 2.3% |
| Bad Speculation | 2.0% |
| Back-End Bound | 9.8% |

## Retiring Detail

| Metric | Value |
|--------|-------|
| Light Operations | 85.8% |
| FP Arithmetic | 62.9% of uOps |
| FP Vector (256-bit) | 62.5% of uOps |
| FP Scalar | 0.0% |
| Memory Operations | 38.9% |
| Heavy Operations | 0.1% |

## Back-End Detail

| Metric | Value |
|--------|-------|
| Memory Bound | 1.0% |
| Core Bound | 8.8% |
| L1 Bound | 0.4% |
| L2 Bound | 0.0% |
| L3 Bound | 0.1% |
| DRAM Bound | 0.1% |
| Store Bound | 0.0% |
| L1 Latency Dependency | 61.6% of clockticks |
| DTLB Load Overhead | 0.1% |
| DTLB Store Overhead | 0.1% |

## Port Utilization

| Port | Usage | Role |
|------|-------|------|
| **Port 0** | **96.0%** | FMA unit 1 |
| **Port 1** | **91.0%** | FMA unit 2 |
| Port 6 | 11.1% | Branch/ALU |
| 3+ Ports Utilized | 96.5% of cycles |
| 2 Ports Utilized | 1.0% |
| 1 Port Utilized | 0.6% |
| 0 Ports Utilized | 0.0% |

## Instruction Mix

| Metric | Value |
|--------|-------|
| Clockticks | 142,425,600,000 |
| Instructions Retired | 755,318,400,000 |
| CPI | 0.189 |
| IPC | 5.30 |
| Vector Capacity Usage (FPU) | 50.0% |

## Front-End Detail

| Metric | Value |
|--------|-------|
| DSB Coverage | 31.0% |
| LSD Coverage | 67.8% |
| DSB Misses | 12.7% |
| ICache Misses | 0.1% |
| Branch Resteers | 0.5% |

## Benchmark Result

| Metric | Value |
|--------|-------|
| ns/call | 118.6 |
| CPE (cycles per element) | 0.277 |
| GFLOP/s | 73.4 |
| FLOPs/call | 8,704 (FMA=2 FLOPs) |

---

## Interpretation

The R=4 t1_dit codelet achieves 85.9% retiring — near theoretical maximum. Both FMA ports are saturated (96%/91%). The codelet is compute-bound, not memory-bound (1.0%). The 8.8% Core Bound is almost entirely from L1 latency dependency (61.6% of clockticks) caused by in-place read-modify-write — store-to-load forwarding adds latency but the OoO engine overlaps it effectively.

The LSD serves 68% of uops — the loop body fits entirely in the Loop Stream Detector, bypassing decode entirely. This is optimal for a tight inner loop.

**Verdict: This codelet is at peak efficiency. No optimization needed at the codelet level.**
