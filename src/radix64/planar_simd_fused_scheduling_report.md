# Planar + SIMD-Fused Scheduling in FFTs

## 1. Introduction

Planar + SIMD-fused scheduling is a high-performance technique used in FFT implementations to minimize memory traffic, reduce pipeline stalls, and fully utilize SIMD (Single Instruction, Multiple Data) execution resources. It represents the next optimization step after loop unrolling, register blocking, and stage merging.

This document explains what it means, why it matters, and how it applies to modern AVX-512 FFT kernels.

---

## 2. Background

A typical FFT implementation (radix-N decomposition) involves several computation stages. Each stage performs:

1. **Butterfly computation** – combines pairs of complex values.
2. **Twiddle multiplication** – multiplies by complex roots of unity.
3. **Reordering** – prepares data for the next stage.

Traditionally, these steps are separated by memory operations (load → compute → store). While simple, this approach creates unnecessary memory traffic and register reloads.

---

## 3. Planar Memory Layout

Complex numbers can be stored in two main ways:

- **Interleaved:** `[Re0, Im0, Re1, Im1, Re2, Im2, …]`
- **Planar (Structure of Arrays, SoA):** 
  - `Re[] = [Re0, Re1, Re2, …]`
  - `Im[] = [Im0, Im1, Im2, …]`

Planar storage is ideal for SIMD because each load or store fetches contiguous values (all reals or all imaginaries). This eliminates the need for shuffle or unpack instructions between every stage.

**Advantages:**
- Perfectly aligned memory access.
- Cleaner SIMD code.
- Reduced instruction count.

---

## 4. SIMD-Fused Scheduling

### Concept
Instead of treating each FFT stage as a separate memory operation, fused scheduling keeps data **in registers** across multiple logical stages.

For example, instead of doing:
```
[Stage 1] → store → [Stage 2] → store → [Stage 3]
```
we compute:
```
load → Stage1+Stage2+Stage3 (in registers) → store
```

### Benefits
- Fewer load/store operations.
- Higher arithmetic intensity.
- Better latency hiding through FMA (Fused Multiply-Add) overlap.

### Requirements
- Enough SIMD registers to hold all intermediate results.
- Predictable data flow between stages (no dynamic indexing).
- Ability to precompute twiddle factors for all sub-stages.

---

## 5. Planar + SIMD Fused Scheduling

When both methods are combined, each block of complex data (e.g., 64 points in a radix-64 FFT) stays entirely in SIMD registers throughout all sub-stages:

```
Load → [Radix-8 butterfly + Twiddle + Combine] → Store
```

All reordering, twiddle application, and butterfly operations are performed using AVX-512 shuffle, permute, and FMA instructions without spilling to memory.

**Typical instruction flow:**
1. Load 8×Re and 8×Im vectors (planar layout).
2. Perform multiple radix-8 steps back-to-back.
3. Apply twiddles via `_mm512_fmaddsub_pd()`.
4. Shuffle vectors using `_mm512_permutex2var_pd()` for stride changes.
5. Store final outputs.

---

## 6. Practical Performance Impact

| Optimization | Description | Typical Gain |
|---------------|--------------|---------------|
| Planar layout | Removes interleave shuffles | +5–10% |
| Fused scheduling | Reduces loads/stores | +10–20% |
| Planar + fused | Combined pipeline | **+20–40%** |

On Intel SKX/ICX (AVX-512), such fusion lets the FFT kernel reach near-peak throughput (2–2.5 FLOPs/cycle/core for double precision).

---

## 7. Example: Radix-64 FFT Fusion

In a radix-64 pipeline composed of eight radix-8 blocks:
- Each radix-8 step fits in registers (16 reals + 16 imags).
- Twiddle multiplications are interleaved after every step.
- Register-level transposes replace memory writes between sub-stages.
- The entire 64-point FFT block is computed from load to store without reloading.

This approach eliminates the intermediate “apply W₆₄” stage used in traditional designs.

---

## 8. When It’s Worth Doing

| Scenario | Recommended? |
|-----------|---------------|
| FFT size ≤ 256 | ❌ No (cache-bound, simpler code is as fast) |
| FFT size ≥ 512 | ✅ Yes (register reuse dominates) |
| Memory bandwidth limited | ✅ Strong gains |
| Register pressure high (≤ AVX2) | ⚠️ Maybe (careful tuning needed) |

---

## 9. Design Notes

- Best implemented per SIMD ISA (AVX2, AVX-512, SVE).
- Works naturally with planar (SoA) buffers.
- Typically uses 28–30 ZMM registers (safe under 32).
- Avoid spilling via careful blocking and reusing temporaries.
- Twiddle constants are best broadcast on-demand.

---

## 10. Summary

**Planar + SIMD fused scheduling** is an advanced but practical technique for high-performance FFTs.  
It trades code simplicity for significant real-world speed gains — often the last major optimization before resorting to algorithmic changes like mixed-radix decomposition or stockham autosort FFTs.

**In short:**  
> Keep it planar, keep it in registers, and fuse as much math as you can before touching memory.
 
---

## 11. References

- Frigo & Johnson, *FFTW: An Adaptive Software Architecture for the FFT* (Proc. ICASSP 1998)  
- Intel Optimization Manual, Vol. 1, “SIMD Data Layouts for FFT and DCT”  
- Heinecke et al., *LIBXSMM and FFT Fusion on Intel Xeon CPUs*, 2019  
- AMD AOCL FFT whitepapers (AVX2/AVX512 backend)  
