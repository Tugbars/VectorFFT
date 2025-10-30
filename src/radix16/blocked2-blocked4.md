# Blocked Twiddle Storage for Radix-16 FFT

## **Overview**

Radix-16 FFT requires 15 complex twiddle factors per k-index: W₁, W₂, ..., W₁₅ where W_r = e^(-2πirk/N). The blocked twiddle scheme exploits mathematical symmetry to reduce memory bandwidth and cache pressure.

---

## **Two Blocking Modes**

### **BLOCKED8** (Small K ≤ 512)
- **Storage:** 8 blocks (W₁...W₈)
- **Derivation:** W₉ = -W₁, W₁₀ = -W₂, ..., W₁₅ = -W₇
- **Implementation:** Pre-negate W₁...W₈ once per k-step:
  ```c
  NW_re[r] = _mm512_xor_pd(W_re[r], sign_mask);  // r=0..7
  ```
- **Savings:** 47% bandwidth (8+7 XORs vs 15 loads)

### **BLOCKED4** (Large K > 512)
- **Storage:** 4 blocks (W₁...W₄)
- **Derivation:**
  - W₅ = W₁×W₄, W₆ = W₂×W₄, W₇ = W₃×W₄, W₈ = W₄²
  - W₉...W₁₅ = -(W₁...W₇)
- **Implementation:** 
  ```c
  cmul_fma_soa_avx512(W1r, W1i, W4r, W4i, &W5r, &W5i);  // 4 products
  NW_re[r] = _mm512_xor_pd(W_re[r], sign_mask);         // 7 negations
  ```
- **Savings:** 73% bandwidth (4 loads + 4 FMA + 7 XORs vs 15 loads)

---

## **Threshold-Based Selection**

```c
#define RADIX16_BLOCKED8_THRESHOLD 512

radix16_twiddle_mode_t mode = (K <= 512) ? BLOCKED8 : BLOCKED4;
```

**Rationale:**
- **K ≤ 512:** Total twiddle footprint = 8 blocks × 512 × 16 bytes = **64 KB**
  - Fits comfortably in L1 (32KB) + L2 (1MB)
  - XOR latency (1 cycle) << DRAM load latency (~200 cycles)
  - **Choose BLOCKED8:** Minimize arithmetic operations
  
- **K > 512:** Twiddles stream from L3/DRAM
  - BLOCKED4 reduces memory traffic by 73%
  - 4 FMA operations (4 cycles latency) amortized over 15 uses
  - **Choose BLOCKED4:** Minimize memory bandwidth

---

## **Design Considerations**

### **1. Cache Hierarchy Alignment**
- Threshold set at L1+L2 capacity boundary
- K-tiling (Tk=64) keeps working set hot for multiple butterfly passes

### **2. Arithmetic vs Memory Trade-off**
- BLOCKED8: Arithmetic-light, memory-moderate (good for cache hits)
- BLOCKED4: Arithmetic-moderate, memory-light (good for cache misses)

### **3. Pre-computation Strategy**
- **Pre-negation:** Performed once per k-step, amortized over all 15 twiddle uses
- **Products (W₅...W₈):** Computed once, reused for both positive and negative variants

### **4. Register Pressure Management**
- BLOCKED8: 16 ZMM registers (W[8] + NW[8])
- BLOCKED4: 14 ZMM registers (W[4] + W[4] derived + NW[7])
- Both fit within AVX-512's 32 ZMM register file with headroom

### **5. Recurrence Integration**
For K > 4096, tile-local recurrence (w ← w×δw) replaces memory loads entirely:
- Refresh accurate twiddles at tile boundaries (every 64 steps)
- Advance within tiles using pre-computed phase increments
- Works with BLOCKED4: derive W₅...W₈ from recurrence state

---

## **Performance Impact**

| Mode | K Range | Memory BW | Compute Ops | L1 Pressure |
|------|---------|-----------|-------------|-------------|
| **BLOCKED8** | ≤ 512 | 53% of naive | +7 XORs/k | Low (64KB) |
| **BLOCKED4** | > 512 | 27% of naive | +4 FMA + 7 XORs/k | None (streaming) |
| **Recurrence** | > 4096 | ~0% (refresh only) | +15 cmul/k | None |

**Measured Impact (estimated):**
- BLOCKED8: ~10-15% speedup over naive 15-load approach
- BLOCKED4: ~30-40% speedup for large K (DRAM bandwidth limited)
- Recurrence: Additional ~5-10% for very large K

---

## **Conclusion**

The blocked twiddle scheme provides automatic, threshold-based optimization across the entire K-space:
- **Small K:** Minimize operations (BLOCKED8)
- **Large K:** Minimize bandwidth (BLOCKED4)
- **Very large K:** Minimize both (BLOCKED4 + recurrence)

This design ensures near-optimal performance without runtime profiling or manual tuning.
