# 🎯 Fused 8×4 Pipeline: Stockham-Style Streaming FFT

This document explains the concept and practical implementation of a **fused 8×4 FFT pipeline**, which eliminates intermediate transposes and temporary buffers by streaming data through consecutive butterfly stages in registers. The result is a **radix-32 Stockham-style FFT kernel** with superior cache locality and minimal memory traffic.

---

## ⚡ Motivation

In the traditional two-pass decomposition (8×4):

1. **Pass 1:** Radix-8 butterflies → write to temp buffer
2. **Transpose:** Data reorganization for second stage
3. **Pass 2:** Radix-4 butterflies → write to final output

This design requires intermediate memory traffic and breaks locality between passes.

The fused approach removes those overheads by streaming results from the first stage directly into the second, keeping all intermediate data **in registers**.

---

## 🔧 Core Concept

### Traditional Flow
```text
input[k*stride8 + g]
   ↓
Radix-8 butterfly (8 samples per group)
   ↓  (writes 8 results per group)
temp[transposed layout]
   ↓  (read again)
Radix-4 butterfly across groups
   ↓
output
```

### Fused Flow
```text
input[k*stride8 + g]
   ↓
Radix-8 butterfly (per group)
   ↓
(4 partial 8-point results now in ZMM regs)
   ↓
Immediately apply radix-4 combine across those 4 groups
   ↓
Store final 32 outputs
```

Each fused iteration handles a complete **32-point FFT tile (8×4)** with:
- **1 read**, **1 write** (no temp buffer)
- **No transpose**
- **Shared twiddles** across both stages

---

## 🧠 Architectural View

Let’s define four radix-8 groups: **A, B, C, D.**

Each produces 8 complex outputs:
```
A0..A7, B0..B7, C0..C7, D0..D7
```

Then, for each frequency lane `m = 0..7`, the final radix-4 stage combines across groups:
```c
[A_m, B_m, C_m, D_m] → radix-4 butterfly with twiddles W32^(m*g)
```

This is mathematically identical to the separate two-pass FFT, except all work stays in registers.

---

## ⚙️ Implementation Sketch (Per 32-Point Tile)

```c
for (size_t k = 0; k < K; k += 8) {

    // 1. Load and form four 8-point groups (interleaved input)
    load_4x8_complex_block(&in_re[k], &in_im[k],
                           &A_re, &B_re, &C_re, &D_re,
                           &A_im, &B_im, &C_im, &D_im);

    // 2. Radix-8 butterfly per group
    radix8_inplace(&A_re, &A_im);
    radix8_inplace(&B_re, &B_im);
    radix8_inplace(&C_re, &C_im);
    radix8_inplace(&D_re, &D_im);

    // 3. Apply twiddles between stages
    apply_twiddles_8x4(&B_re,&B_im, &C_re,&C_im, &D_re,&D_im, k);

    // 4. Radix-4 combine across groups (now fully fused 8×4)
    radix4_across_groups(&A_re,&A_im,
                         &B_re,&B_im,
                         &C_re,&C_im,
                         &D_re,&D_im);

    // 5. Store final 32 results
    store_32_outputs(out_re, out_im, k, A_*,B_*,C_*,D_*);
}
```

---

## 🧩 Key Engineering Points

### 1️⃣ Register Pressure
- 4 groups × (re+im) = **8 ZMM registers**
- Radix-8 per group uses ≈ 4 temporaries → **~12–14 ZMM total**
- AVX-512 provides 32 ZMMs, leaving comfortable headroom.

**Tip:** Compute two groups at a time (A/B, then C/D) and reuse temporary registers aggressively.

### 2️⃣ Twiddle Strategy
Twiddles for the second stage follow the pattern:
```
T(m,g) = W32^(m*g) = {1, W32^m, W32^(2m), W32^(3m)}
```

For power-of-two positions, these reduce to smaller geometries:
- m = 1 → W32 constants
- m = 2 → W16 constants
- m = 4 → W8 constants

Broadcast these from precomputed plan arrays. For others (3,5,6,7), generate on-the-fly:
```c
W2 = cmul(W, W);
W3 = cmul(W2, W);
```

### 3️⃣ DIT/DIF Pairing
Best schedule: **DIF(8)** → **Twiddle Multiply** → **DIT(4)**.
- DIF(8) produces twiddles on outputs.
- DIT(4) consumes them naturally.

### 4️⃣ Data Locality
- Load and store are fully contiguous with planner-ordered layout.
- No gathers or strided access needed.
- Excellent for prefetching.

### 5️⃣ Register Reuse Plan
| Phase | Registers Used | Notes |
|--------|----------------|-------|
| Load groups | 8 | 4 re + 4 im |
| Radix-8 temp | +4–6 | Reuse across A/B and C/D |
| Radix-4 combine | +4 | Row-wise scratch reused |
| Total | ~20–22 ZMM | Fits safely under 32 regs |

### 6️⃣ Small-K Path
For **K ≤ 64**, unroll fully with constants baked in. The fused kernel can outperform a 16×2 even at small sizes.

### 7️⃣ Large-K Path
For **K ≥ 2048**, combine with multi-level cache tiling and non-temporal stores to maintain bandwidth efficiency.

---

## ✅ Summary

| Feature | Benefit |
|----------|----------|
| In-register stage fusion | Eliminates temp write/read |
| Shared twiddles | Fewer loads, better cache |
| Contiguous memory | Perfect prefetching, no gathers |
| Stockham structure | Uniform stride, easy parallelization |
| Safe register use | 20–22 ZMMs peak |

---

## 🚀 Expected Performance Gains

| Optimization | Typical Gain |
|---------------|---------------|
| Remove transpose | +15–25% |
| Shared twiddles (geometric constants) | +5–8% |
| Contiguous layout planner | +5–10% |
| Cache tiling (K>2k) | +10–20% |

Combined, these optimizations can push the fused 8×4 beyond an equivalently optimized 16×2 across nearly all transform sizes.

---

**In short:** this Stockham-style fused 8×4 design streams data seamlessly through both stages, removing memory stalls, leveraging broadcast twiddles, and achieving near-ideal throughput on modern AVX-512 cores.

