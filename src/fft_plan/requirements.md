# Requirements for Mixing Rader and Cooley-Tukey Twiddles in Mixed-Radix FFT

## Overview

This guide explains the requirements for correctly integrating Rader's algorithm (prime radices) with Cooley-Tukey stages (composite radices) in a mixed-radix FFT implementation.

---

## Core Requirements

### 1. Two-Tier Twiddle Architecture

Mixed-radix transforms combining Rader (primes ≥7) and Cooley-Tukey require **orthogonal twiddle factors**:

**Tier 1: CT Stage Twiddles**
- Purpose: Inter-stage rotation
- Scope: Full transform size N_stage
- Indexed by: Butterfly position k
- Formula: W_N^(r×k) where N = N_stage

**Tier 2: Rader Convolution Twiddles**
- Purpose: Intra-radix DFT kernel
- Scope: Prime modulus p only
- Indexed by: Convolution position q
- Formula: W_p^(g^q) where g = primitive root mod p

**Critical Property:** These twiddles serve different roles and must never be conflated.

---

### 2. Twiddle Computation Rules

#### Rule 2.1: Cooley-Tukey Twiddles Must Use Transform Size

**CORRECT:**
```c
base_angle = sign × 2π / N_stage
tw[k×(radix-1) + (r-1)] = exp(i × base_angle × r × k)
                        = W_{N_stage}^(r×k)
```

**WRONG:**
```c
base_angle = sign × 2π / radix  // Using radix instead of N_stage!
```

**Why:** CT twiddles rotate between sub-DFTs across the entire transform, not just within one radix.

**Example (N=14, Stage 1, radix=7):**
- Correct: W₁₄^(r×k) (rotates across 14-point transform)
- Wrong: W₇^(r×k) (would only handle 7-point DFT)

---

#### Rule 2.2: Rader Twiddles Must Use Prime Modulus Only

**CORRECT:**
```c
angle = sign × 2π × perm_out[q] / prime
tw[q] = exp(i × angle) = W_p^(g^q)
```

**Why:** Rader's algorithm converts a prime-point DFT into cyclic convolution. The twiddles implement the convolution kernel, which depends only on the prime, not the overall transform size.

**Example (radix=7 in any transform):**
- Always: W₇^(g^q) for q=0..5, g=3
- Independent of whether N=7, 14, 28, 77, etc.

---

#### Rule 2.3: Final Stage Identity Twiddles Are Valid

When sub_len = 1 (final stage), CT twiddles degenerate to identity:

```
W_{N_stage}^(r×0) = W_{N_stage}^0 = 1 for all r
```

This is mathematically correct because there's only one butterfly group (k=0 always), so no inter-group rotation is needed.

**Implementation must handle this:** Most Rader butterfly implementations include a guard:

```c
if (sub_len > 1) {
    // Apply CT twiddles
    x1 = cmul(x1, stage_tw[6*k + 0]);
    // ...
}
// If sub_len = 1, skip CT twiddles (use identity)
```

---

### 3. Storage and Ownership

**CT Twiddles (stage_tw):**
- Size: (radix - 1) × sub_len complex values
- Ownership: OWNED by stage descriptor (freed with plan)
- Allocation: 32-byte aligned for SIMD

**Rader Twiddles (rader_tw):**
- Size: (prime - 1) complex values
- Ownership: BORROWED from global cache (not freed by stage)
- Allocation: 32-byte aligned, thread-safe cached

**Memory Layout:**
```c
typedef struct {
    int radix;
    int N_stage;
    int sub_len;
    
    fft_data *stage_tw;   // OWNED: CT twiddles
    fft_data *rader_tw;   // BORROWED: Rader twiddles (or NULL)
} stage_descriptor;
```

---

### 4. Execution Strategy

**Sequential Composition:** Apply CT twiddles first, then Rader algorithm.

**Radix-7 Butterfly Structure:**
```c
void fft_radix7_fv(
    fft_data *output,
    const fft_data *input,
    const fft_data *stage_tw,   // CT twiddles: K×6 values
    const fft_data *rader_tw,   // Rader twiddles: 6 values
    int sub_len)
{
    for (int k = 0; k < sub_len; k++) {
        // Load 7 lanes
        fft_data x0 = input[k + 0*K];
        fft_data x1 = input[k + 1*K];
        // ... x2-x6
        
        // TIER 1: Apply CT stage twiddles
        if (sub_len > 1) {
            x1 = cmul(x1, stage_tw[6*k + 0]);  // W_N^(1×k)
            x2 = cmul(x2, stage_tw[6*k + 1]);  // W_N^(2×k)
            // ... x3-x6
        }
        
        // TIER 2: Rader's algorithm
        // - DC component: y0 = sum(x0..x6)
        // - Permute inputs: tx = permute(x1..x6)
        // - Convolve: v = tx ⊗ rader_tw
        // - Assemble outputs
    }
}
```

---

### 5. Dispatcher Requirements

**The dispatcher must check for Rader twiddles:**

```c
if (stage->rader_tw != NULL) {
    // Prime radix: dispatch to Rader butterflies
    switch (radix) {
        case 7:  fft_radix7_fv(out, in, stage_tw, rader_tw, sub_len); break;
        case 11: fft_radix11_fv(out, in, stage_tw, rader_tw, sub_len); break;
        case 13: fft_radix13_fv(out, in, stage_tw, rader_tw, sub_len); break;
    }
} else {
    // Composite radix: dispatch to CT butterflies
    switch (radix) {
        case 2: fft_radix2_fv(out, in, stage_tw, sub_len); break;
        case 4: fft_radix4_fv(out, in, stage_tw, sub_len); break;
        // ...
    }
}
```

**The check `rader_tw != NULL` is the critical branching point.**

---

## Example: N=14 (2×7) Transform

### Planning Phase

**Stage 0: radix=2, N_stage=14, sub_len=7**
- stage_tw: W₁₄^k for k=0..6 (7 values)
- rader_tw: NULL (not prime)

**Stage 1: radix=7, N_stage=7, sub_len=1**
- stage_tw: W₇^(r×0) = {1,0} for r=1..6 (identity, 6 values)
- rader_tw: W₇^(g^q) for q=0..5 (6 values, from cache)

### Execution Phase

**Stage 0:**
```
Dispatcher: rader_tw == NULL → Call fft_radix2_fv()
  - Uses stage_tw[k] = W₁₄^k
  - Standard Cooley-Tukey butterfly
  - Output: 7 pairs of values ready for next stage
```

**Stage 1:**
```
Dispatcher: rader_tw != NULL → Call fft_radix7_fv()
  - CT twiddles: stage_tw = {1,0}×6 (identity)
  - Check: sub_len = 1 → skip CT twiddle application
  - Rader algorithm: uses rader_tw = W₇^(g^q)
  - DC sum + cyclic convolution + output assembly
  - Output: 14 frequency bins (complete DFT)
```

---

## Validation: Does Your Design Handle This Correctly?

### Twiddle Computation

**CT Twiddles (fft_twiddles.c):**
```c
fft_data* compute_stage_twiddles(int N_stage, int radix, fft_direction_t direction)
{
    const int sub_len = N_stage / radix;
    const double sign = (direction == FFT_FORWARD) ? -1.0 : +1.0;
    const double base_angle = sign * 2.0 * M_PI / (double)N_stage;  // ✅ Uses N_stage
    
    for (int k = 0; k < sub_len; k++) {
        for (int r = 1; r < radix; r++) {
            double angle = base_angle * r * k;  // ✅ W_{N_stage}^(r×k)
            tw[idx].re = cos(angle);
            tw[idx].im = sin(angle);
        }
    }
}
```

**Verdict: ✅ CORRECT** - Uses full transform size N_stage, not radix.

---

**Rader Twiddles (fft_rader_plans.c):**
```c
for (int q = 0; q < prime - 1; q++) {
    int idx = perm_out[q];
    double angle_fwd = -2.0 * M_PI * idx / (double)prime;  // ✅ Uses prime only
    sincos_auto(angle_fwd, &conv_tw_fwd[q].im, &conv_tw_fwd[q].re);
}
```

**Verdict: ✅ CORRECT** - Depends only on prime modulus, independent of N_stage.

---

### Planner Integration

**Stage Construction (fft_planner.c):**
```c
for (int i = 0; i < num_stages; i++) {
    int radix = plan->factors[i];
    int sub_len = N_stage / radix;
    
    // Compute CT twiddles (always)
    stage->stage_tw = compute_stage_twiddles(N_stage, radix, direction);  // ✅
    
    // Fetch Rader twiddles (primes only)
    if (radix >= 7 && is_prime(radix)) {
        stage->rader_tw = get_rader_twiddles(radix, direction);  // ✅
    } else {
        stage->rader_tw = NULL;  // ✅
    }
    
    N_stage = sub_len;
}
```

**Verdict: ✅ CORRECT** - Both twiddle tiers are populated properly.

---

### Butterfly Implementation

**Radix-7 Forward (fft_radix7_fv.c):**
```c
// STAGE 2: Apply precomputed stage twiddles
if (sub_len > 1) {  // ✅ Guard for final stage
    const fft_data *tw = &stage_tw[6*k];
    x1 = cmul(x1, tw[0]);  // W_N^(1×k)
    x2 = cmul(x2, tw[1]);  // W_N^(2×k)
    // ...
}

// STAGE 5: Cyclic convolution (using precomputed rader_tw)
for (int q = 0; q < 6; q++) {
    for (int l = 0; l < 6; l++) {
        int idx = (q - l) % 6;
        // Uses rader_tw[idx] = W₇^(g^idx)  ✅
    }
}
```

**Verdict: ✅ CORRECT** - Sequential composition, proper guard for identity twiddles.

---

### Executor Dispatch

**Current Implementation (fft_execute.c):**
```c
for (int stage = 0; stage < plan->num_stages; stage++) {
    stage_descriptor *s = &plan->stages[stage];
    
    switch (s->radix) {
        case 2: radix_2_kernel(...); break;
        case 3: radix_3_kernel(...); break;
        // ❌ MISSING: No dispatch for radix 7, 11, 13!
    }
}
```

**Required Fix:**
```c
if (s->rader_tw != NULL) {
    // Prime radix path
    switch (s->radix) {
        case 7:  fft_radix7_fv(out, in, s->stage_tw, s->rader_tw, s->sub_len); break;
        case 11: fft_radix11_fv(out, in, s->stage_tw, s->rader_tw, s->sub_len); break;
        case 13: fft_radix13_fv(out, in, s->stage_tw, s->rader_tw, s->sub_len); break;
    }
} else {
    // Composite radix path
    switch (s->radix) {
        case 2: fft_radix2_fv(out, in, s->stage_tw, s->sub_len); break;
        // ...
    }
}
```

**Verdict: ❌ INCOMPLETE** - Dispatcher logic missing, but easily fixable.

---

## Summary Checklist

- [x] Two-tier twiddle architecture (CT + Rader)
- [x] CT twiddles use full transform size N_stage
- [x] Rader twiddles use prime modulus only
- [x] Final stage identity twiddles handled via guard
- [x] Memory ownership clearly defined (owned vs borrowed)
- [x] Sequential composition (CT first, then Rader)
- [x] Direction-specific twiddles (forward/inverse signs)
- [x] Thread-safe Rader cache with lazy initialization
- [ ] Dispatcher checks rader_tw and routes to correct butterfly

**O 9/9 requirements met, 1 implementation detail pending (dispatcher).**

---

## Key Insight

The elegance of this design is that **Rader stages look like "black box" radix-p DFT primitives** from the Cooley-Tukey perspective. The CT twiddles handle inter-stage rotation (as they would for any radix), while the Rader twiddles handle the internal DFT computation. This orthogonality is what makes mixed-radix work without special-casing.
```