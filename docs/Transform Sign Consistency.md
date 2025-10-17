# Transform Sign Consistency in Mixed-Radix FFT Implementations

## Table of Contents

1. [The Sign Convention Problem](#the-sign-convention-problem)
2. [Where Sign Errors Hide](#where-sign-errors-hide)
3. [The Rader-Specific Challenge](#the-rader-specific-challenge)
4. [Direction-Dependent Operations](#direction-dependent-operations)
5. [Verification Strategy](#verification-strategy)

---

## The Sign Convention Problem

### DFT Definition

The Discrete Fourier Transform has two conventions that differ **only in the sign of the exponent**:

| Transform | Formula | Exponent Sign |
|-----------|---------|---------------|
| **Forward FFT** | X[k] = Σ x[n] · exp(**-2πi**·kn/N) | **Negative (-)** |
| **Inverse FFT** | x[n] = (1/N) Σ X[k] · exp(**+2πi**·kn/N) | **Positive (+)** |

> **Key Point**: The entire transform's correctness depends on using the correct sign throughout all stages.

### The Mixed-Radix Challenge

| Single-Algorithm FFT | Mixed-Radix FFT |
|---------------------|-----------------|
| ✅ One twiddle table | ⚠️ Multiple twiddle sources |
| ✅ Conjugate once for inverse | ⚠️ Some twiddles need recomputation |
| ✅ Uniform sign handling | ⚠️ Algorithm-specific sign rules |

In mixed-radix implementations combining different algorithms (Cooley-Tukey, Rader, Good-Thomas), each algorithm may:

- ✦ Apply twiddles at different computation stages
- ✦ Use different mathematical structures (convolution vs. direct multiplication)
- ✦ Require different sign-handling strategies

**⚠️ Critical**: If even **one stage** has inconsistent sign convention, the entire transform fails.

---

## Where Sign Errors Hide

### Why Sign Bugs Are Dangerous

| Property | Impact |
|----------|--------|
| **Magnitudes often correct** | Output "looks reasonable" at first glance |
| **Symmetries preserved** | Real input still gives Hermitian output |
| **Phase corruption only** | Errors appear as noise, not obvious failures |
| **Length-specific** | May only fail for certain N values |

> **💡 Detection Difficulty**: Sign errors can pass basic sanity checks while producing completely wrong phase information.

---

### Common Error Locations

#### ❌ Error #1: Wrong Base Angle

```c
// ✅ CORRECT: Negative angle for forward FFT
const double theta = -2.0 * M_PI / (double)N;

// ❌ WRONG: Positive angle (this is inverse FFT!)
const double theta = 2.0 * M_PI / (double)N;
```

**Implementation in the library**:

```c
// In build_twiddles_linear()
const double theta = -2.0 * M_PI / (double)N;  // ✅ Forward convention

// In fft_init() for inverse FFT
if (transform_direction == -1) {
    for (int i = 0; i < twiddle_count; i++) {
        fft_config->twiddles[i].im = -fft_config->twiddles[i].im;  // Conjugate
    }
}
```

---

#### ❌ Error #2: Rotation Direction

Many radix butterflies use **multiplication by ±i** (90° rotation):

| Transform | Operation | Formula | Result |
|-----------|-----------|---------|--------|
| **Forward** | × (-i) | (a + bi) · (-i) | b - ai |
| **Inverse** | × (+i) | (a + bi) · (+i) | -b + ai |

**Correct implementation**:

```c
// ✅ Direction-dependent rotation mask
__m256d rot_mask = (transform_sign == 1) 
    ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)   // Forward: -i
    : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);  // Inverse: +i
```

---

#### ✅ No Error: Per-Stage Twiddles

Standard DIT (Decimation In Time) twiddles are applied before butterfly computation:

```c
// ✅ These inherit sign from global twiddle table
for (int lane = 1; lane < radix; ++lane) {
    x[lane] = cmul(x[lane], stage_tw[k * (radix-1) + (lane-1)]);
}
```

| Property | Behavior |
|----------|----------|
| **Source** | Derived from `fft_obj->twiddles[]` |
| **Sign handling** | Already conjugated for inverse FFT |
| **Stage-specific logic** | ✅ **Not needed** |

---

## The Rader-Specific Challenge

### Why Rader Requires Special Handling

| Standard Cooley-Tukey | Rader's Algorithm |
|-----------------------|-------------------|
| Twiddles: W^k = exp(-2πik/N) | Twiddles: exp(-2πi·g^(-q)/N) |
| Linear index progression | Permuted by primitive root |
| Conjugate once globally | ⚠️ **Must recompute per direction** |
| Works with global table | ⚠️ **Needs separate computation** |

> **⚠️ Critical Difference**: Rader convolution twiddles have a fundamentally different structure than standard FFT twiddles.

---

### Rader Twiddle Computation

The library computes Rader convolution twiddles **at runtime** in each butterfly call:

```c
// ✅ Sign flips with transform direction
const double base_angle = (transform_sign == 1 ? -2.0 : +2.0) * M_PI / 7.0;
                          //                    ↑ Forward    ↑ Inverse

__m256d tw_brd[6];
for (int q = 0; q < 6; ++q) {
    const int output_index = out_perm[q];  // [1, 5, 4, 6, 2, 3]
    double angle = output_index * base_angle;
    sincos(angle, &tw.im, &tw.re);
    tw_brd[q] = _mm256_set_pd(tw.im, tw.re, tw.im, tw.re);
}
```

---

### Why Runtime Computation?

| Cannot Use Global Table Because... | Solution |
|------------------------------------|----------|
| Different mathematical structure (convolution kernel) | Compute separately |
| Output permutation follows g^(-1) powers, not linear | Use permutation array |
| Cannot simply conjugate for inverse direction | Recompute with flipped sign |

**Verification formula**:

| Direction | Twiddle Formula |
|-----------|-----------------|
| **Forward** | twiddle[q] = exp(**-2πi** · output_perm[q] / N) |
| **Inverse** | twiddle[q] = exp(**+2πi** · output_perm[q] / N) |

**Example for N=7**:

```
output_perm = [1, 5, 4, 6, 2, 3]

Forward:
  twiddle[0] = exp(-2πi · 1/7)
  twiddle[1] = exp(-2πi · 5/7)
  twiddle[2] = exp(-2πi · 4/7)
  ...

Inverse: Negate all angles
  twiddle[0] = exp(+2πi · 1/7)
  twiddle[1] = exp(+2πi · 5/7)
  ...
```

---

## Direction-Dependent Operations

### Operations by Category

| Operation Type | Transform Direction | Example |
|----------------|---------------------|---------|
| **Must flip with direction** | ✅ Direction-dependent | W₈ twiddles, ±i rotations |
| **Independent of direction** | ⚠️ Never check direction | DC sum, Nyquist bin |

---

### Example: W₈ Twiddles (Radix-8)

The special twiddles for radix-8 split-radix:

| Twiddle | Forward FFT | Inverse FFT |
|---------|-------------|-------------|
| **W₈⁰** | 1 | 1 |
| **W₈¹** | (1/√2)(1 - i) | (1/√2)(1 + i) |
| **W₈²** | -i | +i |
| **W₈³** | (-1/√2)(1 + i) | (-1/√2)(1 - i) |

**Implementation**:

```c
const double c8 = 0.7071067811865476;  // 1/√2

// W₈^1 multiplication
if (transform_sign == 1) {  // ✅ Forward
    o[1].re = (r + i) * c8;  // (1/√2)(1 - i) multiplication
    o[1].im = (i - r) * c8;
} else {                     // ✅ Inverse
    o[1].re = (r - i) * c8;  // (1/√2)(1 + i) multiplication
    o[1].im = (i + r) * c8;
}
```

---

### Direction-Independent Operations

These **never** check `transform_sign`:

```c
// ✅ DC component: Always sum
y[0] = x[0] + x[1] + x[2] + ... + x[N-1];

// ✅ Nyquist (even N): Always alternate signs
y[N/2] = x[0] - x[1] + x[2] - x[3] + ...;
```

---

### Decision Rule

| Does operation involve... | Check `transform_sign`? |
|---------------------------|-------------------------|
| exp(±2πi·θ) | ✅ **YES** |
| Rotation by ±90° (×i) | ✅ **YES** |
| Real-only arithmetic | ❌ **NO** |
| Symmetry properties | ❌ **NO** |

---

## Verification Strategy

### Test Suite Overview

| Test Type | What It Catches | Priority |
|-----------|-----------------|----------|
| **Impulse Response** | Sign and permutation errors | 🔴 Critical |
| **Parseval's Theorem** | Normalization and energy errors | 🟡 Important |
| **Convolution Theorem** | Forward/inverse consistency | 🟡 Important |
| **Reference Comparison** | All numerical errors | 🔴 Critical |
| **Symmetry Test** | Phase corruption | 🟢 Useful |
| **Per-Radix Isolation** | Stage interaction bugs | 🔴 Critical |

---

### Test #1: Impulse Response (Most Important)

**Concept**: An impulse should reconstruct itself after FFT → IFFT.

```c
// ✅ Correct behavior
x[k] = δ[k]           // Input: impulse at position k
X = FFT(x)            // Forward transform
y = IFFT(X)           // Inverse transform
→ y[k] = δ[k]         // Should recover original impulse

// ❌ Sign error indicator
→ y ≠ δ[k]            // Failed reconstruction means wrong signs
```

**Implementation**:

```c
for (int k = 0; k < N; k++) {
    fft_data input[N] = {0};
    input[k].re = 1.0;  // Impulse at position k
    
    fft_data forward[N], inverse[N];
    fft_exec(fft_fwd, input, forward);
    fft_exec(fft_inv, forward, inverse);
    
    // ✅ Check: inverse[k] ≈ 1.0, all others ≈ 0.0
    assert(fabs(inverse[k].re - 1.0) < 1e-10);
}
```

---

### Test #2: Parseval's Theorem

**Energy must be conserved**:

```
Σ|x[n]|² = (1/N) Σ|X[k]|²
```

| If Test Fails... | Likely Cause |
|------------------|--------------|
| Energy too large | Missing normalization |
| Energy too small | Double normalization |
| Sign-dependent error | Wrong transform direction |

---

### Test #3: Convolution Theorem

**Concept**: Convolution via FFT should match direct convolution.

```
x₁ ⊗ x₂ = IFFT(FFT(x₁) · FFT(x₂))
```

**Sign errors manifest as**:

| Symptom | Interpretation |
|---------|----------------|
| Magnitude correct, phase wrong | Sign error in forward **or** inverse |
| Complete garbage | Sign errors in **both** directions |

---

### Test #4: Cross-Validation

**Gold standard**: Compare against trusted reference (FFTW, NumPy).

```python
import numpy as np

# Generate random complex input
x = np.random.randn(56) + 1j * np.random.randn(56)

# Reference implementation
X_reference = np.fft.fft(x)

# Your library
X_library = your_fft_library(x)

# ✅ Should match to machine precision
assert np.allclose(X_reference, X_library, rtol=1e-10, atol=1e-12)
```

---

### Test #5: Hermitian Symmetry

**For real input**, output must satisfy:

```
X[k] = conj(X[N-k])  for k = 1, 2, ..., N-1
```

| Symmetry Holds | Symmetry Broken |
|----------------|-----------------|
| ✅ Real/imaginary structure correct | ❌ Phase errors from wrong signs |

---

### Test #6: Per-Radix Isolation

**Test each radix independently**, then test combinations:

| Test Sequence | Purpose |
|---------------|---------|
| N = 7, 49, 343 | Verify pure Radix-7 (Rader) |
| N = 8, 64, 512 | Verify pure Radix-8 (split-radix) |
| N = 56, 448, 3136 | Verify Radix-7 × Radix-8 interaction |

> **💡 Debugging Strategy**: If pure radices work but combinations fail, the bug is in **stage composition**, not individual butterflies.

---

## Summary

### Critical Checklist

| Component | Requirement | Status in Library |
|-----------|-------------|-------------------|
| **Base twiddle angle** | Must be negative for forward FFT | ✅ `-2π/N` |
| **Inverse FFT handling** | Conjugate global twiddles once | ✅ Conjugate in `fft_init()` |
| **Rader twiddles** | Recompute with flipped sign | ✅ Runtime computation |
| **W₈ special twiddles** | Check `transform_sign` | ✅ Conditional logic |
| **Rotation operations** | Direction-dependent (±i) | ✅ Mask flips with sign |
| **DC/Nyquist bins** | Direction-independent | ✅ No sign checks |

---

### Key Principles

1. **➊ Global consistency**: All stages must agree on DFT sign convention
2. **➋ Algorithm awareness**: Rader needs special sign handling separate from Cooley-Tukey
3. **➌ Explicit direction checks**: Any `exp(±2πi·θ)` or rotation must test `transform_sign`
4. **➍ Comprehensive testing**: Impulse response and reference comparison are non-negotiable

---

### Why This Matters

Sign errors are **insidious**:

| What Makes Them Dangerous | Why |
|---------------------------|-----|
| Preserve magnitudes | Pass basic validation |
| Maintain symmetries | Real → Hermitian still holds |
| Length-specific | May only fail for N = 15, 35, etc. |
| Phase-only corruption | Appear as "numerical noise" |

**Only rigorous testing with impulse responses, convolution theorems, and cross-validation can ensure correctness across all supported transform lengths and directions.**

Author: Tugbars Heptaskin
