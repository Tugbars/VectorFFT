# W32 Twiddle Factor Corrections - Quick Reference

## The Math Error Summary

Code review correctly identified that J1, J2, and J3 octave mappings were wrong.

## Complete Correction Table

### Forward Transform (FV): W32[k] = exp(-i·2π·k/32)

| k | Octave | Lane | Base j | θ | Correct (wr, wi) | √2 Optimization |
|---|--------|------|--------|---|------------------|-----------------|
| 0 | J0 | 0 | 0 | 0 | (1, 0) | Skip multiply |
| 1 | J0 | 1 | 1 | π/16 | (c₁, -s₁) | CMUL |
| 2 | J0 | 2 | 2 | 2π/16 | (c₂, -s₂) | CMUL |
| 3 | J0 | 3 | 3 | 3π/16 | (c₃, -s₃) | CMUL |
| **4** | **J0** | **4** | **4** | **4π/16** | **(√2/2, -√2/2)** | **(re+im)√2/2, (im-re)√2/2** |
| 5 | J0 | 5 | 5 | 5π/16 | (c₅, -s₅) | CMUL |
| 6 | J0 | 6 | 6 | 6π/16 | (c₆, -s₆) | CMUL |
| 7 | J0 | 7 | 7 | 7π/16 | (c₇, -s₇) | CMUL |
| **8** | **J1** | **0** | **0** | **8π/16** | **(0, -1)** | **(im, -re)** |
| 9 | J1 | 1 | 1 | 9π/16 | (-s₁, -c₁) | CMUL |
| 10 | J1 | 2 | 2 | 10π/16 | (-s₂, -c₂) | CMUL |
| 11 | J1 | 3 | 3 | 11π/16 | (-s₃, -c₃) | CMUL |
| **12** | **J1** | **4** | **4** | **12π/16** | **(-√2/2, -√2/2)** | **-(re+im)√2/2, -(re-im)√2/2** |
| 13 | J1 | 5 | 5 | 13π/16 | (-s₅, -c₅) | CMUL |
| 14 | J1 | 6 | 6 | 14π/16 | (-s₆, -c₆) | CMUL |
| 15 | J1 | 7 | 7 | 15π/16 | (-s₇, -c₇) | CMUL |
| **16** | **J2** | **0** | **0** | **16π/16** | **(-1, 0)** | **(-re, -im)** |
| 17 | J2 | 1 | 1 | 17π/16 | (-c₁, +s₁) | CMUL |
| 18 | J2 | 2 | 2 | 18π/16 | (-c₂, +s₂) | CMUL |
| 19 | J2 | 3 | 3 | 19π/16 | (-c₃, +s₃) | CMUL |
| **20** | **J2** | **4** | **4** | **20π/16** | **(-√2/2, +√2/2)** | **-(re+im)√2/2, (im-re)√2/2** |
| 21 | J2 | 5 | 5 | 21π/16 | (-c₅, +s₅) | CMUL |
| 22 | J2 | 6 | 6 | 22π/16 | (-c₆, +s₆) | CMUL |
| 23 | J2 | 7 | 7 | 23π/16 | (-c₇, +s₇) | CMUL |
| **24** | **J3** | **0** | **0** | **24π/16** | **(0, +1)** | **(-im, re)** |
| 25 | J3 | 1 | 1 | 25π/16 | (+s₁, +c₁) | CMUL |
| 26 | J3 | 2 | 2 | 26π/16 | (+s₂, +c₂) | CMUL |
| 27 | J3 | 3 | 3 | 27π/16 | (+s₃, +c₃) | CMUL |
| **28** | **J3** | **4** | **4** | **28π/16** | **(+√2/2, +√2/2)** | **(im-re)√2/2, (re+im)√2/2** |
| 29 | J3 | 5 | 5 | 29π/16 | (+s₅, +c₅) | CMUL |
| 30 | J3 | 6 | 6 | 30π/16 | (+s₆, +c₆) | CMUL |
| 31 | J3 | 7 | 7 | 31π/16 | (+s₇, +c₇) | CMUL |

### Backward Transform (BV): W32[k] = exp(+i·2π·k/32)

| k | Octave | Lane | Base j | θ | Correct (wr, wi) | √2 Optimization |
|---|--------|------|--------|---|------------------|-----------------|
| **8** | **J1** | **0** | **0** | **8π/16** | **(0, +1)** | **(-im, re)** |
| **12** | **J1** | **4** | **4** | **12π/16** | **(-√2/2, +√2/2)** | **-(re+im)√2/2, (re-im)√2/2** |
| **16** | **J2** | **0** | **0** | **16π/16** | **(-1, 0)** | **(-re, -im)** |
| **20** | **J2** | **4** | **4** | **20π/16** | **(-√2/2, -√2/2)** | **-(re+im)√2/2, -(im-re)√2/2** |
| **24** | **J3** | **0** | **0** | **24π/16** | **(0, -1)** | **(im, -re)** |
| **28** | **J3** | **4** | **4** | **28π/16** | **(+√2/2, -√2/2)** | **(im-re)√2/2, -(re+im)√2/2** |

(Other lanes follow same pattern as FV but with wi sign flipped)

---

## Critical Errors Fixed

### ❌ Original Code Bugs

**J2 Lane 0 (k=16) - WRONG:**
```c
// Original applied +i rotation → (0, 1)
// Should be -1 rotation → (-1, 0)
// Error: Rotates by 90° instead of 180°
```

**J3 Lane 0 (k=24) - WRONG:**
```c
// Original applied (re, -im) 
// FV should be +i → (0, 1)
// BV should be -i → (0, -1)
// Error: Wrong rotation direction
```

**J1 Lane 4 (k=12) - WRONG:**
```c
// Original may have wrong signs in √2/2 optimization
// FV needs (-√2/2, -√2/2)
// BV needs (-√2/2, +√2/2)
```

---

## ✅ Fixed Implementation

### J0 (k=0..7) - Lane 4 Optimization
```c
// FV: W32[4] = exp(-i·π/4) = √2/2 - i·√2/2
tr = _mm512_mul_pd(_mm512_add_pd(x_re[4], x_im[4]), SQRT2_2);
ti = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);

// BV: W32[4] = exp(+i·π/4) = √2/2 + i·√2/2
tr = _mm512_mul_pd(_mm512_add_pd(x_re[4], x_im[4]), SQRT2_2);
ti = _mm512_mul_pd(_mm512_sub_pd(x_re[4], x_im[4]), SQRT2_2);
```

### J1 (k=8..15) - Lane 0 and 4 Optimizations
```c
// Lane 0: FV: W32[8] = exp(-i·π/2) = -i
tr = x_im[0];
ti = _mm512_xor_pd(x_re[0], NEG_ZERO);  // -re

// Lane 0: BV: W32[8] = exp(+i·π/2) = +i
tr = _mm512_xor_pd(x_im[0], NEG_ZERO);  // -im
ti = x_re[0];

// Lane 4: FV: W32[12] = exp(-i·3π/4) = -√2/2 - i·√2/2
tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2);
ti = _mm512_mul_pd(_mm512_xor_pd(_mm512_sub_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2);

// Lane 4: BV: W32[12] = exp(+i·3π/4) = -√2/2 + i·√2/2
tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2);
ti = _mm512_mul_pd(_mm512_sub_pd(x_re[4], x_im[4]), SQRT2_2);
```

### J2 (k=16..23) - Lane 0 and 4 Optimizations
```c
// Lane 0: W32[16] = exp(±i·π) = -1 (SAME FOR FV AND BV!)
x_re[0] = _mm512_xor_pd(x_re[0], NEG_ZERO);
x_im[0] = _mm512_xor_pd(x_im[0], NEG_ZERO);

// Lane 4: FV: W32[20] = exp(-i·5π/4) = -√2/2 + i·√2/2
tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2);
ti = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);

// Lane 4: BV: W32[20] = exp(+i·5π/4) = -√2/2 - i·√2/2
tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2);
ti = _mm512_mul_pd(_mm512_xor_pd(_mm512_sub_pd(x_im[4], x_re[4]), NEG_ZERO), SQRT2_2);
```

### J3 (k=24..31) - Lane 0 and 4 Optimizations
```c
// Lane 0: FV: W32[24] = exp(-i·3π/2) = +i
tr = _mm512_xor_pd(x_im[0], NEG_ZERO);  // -im
ti = x_re[0];

// Lane 0: BV: W32[24] = exp(+i·3π/2) = -i
tr = x_im[0];
ti = _mm512_xor_pd(x_re[0], NEG_ZERO);  // -re

// Lane 4: FV: W32[28] = exp(-i·7π/4) = +√2/2 + i·√2/2
tr = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);
ti = _mm512_mul_pd(_mm512_add_pd(x_re[4], x_im[4]), SQRT2_2);

// Lane 4: BV: W32[28] = exp(+i·7π/4) = +√2/2 - i·√2/2
tr = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);
ti = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2);
```

---

## Verification Formulas

To verify each twiddle factor:
```c
// For k in 0..31:
double theta = direction * 2.0 * M_PI * k / 32.0;
double wr_expected = cos(theta);
double wi_expected = sin(theta);

// Where direction = -1 for FV, +1 for BV
```

Special cases:
```c
W32[0]  = (1, 0)        // Unity
W32[8]  = (0, ±1)       // ±i
W32[16] = (-1, 0)       // -1
W32[24] = (0, ∓1)       // ∓i
W32[4]  = (±√2/2, ...)  // 45° rotations
```

---

## Common Patterns

### Quarter-Turn Identity Application
```
θ + π/2:   cos(θ+π/2) = -sin(θ),  sin(θ+π/2) = +cos(θ)
θ + π:     cos(θ+π)   = -cos(θ),  sin(θ+π)   = -sin(θ)
θ + 3π/2:  cos(θ+3π/2) = +sin(θ), sin(θ+3π/2) = -cos(θ)
```

### Sign Flip for Direction
```
FV (negative exponent): wi = -sin(θ)
BV (positive exponent): wi = +sin(θ)
```

This explains why the implementation has separate `_FV` and `_BV` macros.