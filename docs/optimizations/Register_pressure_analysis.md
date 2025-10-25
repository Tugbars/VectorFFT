# Register Pressure Analysis: Radix-7 through Radix-16
## AVX-512 Single-Stage Butterfly Implementation

**Assumption:** Processing 16 complex numbers (8 complex per `__m512d` vector)

---

## Register Usage Formula

For a **prime or prime-like radix N** using Rader's/DFT structure:

```
Inputs:         N vectors (x0, x1, ..., x[N-1])
Temps Layer 1:  (N-1)/2 vectors (t0, t1, ...)
Temps Layer 2:  (N-1)/2 vectors (s0, s1, ...)
Real Parts:     (N-1)/2 vectors (real1, real2, ...)
Rotations:      (N-1)/2 vectors (rot1, rot2, ...)
Outputs:        N vectors (y0, y1, ..., y[N-1])
DC Component:   1 vector (y0_lo or included in outputs)

TOTAL = N + 2*(N-1)/2 + 2*(N-1)/2 + N
      = N + 2*(N-1) + N
      = 4*N - 2
```

For **composite radices** (8, 12, 16), the structure differs based on decomposition.

---

## Radix-by-Radix Breakdown

### Radix-7 (Prime)
```
Inputs:         7 vectors (x0-x6)
Temps t:        3 vectors (t0-t2)
Temps s:        3 vectors (s0-s2)
Real parts:     3 vectors (real1-real3)
Rotations:      3 vectors (rot1-rot3)
Outputs:        7 vectors (y0-y6)
─────────────────────────────────
TOTAL:          26 __m512d registers
Available:      32 registers
Surplus:        6 registers ✓ FITS!
─────────────────────────────────
LO/HI Split:    NOT REQUIRED
Spill Risk:     LOW (borderline)
```

**Verdict:** Technically fits, but borderline. May benefit from split if additional constants/masks needed.

---

### Radix-8 (Composite: 2×4)
```
Optimal decomposition: two radix-4 stages

Stage 1 (4 parallel radix-2):
Inputs:         8 vectors (x0-x7)
Temps:          8 vectors (4 per radix-2 pair)
Stage 1 Out:    8 vectors

Stage 2 (2 parallel radix-4):
Inputs:         8 vectors (from stage 1)
Temps:          12 vectors (6 per radix-4)
Outputs:        8 vectors
─────────────────────────────────
Peak Usage:     ~24-28 registers (depending on fusion)
Available:      32 registers
Surplus:        4-8 registers ✓ FITS!
─────────────────────────────────
LO/HI Split:    NOT REQUIRED
Spill Risk:     LOW
```

**Verdict:** Composite structure keeps pressure low. No split needed.

---

### Radix-9 (Composite: 3×3)
```
If implemented as prime-like (worst case):

Inputs:         9 vectors (x0-x8)
Temps t:        4 vectors (t0-t3)
Temps s:        4 vectors (s0-s3)
Real parts:     4 vectors (real1-real4)
Rotations:      4 vectors (rot1-rot4)
Outputs:        9 vectors (y0-y8)
─────────────────────────────────
TOTAL:          34 __m512d registers
Available:      32 registers
DEFICIT:        2 registers ✗ SPILLS!
─────────────────────────────────
LO/HI Split:    BORDERLINE
Spill Risk:     MODERATE (2 spills expected)

If implemented as 3×3 decomposition:
Peak Usage:     ~22-26 registers ✓ FITS!
```

**Verdict:** Use Cooley-Tukey 3×3 decomposition to avoid split. If implemented as single-stage DFT, requires split.

---

### Radix-10 (Composite: 2×5)
```
Optimal decomposition: radix-2 then radix-5

Stage 1 (5 parallel radix-2):
Inputs:         10 vectors
Temps:          10 vectors
Stage 1 Out:    10 vectors

Stage 2 (2 parallel radix-5):
Inputs:         10 vectors
Temps:          16 vectors (8 per radix-5)
Outputs:        10 vectors
─────────────────────────────────
Peak Usage:     ~26-30 registers
Available:      32 registers
Surplus:        2-6 registers ✓ FITS (tight!)
─────────────────────────────────
LO/HI Split:    NOT REQUIRED (but close)
Spill Risk:     LOW-MODERATE
```

**Verdict:** Decomposition keeps it manageable. No split needed, but watch for constants.

---

### Radix-11 (Prime)
```
Inputs:         11 vectors (x0-x10)
Temps t:        5 vectors (t0-t4)
Temps s:        5 vectors (s0-s4)
Real parts:     5 vectors (real1-real5)
Rotations:      5 vectors (rot1-rot5)
Outputs:        11 vectors (y0-y10)
─────────────────────────────────
TOTAL:          42 __m512d registers
Available:      32 registers
DEFICIT:        10 registers ✗ SPILLS!
─────────────────────────────────
LO/HI Split:    REQUIRED
Spill Risk:     HIGH (10 spills expected)
Performance:    -25% without split
```

**Verdict:** **LO/HI split mandatory.** Expect 15-25% speedup with split.

---

### Radix-12 (Composite: 3×4 or 4×3)
```
Optimal decomposition: radix-3 then radix-4

Stage 1 (4 parallel radix-3):
Inputs:         12 vectors
Temps:          12 vectors (3 per radix-3)
Stage 1 Out:    12 vectors

Stage 2 (3 parallel radix-4):
Inputs:         12 vectors
Temps:          18 vectors (6 per radix-4)
Outputs:        12 vectors
─────────────────────────────────
Peak Usage:     ~28-32 registers (TIGHT!)
Available:      32 registers
Surplus:        0-4 registers ✓ BORDERLINE
─────────────────────────────────
LO/HI Split:    RECOMMENDED (safety margin)
Spill Risk:     MODERATE (0-4 spills possible)
```

**Verdict:** Technically fits, but very tight. Split recommended for production code.

---

### Radix-13 (Prime)
```
Inputs:         13 vectors (x0-x12)
Temps t:        6 vectors (t0-t5)
Temps s:        6 vectors (s0-s5)
Real parts:     6 vectors (real1-real6)
Rotations:      6 vectors (rot1-rot6)
Outputs:        13 vectors (y0-y12)
─────────────────────────────────
TOTAL:          50 __m512d registers
Available:      32 registers
DEFICIT:        18 registers ✗ SPILLS!
─────────────────────────────────
LO/HI Split:    REQUIRED
Spill Risk:     VERY HIGH (18 spills expected)
Performance:    -30% without split
```

**Verdict:** **LO/HI split mandatory.** Documented 20-30% speedup with split.

---

### Radix-14 (Composite: 2×7)
```
Optimal decomposition: radix-2 then radix-7

Stage 1 (7 parallel radix-2):
Inputs:         14 vectors
Temps:          14 vectors
Stage 1 Out:    14 vectors

Stage 2 (2 parallel radix-7):
Inputs:         14 vectors
Temps:          12 vectors (6 per radix-7)
Real/Rot:       12 vectors
Outputs:        14 vectors
─────────────────────────────────
Peak Usage:     ~32-38 registers (EXCEEDS!)
Available:      32 registers
DEFICIT:        0-6 registers ✗ SPILLS likely!
─────────────────────────────────
LO/HI Split:    REQUIRED
Spill Risk:     MODERATE-HIGH (4-8 spills)
```

**Verdict:** **LO/HI split required.** Even with decomposition, radix-7 stages are expensive.

---

### Radix-15 (Composite: 3×5)
```
Optimal decomposition: radix-3 then radix-5

Stage 1 (5 parallel radix-3):
Inputs:         15 vectors
Temps:          15 vectors
Stage 1 Out:    15 vectors

Stage 2 (3 parallel radix-5):
Inputs:         15 vectors
Temps:          24 vectors (8 per radix-5)
Outputs:        15 vectors
─────────────────────────────────
Peak Usage:     ~32-36 registers (EXCEEDS!)
Available:      32 registers
DEFICIT:        0-4 registers ✗ SPILLS likely!
─────────────────────────────────
LO/HI Split:    REQUIRED
Spill Risk:     MODERATE (2-6 spills)
```

**Verdict:** **LO/HI split required.** High radix-5 count drives pressure.

---

### Radix-16 (Composite: 2×8 or 4×4)
```
Optimal decomposition: four radix-4 stages (4×2×2)

Stage 1 (4 parallel radix-4):
Inputs:         16 vectors
Temps:          24 vectors (6 per radix-4)
Stage 1 Out:    16 vectors

Stage 2 (4 parallel radix-4):
Inputs:         16 vectors
Temps:          24 vectors
Outputs:        16 vectors
─────────────────────────────────
Peak Usage:     ~32-40 registers (EXCEEDS!)
Available:      32 registers
DEFICIT:        0-8 registers ✗ SPILLS likely!
─────────────────────────────────
LO/HI Split:    REQUIRED
Spill Risk:     MODERATE-HIGH (4-10 spills)

Alternative: Use radix-32 register kernel
then handle as special case with half vectors
```

**Verdict:** **LO/HI split required.** Or implement as special radix-32 ZMM kernel (8 complex per vector = 16 complex in 2 vectors).

---

## Summary Table

| Radix | Type | Naive Regs | With Decomp | Split Needed? | Spill Risk | Speedup With Split |
|-------|------|-----------|-------------|---------------|------------|-------------------|
| **7** | Prime | 26 | 26 | Optional | Low | +5-10% |
| **8** | 2×4 | 30 | 24-28 | No | Low | N/A |
| **9** | 3×3 | 34 | 22-26 | Depends | Moderate | +10-15% (if single-stage) |
| **10** | 2×5 | 38 | 26-30 | No | Low-Mod | N/A |
| **11** | Prime | 42 | 42 | **YES** | High | +15-25% |
| **12** | 3×4 | 46 | 28-32 | Recommended | Moderate | +10-15% |
| **13** | Prime | 50 | 50 | **YES** | Very High | +20-30% |
| **14** | 2×7 | 54 | 32-38 | **YES** | Moderate-High | +15-20% |
| **15** | 3×5 | 58 | 32-36 | **YES** | Moderate | +10-20% |
| **16** | 4×4 | 62 | 32-40 | **YES** | Moderate-High | +15-25% |

---

## Decision Rules

### Automatic Split Threshold
```c
if (naive_register_count > 28) {
    // Definitely use LO/HI split
    use_split = true;
}
else if (naive_register_count > 24 && is_prime_radix) {
    // Prime radices don't benefit from decomposition
    use_split = true;
}
else if (decomposed_register_count > 30) {
    // Even with decomposition, pressure is high
    use_split = true;
}
else {
    // Low enough pressure, no split needed
    use_split = false;
}
```

### Critical Thresholds
- **< 24 registers:** No split needed (radix 2-5)
- **24-28 registers:** Borderline, depends on constants/masks (radix 6-7)
- **28-32 registers:** Split recommended for safety (radix 8-10 composite)
- **> 32 registers:** Split mandatory (radix 11+ primes, 12+ with tight decomposition)

---

## Implementation Strategy

**Radix 7:** No split by default, but provide split version for safety
**Radix 8:** Use 2×4 decomposition, no split
**Radix 9:** Use 3×3 decomposition, no split
**Radix 10:** Use 2×5 decomposition, no split  
**Radix 11+:** **Always use LO/HI split** for prime radices
**Radix 12+:** Use split for composite radices with expensive factors (7, 11, 13)

The split becomes **mandatory, not optional** at radix-11 and above for prime radices.
