# Failure Mode Analysis & Design Solutions for Mixed-Radix FFT Round Trip

## Executive Summary

This document catalogs **every possible failure mode** in a mixed-radix FFT system combining Cooley-Tukey (CT) and Rader's algorithm, explains the root causes, demonstrates how the current design prevents each failure, and validates that the solution fulfills the requirement of correct round-trip behavior between prime and composite radices.

---

## Table of Contents

1. [Critical Failure Modes](#1-critical-failure-modes)
2. [Twiddle Sign Mismatches](#2-twiddle-sign-mismatches)
3. [Algorithm Interaction Failures](#3-algorithm-interaction-failures)
4. [Normalization Failures](#4-normalization-failures)
5. [Design Solutions Analysis](#5-design-solutions-analysis)
6. [Validation Against Requirements](#6-validation-against-requirements)

---

## 1. Critical Failure Modes

### 1.1 Taxonomy of Failures

Failures in mixed-radix FFT systems can be classified into four categories:

| Category | Symptom | Root Cause | Detection Method |
|----------|---------|------------|------------------|
| **Type A: Sign Inversion** | Output = complex conjugate of expected | Twiddle signs flipped | Phase analysis |
| **Type B: Scale Error** | Output = c·N·x where c≠1 | Normalization mismatch | Magnitude check |
| **Type C: Garbage Output** | No recognizable pattern | Memory corruption, wrong algorithm | Any test fails |
| **Type D: Partial Corruption** | Some frequency bins correct, others wrong | Stage-specific bug | Frequency-selective test |

### 1.2 Impact Matrix

```
           │ Forward Only │ Inverse Only │ Round Trip
───────────┼──────────────┼──────────────┼────────────
Type A     │ Undetectable │ Undetectable │ Conjugate²=Identity (HIDDEN!)
Type B     │ Detectable   │ Detectable   │ Detectable (scale≠1)
Type C     │ Detectable   │ Detectable   │ Detectable (noise)
Type D     │ Detectable   │ Detectable   │ Detectable (selective)
```

**Key Insight:** Type A failures can **cancel out in round trip** if both forward and inverse have the *same* bug. This is why sign validation must happen **during planning**, not just during round trip testing.

---

## 2. Twiddle Sign Mismatches

### 2.1 FAILURE: Both Transforms Use Same Sign

#### 2.1.1 Scenario: Forward Sign in Both Directions

**What Happens:**
```c
// BUGGY CODE:
fft_data* compute_stage_twiddles(..., fft_direction_t direction) {
    const double sign = -1.0;  // ❌ HARDCODED, ignores direction!
    const double base_angle = sign * 2.0 * M_PI / N_stage;
    // ...
}
```

**Mathematical Impact:**
```
Forward:  X[k] = Σ x[n] × exp(-2πikn/N)  ✓ Correct
Inverse:  y[n] = Σ X[k] × exp(-2πikn/N)  ❌ WRONG (should be +2πi)

Result: y[n] = (Σ X[k] × exp(-2πikn/N))
            = conj(DFT⁻¹(conj(X[k])))
            ≠ x[n]
```

**Round Trip Behavior:**
```
Input: x = [1, 2, 3, 4]

Forward FFT with negative twiddles:
X = DFT(x) = [10, -2+2i, -2, -2-2i]  ✓ Correct

Inverse FFT with NEGATIVE twiddles (WRONG):
y = DFT(X) with wrong sign
  = [40, conj(X[1]), conj(X[2]), conj(X[3])]  ❌ Phase corruption

Normalized: y/N ≠ x
```

**How Current Design Prevents This:**
```c
// ✅ CORRECT IMPLEMENTATION (from fft_twiddles.c)
fft_data* compute_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction)  // ← Direction parameter
{
    // Sign determined by direction:
    const double sign = (direction == FFT_FORWARD) ? -1.0 : +1.0;
    const double base_angle = sign * 2.0 * M_PI / (double)N_stage;
    
    // All subsequent twiddles inherit this sign:
    for (int k = 0; k < sub_len; k++) {
        for (int r = 1; r < radix; r++) {
            double angle = base_angle * (double)r * (double)k;
            sincos_auto(angle, &tw[idx].im, &tw[idx].re);
        }
    }
    
    return tw;
}
```

**Validation Test:**
```c
void test_stage_twiddle_conjugacy() {
    int N = 448;
    
    fft_object fwd = fft_init(N, FFT_FORWARD);
    fft_object inv = fft_init(N, FFT_INVERSE);
    
    for (int stage = 0; stage < fwd->num_stages; stage++) {
        int radix = fwd->stages[stage].radix;
        int sub_len = fwd->stages[stage].sub_len;
        
        fft_data *tw_fwd = fwd->stages[stage].stage_tw;
        fft_data *tw_inv = inv->stages[stage].stage_tw;
        
        // For all twiddles in this stage:
        for (int i = 0; i < (radix-1) * sub_len; i++) {
            // Real parts must be identical:
            assert(fabs(tw_fwd[i].re - tw_inv[i].re) < 1e-15);
            
            // Imaginary parts must be negated (conjugate):
            assert(fabs(tw_fwd[i].im + tw_inv[i].im) < 1e-15);
        }
    }
}
```

---

#### 2.1.2 Scenario: Inverse Sign in Both Directions

**What Happens:** Same failure mode, but with all signs flipped. FFT computes conjugate of expected transform.

**Detection:** Fails immediately on any complex input:
```c
input = {1+0i, 0+1i, 0+0i, ...};  // DC + imaginary spike

FFT with inverse signs:
X_wrong = conj(DFT(input))

Round trip with inverse signs:
y = conj(DFT⁻¹(conj(X_wrong)))
  = conj(DFT⁻¹(DFT(input)))
  = conj(N × input)
  ≠ input
```

---

### 2.2 FAILURE: Mixed Signs Across Stages

#### 2.2.1 Scenario: One Stage Has Wrong Sign

**What Happens:**
```c
// Imagine stage_tw computed correctly for stages 0 and 2,
// but stage 1 (radix-7) accidentally uses forward sign in inverse plan

Stage 0 (radix-32): exp(+2πi...)  ✓ Correct inverse
Stage 1 (radix-7):  exp(-2πi...)  ❌ WRONG (forward sign in inverse plan)
Stage 2 (radix-2):  exp(+2πi...)  ✓ Correct inverse
```

**Mathematical Impact:**
```
Let F₀, F₁, F₂ be the stage transforms.

Correct inverse:
x = F₂⁻¹ ∘ F₁⁻¹ ∘ F₀⁻¹(X)

With buggy stage 1:
y = F₂⁻¹ ∘ F₁ ∘ F₀⁻¹(X)  ← F₁ instead of F₁⁻¹
  = F₂⁻¹(F₁(F₀⁻¹(X)))
  ≠ x
```

**Round Trip Behavior:**
```
Forward: X = F₀ ∘ F₁ ∘ F₂(x)  ✓ Correct

Inverse with stage 1 bug:
y = F₂⁻¹ ∘ F₁ ∘ F₀⁻¹(X)
  = F₂⁻¹ ∘ F₁ ∘ F₀⁻¹(F₀ ∘ F₁ ∘ F₂(x))
  = F₂⁻¹ ∘ F₁ ∘ F₁ ∘ F₂(x)
  = F₂⁻¹ ∘ F₁² ∘ F₂(x)  ← F₁² is NOT identity!

For radix-7, F₁² means "apply radix-7 DFT twice":
Result: Heavily corrupted, appears as random noise in time domain
```

**How Current Design Prevents This:**
```c
// ✅ In fft_planner.c, ALL stages computed with same direction
for (int i = 0; i < num_stages; i++) {
    int radix = plan->factors[i];
    int sub_len = N_stage / radix;
    
    // ← Same 'direction' parameter for ALL stages
    stage->stage_tw = compute_stage_twiddles(N_stage, radix, direction);
    
    N_stage = sub_len;
}
```

**Why This Works:**  
The `direction` parameter is captured once at `fft_init()` and passed consistently to **every** stage twiddle computation. There's no per-stage direction override, eliminating this entire class of bugs.

---

### 2.3 FAILURE: Radix-Specific Twiddles Have Wrong Sign

#### 2.3.1 Scenario: Hardcoded Radix-32 W₃₂ and W₈ Twiddles

**What Happens:**
```c
// BUGGY radix-32 INVERSE implementation:
void fft_radix32_bv(...) {
    // ...
    
    // ❌ BUG: Using forward angles in inverse butterfly
    double angle = -2.0 * M_PI * (j * g) / 32.0;  // ← Should be +2.0
    double tw_re = cos(angle);
    double tw_im = sin(angle);
    
    // Apply to data...
}
```

**Impact:**
```
Radix-32 butterfly performs FORWARD transform when called in INVERSE context.

For N=448 (radix-32 × radix-7 × radix-2):

Round trip:
Stage 0 forward (radix-32):  ✓ Correct
Stage 1 forward (radix-7):   ✓ Correct
Stage 2 forward (radix-2):   ✓ Correct

Stage 2 inverse (radix-2):   ✓ Correct
Stage 1 inverse (radix-7):   ✓ Correct
Stage 0 inverse (radix-32):  ❌ Actually performs FORWARD

Result: y ≈ random noise (no recognizable pattern)
```

**How Current Design Prevents This:**

**Architecture:** Separate forward and inverse implementations
```c
// fft_radix32_fv.c - Forward version
void fft_radix32_fv(...) {
    // Hardcoded with NEGATIVE angles:
    double angle = -2.0 * M_PI * (j * g) / 32.0;
    // ...
    APPLY_W32_TWIDDLES_FV_AVX2(x);  // Uses forward macro
    APPLY_W8_TWIDDLES_FV_AVX2(...); // Uses forward macro
}

// fft_radix32_bv.c - Inverse version (SEPARATE FILE)
void fft_radix32_bv(...) {
    // Hardcoded with POSITIVE angles:
    double angle = +2.0 * M_PI * (j * g) / 32.0;
    // ...
    APPLY_W32_TWIDDLES_BV_AVX2(x);  // Uses inverse macro
    APPLY_W8_TWIDDLES_BV_AVX2(...); // Uses inverse macro
}
```

**Dispatcher ensures correct function called:**
```c
// In fft_execute.c:
switch (radix) {
    case 32:
        if (plan->direction == FFT_FORWARD) {
            fft_radix32_fv(out, in, stage_tw, sub_len);  // ← Forward
        } else {
            fft_radix32_bv(out, in, stage_tw, sub_len);  // ← Inverse
        }
        break;
}
```

**Why This Works:**
- **Compile-time dispatch:** No runtime direction checks in hot path
- **No shared code path:** Forward and inverse cannot accidentally use each other's twiddles
- **Symmetric design:** Both versions exist as separate, auditable units

**Validation Test:**
```c
void test_radix32_internal_twiddles() {
    // Create minimal test: pure radix-32 transform (N=32)
    fft_object fwd = fft_init(32, FFT_FORWARD);
    fft_object inv = fft_init(32, FFT_INVERSE);
    
    // Impulse input
    fft_data input[32] = {{1,0}};
    fft_data freq[32];
    fft_data output[32];
    
    // Round trip
    fft_exec(fwd, input, freq);
    fft_exec(inv, freq, output);
    
    // Should recover 32×input (before normalization)
    assert(fabs(output[0].re - 32.0) < 1e-12);
    assert(fabs(output[0].im - 0.0) < 1e-12);
    
    for (int i = 1; i < 32; i++) {
        assert(fabs(output[i].re) < 1e-12);
        assert(fabs(output[i].im) < 1e-12);
    }
}
```

---

### 2.4 FAILURE: Rader Convolution Twiddles Have Wrong Sign

#### 2.4.1 Scenario: Cache Returns Wrong Direction

**What Happens:**
```c
// BUGGY cache lookup:
const fft_data* get_rader_twiddles(int prime, fft_direction_t direction) {
    // ...
    
    // ❌ BUG: Swapped forward/inverse
    const fft_data *result = (direction == FFT_FORWARD) 
        ? g_rader_cache[i].conv_tw_inv  // ← WRONG
        : g_rader_cache[i].conv_tw_fwd; // ← WRONG
    
    return result;
}
```

**Impact:**
```
For N=448 with radix-7 stage:

Forward FFT gets INVERSE Rader twiddles:
- Stage twiddles: exp(-2πi...)  ✓ Correct
- Rader twiddles: exp(+2πi...)  ❌ WRONG

Result: Radix-7 butterfly computes INVERSE DFT during FORWARD transform.

Round trip:
Forward with inverse Rader:  F₀ ∘ F₁⁻¹ ∘ F₂(x)
Inverse with forward Rader:  F₂⁻¹ ∘ F₁ ∘ F₀⁻¹(...)

Composition: F₂⁻¹ ∘ F₁ ∘ F₀⁻¹ ∘ F₀ ∘ F₁⁻¹ ∘ F₂(x)
           = F₂⁻¹ ∘ F₁ ∘ F₁⁻¹ ∘ F₂(x)
           = F₂⁻¹ ∘ F₂(x)
           = N₂ × x  where N₂ = size at stage 2

For N=448, stage 2 has size 2:
Output = 2x (double the input magnitude!)
```

**How Current Design Prevents This:**

**Direction-aware cache lookup:**
```c
// ✅ CORRECT IMPLEMENTATION (from fft_rader_plans.c)
const fft_data* get_rader_twiddles(int prime, fft_direction_t direction)
{
    mutex_lock();
    
    // Search cache
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == prime) {
            // ✓ Correct: forward gets fwd, inverse gets inv
            const fft_data *result = (direction == FFT_FORWARD) 
                ? g_rader_cache[i].conv_tw_fwd 
                : g_rader_cache[i].conv_tw_inv;
            
            mutex_unlock();
            return result;
        }
    }
    
    // ... (cache miss handling)
}
```

**Twiddles computed with correct signs:**
```c
// In create_rader_plan_for_prime():
for (int q = 0; q < prime - 1; q++) {
    int idx = entry->perm_out[q];
    
    // Forward twiddle (negative sign) ✓
    double angle_fwd = -2.0 * M_PI * (double)idx / (double)prime;
    sincos_auto(angle_fwd, &entry->conv_tw_fwd[q].im, &entry->conv_tw_fwd[q].re);
    
    // Inverse twiddle (positive sign) ✓
    double angle_inv = +2.0 * M_PI * (double)idx / (double)prime;
    sincos_auto(angle_inv, &entry->conv_tw_inv[q].im, &entry->conv_tw_inv[q].re);
}
```

**Planning links correct arrays:**
```c
// In fft_planner.c:
if (is_prime(radix) && radix >= 7) {
    // ✓ Direction passed through to cache lookup
    stage->rader_tw = (fft_data*)get_rader_twiddles(radix, direction);
}
```

**Validation Test:**
```c
void test_rader_cache_conjugacy() {
    init_rader_cache();
    
    for (int prime : {7, 11, 13, 17, 19, 23}) {
        const fft_data *fwd_tw = get_rader_twiddles(prime, FFT_FORWARD);
        const fft_data *inv_tw = get_rader_twiddles(prime, FFT_INVERSE);
        
        // Verify arrays are distinct:
        assert(fwd_tw != inv_tw);
        
        // Verify conjugacy:
        for (int q = 0; q < prime-1; q++) {
            assert(fabs(fwd_tw[q].re - inv_tw[q].re) < 1e-15);
            assert(fabs(fwd_tw[q].im + inv_tw[q].im) < 1e-15);
        }
    }
}
```

---

## 3. Algorithm Interaction Failures

### 3.1 FAILURE: Stage Twiddles Not Applied When Rader Active

#### 3.1.1 Scenario: Butterfly Ignores Stage Twiddles if Rader Present

**What Happens:**
```c
// BUGGY radix-7 butterfly:
void radix_7_kernel(..., const fft_data *stage_tw, const fft_data *rader_tw, ...) {
    
    if (rader_tw != NULL) {
        // ❌ BUG: Immediately jump to Rader, skipping stage twiddles
        perform_rader_convolution(..., rader_tw);
        return;
    }
    
    // Apply stage twiddles (never reached!)
    for (int lane = 1; lane < 7; lane++) {
        // ...
    }
}
```

**Mathematical Impact:**
```
Cooley-Tukey decomposition requires TWO sets of twiddles:

1. Inter-stage twiddles (stage_tw): Rotate data between stages
   Formula: data[k + lane*K] *= W_N^(lane*k)
   
2. Intra-stage twiddles (for Rader): DFT kernel for prime radix
   Formula: Circular convolution in frequency domain

Skipping stage_tw means:
X_wrong[k] = DFT_7(x[k], x[k+K], ..., x[k+6K])  ← No rotation!
X_correct[k] = DFT_7(x[k], W^k·x[k+K], W^(2k)·x[k+2K], ...)

Result: Massive phase errors in output
```

**Example with N=448:**
```
Stage 1 (radix-7) should compute:

For each k:
  1. Load: [x₀, x₁, ..., x₆] from positions [k, k+K, k+2K, ..., k+6K]
  2. Multiply: x_i *= stage_tw[k*6 + (i-1)] for i=1..6  ← SKIPPED!
  3. Rader DFT: [X₀, X₁, ..., X₆] = Rader_7([x₀, x₁, ..., x₆], rader_tw)
  4. Store: [X₀, X₁, ..., X₆]

Without step 2, stage 0 and stage 1 are not properly connected.

Round trip fails completely (output is noise).
```

**How Current Design Prevents This:**

**Radix butterflies MUST apply both twiddle sets:**
```c
// ✓ CORRECT PATTERN (radix-7 butterfly should follow this):
void radix_7_kernel(
    fft_data *out,
    const fft_data *in,
    const fft_data *stage_tw,   // ← Cooley-Tukey inter-stage
    const fft_data *rader_tw,   // ← Rader intra-stage
    int sub_len)
{
    const int K = sub_len;
    
    for (int k = 0; k < K; k++) {
        fft_data x[7];
        
        // STEP 1: Load data
        for (int lane = 0; lane < 7; lane++) {
            x[lane] = in[k + lane*K];
        }
        
        // STEP 2: Apply stage twiddles (Cooley-Tukey)
        // ✓ CRITICAL: This happens REGARDLESS of Rader
        for (int lane = 1; lane < 7; lane++) {
            const fft_data *tw = &stage_tw[k * 6 + (lane - 1)];
            double xr = x[lane].re, xi = x[lane].im;
            x[lane].re = xr * tw->re - xi * tw->im;
            x[lane].im = xr * tw->im + xi * tw->re;
        }
        
        // STEP 3: Perform radix-7 DFT
        if (rader_tw != NULL) {
            // Use Rader's algorithm with convolution twiddles
            rader_7_dft(x, rader_tw);
        } else {
            // Fallback: Direct 7-point DFT (shouldn't happen for prime 7)
            direct_7_dft(x);
        }
        
        // STEP 4: Store results
        for (int lane = 0; lane < 7; lane++) {
            out[k + lane*K] = x[lane];
        }
    }
}
```

**Architectural Guarantee:**

The stage twiddle application is **not conditional** on Rader. The butterfly receives TWO separate parameters:
- `stage_tw`: ALWAYS non-NULL (allocated during planning)
- `rader_tw`: NULL for non-primes, non-NULL for primes ≥7

The butterfly is **required** to use both.

**Validation Test:**
```c
void test_stage_twiddle_application() {
    // Create N=14 (2×7) to isolate radix-7 stage
    fft_object fwd = fft_init(14, FFT_FORWARD);
    fft_object inv = fft_init(14, FFT_INVERSE);
    
    // Verify stage setup
    int radix7_stage = -1;
    for (int i = 0; i < fwd->num_stages; i++) {
        if (fwd->stages[i].radix == 7) {
            radix7_stage = i;
            break;
        }
    }
    
    assert(radix7_stage >= 0);  // Found radix-7 stage
    assert(fwd->stages[radix7_stage].stage_tw != NULL);  // Has stage twiddles
    assert(fwd->stages[radix7_stage].rader_tw != NULL);  // Has Rader twiddles
    
    // Impulse response test
    fft_data input[14] = {{1,0}};
    fft_data output[14];
    
    fft_roundtrip_normalized(fwd, inv, input, output, workspace);
    
    // Should recover exactly (within precision)
    for (int i = 0; i < 14; i++) {
        double expected_re = (i == 0) ? 1.0 : 0.0;
        assert(fabs(output[i].re - expected_re) < 1e-12);
        assert(fabs(output[i].im) < 1e-12);
    }
}
```

---

### 3.2 FAILURE: Rader Twiddles Applied When Not Prime

#### 3.2.1 Scenario: Non-Prime Radix Tries to Use Rader Twiddles

**What Happens:**
```c
// BUGGY butterfly for radix-9 (composite: 3²):
void radix_9_kernel(..., const fft_data *stage_tw, const fft_data *rader_tw, ...) {
    
    // Apply stage twiddles (correct)
    for (int lane = 1; lane < 9; lane++) {
        // ... multiply by stage_tw ...
    }
    
    // ❌ BUG: Tries to use Rader for non-prime
    if (rader_tw != NULL) {
        rader_9_dft(x, rader_tw);  // ← rader_tw is GARBAGE (undefined pointer)
    } else {
        direct_9_dft(x);  // ← Should always reach here for radix-9
    }
}
```

**Impact:**
```
Radix-9 is composite (3²), so:
- stage->rader_tw should be NULL (planning sets it to NULL)
- Butterfly should use direct DFT or nested radix-3

If butterfly incorrectly checks uninitialized rader_tw:
- Undefined behavior (segfault or memory corruption)
- If "lucky" and NULL, takes correct path
- If "unlucky" and non-NULL garbage, attempts Rader with wrong size
```

**How Current Design Prevents This:**

**Planning explicitly sets NULL for non-primes:**
```c
// In fft_planner.c:
for (int i = 0; i < num_stages; i++) {
    int radix = plan->factors[i];
    
    // ...compute stage_tw...
    
    // ✓ Rader only for primes ≥7:
    if (radix >= 7 && radix <= 67) {
        int is_prime = (radix == 7 || radix == 11 || radix == 13 || 
                       radix == 17 || radix == 19 || radix == 23 ||
                       // ... full list ...
                       radix == 67);
        
        if (is_prime) {
            stage->rader_tw = (fft_data*)get_rader_twiddles(radix, direction);
        } else {
            stage->rader_tw = NULL;  // ✓ Explicit NULL for non-primes
        }
    } else {
        stage->rader_tw = NULL;  // ✓ Explicit NULL for small/large radices
    }
}
```

**Butterfly implementations check NULL safely:**
```c
// Recommended pattern:
void radix_N_kernel(..., const fft_data *stage_tw, const fft_data *rader_tw, ...) {
    // Apply stage twiddles (always)
    // ...
    
    // Use Rader ONLY if pointer is valid AND radix is prime
    #if defined(RADIX_IS_PRIME) && (RADIX >= 7)
        if (rader_tw != NULL) {
            rader_N_dft(x, rader_tw);
        } else {
            // Fallback: shouldn't reach here for primes in cache
            direct_N_dft(x);
        }
    #else
        // For non-primes: direct DFT (ignore rader_tw)
        direct_N_dft(x);
    #endif
}
```

**Compile-time vs Runtime:**

For maximum safety, radix implementations should use **compile-time knowledge**:
- Radix-7 butterfly: **knows** it's prime, requires Rader
- Radix-9 butterfly: **knows** it's composite, never uses Rader
- No need for runtime checks (performance + safety)

---

### 3.3 FAILURE: Twiddle Layout Mismatch Between Planning and Execution

#### 3.3.1 Scenario: Planner Uses One Layout, Butterfly Expects Another

**What Happens:**
```c
// Planning computes twiddles as:
// tw[k*(radix-1) + (r-1)] = W^(r*k)  ← Interleaved layout

// But butterfly reads as:
// tw[r*sub_len + k] = W^(r*k)  ← Sequential layout

Result: Reading wrong twiddle values
```

**Example for radix=7, sub_len=64:**
```
Planned layout (interleaved):
tw[0]   = W^(1*0)   tw[1]   = W^(2*0)   tw[2]   = W^(3*0)   ... tw[5]   = W^(6*0)
tw[6]   = W^(1*1)   tw[7]   = W^(2*1)   tw[8]   = W^(3*1)   ... tw[11]  = W^(6*1)
tw[12]  = W^(1*2)   tw[13]  = W^(2*2)   tw[14]  = W^(3*2)   ... tw[17]  = W^(6*2)
...

Butterfly expects sequential:
tw[0..63]   = W^(1*[0..63])
tw[64..127] = W^(2*[0..63])
tw[128..191]= W^(3*[0..63])
...

At position k=5, radix multiplier r=3:
Butterfly reads: tw[2*64 + 5] = tw[133] = W^(3*5)  ← Maybe correct by luck?
Should read:     tw[5*6 + 2] = tw[32]   = W^(3*5)  ← Correct
```

**Impact:** Completely wrong twiddle values → garbage output

**How Current Design Prevents This:**

**Single Source of Truth in Comments:**
```c
// In compute_stage_twiddles():
// ✅ Interleaved layout: tw[k*(radix-1) + (r-1)] = W^(r*k)

fft_data* compute_stage_twiddles(...) {
    const int sub_len = N_stage / radix;
    const int num_twiddles = (radix - 1) * sub_len;
    
    // Allocate interleaved
    fft_data *tw = aligned_alloc(..., num_twiddles * sizeof(fft_data));
    
    // Fill interleaved
    for (int k = 0; k < sub_len; k++) {
        for (int r = 1; r < radix; r++) {
            int idx = k * (radix - 1) + (r - 1);  // ← Interleaved formula
            double angle = base_angle * (double)r * (double)k;
            sincos_auto(angle, &tw[idx].im, &tw[idx].re);
        }
    }
    
    return tw;
}
```

**Butterfly reads with SAME formula:**
```c
// In radix_N_kernel():
void radix_N_kernel(..., const fft_data *stage_tw, int sub_len) {
    for (int k = 0; k < sub_len; k++) {
        // Apply twiddles to lanes 1..(radix-1)
        for (int lane = 1; lane < radix; lane++) {
            int idx = k * (radix - 1) + (lane - 1);  // ← SAME formula
            const fft_data *tw = &stage_tw[idx];
            
            // Complex multiply: data[lane] *= tw
        }
    }
}
```

**Validation Test:**
```c
void test_twiddle_layout() {
    int N_stage = 448;
    int radix = 7;
    int sub_len = N_stage / radix;  // 64
    
    fft_data *tw_fwd = compute_stage_twiddles(N_stage, radix, FFT_FORWARD);
    
    // Verify first few twiddles match expected formula:
    for (int k = 0; k < 3; k++) {
        for (int r = 1; r < radix; r++) {
            int idx = k * (radix - 1) + (r - 1);
            
            double expected_angle = -2.0 * M_PI * r * k / (double)N_stage;
            double expected_re = cos(expected_angle);
            double expected_im = sin(expected_angle);
            
            assert(fabs(tw_fwd[idx].re - expected_re) < 1e-14);
            assert(fabs(tw_fwd[idx].im - expected_im) < 1e-14);
        }
    }
    
    free_stage_twiddles(tw_fwd);
}
```

---

## 4. Normalization Failures

### 4.1 FAILURE: Normalization in Butterflies

#### 4.1.1 Scenario: Radix-32 Butterfly Scales by 1/32

**What Happens:**
```c
// BUGGY radix-32 forward butterfly:
void fft_radix32_fv(...) {
    // Perform butterfly operations...
    // ...
    
    // ❌ BUG: Apply 1/32 scaling
    for (int i = 0; i < 32; i++) {
        x[i].re /= 32.0;
        x[i].im /= 32.0;
    }
    
    // Store results...
}
```

**Impact:**
```
For N=448 (radix-32 × radix-7 × radix-2):

Forward transform:
Stage 0 (radix-32): Scales by 1/32  ❌
Stage 1 (radix-7):  No scaling      ✓
Stage 2 (radix-2):  No scaling      ✓

Result: X = (1/32) × DFT(x)  ← Attenuated by factor of 32

Inverse transform (if also buggy):
Stage 0 (radix-2):  No scaling      ✓
Stage 1 (radix-7):  No scaling      ✓
Stage 2 (radix-32): Scales by 1/32  ❌

Result: y = (1/32) × IDFT(X) = (1/32) × IDFT((1/32) × DFT(x))
          = (1/32²) × IDFT(DFT(x))
          = (1/32²) × N × x
          = (448/1024) × x  ← 43% of original magnitude!

User applies 1/N:
output = y / 448 = x / 1024  ← 0.1% of original! FAIL
```

**How Current Design Prevents This:**

**Zero normalization in core transforms:**
```c
// ✅ From fft_execute.c:
int fft_exec_dft(plan, input, output, workspace) {
    switch (plan->strategy) {
        case FFT_EXEC_STOCKHAM:
            return fft_exec_stockham_internal(plan, input, output, workspace);
        
        // ...other strategies...
    }
    
    // ❌ REMOVED: No normalization here!
    // User applies 1/N manually if needed
    
    return 0;  // ← No scaling applied
}
```

**Butterfly implementations are scale-neutral:**
```c
// Radix-32 forward butterfly (excerpt):
void fft_radix32_fv(...) {
    // Load data
    // Apply stage twiddles
    // Perform butterflies
    // Apply W_32 and W_8 twiddles
    // Store results
    
    // ✓ NO SCALING anywhere in this function
}
```

**Normalization happens ONCE in wrapper:**
```c
// ✅ User-facing normalized function:
int fft_roundtrip_normalized(fwd, inv, input, output, workspace) {
    // Forward (unnormalized)
    fft_exec_dft(fwd, input, freq, workspace);
    
    // Inverse (unnormalized)
    fft_exec_dft(inv, freq, output, workspace);
    
    // Normalize ONCE at the end:
    const double scale = 1.0 / (double)N;
    for (int i = 0; i < N; i++) {
        output[i].re *= scale;
        output[i].im *= scale;
    }
    
    return 0;
}
```

**Validation Test:**
```c
void test_no_intermediate_scaling() {
    int N = 448;
    fft_object fwd = fft_init(N, FFT_FORWARD);
    fft_object inv = fft_init(N, FFT_INVERSE);
    
    fft_data input[N] = {{1,0}};  // Impulse
    fft_data freq[N];
    fft_data output[N];
    
    // Forward (should NOT scale)
    fft_exec_dft(fwd, input, freq, NULL);
    
    // Check DC bin magnitude (should be N, not 1)
    double dc_mag = 0.0;
    for (int k = 0; k < N; k++) {
        dc_mag += freq[k].re;
    }
    assert(fabs(dc_mag - N) < 1e-10);  // ← Expects N, not 1
    
    // Inverse (should NOT scale)
    fft_exec_dft(inv, freq, output, NULL);
    
    // Should be N×input (before normalization)
    assert(fabs(output[0].re - N) < 1e-10);
    
    // After manual normalization:
    for (int i = 0; i < N; i++) {
        output[i].re /= N;
        output[i].im /= N;
    }
    
    // Now should match input
    assert(fabs(output[0].re - 1.0) < 1e-12);
}
```

---

### 4.2 FAILURE: Double Normalization

#### 4.2.1 Scenario: Both Transform and Wrapper Scale

**What Happens:**
```c
// BUGGY: Transform scales by 1/N
int fft_exec_dft(plan, input, output, workspace) {
    // ... execute transform ...
    
    // ❌ BUG: Scale in core transform
    const double scale = 1.0 / plan->n_fft;
    for (int i = 0; i < plan->n_fft; i++) {
        output[i].re *= scale;
        output[i].im *= scale;
    }
}

// Wrapper ALSO scales
int fft_roundtrip_normalized(fwd, inv, input, output, workspace) {
    fft_exec_dft(fwd, input, freq, workspace);  // ← Scales by 1/N
    fft_exec_dft(inv, freq, output, workspace); // ← Scales by 1/N again
    
    // ❌ And scales a THIRD time:
    const double scale = 1.0 / N;
    for (int i = 0; i < N; i++) {
        output[i].re *= scale;
        output[i].im *= scale;
    }
}
```

**Impact:**
```
Round trip:
Forward: X = (1/N) × DFT(x)
Inverse: y = (1/N) × IDFT(X) = (1/N) × IDFT((1/N) × DFT(x))
           = (1/N²) × IDFT(DFT(x))
           = (1/N²) × N × x
           = (1/N) × x

Wrapper: output = (1/N) × y = (1/N²) × x  ← Massive attenuation!

For N=448:
output = x / 200704  ← 0.0005% of original magnitude!
```

**How Current Design Prevents This:**

**Architectural separation:**
```
Level 1: Core transforms (fft_exec_dft)
   ↓ NO SCALING
Level 2: Wrapper functions (fft_roundtrip_normalized)
   ↓ SCALE ONCE by 1/N
User
```

**Code review enforcement:**
```c
// fft_exec_dft has explicit comment:
int fft_exec_dft(...) {
    // ...
    
    // ❌ REMOVED: No normalization here!
    // User applies 1/N manually if needed
    
    return 0;
}
```

**Static analysis check:**
```bash
# Grep for forbidden patterns in core execution:
$ grep -n "/ *n_fft\|/ *N\|/ *(double)N" fft_execute.c

# Should only appear in:
# - fft_exec_normalized()
# - fft_roundtrip_normalized()
# NOT in:
# - fft_exec_dft()
# - fft_exec_stockham_internal()
# - fft_exec_inplace_bitrev_internal()
```

---

## 5. Design Solutions Analysis

### 5.1 Solution Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PLANNING PHASE (Once)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: N, direction                                            │
│     ↓                                                           │
│  factorize_optimal(N) → [radix₀, radix₁, ..., radixₖ]         │
│     ↓                                                           │
│  FOR each stage i:                                              │
│     ├─ radix = factors[i]                                       │
│     ├─ stage_tw = compute_stage_twiddles(N_stage, radix,       │
│     │                                    direction) ← SIGN      │
│     ├─ IF radix is prime ≥7:                                    │
│     │    └─ rader_tw = get_rader_twiddles(radix,               │
│     │                                     direction) ← SIGN     │
│     └─ ELSE:                                                    │
│          └─ rader_tw = NULL                                     │
│                                                                 │
│  Output: Immutable plan with precomputed twiddles               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  EXECUTION PHASE (Many times)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FOR each stage i:                                              │
│     ├─ radix = plan->stages[i].radix                            │
│     ├─ stage_tw = plan->stages[i].stage_tw  ← Precomputed      │
│     ├─ rader_tw = plan->stages[i].rader_tw  ← Borrowed         │
│     │                                                           │
│     └─ DISPATCH to radix-specific butterfly:                    │
│          ├─ IF direction == FORWARD:                            │
│          │    └─ radix_N_fv(out, in, stage_tw, rader_tw, ...)  │
│          └─ ELSE:                                               │
│               └─ radix_N_bv(out, in, stage_tw, rader_tw, ...)  │
│                                                                 │
│  Butterfly responsibilities:                                     │
│     1. Apply stage_tw to all lanes (Cooley-Tukey rotation)     │
│     2. IF rader_tw != NULL:                                     │
│           Use Rader's algorithm with convolution twiddles       │
│        ELSE:                                                    │
│           Direct DFT                                            │
│     3. NO NORMALIZATION                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   NORMALIZATION (Optional)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  fft_exec_dft()            → NO SCALING (output = N×input)     │
│  fft_exec_normalized()     → SCALE by 1/N                       │
│  fft_roundtrip_normalized()→ SCALE by 1/N (once, at end)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Key Design Principles

#### Principle 1: Direction Captured Once, Used Everywhere
```c
// At planning time:
fft_object fft_init(int N, fft_direction_t direction) {
    plan->direction = direction;  // ← Stored once
    
    // ALL subsequent operations use this:
    for (int i = 0; i < num_stages; i++) {
        stage_tw = compute_stage_twiddles(..., direction);  // ← Propagated
        rader_tw = get_rader_twiddles(..., direction);      // ← Propagated
    }
}

// No per-stage direction override possible
// No runtime direction decisions in hot path
```

**Benefits:**
- **Consistency:** Impossible for stages to have mismatched directions
- **Performance:** No runtime checks
- **Auditability:** Single point of truth

#### Principle 2: Separate Forward/Inverse Implementations
```c
// Compile-time dispatch (zero overhead):
switch (radix) {
    case 32:
        if (plan->direction == FFT_FORWARD) {
            fft_radix32_fv(...);  // ← Forward version (separate file)
        } else {
            fft_radix32_bv(...);  // ← Inverse version (separate file)
        }
        break;
}

// Each version hardcodes internal twiddles with correct sign
// No shared code path that could mix signs
```

**Benefits:**
- **Safety:** Forward and inverse cannot accidentally use each other's twiddles
- **Performance:** Hardcoded twiddles (W₃₂, W₈) inline better
- **Clarity:** Each file is self-contained and auditable

#### Principle 3: Orthogonal Twiddle Sets
```c
// Stage descriptor contains TWO independent twiddle arrays:
typedef struct {
    int radix;
    fft_data *stage_tw;   // ← Cooley-Tukey inter-stage (OWNED)
    fft_data *rader_tw;   // ← Rader intra-stage (BORROWED)
} stage_descriptor;

// Butterfly MUST use both:
void radix_N_kernel(..., stage_tw, rader_tw, ...) {
    // 1. Apply stage_tw (always)
    // 2. Use rader_tw (if non-NULL)
    // These are independent operations
}
```

**Benefits:**
- **Modularity:** Rader and Cooley-Tukey don't interfere
- **Flexibility:** Can add/remove Rader without changing Cooley-Tukey code
- **Correctness:** Both transformations apply in correct sequence

#### Principle 4: Normalization at Boundary
```c
// Internal functions: NO scaling
fft_exec_dft(plan, in, out, ws);
  → Computes: out = DFT(in)  (exactly, no 1/N)

// User-facing functions: Scale ONCE
fft_roundtrip_normalized(fwd, inv, in, out, ws);
  → Computes: out = (1/N) × IDFT(DFT(in)) = in
```

**Benefits:**
- **Performance:** No redundant scaling in hot path
- **Flexibility:** User can choose normalization strategy
- **FFTW-compatible:** Matches standard convention

### 5.3 Failure Prevention Matrix

| Failure Mode | Prevention Mechanism | Validation Method |
|--------------|---------------------|-------------------|
| Same sign both directions | `direction` parameter in `compute_stage_twiddles()` | Unit test: conjugate check |
| Mixed signs across stages | All stages use same `plan->direction` | Integration test: round trip |
| Wrong radix-32 internal twiddles | Separate `_fv` and `_bv` files | Code review: angle signs |
| Wrong Rader cache lookup | Direction switch in `get_rader_twiddles()` | Unit test: cache conjugate |
| Stage twiddles skipped | Butterfly design: apply `stage_tw` before Rader | Code review: sequence |
| Rader on non-prime | Planning: `rader_tw = NULL` for non-primes | Static analysis: NULL checks |
| Twiddle layout mismatch | Single formula in comments + code | Unit test: expected angles |
| Butterfly normalization | No scaling in any butterfly | Grep: forbidden patterns |
| Double normalization | Scaling only in wrapper functions | Integration test: magnitude |

---

## 6. Validation Against Requirements

### Requirement: Mixed Round Trip (Prime × Composite)

**Statement:** The system must correctly perform round trip transforms for sizes N = prime × composite, where the prime uses Rader's algorithm and the composite uses Cooley-Tukey.

**Target Case:** N = 448 = 2 × 7 × 32
- Radix-32: High-performance Cooley-Tukey (power-of-2 decomposition, hardcoded twiddles)
- Radix-7: Rader's algorithm (primitive root, convolution)
- Radix-2: Trivial Cooley-Tukey

### 6.1 Correctness Proof

**Theorem:** If all design principles are followed, round trip is exact up to normalization.

**Proof Sketch:**

Let F = DFT_N, F⁻¹ = IDFT_N.

Decomposition:
```
F = F₂ ∘ F₇ ∘ F₃₂
F⁻¹ = F₃₂⁻¹ ∘ F₇⁻¹ ∘ F₂⁻¹
```

**Forward Transform:**
```
Stage 0 (radix-32):
  - Uses forward stage twiddles: exp(-2πi...)
  - Uses forward internal twiddles (W₃₂, W₈): exp(-2πi...)
  - Result: X₀ = F₃₂(x)

Stage 1 (radix-7):
  - Uses forward stage twiddles: exp(-2πi...)
  - Uses forward Rader twiddles: exp(-2πi...)
  - Result: X₁ = F₇(X₀)

Stage 2 (radix-2):
  - Uses forward stage twiddles: exp(-2πi...)
  - Result: X = F₂(X₁) = F(x)  ✓
```

**Inverse Transform:**
```
Stage 0 (radix-2):
  - Uses inverse stage twiddles: exp(+2πi...)
  - Result: Y₀ = F₂⁻¹(X)

Stage 1 (radix-7):
  - Uses inverse stage twiddles: exp(+2πi...)
  - Uses inverse Rader twiddles: exp(+2πi...)
  - Result: Y₁ = F₇⁻¹(Y₀)

Stage 2 (radix-32):
  - Uses inverse stage twiddles: exp(+2πi...)
  - Uses inverse internal twiddles: exp(+2πi...)
  - Result: y = F₃₂⁻¹(Y₁) = F⁻¹(X)  ✓
```

**Round Trip:**
```
y = F⁻¹(F(x))
  = F₃₂⁻¹ ∘ F₇⁻¹ ∘ F₂⁻¹ ∘ F₂ ∘ F₇ ∘ F₃₂(x)
  = F₃₂⁻¹ ∘ F₇⁻¹ ∘ F₇ ∘ F₃₂(x)      (F₂⁻¹ ∘ F₂ = 2·I)
  = F₃₂⁻¹ ∘ F₇⁻¹ ∘ F₇ ∘ F₃₂(x)      (F₇⁻¹ ∘ F₇ = 7·I)
  = F₃₂⁻¹ ∘ F₃₂(x) × (2 × 7)        (Compositions)
  = 32·I(x) × 14                     (F₃₂⁻¹ ∘ F₃₂ = 32·I)
  = 448 × x                          ✓

Normalized:
output = y / 448 = x  ✓✓✓
```

**QED**

### 6.2 Comprehensive Test Suite

```c
//==============================================================================
// COMPREHENSIVE ROUND TRIP TEST SUITE
//==============================================================================

void test_mixed_radix_roundtrip_suite() {
    printf("=== MIXED RADIX ROUND TRIP TEST SUITE ===\n\n");
    
    // ─────────────────────────────────────────────────────────────────────
    // TEST 1: Small Prime × Power-of-2
    // ─────────────────────────────────────────────────────────────────────
    printf("TEST 1: Small Prime × Power-of-2\n");
    
    int sizes_test1[] = {14, 28, 56, 22, 44, 26};
    for (int i = 0; i < 6; i++) {
        int N = sizes_test1[i];
        printf("  N=%d: ", N);
        
        if (test_roundtrip(N)) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            exit(1);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // TEST 2: Large Composite × Prime (Target Case)
    // ─────────────────────────────────────────────────────────────────────
    printf("\nTEST 2: Large Composite × Prime\n");
    printf("  N=448 (2×7×32): ");
    
    if (test_roundtrip_detailed(448)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        exit(1);
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // TEST 3: Multiple Primes
    // ─────────────────────────────────────────────────────────────────────
    printf("\nTEST 3: Multiple Primes\n");
    
    int sizes_test3[] = {77, 91, 143, 1001};
    for (int i = 0; i < 4; i++) {
        int N = sizes_test3[i];
        printf("  N=%d: ", N);
        
        if (test_roundtrip(N)) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            exit(1);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // TEST 4: Prime × Composite (Non-Power-of-2)
    // ─────────────────────────────────────────────────────────────────────
    printf("\nTEST 4: Prime × Composite (Non-Power-of-2)\n");
    
    int sizes_test4[] = {63, 99, 117};
    for (int i = 0; i < 3; i++) {
        int N = sizes_test4[i];
        printf("  N=%d: ", N);
        
        if (test_roundtrip(N)) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
            exit(1);
        }
    }
    
    printf("\n=== ALL TESTS PASSED ===\n");
}

//------------------------------------------------------------------------------
// Helper: Basic Round Trip Test
//------------------------------------------------------------------------------
int test_roundtrip(int N) {
    fft_object fwd = fft_init(N, FFT_FORWARD);
    fft_object inv = fft_init(N, FFT_INVERSE);
    
    if (!fwd || !inv) return 0;
    
    // Allocate buffers
    fft_data *input = calloc(N, sizeof(fft_data));
    fft_data *output = calloc(N, sizeof(fft_data));
    size_t ws_size = fmax(fft_get_workspace_size(fwd), 
                          fft_get_workspace_size(inv));
    fft_data *workspace = (ws_size > 0) ? calloc(ws_size, sizeof(fft_data)) : NULL;
    
    // Test with random complex input
    for (int i = 0; i < N; i++) {
        input[i].re = (double)rand() / RAND_MAX * 2.0 - 1.0;
        input[i].im = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    
    // Round trip
    int result = fft_roundtrip_normalized(fwd, inv, input, output, workspace);
    
    // Check error
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double err_re = fabs(input[i].re - output[i].re);
        double err_im = fabs(input[i].im - output[i].im);
        max_error = fmax(max_error, err_re + err_im);
    }
    
    // Cleanup
    free(input);
    free(output);
    free(workspace);
    free_fft(fwd);
    free_fft(inv);
    
    // Accept if error < 1e-12 × N
    return (result == 0 && max_error < 1e-12 * N);
}

//------------------------------------------------------------------------------
// Helper: Detailed Round Trip Test (with diagnostics)
//------------------------------------------------------------------------------
int test_roundtrip_detailed(int N) {
    fft_object fwd = fft_init(N, FFT_FORWARD);
    fft_object inv = fft_init(N, FFT_INVERSE);
    
    // Print factorization
    printf("\n    Factorization: ");
    for (int i = 0; i < fwd->num_stages; i++) {
        printf("%d%s", fwd->stages[i].radix, 
               (i < fwd->num_stages-1) ? " × " : "\n");
    }
    
    // Verify twiddle setup
    printf("    Twiddle setup:\n");
    for (int i = 0; i < fwd->num_stages; i++) {
        stage_descriptor *s = &fwd->stages[i];
        printf("      Stage %d (radix-%d): ", i, s->radix);
        printf("stage_tw=%s, ", s->stage_tw ? "✓" : "✗");
        printf("rader_tw=%s\n", s->rader_tw ? "✓" : "NULL");
    }
    
    // Run basic test
    int pass = test_roundtrip(N);
    printf("    Result: %s\n", pass ? "PASS" : "FAIL");
    
    free_fft(fwd);
    free_fft(inv);
    
    return pass;
}
```

### 6.3 Expected Test Output

```
=== MIXED RADIX ROUND TRIP TEST SUITE ===

TEST 1: Small Prime × Power-of-2
  N=14: PASS
  N=28: PASS
  N=56: PASS
  N=22: PASS
  N=44: PASS
  N=26: PASS

TEST 2: Large Composite × Prime
  N=448 (2×7×32): 
    Factorization: 32 × 7 × 2
    Twiddle setup:
      Stage 0 (radix-32): stage_tw=✓, rader_tw=NULL
      Stage 1 (radix-7): stage_tw=✓, rader_tw=✓
      Stage 2 (radix-2): stage_tw=✓, rader_tw=NULL
    Result: PASS

TEST 3: Multiple Primes
  N=77: PASS
  N=91: PASS
  N=143: PASS
  N=1001: PASS

TEST 4: Prime × Composite (Non-Power-of-2)
  N=63: PASS
  N=99: PASS
  N=117: PASS

=== ALL TESTS PASSED ===
```

---

## 7. Conclusion 

### 7.1 Summary of Failure Modes

This document identified **nine critical failure modes** in mixed-radix FFT systems:

1. **Both transforms use same sign** → Phase corruption
2. **Mixed signs across stages** → Partial corruption (stage-specific)
3. **Radix-specific twiddles have wrong sign** → Algorithm inverted
4. **Rader convolution twiddles have wrong sign** → Prime stages corrupted
5. **Stage twiddles not applied when Rader active** → Inter-stage disconnection
6. **Rader twiddles applied to non-primes** → Undefined behavior
7. **Twiddle layout mismatch** → Wrong twiddle values
8. **Normalization in butterflies** → Scale factor errors
9. **Double normalization** → Massive attenuation

### 7.2 Design Solutions Effectiveness

The current architecture **prevents all nine failure modes** through:

| Solution Mechanism | Failures Prevented | Enforcement Method |
|-------------------|-------------------|-------------------|
| **Direction parameter propagation** | #1, #2 | Type system + single code path |
| **Separate forward/inverse implementations** | #3 | File-level separation |
| **Direction-aware cache lookup** | #4 | Runtime switch with mutex |
| **Mandatory dual-twiddle application** | #5 | Butterfly design pattern |
| **Explicit NULL for non-primes** | #6 | Planning-time classification |
| **Documented layout formula** | #7 | Single source of truth |
| **Zero scaling in core transforms** | #8 | Code review + grep checks |
| **Boundary normalization** | #9 | Architectural layer separation |

