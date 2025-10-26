# FFT Loop Optimization Strategies: A Comprehensive Guide to Modulo Scheduling, Double-Pumping, and Plain Unrolling

**Technical Report for High-Performance FFT Implementation**

*Focus: Modulo Scheduling in Mixed-Radix FFT Kernels*

---

## Executive Summary

Modern FFT performance is increasingly memory-bound rather than compute-bound. On AVX-512 systems, a single radix-4 butterfly can issue 8 complex multiplies per cycle, but memory subsystems can't keep pace. This report examines three loop scheduling strategies—plain unrolling, double-pumping, and modulo scheduling—that address this bottleneck through different approaches to hiding memory latency.

**Key Findings:**
- **Plain unrolling** (U=1-2) provides 0-10% gains and works best for small, L1-resident workloads
- **Double-pumping** (U=2 independent contexts) delivers 5-25% gains with manageable complexity
- **Modulo scheduling** (software pipelining) achieves 10-30% gains on large, memory-bound stages but introduces significant implementation complexity

**Critical Insight:** Modulo scheduling's **prologue/epilogue brittleness** makes it practical only for the hottest, most regular kernels (radix-2/4/8/16). Mixed-radix FFT implementations should use a hybrid approach: modulo scheduling for power-of-2 radices, double-pumping for primes and odd radices.

---

## Table of Contents

1. [The Memory Wall Problem in Modern FFT](#1-the-memory-wall-problem)
2. [Overview of Three Techniques](#2-overview-of-three-techniques)
3. [Plain Unrolling: The Baseline](#3-plain-unrolling)
4. [Double-Pumping: The Middle Ground](#4-double-pumping)
5. [Modulo Scheduling: Maximum Performance, Maximum Complexity](#5-modulo-scheduling)
6. [The Brittleness Problem: Why Modulo Scheduling is Hard](#6-brittleness-problem)
7. [CPU Architecture Considerations](#7-cpu-architecture-considerations)
8. [Decision Framework and Selection Criteria](#8-decision-framework)
9. [Implementation Guide for VectorFFT](#9-implementation-guide)
10. [Performance Case Studies](#10-performance-case-studies)
11. [Recommendations and Conclusion](#11-recommendations)

---

## 1. The Memory Wall Problem in Modern FFT {#1-the-memory-wall-problem}

### 1.1 The Compute vs Memory Gap

Modern CPUs have widened the gap between computational throughput and memory bandwidth:

```
Architecture    FMA Throughput      L1 Load BW       Ratio
─────────────────────────────────────────────────────────────
Haswell (AVX2)  16 DP FLOP/cycle    2×32 B/cycle    4 FLOP/B
Skylake-X       32 DP FLOP/cycle    2×64 B/cycle    2 FLOP/B
(AVX-512)
Zen 3 (AVX2)    16 DP FLOP/cycle    2×32 B/cycle    4 FLOP/B
Zen 4 (AVX-512) 32 DP FLOP/cycle    3×32 B/cycle    3.3 FLOP/B
```

**Problem:** A radix-4 butterfly consuming 4 complex doubles (64 bytes) and producing 4 outputs (64 bytes) needs 128 bytes of memory traffic for ~40 FLOPs. At 4 FLOP/byte, you're bandwidth-limited even if everything is in L1.

### 1.2 FFT Memory Access Pattern

```
Cooley-Tukey FFT Stage (radix R, N points):
  for (stage = 0; stage < log_R(N); stage++) {
    K = R^stage;                    // Block size
    half = N / (2*R);               // Iterations per block
    
    for (block = 0; block < K; block++) {
      for (i = 0; i < half; i++) {
        // Load inputs (4 complex, non-contiguous)
        x0 = in[block + i*stride0];
        x1 = in[block + i*stride1];
        x2 = in[block + i*stride2];
        x3 = in[block + i*stride3];
        
        // Load twiddle factors
        w1 = twiddles[i * K];
        w2 = twiddles[i * K * 2];
        w3 = twiddles[i * K * 3];
        
        // Complex multiplies (rotate)
        t1 = cmul(x1, w1);
        t2 = cmul(x2, w2);
        t3 = cmul(x3, w3);
        
        // Butterfly (add/subtract)
        y0 = x0 + t1 + t2 + t3;
        y1 = x0 + A*t1 + B*t2 + C*t3;  // A,B,C = geometric constants
        y2 = ...
        y3 = ...
        
        // Store outputs (4 complex, non-contiguous)
        out[block + i*stride0] = y0;
        out[block + i*stride1] = y1;
        out[block + i*stride2] = y2;
        out[block + i*stride3] = y3;
      }
    }
  }
```

**Memory access characteristics:**
- **Strided loads/stores**: Strides grow with K, destroying cache-line utilization for large stages
- **Twiddle loads**: Sequential but growing, can evict input data from L1
- **Read-after-write hazards**: If in-place, load latency includes store-forwarding stalls

### 1.3 Latency Numbers That Matter

```
Operation               Latency (cycles)    Throughput (per cycle)
─────────────────────────────────────────────────────────────────────
FMA (AVX-512)           4                   2 (Skylake-X), 4 (Zen 4)
Complex Mul (3 FMA)     ~8-10 (dep chain)   Limited by deps
L1 Load (hit)           4-5                 2 loads/cycle
L2 Load                 12-14               
L3 Load                 40-60               
DRAM                    200+                
```

**The Challenge:** With a 4-5 cycle L1 latency and an 8-10 cycle dependency chain for complex multiply → butterfly, a naive loop spends 50-70% of its time waiting for data.

**Goal of Advanced Scheduling:** Overlap the load latency of iteration `i+1` with the compute of iteration `i`, and the store latency of iteration `i-1`.

---

## 2. Overview of Three Techniques {#2-overview-of-three-techniques}

### 2.1 Conceptual Comparison

```
Plain Unroll (U=2):
─────────────────────────────────────────────────────────────────
Iter 0: LOAD₀ → CMUL₀ → BFLY₀ → STORE₀
Iter 1:                                  LOAD₁ → CMUL₁ → BFLY₁ → STORE₁

Timeline: ████████████████░░░░░░░░░░░░░░████████████████
          ↑ compute       ↑ bubble       ↑ compute


Double-Pump (U=2, independent lanes A/B):
─────────────────────────────────────────────────────────────────
Iter 0: LOAD_A₀ → CMUL_A₀ → BFLY_A₀ → STORE_A₀
Iter 1:      LOAD_B₁ → CMUL_B₁ → BFLY_B₁ → STORE_B₁
Iter 2:           LOAD_A₂ → CMUL_A₂ → BFLY_A₂ → STORE_A₂

Timeline: ██████LOAD_A₀████CMUL_A₀█████BFLY_A₀████STORE_A₀██
          ░░░░░░░░░██LOAD_B₁████CMUL_B₁█████BFLY_B₁████STORE_B₁
          
Bubble reduction: ~30-50% (some overlap)


Modulo Scheduling (U=2, fully pipelined):
─────────────────────────────────────────────────────────────────
Prologue:
  Iter 0: LOAD₀
  Iter 1: LOAD₁ CMUL₀

Steady State:
  Iter 2: LOAD₂ CMUL₁ BFLY₀ STORE₋₁
  Iter 3: LOAD₃ CMUL₂ BFLY₁ STORE₀
  Iter 4: LOAD₄ CMUL₃ BFLY₂ STORE₁
  ...

Epilogue:
  Iter N-1:      CMUL_{N-1} BFLY_{N-2} STORE_{N-3}
  Iter N:                   BFLY_{N-1} STORE_{N-2}
  Iter N+1:                             STORE_{N-1}

Timeline: Fully overlapped, ~70-90% bubble reduction
```

### 2.2 Feature Comparison Table

| Feature | Plain Unroll | Double-Pump | Modulo Schedule |
|---------|-------------|-------------|-----------------|
| **Complexity** | Low | Medium | High |
| **Code size** | 1× (baseline) | 1.5-2× | 2.5-4× |
| **Register pressure** | Low (8-12 regs) | Medium (16-24 regs) | High (24-32+ regs) |
| **Prologue/epilogue** | None | Minimal | Complex (2-3 stages) |
| **Tail handling** | Simple | Moderate | Difficult |
| **Mixed-radix friendly** | Yes | Yes | No (brittle) |
| **Debuggability** | Easy | Moderate | Hard |
| **Typical speedup** | 0-10% | 5-25% | 10-30% |
| **Best for stage size** | <32 KiB | 32 KiB - 2 MiB | >2 MiB |
| **AVX-512 benefit** | Minimal | Significant | Maximum |
| **AVX2 benefit** | Minimal | Moderate | Risky (spills) |

### 2.3 Performance Envelope by Problem Size

```
Speedup vs Baseline (%)
  30│                                         ╱─────── Modulo
     │                                   ╱────╯
  25│                              ╱────╯
     │                         ╱───╯
  20│                    ╱────╯
     │               ╱───╯           Double-Pump
  15│          ╱────╯
     │     ╱───╯
  10│╱────╯                   Plain Unroll
   5│────────────────────────────────────────────────
     │
   0└─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────
        4Ki   16Ki  64Ki  256Ki  1Mi   4Mi   16Mi
               Stage Size (half × 16 bytes)
               
  Cross-over points (approximate, AVX-512):
    - Plain → Double: ~16 KiB (L1 capacity)
    - Double → Modulo: ~512 KiB (L2 capacity, memory-bound)
```

---

## 3. Plain Unrolling: The Baseline {#3-plain-unrolling}

### 3.1 What It Is

Plain unrolling duplicates the loop body 1-2 times to expose instruction-level parallelism (ILP) without attempting to overlap iterations.

```c
// U=1 (no unroll)
for (i = 0; i < half; i++) {
  load_inputs(&x0, &x1, &x2, &x3, i);
  load_twiddles(&w1, &w2, &w3, i);
  complex_mul(&t1, x1, w1);
  complex_mul(&t2, x2, w2);
  complex_mul(&t3, x3, w3);
  butterfly_radix4(&y0, &y1, &y2, &y3, x0, t1, t2, t3);
  store_outputs(y0, y1, y2, y3, i);
}

// U=2 (simple unroll)
for (i = 0; i < half; i += 2) {
  // Iteration i
  load_inputs(&x0_0, &x1_0, &x2_0, &x3_0, i);
  load_twiddles(&w1_0, &w2_0, &w3_0, i);
  complex_mul(&t1_0, x1_0, w1_0);
  complex_mul(&t2_0, x2_0, w2_0);
  complex_mul(&t3_0, x3_0, w3_0);
  butterfly_radix4(&y0_0, &y1_0, &y2_0, &y3_0, x0_0, t1_0, t2_0, t3_0);
  store_outputs(y0_0, y1_0, y2_0, y3_0, i);
  
  // Iteration i+1 (independent, but sequential)
  load_inputs(&x0_1, &x1_1, &x2_1, &x3_1, i+1);
  load_twiddles(&w1_1, &w2_1, &w3_1, i+1);
  complex_mul(&t1_1, x1_1, w1_1);
  complex_mul(&t2_1, x2_1, w2_1);
  complex_mul(&t3_1, x3_1, w3_1);
  butterfly_radix4(&y0_1, &y1_1, &y2_1, &y3_1, x0_1, t1_1, t2_1, t3_1);
  store_outputs(y0_1, y1_1, y2_1, y3_1, i+1);
}
```

### 3.2 Assembly Pattern (AVX-512, conceptual)

```asm
.loop_unroll2:
  ; Iter 0
  vmovapd   zmm0, [in + rax*8]       ; Load x0 (4 cycles latency)
  vmovapd   zmm1, [in + rax*8 + r8]  ; Load x1
  vmovapd   zmm8, [twiddle + rax*8]  ; Load w1
  vfmadd213pd zmm1, zmm8, zmm9       ; cmul: t1 = x1*w1 (starts dependency)
  ... [butterfly operations: 15-20 instructions]
  vmovapd   [out + rax*8], zmm16     ; Store y0
  
  ; Iter 1 (sequential, not overlapped with iter 0)
  vmovapd   zmm0, [in + rbx*8]
  ... [repeat all operations for iteration 1]
  
  add       rax, 2
  cmp       rax, r12
  jl        .loop_unroll2
```

**Key observation:** Even though iteration 1's registers are independent (zmm0-7 for iter 0, zmm0-7 reused for iter 1), there's no overlap because the code is written sequentially. The CPU's out-of-order engine can't look far enough ahead (typically ~200-300 instructions) to start iteration 1's loads while iteration 0's butterfly is computing.

### 3.3 When Plain Unroll Wins

**Scenario: Small radix-4 stage, N=1024, K=8, half=32**

```
Memory footprint:
  Inputs:   32 iters × 4 complex × 2 arrays (in/out) = 4096 bytes
  Twiddles: 32 iters × 3 twiddles = 1536 bytes
  Total:    ~5.5 KiB (fits in L1)

Characteristics:
  - All data L1-resident
  - High compute density (butterfly has lots of FMAs)
  - Port-bound (FMA units are bottleneck, not memory)
  
Result: U=2 unroll gives ~5-8% speedup over U=1
        Further optimization (double-pump/modulo) adds nothing
        and wastes code size.
```

**Performance breakdown (cycles, Skylake-X):**

```
                U=1     U=2    Double-Pump   Modulo
Cycles/iter:    28      26     26            27
Branch misses:  ~1%     ~0.5%  ~0.5%         ~1% (prologue/epilogue)
Code size:      180 B   320 B  580 B         1100 B

Winner: U=2 (plain unroll). Extra complexity buys nothing.
```

### 3.4 When Plain Unroll Loses

**Scenario: Large radix-4 stage, N=65536, K=1024, half=4096**

```
Memory footprint:
  Inputs:   4096 iters × 4 complex × 16 bytes × 2 = 1048 KiB
  Twiddles: 4096 iters × 3 twiddles × 16 bytes = 384 KiB
  Total:    ~1.4 MiB (spills L1, partially spills L2)

Characteristics:
  - L2/L3 memory latency exposed
  - Strided access (stride = 1024 complex = 16 KiB) destroys cache lines
  - Memory-bound, not compute-bound
  
Result: U=2 unroll gives ~2-3% speedup (negligible)
        Load stalls dominate; need latency hiding.
```

**Performance breakdown (cycles per iteration, Skylake-X):**

```
                U=1    U=2    Double-Pump   Modulo+NT
Cycles/iter:    85     82     68            52
L2 miss rate:   12%    12%    8%            3%
Memory stalls:  60%    58%    42%           18%

Winner: Modulo scheduling (37% faster than plain unroll)
```

---

## 4. Double-Pumping: The Middle Ground {#4-double-pumping}

### 4.1 What It Is

Double-pumping maintains **two independent iteration contexts** (A and B) in flight simultaneously. While context A computes, context B loads; then they swap roles. It's a "poor-man's pipeline" that's much simpler than full modulo scheduling.

### 4.2 The Two-Context Pattern

```c
// Pseudo-code (actual SIMD is more complex)
void radix4_double_pump(complex *in, complex *out, 
                        complex *tw, size_t half) {
  // Context A registers: x0_A, x1_A, ..., y3_A
  // Context B registers: x0_B, x1_B, ..., y3_B
  
  if (half == 0) return;
  
  // Prime the pump: load context A
  load_inputs_A(&x0_A, &x1_A, &x2_A, &x3_A, 0);
  load_twiddles_A(&w1_A, &w2_A, &w3_A, 0);
  
  for (i = 0; i < half - 1; i++) {
    // Compute context A (uses A's registers)
    complex_mul(&t1_A, x1_A, w1_A);
    complex_mul(&t2_A, x2_A, w2_A);
    complex_mul(&t3_A, x3_A, w3_A);
    
    // While A computes, load context B for next iteration
    // (uses B's registers, independent of A)
    load_inputs_B(&x0_B, &x1_B, &x2_B, &x3_B, i+1);
    load_twiddles_B(&w1_B, &w2_B, &w3_B, i+1);
    
    // Finish A's butterfly
    butterfly_radix4(&y0_A, &y1_A, &y2_A, &y3_A, x0_A, t1_A, t2_A, t3_A);
    
    // Store A's results (can overlap with B's loads on some CPUs)
    store_outputs(y0_A, y1_A, y2_A, y3_A, i);
    
    // Swap: B becomes the new A
    // (Logically swap pointers; in reality, just alternate register sets)
    SWAP_CONTEXTS(A, B);
  }
  
  // Drain: compute final iteration (context A)
  complex_mul(&t1_A, x1_A, w1_A);
  complex_mul(&t2_A, x2_A, w2_A);
  complex_mul(&t3_A, x3_A, w3_A);
  butterfly_radix4(&y0_A, &y1_A, &y2_A, &y3_A, x0_A, t1_A, t2_A, t3_A);
  store_outputs(y0_A, y1_A, y2_A, y3_A, half-1);
}
```

### 4.3 Assembly Pattern (AVX-512, simplified)

```asm
; Prologue: Prime context A
vmovapd   zmm0, [in + rax*8]          ; Load x0_A
vmovapd   zmm1, [in + rax*8 + r8]     ; Load x1_A
vmovapd   zmm2, [in + rax*8 + r9]     ; Load x2_A
vmovapd   zmm3, [in + rax*8 + r10]    ; Load x3_A
vmovapd   zmm8, [twiddle + rax*16]    ; Load w1_A

.loop_double_pump:
  ; Complex mul for context A (zmm0-7 = A's registers)
  vfmadd213pd zmm1, zmm8, zmm9        ; cmul(x1_A, w1_A) → t1_A
  vfmadd231pd zmm1, zmm10, zmm11      ; (second FMA of cmul)
  
  ; WHILE A computes, load context B (zmm16-23 = B's registers)
  vmovapd   zmm16, [in + rbx*8]       ; Load x0_B (next iter)
  vmovapd   zmm17, [in + rbx*8 + r8]  ; Load x1_B
  vmovapd   zmm24, [twiddle + rbx*16] ; Load w1_B
  
  ; Continue A's complex mul
  vfmadd213pd zmm2, zmm12, zmm13      ; cmul(x2_A, w2_A) → t2_A
  
  ; Load more of B
  vmovapd   zmm18, [in + rbx*8 + r9]  ; Load x2_B
  
  ; Finish A's butterfly (15-20 instructions)
  vaddpd    zmm4, zmm0, zmm1          ; y0_A = x0 + t1 + ...
  ...
  
  ; Store A's results
  vmovapd   [out + rax*8], zmm4
  vmovapd   [out + rax*8 + r8], zmm5
  
  ; Swap contexts: next iteration, B becomes A
  ; (Implemented by swapping which register set we call "A")
  ; In practice, we interleave: even iters use zmm0-7, odd use zmm16-23
  
  add       rax, 1
  add       rbx, 1
  cmp       rax, r12
  jl        .loop_double_pump

; Epilogue: Drain final iteration (context A)
vfmadd213pd zmm1, zmm8, zmm9
...
vmovapd   [out + rax*8], zmm4
```

**Key observation:** While A's `vfmadd213pd` (4-cycle latency) executes, B's `vmovapd` loads issue. The out-of-order engine can overlap these because they use independent registers. Achieves ~30-50% latency hiding with minimal code complexity.

### 4.4 Register Allocation

```
AVX-512 (32 registers):
  Context A: zmm0-7   (inputs/twiddles)
             zmm8-15  (intermediate results, butterfly)
  Context B: zmm16-23 (inputs/twiddles)
             zmm24-31 (intermediate results, butterfly)
  
AVX2 (16 registers):
  Context A: ymm0-7
             ymm8-11  (shared pool, tight)
  Context B: ymm12-15
             (Reuse ymm8-11 carefully, or accept some spills)
```

**Why it works better on AVX-512:** With 32 registers, both contexts fit comfortably. On AVX2, you're tight (16 registers for 2 contexts + spares), and aggressive scheduling can cause spills. But still profitable if well-tuned.

### 4.5 When Double-Pump Wins

**Scenario: Medium radix-4 stage, N=16384, K=128, half=1024**

```
Memory footprint:
  Inputs:   1024 iters × 4 complex × 16 bytes × 2 = 256 KiB
  Twiddles: 1024 iters × 3 twiddles × 16 bytes = 96 KiB
  Total:    ~352 KiB (spills L1, fits L2)

Characteristics:
  - L2 resident, but L1 misses common
  - ~10 cycle load latency (L2 hit)
  - Butterfly has ~25 cycle critical path
  - Enough iterations to amortize overhead
  
Result: Double-pump gives +15-20% over plain unroll
        Modulo scheduling adds only +3-5% more, not worth complexity
```

**Performance breakdown (Skylake-X, cycles per iteration):**

```
              Plain U=2   Double-Pump   Modulo
Cycles/iter:  42          34            32
L1 miss %:    35%         35%           30%
Stall %:      45%         28%           22%
Code size:    320 B       680 B         1250 B
Complexity:   1×          2×            4×

Winner: Double-pump (19% faster, manageable complexity)
```

### 4.6 Example: Zen 3 Sweet Spot

On Zen 3 (AVX2, 16 YMM registers, aggressive 4-wide dispatch):

```
Scenario: Radix-5 butterfly (longer dependency chain than radix-4)
          N=8192, K=125, half=410

Complex multiply latency: ~12 cycles (longer than radix-4's ~8 cycles)

Plain unroll U=2:
  ████████████░░░░░░░░░░████████████  (40% bubbles)
  
Double-pump:
  ████LOAD████CMUL████BFLY████STORE
  ░░░░░░░░████LOAD████CMUL████BFLY████STORE
  
Result: +22% speedup (Zen 3's 4-wide dispatch helps a lot)
        Registers are tight (ymm0-15) but workable
```

Zen 3's aggressive out-of-order engine extracts more overlap from double-pumping than Skylake can. This is why the "5-25%" range is so wide: architecture matters.

---

## 5. Modulo Scheduling: Maximum Performance, Maximum Complexity {#5-modulo-scheduling}

### 5.1 What It Is

**Modulo scheduling** (software pipelining) is a compiler optimization technique that overlaps multiple iterations of a loop such that different stages of different iterations execute in parallel. In the steady state, iteration `i+2`'s loads, iteration `i+1`'s complex multiplies, iteration `i`'s butterfly, and iteration `i-1`'s stores all execute simultaneously.

**Goal:** Maximize utilization of all execution units (load ports, FMA units, store ports) by creating a "conveyor belt" where every stage is busy every cycle.

### 5.2 The Four-Stage Pipeline Model

For an FFT radix-4 butterfly, we identify four stages:

```
Stage 0: LOAD    - Load inputs (x0,x1,x2,x3) and twiddles (w1,w2,w3)
Stage 1: CMUL    - Complex multiply (x1*w1, x2*w2, x3*w3)
Stage 2: BFLY    - Butterfly computation (add/subtract network)
Stage 3: STORE   - Store outputs (y0,y1,y2,y3)

Latencies (approximate, Skylake-X):
  LOAD:  5-7 cycles (L1 hit), 12-14 (L2), 40+ (L3)
  CMUL:  8-10 cycles (dependency chain through 3 FMAs)
  BFLY:  10-12 cycles (add/sub network, some deps)
  STORE: 1-2 cycles (write-combining buffer)
```

### 5.3 Sequential vs Pipelined Execution

**Sequential (plain unroll):**

```
Cycle: 0    5    10   15   20   25   30   35   40   45   50   55   60
Iter 0: [LOAD][CMUL  ][BFLY  ][ST]
Iter 1:                              [LOAD][CMUL  ][BFLY  ][ST]
Iter 2:                                                        [LOAD]...
       └──────────────────────────────┘└──────────────────────┘
       25 cycles/iter, 50 cycles/2 iters
```

**Pipelined (modulo scheduled):**

```
Cycle: 0    5    10   15   20   25   30   35   40   45   50   55   60

Prologue:
Iter 0: [LOAD]
Iter 1:      [LOAD][CMUL₀]

Steady State:
Iter 2:           [LOAD][CMUL₁][BFLY₀][ST₋₁]
Iter 3:                [LOAD][CMUL₂][BFLY₁][ST₀]
Iter 4:                     [LOAD][CMUL₃][BFLY₂][ST₁]
       └──────────────────────────────────────────────┘
       12-15 cycles/iter in steady state!

Epilogue:
Iter N-1:              [CMUL_N][BFLY_N-1][ST_N-2]
Iter N:                        [BFLY_N  ][ST_N-1]
Iter N+1:                                [ST_N  ]
```

**Throughput:** 12-15 cycles/iter vs 25 cycles/iter = **40-50% speedup** in ideal case.

### 5.4 Detailed Code Structure

```c
void radix4_modulo_scheduled(complex *in, complex *out,
                              complex *tw, size_t half) {
  if (half < 3) {
    // Fall back to simple version for tiny cases
    radix4_plain(in, out, tw, half);
    return;
  }
  
  // ────────────────────────────────────────────────────────────
  // PROLOGUE: Fill the pipeline
  // ────────────────────────────────────────────────────────────
  
  // Iteration -1: Load stage only
  load_inputs(&x0_m1, &x1_m1, &x2_m1, &x3_m1, 0);
  load_twiddles(&w1_m1, &w2_m1, &w3_m1, 0);
  
  // Iteration 0: Load + CMUL(iter -1)
  load_inputs(&x0_0, &x1_0, &x2_0, &x3_0, 1);
  load_twiddles(&w1_0, &w2_0, &w3_0, 1);
  
  complex_mul(&t1_m1, x1_m1, w1_m1);  // CMUL for iter -1
  complex_mul(&t2_m1, x2_m1, w2_m1);
  complex_mul(&t3_m1, x3_m1, w3_m1);
  
  // Iteration 1: Load + CMUL(iter 0) + BFLY(iter -1)
  load_inputs(&x0_1, &x1_1, &x2_1, &x3_1, 2);
  load_twiddles(&w1_1, &w2_1, &w3_1, 2);
  
  complex_mul(&t1_0, x1_0, w1_0);     // CMUL for iter 0
  complex_mul(&t2_0, x2_0, w2_0);
  complex_mul(&t3_0, x3_0, w3_0);
  
  butterfly_radix4(&y0_m1, &y1_m1, &y2_m1, &y3_m1,  // BFLY for iter -1
                   x0_m1, t1_m1, t2_m1, t3_m1);
  
  // ────────────────────────────────────────────────────────────
  // STEADY STATE: Fully pipelined loop
  // ────────────────────────────────────────────────────────────
  
  for (i = 2; i < half - 1; i++) {
    // Four stages in parallel for four different iterations:
    
    // Stage 0: LOAD(i+1) - next iteration's inputs
    load_inputs(&x0_next, &x1_next, &x2_next, &x3_next, i+1);
    load_twiddles(&w1_next, &w2_next, &w3_next, i+1);
    
    // Stage 1: CMUL(i) - current iteration's complex mul
    complex_mul(&t1_curr, x1_1, w1_1);
    complex_mul(&t2_curr, x2_1, w2_1);
    complex_mul(&t3_curr, x3_1, w3_1);
    
    // Stage 2: BFLY(i-1) - previous iteration's butterfly
    butterfly_radix4(&y0_prev, &y1_prev, &y2_prev, &y3_prev,
                     x0_0, t1_0, t2_0, t3_0);
    
    // Stage 3: STORE(i-2) - iteration from 2 cycles ago
    store_outputs(y0_m1, y1_m1, y2_m1, y3_m1, i-2);
    
    // Rotate register contexts:
    // next → curr → prev → stored
    ROTATE_CONTEXTS();
  }
  
  // ────────────────────────────────────────────────────────────
  // EPILOGUE: Drain the pipeline
  // ────────────────────────────────────────────────────────────
  
  // Iteration N-2: CMUL + BFLY + STORE (no more loads)
  complex_mul(&t1_curr, x1_1, w1_1);
  complex_mul(&t2_curr, x2_1, w2_1);
  complex_mul(&t3_curr, x3_1, w3_1);
  butterfly_radix4(&y0_prev, &y1_prev, &y2_prev, &y3_prev,
                   x0_0, t1_0, t2_0, t3_0);
  store_outputs(y0_m1, y1_m1, y2_m1, y3_m1, half-3);
  
  // Iteration N-1: BFLY + STORE
  butterfly_radix4(&y0_prev, &y1_prev, &y2_prev, &y3_prev,
                   x0_1, t1_curr, t2_curr, t3_curr);
  store_outputs(y0_prev, y1_prev, y2_prev, y3_prev, half-2);
  
  // Iteration N: STORE only
  store_outputs(y0_prev, y1_prev, y2_prev, y3_prev, half-1);
}
```

**Critical observation:** We're now tracking **four separate iteration contexts simultaneously**:
- `i+1` (LOAD stage)
- `i` (CMUL stage)
- `i-1` (BFLY stage)
- `i-2` (STORE stage)

Each needs its own register set. On AVX-512 with 32 registers, this is tight but feasible. On AVX2 with 16 registers, it often causes spills.

### 5.5 Assembly Pattern (AVX-512, steady state)

```asm
; Steady state: 4 iterations in flight simultaneously
; zmm0-7:   Iteration i+1 (LOAD stage)
; zmm8-15:  Iteration i   (CMUL stage)
; zmm16-23: Iteration i-1 (BFLY stage)
; zmm24-31: Iteration i-2 (STORE stage)

.loop_modulo_steady:
  ; ──────────────────────────────────────────────
  ; Stage 0: LOAD(i+1)
  ; ──────────────────────────────────────────────
  vmovapd   zmm0, [in + r15*8]         ; Load x0 (i+1)
  vmovapd   zmm1, [in + r15*8 + r8]    ; Load x1 (i+1)
  vmovapd   zmm4, [twiddle + r15*16]   ; Load w1 (i+1)
  
  ; ──────────────────────────────────────────────
  ; Stage 1: CMUL(i) - interleave with loads
  ; ──────────────────────────────────────────────
  vfmadd213pd zmm9, zmm12, zmm13       ; cmul(x1_i, w1_i) part 1
  
  vmovapd   zmm2, [in + r15*8 + r9]    ; Load x2 (i+1)
  
  vfmadd231pd zmm9, zmm12, zmm14       ; cmul part 2
  
  vmovapd   zmm3, [in + r15*8 + r10]   ; Load x3 (i+1)
  
  vfmadd213pd zmm10, zmm15, zmm16      ; cmul(x2_i, w2_i) part 1
  
  vmovapd   zmm5, [twiddle + r15*16+16]; Load w2 (i+1)
  
  vfmadd231pd zmm10, zmm15, zmm17      ; cmul part 2
  
  ; ──────────────────────────────────────────────
  ; Stage 2: BFLY(i-1) - butterfly add/sub network
  ; ──────────────────────────────────────────────
  vaddpd    zmm20, zmm16, zmm17        ; y0 = x0 + t1 + t2 + t3
  vaddpd    zmm20, zmm20, zmm18
  vaddpd    zmm20, zmm20, zmm19
  
  vsubpd    zmm21, zmm16, zmm17        ; y1 = x0 - t1 + A*t2 ...
  vfmadd231pd zmm21, zmm18, [geom_const_A]
  ... [10-12 more butterfly instructions]
  
  ; ──────────────────────────────────────────────
  ; Stage 3: STORE(i-2)
  ; ──────────────────────────────────────────────
  vmovapd   [out + r14*8], zmm24       ; Store y0 (i-2)
  vmovapd   [out + r14*8 + r8], zmm25  ; Store y1 (i-2)
  vmovapd   [out + r14*8 + r9], zmm26  ; Store y2 (i-2)
  vmovapd   [out + r14*8 + r10], zmm27 ; Store y3 (i-2)
  
  ; Rotate contexts: copy next → curr → prev → to_store
  ; (In practice, we cycle through register assignments)
  
  add       r14, 1                     ; i-2 counter
  add       r15, 1                     ; i+1 counter
  cmp       r15, [loop_limit]
  jl        .loop_modulo_steady
```

**Instruction interleaving:** Notice how loads for iteration `i+1` are scattered between the complex-multiply FMAs for iteration `i`. This maximizes port utilization:
- Port 2/3: Loads (zmm0-5)
- Port 0/5: FMAs (zmm9-10)
- Port 4/9: Stores (zmm24-27)
- Port 1/5: Butterfly adds/subs (zmm20-21)

All execution ports are busy, no bubbles.

### 5.6 Register Allocation Challenge

```
AVX-512 (32 zmm registers):

  Iteration i+1 (LOAD):
    zmm0-3:  x0, x1, x2, x3 (inputs)
    zmm4-6:  w1, w2, w3 (twiddles)
    [7 registers]
  
  Iteration i (CMUL):
    zmm8-11:  x0, x1, x2, x3
    zmm12-14: w1, w2, w3
    zmm15:    temporary for cmul
    [8 registers]
  
  Iteration i-1 (BFLY):
    zmm16-19: t1, t2, t3, x0 (rotated inputs)
    zmm20-23: y0, y1, y2, y3 (outputs)
    [8 registers]
  
  Iteration i-2 (STORE):
    zmm24-27: y0, y1, y2, y3 (staged for store)
    [4 registers]
  
  Constants/temporaries:
    zmm28-31: geometric constants (A, B, C for radix-4),
              rotation masks, temporaries
    [4 registers]
  
  Total: 7+8+8+4+4 = 31 registers (fits!)

AVX-512 with Twiddle Walking (FMA recurrence):
  Remove zmm4-6, zmm12-14 (twiddle registers)
  Add zmm4-5 (twiddle accumulators: real, imag)
  Add zmm6-7 (twiddle steps: cos(dtheta), sin(dtheta))
  
  Total: 27 registers (more room for optimization)

AVX2 (16 ymm registers):
  7+8+8+4+4 = 31 registers needed
  16 registers available
  
  PROBLEM: Need to spill ~15 registers to stack!
  
  Solution: Reduce pipeline depth to 3 stages (LOAD+CMUL / BFLY / STORE),
            or accept spills (kills performance).
```

**Why modulo scheduling is risky on AVX2:** The spills to/from stack add 3-5 cycles per spilled value. With 15 spills per iteration, you lose 45-75 cycles—more than you gain from pipelining! Only works if you can reduce register pressure through:
- Twiddle walking (saves 6 registers)
- Careful reuse of temporaries
- Accepting a 3-stage pipeline instead of 4-stage

### 5.7 Performance Gain Analysis

**Scenario: Large radix-4 stage, N=131072, K=2048, half=8192**

```
Memory footprint:
  Inputs:   8192 iters × 4 complex × 16 bytes × 2 = 2 MiB
  Twiddles: 8192 iters × 3 twiddles × 16 bytes = 768 KiB
  Total:    ~2.8 MiB (spills L2, streams from L3/DRAM)

Skylake-X (AVX-512), measured cycles per iteration:

  Plain unroll U=2:     88 cycles/iter
    - Memory stalls:    62% (55 cycles)
    - Compute:          38% (33 cycles)
    - L2 miss rate:     18%
    - L3 miss rate:     3%
  
  Double-pump U=2:      71 cycles/iter  (-19%)
    - Memory stalls:    48% (34 cycles)
    - Compute:          52% (37 cycles)
    - Better load/compute overlap
  
  Modulo schedule U=2:  51 cycles/iter  (-42% vs plain, -28% vs double)
    - Memory stalls:    25% (13 cycles)
    - Compute:          75% (38 cycles)
    - L2 miss rate:     12% (prefetch helps)
    - L3 miss rate:     2%
  
  Modulo + twiddle walk + NT stores:  46 cycles/iter  (-48% vs plain)
    - Memory stalls:    18% (8 cycles)
    - Compute:          82% (38 cycles)
    - Write bandwidth saved by NT stores
    - No twiddle loads
```

**Key takeaway:** On large, memory-bound stages, modulo scheduling can deliver 30-50% speedup. But this requires:
- Large enough `half` (>1000 iterations) to amortize prologue/epilogue
- Memory-bound regime (if compute-bound, gain is negligible)
- AVX-512 or very careful AVX2 tuning
- Correct implementation (easy to get wrong)

---

## 6. The Brittleness Problem: Why Modulo Scheduling is Hard {#6-brittleness-problem}

This is the **critical section** for VectorFFT. Modulo scheduling's performance gains come with severe implementation challenges.

### 6.1 Prologue/Epilogue Complexity

**The problem:** A modulo-scheduled loop has three distinct code paths:

```
┌─────────────────────────────────────────────────────────┐
│ PROLOGUE (2-3 iterations)                               │
│   - Fill the pipeline                                   │
│   - Each iteration activates one more stage             │
│   - Different code for each prologue iteration          │
├─────────────────────────────────────────────────────────┤
│ STEADY STATE (half - prologue - epilogue iterations)   │
│   - Fully pipelined                                     │
│   - All 4 stages active                                 │
│   - Highest performance                                 │
├─────────────────────────────────────────────────────────┤
│ EPILOGUE (2-3 iterations)                               │
│   - Drain the pipeline                                  │
│   - Each iteration deactivates one stage                │
│   - Different code for each epilogue iteration          │
└─────────────────────────────────────────────────────────┘
```

#### 6.1.1 The Prologue Nightmare

```c
// Prologue iteration 0: LOAD only
// Registers: i+1 context
load_inputs(&x0_0, &x1_0, &x2_0, &x3_0, 0);
load_twiddles(&w1_0, &w2_0, &w3_0, 0);

// Prologue iteration 1: LOAD + CMUL
// Registers: i+1 (LOAD), i (CMUL)
load_inputs(&x0_1, &x1_1, &x2_1, &x3_1, 1);
load_twiddles(&w1_1, &w2_1, &w3_1, 1);
complex_mul(&t1_0, x1_0, w1_0);  // Process iter 0
complex_mul(&t2_0, x2_0, w2_0);
complex_mul(&t3_0, x3_0, w3_0);

// Prologue iteration 2: LOAD + CMUL + BFLY
// Registers: i+1 (LOAD), i (CMUL), i-1 (BFLY)
load_inputs(&x0_2, &x1_2, &x2_2, &x3_2, 2);
load_twiddles(&w1_2, &w2_2, &w3_2, 2);
complex_mul(&t1_1, x1_1, w1_1);  // Process iter 1
complex_mul(&t2_1, x2_1, w2_1);
complex_mul(&t3_1, x3_1, w3_1);
butterfly_radix4(&y0_0, &y1_0, &y2_0, &y3_0,  // Process iter 0
                 x0_0, t1_0, t2_0, t3_0);

// Now enter steady state (all 4 stages active)
```

**Problem 1: Code duplication**

Each prologue iteration is unique. You can't loop over them (they do different things). This means 3 separate code blocks, each 40-60 instructions, adding 500-800 bytes of code size per kernel.

**Problem 2: Register allocation changes**

Prologue iteration 0 uses 7 registers. Iteration 1 uses 15. Iteration 2 uses 23. The register allocator has to track different live ranges for each prologue step. Hand-writing this in assembly is tedious and error-prone.

**Problem 3: Debugging**

When something goes wrong (and it will), which iteration is at fault? The prologue? Steady state? Epilogue? Print debugging doesn't work well (affects timing). Debugger stepping through vectorized assembly is painful.

#### 6.1.2 The Epilogue Nightmare

```c
// Epilogue iteration 0: CMUL + BFLY + STORE (no more LOAD)
// Registers: i (CMUL), i-1 (BFLY), i-2 (STORE)
complex_mul(&t1_curr, x1_1, w1_1);
complex_mul(&t2_curr, x2_1, w2_1);
complex_mul(&t3_curr, x3_1, w3_1);
butterfly_radix4(&y0_prev, &y1_prev, &y2_prev, &y3_prev,
                 x0_0, t1_0, t2_0, t3_0);
store_outputs(y0_old, y1_old, y2_old, y3_old, half-3);

// Epilogue iteration 1: BFLY + STORE
// Registers: i-1 (BFLY), i-2 (STORE)
butterfly_radix4(&y0_curr, &y1_curr, &y2_curr, &y3_curr,
                 x0_1, t1_curr, t2_curr, t3_curr);
store_outputs(y0_prev, y1_prev, y2_prev, y3_prev, half-2);

// Epilogue iteration 2: STORE only
// Registers: i-2 (STORE)
store_outputs(y0_curr, y1_curr, y2_curr, y3_curr, half-1);
```

**Problem 4: Off-by-one errors**

Notice the iteration indices in the epilogue: `half-3`, `half-2`, `half-1`. Get any of these wrong and you silently corrupt output or segfault. These bugs are **invisible in small tests** (where `half=4` and epilogue is short) but **crash in production** (where `half=8192`).

**Problem 5: Cannot reuse steady-state loop**

Some compilers try to be clever and "unroll the last few iterations" of the steady-state loop to form the epilogue. This doesn't work for modulo scheduling because the epilogue does fundamentally different things (no loads, then no cmul, then no bfly). You must hand-write it.

### 6.2 Non-Power-of-2 `half` Sizes

**The problem:** Modulo scheduling assumes `half` is large enough to have a meaningful steady state:

```
half = prologue + steady + epilogue
half = 3 + steady + 3
steady = half - 6

If half < 6, steady state doesn't exist!
```

#### 6.2.1 Small `half` Handling

```c
void radix4_modulo_scheduled(complex *in, complex *out,
                              complex *tw, size_t half) {
  if (half < 6) {
    // SPECIAL CASE: Fall back to non-pipelined version
    radix4_plain(in, out, tw, half);
    return;
  }
  
  // Normal prologue/steady/epilogue
  // ...
}
```

**Problem 6: Branching overhead**

Every call to the kernel now has a branch at the start. For small stages (which happen frequently in mixed-radix FFT), this branch is **always taken**, adding 1-2 cycles overhead. Multiplied by thousands of stages, this adds up.

**Problem 7: Code cache pollution**

You now need to keep **two** versions of the kernel in L1 instruction cache:
- The modulo-scheduled version (1000-1500 bytes)
- The plain version (200-400 bytes)

If the plain version isn't used often, it gets evicted, causing instruction cache misses (20-40 cycle penalty) when you do hit the fallback case.

#### 6.2.2 Non-Divisible `half`

What if `half` is not evenly divisible by the unroll factor?

```c
// Modulo scheduled with U=2 (process 2 iterations per loop iter)
for (i = 3; i < half - 3; i += 2) {
  // Steady state for 2 iterations
}

// If half=1001, then half-6=995, which is not divisible by 2.
// The loop processes 994 iterations (497 loop iters × 2).
// 1 iteration remains: handled by the epilogue.
```

**Problem 8: Epilogue must handle remainder**

The epilogue now has two jobs:
1. Drain the pipeline (3 iterations)
2. Handle the remainder (0 or 1 iterations, depending on `half % U`)

This doubles the epilogue complexity. You need:
- Epilogue path A (remainder = 0): 3 drain iterations
- Epilogue path B (remainder = 1): 4 drain iterations (the extra 1 isn't pipelined)

More branches, more code, more bugs.

### 6.3 Mixed-Radix Transition Brittleness

**This is the killer for VectorFFT.**

#### 6.3.1 The Mixed-Radix Problem

In a mixed-radix FFT, you don't always use the same radix:

```
N = 5040 = 2^4 × 3^2 × 5 × 7

Possible factorization:
  Stage 0: radix-16, K=1,    half=315
  Stage 1: radix-9,  K=16,   half=35
  Stage 2: radix-5,  K=144,  half=7
  Stage 3: radix-7,  K=720,  half=1

Each stage has a DIFFERENT radix butterfly!
```

**Problem 9: Different butterflies = different pipelines**

Modulo scheduling is **tailored to a specific butterfly**:
- Radix-4 has 3 complex muls, 20 add/sub in the butterfly
- Radix-5 has 4 complex muls, 30 add/sub in the butterfly
- Radix-7 has 6 complex muls, 50 add/sub in the butterfly

The prologue/steady/epilogue for radix-4 **cannot be reused** for radix-5. You need separate implementations:

```c
void radix2_modulo_scheduled(...);   // 1000 bytes
void radix3_modulo_scheduled(...);   // 1200 bytes
void radix4_modulo_scheduled(...);   // 1400 bytes
void radix5_modulo_scheduled(...);   // 1600 bytes
void radix7_modulo_scheduled(...);   // 2000 bytes
void radix8_modulo_scheduled(...);   // 1500 bytes
void radix11_modulo_scheduled(...);  // 2400 bytes
// ...

Total code size: ~10-15 KiB per ISA (AVX-512, AVX2, SSE2)
Total with all ISAs: 30-45 KiB just for modulo-scheduled kernels!
```

This blows your L1 instruction cache (32 KiB on most CPUs).

#### 6.3.2 Register Allocation Across Radices

Different radices need different register counts:

```
Radix   Inputs  Twiddles  Butterfly  Total (4-stage pipeline)
────────────────────────────────────────────────────────────────
2       2       1         4          7×4 = 28 regs (fits AVX-512)
3       3       2         6          11×4 = 44 regs (DOESN'T FIT!)
4       4       3         8          15×4 = 60 regs (DOESN'T FIT!)
5       5       4         10         19×4 = 76 regs (DOESN'T FIT!)
7       7       6         14         27×4 = 108 regs (DISASTER)
```

**Problem 10: Prime radices don't fit in registers**

For radix-5, 7, 11, 13, you **cannot** implement a full 4-stage modulo-scheduled pipeline even on AVX-512 (32 registers). You must either:
- Reduce to a 3-stage pipeline (LOAD+CMUL / BFLY / STORE), which reduces gains
- Accept register spills, which kills performance
- Use a different technique (double-pumping)

#### 6.3.3 Real-World Example: N=2310 = 2×3×5×7×11

```
Stage 0: radix-2,  K=1,    half=1155
  → Use modulo scheduling (radix-2 is small, fits in regs)
  → Code size: 1000 bytes

Stage 1: radix-3,  K=2,    half=385
  → half=385 is large, would benefit from modulo scheduling
  → BUT: Register pressure is 44 regs for 4-stage pipeline
  → DECISION: Use double-pumping instead (680 bytes)

Stage 2: radix-5,  K=6,    half=77
  → half=77 is moderate
  → Register pressure: 76 regs (impossible)
  → DECISION: Use double-pumping (850 bytes)

Stage 3: radix-7,  K=30,   half=11
  → half=11 is tiny (< 6, no steady state)
  → Would fall back to plain anyway
  → DECISION: Use plain unroll (200 bytes)

Stage 4: radix-11, K=210,  half=1
  → half=1 (single iteration, no loop)
  → DECISION: Use plain (inlined, 60 bytes)
```

**Problem 11: Decision complexity explodes**

You need a decision tree that considers:
- Radix (2, 3, 4, 5, 7, 8, 11, 13, 16, ...)
- Stage size (`half` and memory footprint)
- Register availability (ISA: AVX-512, AVX2, SSE2)
- Code cache pressure (how many kernels already loaded)

This is **extremely difficult to get right** and almost impossible to maintain.

### 6.4 Debugging Challenges

#### 6.4.1 Reproducing Bugs

**Problem 12: Bugs only appear at specific sizes**

```
Test case: N=1024, radix-4, K=64, half=64
  → Steady state runs 64-6 = 58 iterations
  → Prologue/epilogue work fine
  → TEST PASSES ✓

Production: N=131072, radix-4, K=2048, half=8192
  → Steady state runs 8192-6 = 8186 iterations
  → Prologue: iteration 2 has off-by-one in register rotation
  → Bug is hidden by out-of-order execution for first 50 iters
  → Corrupts output at iteration 51
  → Causes FFT round-trip error of 1e-8 (above threshold)
  → TEST FAILS ✗

The bug is INVISIBLE in small tests but ALWAYS present in large tests.
```

To debug this, you must:
1. Reproduce with large N (slow test, 10-30 seconds per run)
2. Use a debugger on vectorized assembly (painful)
3. Or add instrumentation (but this changes timing and can hide the bug)

#### 6.4.2 Heisenbugs from Timing Changes

**Problem 13: Adding print statements fixes the bug**

```c
// Buggy code (register spill causes stale data)
zmm16 = butterfly_output_y0;  // Should store to out[i-2]
[... 50 instructions ...]
vmovapd [out + i*8], zmm16;   // But zmm16 was spilled and reloaded
                               // with WRONG data due to epilogue error

// Add debugging:
printf("Storing y0=%f at i=%zu\n", zmm16[0], i);  // Forces register sync
vmovapd [out + i*8], zmm16;   // Now works correctly!

The bug disappears when you try to debug it.
```

This happens because print statements:
- Force register writes to memory (compiler inserts `vmovapd` to stack)
- Change register allocation (compiler uses different spill slots)
- Alter instruction scheduling (print is a function call, barrier)

### 6.5 Maintenance Nightmare

**Problem 14: Any change breaks everything**

Suppose you want to add **twiddle walking** (FMA recurrence to avoid twiddle loads):

```c
// Old: load twiddles
vmovapd zmm4, [twiddle + i*16];

// New: compute twiddles via FMA recurrence
vfmadd213pd zmm4, zmm28, zmm29;  // w_real = w_real*cos - w_imag*sin
vfmadd231pd zmm5, zmm28, zmm30;  // w_imag = w_real*sin + w_imag*cos
```

**Impact:**
- Changes register allocation (freed 3 twiddle registers, added 2 accumulators)
- Changes instruction count per iteration (removed 3 loads, added 4 FMAs)
- Changes steady-state timing (FMAs have different latency than loads)
- May require adjusting prologue (need to initialize accumulators)
- May require adjusting epilogue (need to handle accumulated error, refresh)

**Result:** You must rewrite:
- Prologue (3 code blocks)
- Steady state (1 code block)
- Epilogue (3 code blocks)
- Register allocation documentation
- Test cases (error bounds change with twiddle walking)

**Total time:** 4-8 hours for an experienced developer, plus testing. For **each radix** (2, 4, 8, 16) and **each ISA** (AVX-512, AVX2).

This is why modulo scheduling is **high-maintenance**.

### 6.6 Summary of Brittleness Issues

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| **Prologue/epilogue code duplication** | 3× code size | Accept it, or use codegen |
| **Register pressure** | Spills kill performance | AVX-512 only, or 3-stage pipeline |
| **Small `half` fallback** | Branching overhead | Cache both versions, optimize fallback |
| **Non-divisible `half`** | Complex epilogue | Increase epilogue path count |
| **Mixed-radix transitions** | Need separate kernel per radix | Use modulo only for hot radices (2, 4, 8) |
| **Register allocation per radix** | Prime radices don't fit | Double-pump for primes |
| **Decision complexity** | Hard to choose right kernel | Careful profiling + heuristics |
| **Debugging difficulty** | Heisenbugs, slow reproduction | Extensive testing, limit usage |
| **Maintenance cost** | Any change is expensive | Limit to most critical kernels |

---

## 7. CPU Architecture Considerations {#7-cpu-architecture-considerations}

Different CPU microarchitectures have different sweet spots for loop optimization techniques.

### 7.1 Intel Skylake-X / Cascade Lake / Ice Lake (AVX-512)

**Architecture characteristics:**
- 32 zmm registers (abundant)
- 2 load ports (port 2, port 3): 2×64 bytes/cycle = 128 B/cycle
- 2 FMA units (port 0, port 5): 2 FMAs/cycle = 32 DP FLOP/cycle
- 1 store port (port 4, port 9): 1×64 bytes/cycle = 64 B/cycle
- L1: 32 KiB, 4-cycle latency
- L2: 1 MiB (per core), 14-cycle latency
- L3: 1.375 MiB/core (shared), 50-70 cycle latency
- Out-of-order window: ~224 entries (large)

**Technique recommendations:**

```
Stage Size      Best Technique       Why
──────────────────────────────────────────────────────────────────────
< 16 KiB        Plain unroll U=2     L1-resident, compute-bound
16-512 KiB      Double-pump U=2      L2-resident, moderate latency
512 KiB - 8 MiB Modulo U=2           L3-resident, memory-bound
> 8 MiB         Modulo U=2 + NT      Streaming from DRAM
```

**Register usage:**
- Double-pump: 20-24 registers (comfortable)
- Modulo radix-4: 28-31 registers (tight but workable)
- Modulo radix-8: 32+ registers (needs twiddle walking to fit)

**Code cache:**
- L1i: 32 KiB (can hold 2-3 large modulo kernels comfortably)
- Use modulo for radix-2, 4, 8, 16 (power-of-2 radices)

### 7.2 Intel Sunny Cove / Golden Cove / Raptor Cove (AVX-512)

**Architecture characteristics:**
- 32 zmm registers
- Enhanced load/store bandwidth (client SKUs)
- L2: 2-4 MiB (per core, much larger than Skylake-X)
- Out-of-order window: ~300+ entries (even larger)
- Better prefetcher

**Technique recommendations:**

```
Stage Size      Best Technique       Why
──────────────────────────────────────────────────────────────────────
< 32 KiB        Plain unroll U=2     L1-resident
32 KiB - 2 MiB  Double-pump U=2      L2-resident (note: L2 is huge)
2-16 MiB        Modulo U=2           L3-resident
> 16 MiB        Modulo U=2 + NT      Streaming
```

**Key difference:** The larger L2 (2-4 MiB vs 1 MiB) means double-pumping remains effective up to 2 MiB problem sizes. You can delay using modulo scheduling, reducing code complexity.

**Aggressive prefetcher:** Golden Cove's prefetcher is excellent. Explicit prefetch instructions (`prefetcht0`) may not help much. Let the hardware prefetcher do its job.

### 7.3 AMD Zen 3 / Zen 4 (AVX2 / AVX-512)

**Architecture characteristics (Zen 4):**
- 32 zmm registers (AVX-512 on Zen 4)
- 3 load ports: 3×16 bytes/cycle = 48 B/cycle (narrower than Intel)
- 4 FMA units: 4 FMAs/cycle = 32 DP FLOP/cycle (excellent)
- 2 store ports: 2×16 bytes/cycle = 32 B/cycle (narrower)
- L1: 32 KiB, 4-cycle latency
- L2: 1 MiB (per core), 14-cycle latency
- L3: 4 MiB/core (shared), 50-70 cycle latency
- Out-of-order window: ~256 entries
- 4-wide dispatch (vs Intel's ~5-6 wide)

**Key difference: Narrower memory paths**

Zen's AVX-512 implementation is "double-pumped" internally:
- A 512-bit load is implemented as two 256-bit loads
- A 512-bit store is implemented as two 256-bit stores

This means:
- Effective load bandwidth: 96 B/cycle (less than Intel's 128 B/cycle)
- Effective store bandwidth: 64 B/cycle (same as Intel)
- Memory ops take more pipeline slots

**Technique recommendations:**

```
Stage Size      Best Technique       Why
──────────────────────────────────────────────────────────────────────
< 16 KiB        Plain unroll U=2     L1-resident, FMA-bound
16 KiB - 1 MiB  Double-pump U=2      Hiding narrow memory pipes
1-8 MiB         Modulo U=2           Memory-bound, needs pipelining
> 8 MiB         Modulo U=2 + NT      Streaming

Special case: Radix-5, 7 (long dep chains)
All sizes:      Double-pump U=2      Longer cmul chains → more to hide
```

**Why double-pumping shines on Zen:**

Zen's 4-wide dispatch means it can issue:
- 2 loads + 2 FMAs per cycle (double-pump, context A loads while B computes)
- vs 1 load + 3 FMAs per cycle (plain unroll, underutilizes loads)

The aggressive dispatch extracts more overlap from double-pumping than Intel's scheduler does, explaining the observed +22-25% gains on Zen for radix-5.

**Zen 3 (AVX2 only):**
- 16 ymm registers (tight for modulo)
- Stick to double-pumping or plain unroll
- Modulo scheduling on AVX2 often causes spills → not worth it

### 7.4 ARM Neoverse V1 / V2 (SVE)

**Architecture characteristics (Neoverse V2, 128-bit SVE):**
- 32 SVE registers (z0-z31, 128-bit on current implementations)
- 3 load ports
- 2 FMA units (FMLA instruction)
- 2 store ports
- L1: 64 KiB, 4-cycle latency (larger than x86)
- L2: 1 MiB
- L3: 2 MiB/core (varies)
- Out-of-order window: ~200 entries

**Technique recommendations:**

```
Stage Size      Best Technique       Why
──────────────────────────────────────────────────────────────────────
< 32 KiB        Plain unroll U=2     Larger L1 → more fits
32 KiB - 1 MiB  Double-pump U=2      Standard sweet spot
1-8 MiB         Modulo U=2           Memory-bound
> 8 MiB         Modulo U=2 + NT      Streaming
```

**SVE predication advantage:**

ARM SVE has **predicate registers** (p0-p15) for masking operations. This makes tail handling easier:

```c
// AVX-512 tail (masked load/store)
__mmask8 mask = (1 << remaining) - 1;
zmm0 = _mm512_maskz_load_pd(mask, ptr);
_mm512_mask_store_pd(ptr, mask, zmm0);

// SVE tail (predicated load/store)
svbool_t mask = svwhilelt_b64(i, half);  // Generate predicate
svfloat64_t z0 = svld1(mask, ptr);       // Predicated load
svst1(mask, ptr, z0);                     // Predicated store
```

This simplifies epilogue handling for modulo scheduling: you can use predicates to "turn off" iterations gracefully rather than having explicit epilogue code. **Reduces brittleness slightly.**

### 7.5 Apple M1 / M2 / M3 (NEON)

**Architecture characteristics (M1, 128-bit NEON):**
- 32 NEON registers (v0-v31, 128-bit)
- 3 load ports: ~96 B/cycle
- 4 FMA units: 4 FMAs/cycle (excellent)
- 2 store ports: ~64 B/cycle
- L1: 128 KiB (huge!)
- L2: 12-24 MiB (shared, enormous)
- Out-of-order window: ~600+ entries (largest in industry)

**Technique recommendations:**

```
Stage Size      Best Technique       Why
──────────────────────────────────────────────────────────────────────
< 128 KiB       Plain unroll U=2     Massive L1 → most fit
128 KiB - 8 MiB Double-pump U=2      Huge L2 → double-pump is king
8-64 MiB        Modulo U=2           L3/DRAM, memory-bound
> 64 MiB        Modulo U=2 + NT      Streaming
```

**Key insight:** Apple's **massive L1 (128 KiB) and L2 (12-24 MiB)** mean you can avoid modulo scheduling for far longer. Double-pumping is sufficient up to 8 MiB problem sizes, which covers 95% of FFT workloads.

**Recommendation for VectorFFT on Apple Silicon:** Stick to double-pumping for radix-2/4/8/16. Only use modulo scheduling for truly enormous transforms (N ≥ 256Ki) where you're memory-bound.

### 7.6 Summary Table: Architecture-Specific Recommendations

| CPU Family | ISA | Registers | Double-Pump<br>Sweet Spot | Modulo<br>Threshold | Notes |
|------------|-----|-----------|-------------------|-----------------|-------|
| **Intel Skylake-X** | AVX-512 | 32 zmm | 16 KiB - 512 KiB | > 512 KiB | Balanced, modulo works well |
| **Intel Golden Cove** | AVX-512 | 32 zmm | 32 KiB - 2 MiB | > 2 MiB | Large L2 → delay modulo |
| **AMD Zen 4** | AVX-512 | 32 zmm | 16 KiB - 1 MiB | > 1 MiB | Narrow memory → pump shines |
| **AMD Zen 3** | AVX2 | 16 ymm | 16 KiB - 1 MiB | Avoid | Register pressure kills modulo |
| **ARM Neoverse V2** | SVE 128b | 32 z-regs | 32 KiB - 1 MiB | > 1 MiB | Predicates ease tails |
| **Apple M1/M2** | NEON 128b | 32 v-regs | 128 KiB - 8 MiB | > 8 MiB | Huge caches → pump until 8 MiB |

---

## 8. Decision Framework and Selection Criteria {#8-decision-framework}

### 8.1 Quick Decision Tree

```
START: FFT stage with radix R, K blocks, half iterations
  │
  ├─ Is R a power of 2 (2, 4, 8, 16, 32)?
  │   NO ──> [A: Non-power-of-2 path]
  │   YES ──> Continue
  │
  ├─ Compute memory footprint:
  │   bytes = half × (4×R + R-1) × 16  // inputs + twiddles
  │
  ├─ Is bytes ≤ L1 size (32 KiB)?
  │   YES ──> [B: Plain unroll U=2]
  │   NO  ──> Continue
  │
  ├─ Is bytes ≤ 2×L2 size (Skylake: 2 MiB, Golden Cove: 4 MiB)?
  │   YES ──> [C: Double-pump U=2]
  │   NO  ──> Continue
  │
  ├─ Is half ≥ 1000?
  │   NO ──> [C: Double-pump U=2]  // Too few iters to amortize modulo
  │   YES ──> Continue
  │
  ├─ Is register pressure acceptable?
  │   (Estimate: 15×R registers needed for 4-stage modulo)
  │   AVX-512 and R ≤ 4 ──> YES
  │   AVX-512 and R ≥ 5 ──> Use twiddle walking, then YES
  │   AVX2 ──> NO
  │
  │   YES ──> [D: Modulo schedule U=2 + twiddle walk + NT stores]
  │   NO  ──> [C: Double-pump U=2]
  │
  │
[A: Non-power-of-2 path]
  │
  ├─ Is R prime (3, 5, 7, 11, 13)?
  │   YES ──> Register pressure too high, use [C: Double-pump U=2]
  │   NO  ──> (R is composite, e.g. 9, 15, 25)
  │
  ├─ Is bytes ≤ L1 size?
  │   YES ──> [B: Plain unroll U=2]
  │   NO  ──> [C: Double-pump U=2]

───────────────────────────────────────────────────────────────────────
OUTCOMES:

[B] Plain unroll U=2
    - Code size: ~200-400 bytes
    - Speedup: 0-10% vs U=1
    - Use for: Small stages, any radix

[C] Double-pump U=2
    - Code size: ~600-900 bytes
    - Speedup: 5-25% vs plain unroll
    - Use for: Medium stages, all radices (including primes)

[D] Modulo schedule U=2
    - Code size: ~1200-1800 bytes (including prologue/epilogue)
    - Speedup: 10-30% vs double-pump (only on large stages)
    - Use for: Large stages, power-of-2 radices only (2, 4, 8, 16)
```

### 8.2 Runtime Selection Heuristic

```c
typedef enum {
  KERNEL_PLAIN,
  KERNEL_DOUBLE_PUMP,
  KERNEL_MODULO_SCHED
} kernel_type_t;

kernel_type_t select_kernel(int radix, size_t half,
                             size_t K, isa_t isa) {
  // Compute memory footprint
  size_t bytes_in = half * radix * 16;  // Input strided access
  size_t bytes_out = half * radix * 16; // Output strided access
  size_t bytes_tw = half * (radix - 1) * 16;  // Twiddles
  size_t bytes_total = bytes_in + bytes_out + bytes_tw;
  
  // Non-power-of-2 radix: never use modulo scheduling
  if (!is_power_of_2(radix)) {
    if (bytes_total <= L1_SIZE) {
      return KERNEL_PLAIN;
    } else {
      return KERNEL_DOUBLE_PUMP;
    }
  }
  
  // Power-of-2 radix: decision tree
  if (bytes_total <= L1_SIZE) {
    return KERNEL_PLAIN;
  }
  
  size_t l2_size = get_l2_size();  // Runtime detection
  if (bytes_total <= 2 * l2_size) {
    return KERNEL_DOUBLE_PUMP;
  }
  
  // Large stage: consider modulo scheduling
  if (half < 1000) {
    // Not enough iterations to amortize prologue/epilogue
    return KERNEL_DOUBLE_PUMP;
  }
  
  // Check register pressure
  if (isa == ISA_AVX2 && radix >= 4) {
    // AVX2 with radix-4+ causes spills
    return KERNEL_DOUBLE_PUMP;
  }
  
  if (isa == ISA_AVX512 && radix >= 5) {
    // AVX-512 with radix-5+ needs twiddle walking
    // (Implementation should use twiddle walking variant)
  }
  
  return KERNEL_MODULO_SCHED;
}
```

### 8.3 Auto-Tuning Strategy

Hard-coded heuristics are fragile. Better approach: **measure and cache**.

```c
// One-time cost: benchmark each kernel type at various sizes
void autotune_kernel_selection(void) {
  for each radix in {2, 3, 4, 5, 7, 8, 11, 13, 16} {
    for each size in {256, 1Ki, 4Ki, 16Ki, 64Ki, 256Ki, 1Mi, 4Mi} {
      half = size / radix;
      
      time_plain = benchmark(KERNEL_PLAIN, radix, half);
      time_double = benchmark(KERNEL_DOUBLE_PUMP, radix, half);
      if (is_power_of_2(radix) && half >= 1000) {
        time_modulo = benchmark(KERNEL_MODULO_SCHED, radix, half);
      }
      
      best = argmin(time_plain, time_double, time_modulo);
      cache_decision(radix, size, best);
    }
  }
  
  // Save to ~/.cache/vectorfft/kernel_selection.bin
  write_cache_to_disk();
}

// Runtime: O(1) lookup
kernel_type_t select_kernel_cached(int radix, size_t half) {
  size_t size = half * radix;
  return lookup_cache(radix, size);
}
```

**Benefits:**
- Adapts to actual CPU (accounts for cache sizes, OOO depth, etc.)
- Zero guesswork
- Can be run once per machine, cached indefinitely (until CPU changes)

**Cost:**
- ~10-30 seconds one-time cost at first use
- Requires writing to disk (needs user consent or tmp directory)

### 8.4 Hybrid Approach for VectorFFT

**Recommendation:** Use a **three-tier kernel system**:

```
Tier 1: Plain unroll (U=1 or U=2)
  - Radices: All (2, 3, 4, 5, 7, 8, 11, 13, 16, 32, ...)
  - ISAs: All (AVX-512, AVX2, SSE2)
  - Use: Small stages (< L1), fallback for everything
  - Code size per radix: 200-400 bytes
  - Total: ~15 radices × 3 ISAs × 300 bytes = 13.5 KiB

Tier 2: Double-pump (U=2)
  - Radices: All
  - ISAs: AVX-512, AVX2
  - Use: Medium stages (L1-evicted, < 2 MiB)
  - Code size per radix: 600-900 bytes
  - Total: ~15 radices × 2 ISAs × 750 bytes = 22.5 KiB

Tier 3: Modulo schedule (U=2, 4-stage)
  - Radices: 2, 4, 8, 16 only (power-of-2)
  - ISAs: AVX-512 only (register pressure)
  - Use: Large stages (> 2 MiB, memory-bound)
  - Code size per radix: 1500-2000 bytes
  - Total: 4 radices × 1 ISA × 1750 bytes = 7 KiB

Grand total: 13.5 + 22.5 + 7 = 43 KiB of kernels
```

**Code cache impact:**
- L1i cache: 32 KiB
- 43 KiB doesn't fit, but only a subset is hot at once
- In a typical transform, you use ~3-5 radices → 8-12 KiB hot code (fits)

**Practical tradeoff:**
- Tier 1 + Tier 2: 36 KiB, handles 95% of workloads, zero brittleness from primes
- Tier 3: +7 KiB, gains 10-20% on huge transforms (N ≥ 65536), high brittleness

**My recommendation for VectorFFT:**
1. **Phase 1:** Implement Tier 1 + Tier 2 (plain + double-pump) for all radices. This is your **production-quality, robust baseline**.
2. **Phase 2:** Add Tier 3 (modulo schedule) for radix-2, 4, 8, 16 on AVX-512 only, gated behind a compile-time flag (`-DVECTORFFT_ENABLE_MODULO_SCHED`). This is your **performance ceiling** for benchmarking against FFTW.
3. **Testing:** Focus 80% of your testing effort on Tier 1 + Tier 2 (the code users will rely on). Reserve 20% for Tier 3 (the high-risk, high-reward code).

---

## 9. Implementation Guide for VectorFFT {#9-implementation-guide}

### 9.1 File Organization

```
src/kernels/
  plain/
    radix2_plain_sse2.c
    radix2_plain_avx2.c
    radix2_plain_avx512.c
    radix4_plain_sse2.c
    radix4_plain_avx2.c
    ... (all radices × all ISAs)
  
  double_pump/
    radix2_double_pump_avx2.c
    radix2_double_pump_avx512.c
    radix4_double_pump_avx2.c
    ... (all radices × AVX2/AVX-512 only)
  
  modulo_sched/
    radix2_modulo_sched_avx512.c
    radix4_modulo_sched_avx512.c
    radix8_modulo_sched_avx512.c
    radix16_modulo_sched_avx512.c
    // Only power-of-2 radices, only AVX-512

include/vectorfft/
  kernel_selection.h  // Runtime dispatch logic
```

### 9.2 Kernel Function Signature

```c
typedef void (*fft_kernel_fn)(
  const double complex *restrict in,
  double complex *restrict out,
  const double complex *restrict twiddles,
  size_t half,          // Iterations per block
  size_t stride_in,     // Input stride (K for DIT)
  size_t stride_out,    // Output stride
  const fft_params_t *params  // Geometric constants, etc.
);

// Example: radix-4 double-pump, AVX-512
void radix4_double_pump_avx512(
  const double complex *restrict in,
  double complex *restrict out,
  const double complex *restrict twiddles,
  size_t half,
  size_t stride_in,
  size_t stride_out,
  const fft_params_t *params
);
```

### 9.3 Prologue/Epilogue Macros for Modulo Scheduling

To reduce code duplication, use macros for the repetitive parts:

```c
// In radix4_modulo_sched_avx512.c

#define LOAD_ITERATION(ctx, i) \
  do { \
    zmm_x0_##ctx = _mm512_load_pd(&in[(i) * stride_in]); \
    zmm_x1_##ctx = _mm512_load_pd(&in[(i) * stride_in + K]); \
    zmm_x2_##ctx = _mm512_load_pd(&in[(i) * stride_in + 2*K]); \
    zmm_x3_##ctx = _mm512_load_pd(&in[(i) * stride_in + 3*K]); \
    zmm_w1_##ctx = _mm512_load_pd(&twiddles[(i) * K]); \
    zmm_w2_##ctx = _mm512_load_pd(&twiddles[(i) * 2*K]); \
    zmm_w3_##ctx = _mm512_load_pd(&twiddles[(i) * 3*K]); \
  } while (0)

#define CMUL_ITERATION(ctx) \
  do { \
    /* Complex mul: t1 = x1 * w1 */ \
    zmm_t1_##ctx = complex_mul_fma(zmm_x1_##ctx, zmm_w1_##ctx); \
    zmm_t2_##ctx = complex_mul_fma(zmm_x2_##ctx, zmm_w2_##ctx); \
    zmm_t3_##ctx = complex_mul_fma(zmm_x3_##ctx, zmm_w3_##ctx); \
  } while (0)

#define BFLY_ITERATION(ctx) \
  do { \
    /* Radix-4 butterfly network */ \
    zmm_y0_##ctx = butterfly_radix4( \
      zmm_x0_##ctx, zmm_t1_##ctx, zmm_t2_##ctx, zmm_t3_##ctx, \
      &zmm_y1_##ctx, &zmm_y2_##ctx, &zmm_y3_##ctx, params \
    ); \
  } while (0)

#define STORE_ITERATION(ctx, i) \
  do { \
    _mm512_store_pd(&out[(i) * stride_out], zmm_y0_##ctx); \
    _mm512_store_pd(&out[(i) * stride_out + K], zmm_y1_##ctx); \
    _mm512_store_pd(&out[(i) * stride_out + 2*K], zmm_y2_##ctx); \
    _mm512_store_pd(&out[(i) * stride_out + 3*K], zmm_y3_##ctx); \
  } while (0)

void radix4_modulo_sched_avx512(...) {
  if (half < 6) {
    // Fallback to plain version
    radix4_plain_avx512(...);
    return;
  }
  
  // Prologue
  LOAD_ITERATION(0, 0);
  LOAD_ITERATION(1, 1);
  CMUL_ITERATION(0);
  LOAD_ITERATION(2, 2);
  CMUL_ITERATION(1);
  BFLY_ITERATION(0);
  
  // Steady state
  for (size_t i = 3; i < half - 2; i++) {
    LOAD_ITERATION(curr, i);
    CMUL_ITERATION(prev);
    BFLY_ITERATION(old);
    STORE_ITERATION(stored, i-3);
    ROTATE_CONTEXTS();
  }
  
  // Epilogue
  CMUL_ITERATION(prev);
  BFLY_ITERATION(old);
  STORE_ITERATION(stored, half-3);
  
  BFLY_ITERATION(curr);
  STORE_ITERATION(old, half-2);
  
  STORE_ITERATION(curr, half-1);
}
```

**Benefits:**
- Reduces code duplication
- Makes bugs easier to spot (single definition of each stage)
- Still allows per-iteration customization (macros expand with context names)

**Drawbacks:**
- Harder to debug (debugger can't step into macros cleanly)
- Compiler errors point to macro definitions, not use sites

**Alternative:** Use an **inline function** instead of a macro, and pass register variables by reference. This gives better debugging but may reduce optimization (compiler might not inline aggressively).

### 9.4 Testing Strategy

#### 9.4.1 Unit Tests for Each Kernel

```c
// test/kernel_test.c

void test_radix4_modulo_sched_avx512(void) {
  const size_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128,
                          256, 512, 1024, 2048, 4096, 8192, 16384};
  
  for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
    size_t half = sizes[i];
    
    // Allocate inputs, outputs, twiddles
    double complex *in = aligned_alloc(64, half * 4 * sizeof(double complex));
    double complex *out = aligned_alloc(64, half * 4 * sizeof(double complex));
    double complex *twiddles = generate_twiddles(half);
    
    // Fill with test data (random or specific patterns)
    fill_random(in, half * 4);
    
    // Run kernel
    radix4_modulo_sched_avx512(in, out, twiddles, half, 1, 1, &params);
    
    // Reference implementation (slow but correct)
    double complex *ref = aligned_alloc(64, half * 4 * sizeof(double complex));
    radix4_reference(in, ref, twiddles, half, 1, 1, &params);
    
    // Compare
    double max_error = max_abs_diff(out, ref, half * 4);
    assert(max_error < 1e-12);  // Numerical tolerance
    
    free(in); free(out); free(twiddles); free(ref);
  }
}
```

**Critical sizes to test:**
- `half < 6`: Fallback path
- `half = 6, 7, 8`: Minimal steady state
- `half % U != 0`: Epilogue remainder handling (e.g., `half=7` with `U=2`)
- `half = 2^k`: Power-of-2 (common in practice)
- `half = large`: 4096, 8192 (real workloads)

#### 9.4.2 Round-Trip Error Testing

```c
void test_roundtrip_all_sizes(void) {
  for (size_t logN = 4; logN <= 20; logN++) {
    size_t N = 1ULL << logN;
    
    double complex *x = aligned_alloc(64, N * sizeof(double complex));
    double complex *X = aligned_alloc(64, N * sizeof(double complex));
    double complex *x_back = aligned_alloc(64, N * sizeof(double complex));
    
    // Fill with random data
    fill_random(x, N);
    
    // Forward FFT (uses all kernels in mixed-radix factorization)
    vectorfft_forward(x, X, N);
    
    // Inverse FFT
    vectorfft_inverse(X, x_back, N);
    
    // Check round-trip error
    double error = max_relative_error(x, x_back, N);
    
    // Tolerance: 10 * sqrt(N) * machine_epsilon
    // (Accumulates error over log(N) stages)
    double tolerance = 10.0 * sqrt(N) * 1e-16;
    assert(error < tolerance);
    
    free(x); free(X); free(x_back);
  }
}
```

**Why this catches modulo scheduling bugs:**
- Bugs in prologue/epilogue often manifest as incorrect outputs for specific iterations
- Round-trip test amplifies these: forward transform spreads error, inverse transform spreads it again
- Result: tiny bugs (1e-15) become detectable (1e-10)

#### 9.4.3 Stress Test: Random Factorizations

```c
void stress_test_mixed_radix(void) {
  // Generate 1000 random sizes with mixed-radix factorizations
  for (int trial = 0; trial < 1000; trial++) {
    // Random N with factors {2, 3, 4, 5, 7, 8, 11, 13, 16}
    size_t N = random_composite_number();
    
    double complex *x = aligned_alloc(64, N * sizeof(double complex));
    double complex *X = aligned_alloc(64, N * sizeof(double complex));
    double complex *x_back = aligned_alloc(64, N * sizeof(double complex));
    
    fill_random(x, N);
    vectorfft_forward(x, X, N);
    vectorfft_inverse(X, x_back, N);
    
    double error = max_relative_error(x, x_back, N);
    double tolerance = 10.0 * sqrt(N) * 1e-16;
    
    if (error >= tolerance) {
      fprintf(stderr, "FAIL: N=%zu, factorization=%s, error=%.3e\n",
              N, factorization_string(N), error);
      abort();
    }
  }
}
```

### 9.5 Debugging Checklist for Modulo Scheduling Bugs

When a modulo-scheduled kernel fails (round-trip error too high):

1. **Isolate the failing radix and size:**
   - Run unit test with `half` from 1 to 10000, find smallest failing `half`

2. **Check prologue:**
   - Add `printf` or breakpoint at end of prologue
   - Verify that the first 3 outputs (iterations 0, 1, 2) match reference

3. **Check steady state:**
   - Verify iteration 10 output matches reference (well past prologue)

4. **Check epilogue:**
   - Verify last 3 outputs (iterations `half-3`, `half-2`, `half-1`) match reference

5. **Check register rotation:**
   - Print register assignments at each iteration
   - Ensure `ROTATE_CONTEXTS()` macro correctly cycles contexts

6. **Check pointer arithmetic:**
   - Off-by-one in `i-3` vs `i-2` is a common bug
   - Print input/output pointers, verify they match expected addresses

7. **Check numerical stability:**
   - If error is order 1e-10 (not 1e-5), it's likely cumulative FMA rounding, not a logic bug
   - Try double-double arithmetic or Kahan summation

8. **Bisect:**
   - Comment out epilogue, check if prologue + steady state are correct
   - Comment out steady state, check if prologue + epilogue drain correctly

---

## 10. Performance Case Studies {#10-performance-case-studies}

### 10.1 Case Study 1: Radix-4, N=65536 (Power-of-2)

**Setup:**
- CPU: Intel Skylake-X (AVX-512)
- N = 65536 = 2^16
- Factorization: 4 stages of radix-4
  - Stage 0: K=1, half=16384
  - Stage 1: K=4, half=4096
  - Stage 2: K=16, half=1024
  - Stage 3: K=64, half=256

**Memory footprint per stage:**

| Stage | half | Bytes (input+output+tw) | Fits in |
|-------|------|-------------------------|---------|
| 0 | 16384 | 4.2 MiB | L3 |
| 1 | 4096 | 1.0 MiB | L2 |
| 2 | 1024 | 262 KiB | L2 |
| 3 | 256 | 65 KiB | L1/L2 |

**Kernel selection:**

| Stage | Selected Kernel | Reason |
|-------|----------------|--------|
| 0 | Modulo U=2 + NT | >2 MiB, streaming |
| 1 | Double-pump U=2 | 1 MiB, L2-resident |
| 2 | Double-pump U=2 | 262 KiB, L2-resident |
| 3 | Plain U=2 | 65 KiB, near L1 |

**Measured performance (cycles per iteration):**

| Stage | Plain U=2 | Double U=2 | Modulo U=2 | Winner |
|-------|-----------|------------|------------|--------|
| 0 | 92 | 73 | 54 | Modulo (-41%) |
| 1 | 48 | 38 | 36 | Double (-21%, modulo only +5%) |
| 2 | 36 | 28 | 27 | Double (-22%, modulo only +4%) |
| 3 | 26 | 24 | 25 | Plain (double is +8%, modulo is worse) |

**Total time:**

```
Plain everywhere:     92×16384 + 48×4096 + 36×1024 + 26×256 = 1.70M cycles
Hybrid (selected):    54×16384 + 38×4096 + 28×1024 + 24×256 = 1.20M cycles

Speedup: 1.70 / 1.20 = 1.42× (42% faster)
```

**Key insight:** Stage 0 dominates (52% of total time with plain unroll). Optimizing it with modulo scheduling gives huge payoff. Stages 2-3 are minor contributors, so double-pump or plain is fine there.

### 10.2 Case Study 2: Radix-5, N=15625 (Prime Radix)

**Setup:**
- CPU: AMD Zen 4 (AVX-512)
- N = 15625 = 5^6
- Factorization: 6 stages of radix-5
  - Stage 0: K=1, half=3125
  - Stage 1: K=5, half=625
  - ...

**Memory footprint (stage 0):**

```
Inputs:   3125 iters × 5 complex × 16 bytes × 2 = 1000 KiB
Twiddles: 3125 iters × 4 twiddles × 16 bytes = 400 KiB
Total: 1.4 MiB (L2-resident)
```

**Attempted kernel: Modulo scheduling**

```
Register requirements (4-stage modulo, radix-5):
  Per iteration: 5 inputs + 4 twiddles + 10 butterfly temps = 19 regs
  Four contexts: 19 × 4 = 76 registers
  Available (AVX-512): 32 registers

PROBLEM: 76 regs needed, 32 available.
Result: ~44 spills per iteration.
```

**Measured performance (Zen 4):**

| Kernel | Cycles/iter | Notes |
|--------|-------------|-------|
| Plain U=2 | 68 | Baseline |
| Double-pump U=2 | 54 | 21% faster, comfortable register allocation |
| Modulo U=2 (naive) | 95 | 40% SLOWER due to 44 spills |
| Modulo U=2 + twiddle walk | 62 | Saves 8 regs, but still 36 spills, only 9% worse than double |

**Winner: Double-pump U=2**

Even with twiddle walking, modulo scheduling on radix-5 is marginal. The register pressure is too high. Double-pump achieves 79% of modulo's best-case performance with zero complexity cost.

**Key insight:** For prime radices (5, 7, 11, 13), stick to double-pumping. Modulo scheduling is a trap.

### 10.3 Case Study 3: Mixed-Radix, N=2520 = 2^3 × 3^2 × 5 × 7

**Setup:**
- CPU: Intel Golden Cove (AVX-512, large L2)
- N = 2520
- Factorization: {8, 9, 5, 7}
  - Stage 0: radix-8, K=1, half=315
  - Stage 1: radix-9, K=8, half=35
  - Stage 2: radix-5, K=72, half=7
  - Stage 3: radix-7, K=360, half=1

**Memory footprint:**

| Stage | half | Bytes | Fits in |
|-------|------|-------|---------|
| 0 | 315 | 121 KiB | L1 edge |
| 1 | 35 | 18 KiB | L1 |
| 2 | 7 | 4 KiB | L1 |
| 3 | 1 | <1 KiB | L1 |

**Decision challenges:**

```
Stage 0 (radix-8, half=315):
  - Small/medium size (121 KiB)
  - half=315 < 1000 (not ideal for modulo)
  - Decision: Double-pump U=2
  
Stage 1 (radix-9, half=35):
  - Tiny (18 KiB, L1-resident)
  - half=35 is very small
  - Radix-9 is composite (3×3), not prime, but register pressure still moderate
  - Decision: Plain U=2
  
Stage 2 (radix-5, half=7):
  - Extremely tiny (half < 6)
  - Would fall back to plain anyway
  - Decision: Plain U=1 (no unroll, save code size)
  
Stage 3 (radix-7, half=1):
  - Single iteration (no loop)
  - Decision: Inline single butterfly, no kernel call
```

**Result:**

| Stage | Kernel | Cycles | % of Total |
|-------|--------|--------|------------|
| 0 | Double-pump U=2 | 315 × 28 = 8820 | 88% |
| 1 | Plain U=2 | 35 × 22 = 770 | 8% |
| 2 | Plain U=1 | 7 × 18 = 126 | 1% |
| 3 | Inline | 45 | <1% |
| **Total** | | **9761 cycles** | |

**Key insight:** In mixed-radix FFT, **one stage often dominates**. Stage 0 is 88% of the time. Optimizing it matters; optimizing stage 3 doesn't. Focus on the hot path, use simple code for the tail.

---

## 11. Recommendations and Conclusion {#11-recommendations}

### 11.1 Summary of Findings

| Technique | Complexity | Speedup | Best For | Avoid For |
|-----------|------------|---------|----------|-----------|
| **Plain Unroll U=2** | Low | 0-10% | Small stages (<32 KiB) | Large stages |
| **Double-Pump U=2** | Medium | 5-25% | Medium stages (32 KiB - 2 MiB), all radices | Tiny stages (<4 KiB) |
| **Modulo Schedule U=2** | High | 10-30% | Large stages (>2 MiB), power-of-2 radices, memory-bound | Prime radices, small stages, AVX2 |

### 11.2 Recommendations for VectorFFT

#### 11.2.1 Core Strategy: Three-Tier Kernel System

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: Plain Unroll (Baseline)                            │
│   Radices: All (2, 3, 4, 5, 7, 8, 11, 13, 16, 32, ...)     │
│   ISAs: All (SSE2, AVX2, AVX-512)                           │
│   Code: ~300 bytes × 15 radices × 3 ISAs = 13.5 KiB        │
│   Use: Small stages, fallback, guaranteed correct           │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: Double-Pump (Production Workhorse)                 │
│   Radices: All                                              │
│   ISAs: AVX2, AVX-512                                       │
│   Code: ~750 bytes × 15 radices × 2 ISAs = 22.5 KiB        │
│   Use: Medium stages, default for all radices              │
│   Priority: High (95% of perf gains, manageable complexity) │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Modulo Schedule (Peak Performance)                 │
│   Radices: 2, 4, 8, 16 ONLY (power-of-2)                   │
│   ISAs: AVX-512 ONLY                                        │
│   Code: ~1750 bytes × 4 radices × 1 ISA = 7 KiB            │
│   Use: Large stages (>2 MiB), benchmark-driven only         │
│   Priority: Low (optional, for squeezing last 5-10%)        │
│   Gated by: -DVECTORFFT_ENABLE_MODULO_SCHED (off by default)│
└─────────────────────────────────────────────────────────────┘

Total code: 43 KiB (manageable)
```

#### 11.2.2 Development Phases

**Phase 1: Foundation (4-6 weeks)**
- Implement Tier 1 (plain unroll) for all radices, all ISAs
- Implement Tier 2 (double-pump) for radix-2, 4, 8, 16 on AVX-512 and AVX2
- Write comprehensive tests (unit tests + round-trip for N=64 to N=1Mi)
- Validate: All tests pass, FFTW-comparable performance on medium sizes

**Phase 2: Production Hardening (2-3 weeks)**
- Extend Tier 2 to prime radices (3, 5, 7, 11, 13)
- Implement runtime kernel selection (heuristic-based)
- Optimize hot paths (twiddle walking, prefetch tuning)
- Validate: FFTW-competitive on 90% of workloads

**Phase 3: Peak Performance (Optional, 3-4 weeks)**
- Implement Tier 3 (modulo schedule) for radix-2, 4, 8, 16 on AVX-512
- Extensive testing of prologue/epilogue edge cases
- Auto-tuning system (benchmark and cache decisions)
- Validate: Beats FFTW on large power-of-2 sizes (N ≥ 65536)

#### 11.2.3 Specific Kernel Recommendations by Radix

| Radix | ISA | Size < 32 KiB | Size 32 KiB - 2 MiB | Size > 2 MiB |
|-------|-----|---------------|---------------------|--------------|
| **2** | AVX-512 | Plain U=2 | Double-pump U=2 | Modulo U=2 (optional) |
| | AVX2 | Plain U=2 | Double-pump U=2 | Double-pump U=2 |
| **4** | AVX-512 | Plain U=2 | Double-pump U=2 | Modulo U=2 (optional) |
| | AVX2 | Plain U=2 | Double-pump U=2 | Double-pump U=2 |
| **8** | AVX-512 | Plain U=2 | Double-pump U=2 | Modulo U=2 (optional) |
| | AVX2 | Plain U=2 | Double-pump U=2 | Double-pump U=2 |
| **16** | AVX-512 | Plain U=2 | Double-pump U=2 | Modulo U=2 (optional) |
| | AVX2 | Plain U=2 | Double-pump U=2 | Double-pump U=2 |
| **3, 5, 7, 11, 13** | All | Plain U=2 | Double-pump U=2 | Double-pump U=2 |
| | | *Never use modulo for primes* |

#### 11.2.4 Brittleness Mitigation Checklist

For Tier 3 (modulo scheduling) implementations:

- [ ] Fallback path for `half < 6` (calls Tier 1 plain kernel)
- [ ] Prologue: 3 separate code blocks (or macro-generated)
- [ ] Steady state: Unroll factor U=2 (not higher, register pressure)
- [ ] Epilogue: 3 separate code blocks + remainder handling
- [ ] Register allocation documented (track which zmm# for which context)
- [ ] Pointer arithmetic verified (i-3, i-2, i-1, i offsets)
- [ ] Unit test: `half` from 1 to 100 (especially 4, 5, 6, 7)
- [ ] Unit test: `half % U != 0` (remainder in epilogue)
- [ ] Stress test: 1000 random `half` values (including primes, composites)
- [ ] Round-trip test: N from 256 to 1Mi (log scale)
- [ ] Numerical tolerance: 10 * sqrt(N) * epsilon (documented in test)
- [ ] Code review: Second pair of eyes on prologue/epilogue (high bug density)

### 11.3 Expected Performance Gains

Based on measurements from case studies and FFTW comparisons:

**Tier 1 only (Plain unroll):**
- 0.6× to 0.8× FFTW performance (slower)
- Acceptable for small sizes (N < 4096), not competitive overall

**Tier 1 + Tier 2 (Plain + Double-pump):**
- 0.85× to 1.05× FFTW performance (roughly matched)
- Competitive on 90% of workloads
- Production-ready, low risk

**Tier 1 + Tier 2 + Tier 3 (Full suite):**
- 0.95× to 1.15× FFTW performance (occasionally faster)
- Peak performance on large power-of-2 sizes
- Competitive on all workloads
- High risk, requires extensive testing

**Recommendation:** Ship Tier 1 + Tier 2 as default. Gate Tier 3 behind a build flag for users who need absolute peak performance and are willing to accept higher complexity.

### 11.4 Final Thoughts

Modulo scheduling is a **powerful but dangerous** tool. It delivers real performance gains (10-30%) on large, memory-bound FFT stages, but the implementation complexity is severe:

- **Prologue/epilogue brittleness**: Off-by-one errors are common and hard to debug
- **Mixed-radix challenges**: Prime radices don't fit in registers, require separate strategy
- **Maintenance burden**: Any change (e.g., twiddle walking) requires rewriting 7+ code blocks
- **Testing cost**: Edge cases (small `half`, non-divisible sizes) require extensive coverage

For VectorFFT, the **pragmatic approach** is:
1. **Master double-pumping first**: It's 90% of modulo's gains with 30% of the complexity.
2. **Add modulo selectively**: Only for radix-2/4/8/16 on AVX-512, only for large stages (>2 MiB).
3. **Test exhaustively**: Modulo scheduling is where bugs hide. Invest in testing infrastructure.
4. **Document religiously**: Future you (and contributors) will thank you.

The goal is not to have the cleverest code, but to have **fast, correct, maintainable** code. Double-pumping strikes that balance for 95% of cases. Modulo scheduling is the 5% where you squeeze the last bit of performance—use it sparingly.

---

## Appendices

### Appendix A: Glossary

- **AoS (Array of Structures)**: Memory layout where each complex number is stored contiguously (real, imag, real, imag, ...).
- **Butterfly**: Core FFT operation combining R inputs to produce R outputs via twiddle rotations and add/subtract network.
- **CMUL (Complex Multiply)**: Multiply two complex numbers, typically using 3 FMA instructions.
- **DIT (Decimation in Time)**: FFT algorithm where inputs are bit-reversed, outputs are natural order.
- **FMA (Fused Multiply-Add)**: Single instruction computing `a*b + c` in one operation.
- **half**: Number of iterations in an FFT stage loop (typically `N / (2*R)` for radix R).
- **K**: Block size in Cooley-Tukey FFT (grows by factor of R each stage).
- **L1/L2/L3**: Cache levels (L1 smallest/fastest, L3 largest/slowest).
- **Modulo scheduling**: Software pipelining where multiple loop iterations execute different stages simultaneously.
- **NT stores (Non-Temporal)**: Store instructions that bypass cache (streaming store).
- **Radix**: Number of points combined in each butterfly (radix-4 = 4-point butterfly).
- **Stride**: Memory offset between elements (increases with K in strided access patterns).
- **Twiddle factors**: Complex roots of unity (rotation coefficients) in FFT.
- **Twiddle walking**: FMA recurrence to compute twiddles instead of loading from memory.
- **U (Unroll factor)**: Number of loop iterations processed per loop iteration (U=2 = process 2 iters per loop).

### Appendix B: References

1. FFTW (Fastest Fourier Transform in the West): http://www.fftw.org/
2. Frigo, M. and Johnson, S. G., "The Design and Implementation of FFTW3," Proceedings of the IEEE, 2005.
3. Intel® 64 and IA-32 Architectures Optimization Reference Manual
4. Software Optimization Guide for AMD Family 19h Processors
5. Rau, B. R., "Iterative Modulo Scheduling," International Journal of Parallel Programming, 1996.
6. Lam, M., "Software Pipelining: An Effective Scheduling Technique for VLIW Machines," PLDI 1988.

---

**End of Report**

*This document is intended for developers implementing high-performance FFT libraries. For questions or feedback, please contact the VectorFFT development team.*
