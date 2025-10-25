/**
 * @file TUNING_QUICK_REFERENCE.md
 * @brief Quick Reference for Tuning Radix-16 Optimized FFT
 */

# Radix-16 Optimized FFT - Tuning Quick Reference

## 🎛️ Key Parameters to Tune

### 1. Prefetch Distances (`fft_radix16_uniform_optimized.h`)

```c
// Lines 42-54: Multi-level prefetch distances
// Measured in "butterflies ahead" (not bytes)

// ===== AVX-512 =====
#define PREFETCH_L1_DISTANCE_AVX512    8    // Tune: 6-12
#define PREFETCH_L2_DISTANCE_AVX512    32   // Tune: 24-48  
#define PREFETCH_L3_DISTANCE_AVX512    128  // Tune: 96-192

// ===== AVX2 =====
#define PREFETCH_L1_DISTANCE_AVX2      8    // Tune: 6-12
#define PREFETCH_L2_DISTANCE_AVX2      40   // Tune: 32-56
#define PREFETCH_L3_DISTANCE_AVX2      160  // Tune: 128-224

// ===== SSE2 =====
#define PREFETCH_L1_DISTANCE_SSE2      8    // Tune: 6-12
#define PREFETCH_L2_DISTANCE_SSE2      32   // Tune: 24-48
#define PREFETCH_L3_DISTANCE_SSE2      128  // Tune: 96-192
```

**How to tune:**
1. Start with defaults
2. Run: `perf stat -e L1-dcache-load-misses,LLC-load-misses ./bench`
3. Increase distance if seeing many misses
4. Decrease if prefetch bandwidth saturated

**CPU-specific starting points:**
- **Intel Ice Lake / Sapphire Rapids**: Use defaults
- **Intel older (Skylake)**: Reduce L3 by 25% (→96)
- **AMD Zen 3**: Increase all by 25% (L1→10, L2→40, L3→160)
- **AMD Zen 4**: Increase all by 50% (L1→12, L2→48, L3→192)

---

### 2. Cache Sizes (`fft_radix16_uniform_optimized.h`)

```c
// Lines 74-76: Cache blocking thresholds
// Measured in BYTES

#define L1_CACHE_SIZE       (32 * 1024)       // Tune: 24-48 KB
#define L2_CACHE_SIZE       (512 * 1024)      // Tune: 256KB-2MB
#define L3_CACHE_SIZE       (32 * 1024 * 1024) // Tune: 8-128 MB
```

**How to find your cache sizes:**

```bash
# Linux
cat /sys/devices/system/cpu/cpu0/cache/index0/size  # L1D
cat /sys/devices/system/cpu/cpu0/cache/index2/size  # L2
cat /sys/devices/system/cpu/cpu0/cache/index3/size  # L3

# macOS
sysctl hw.l1dcachesize hw.l2cachesize hw.l3cachesize

# Windows (PowerShell)
Get-WmiObject -Class Win32_CacheMemory
```

**Common values:**
- **Consumer CPUs**: L1=32KB, L2=512KB, L3=8-32MB
- **HEDT/Server**: L1=32-48KB, L2=1-2MB, L3=32-128MB

---

### 3. Cache Work Factor (`fft_radix16_uniform_optimized.h`)

```c
// Line 79: Safety margin for cache blocking
#define CACHE_WORK_FACTOR   0.625  // Tune: 0.5-0.75
```

**What it does:** Fraction of cache to use (leaves room for twiddles, temporaries)

**How to tune:**
- **0.5**: Very conservative, good for complex pipelines
- **0.625**: Default, balanced
- **0.75**: Aggressive, use if twiddles are small

---

### 4. Streaming Threshold (`fft_radix16_uniform_optimized.h`)

```c
// Line 117: When to use non-temporal stores
#define STREAM_THRESHOLD_R16 4096  // K >= 4096 → streaming stores
```

**What it does:** Bypasses cache for outputs when K is large

**How to tune:**
- Measure cache size in complex doubles: `L3_SIZE / 16`
- Set threshold to ~50-75% of that
- Example: 32MB L3 → 2M complex → K=125000 → use 4096-8192

**Rule of thumb:**
- Small L3 (8-16MB): 2048-4096
- Medium L3 (32-64MB): 4096-8192
- Large L3 (128MB+): 8192-16384

---

### 5. Parallel Thresholds (`fft_radix16_uniform_optimized.h`)

```c
// Lines 122-132: When to use multiple threads
#if defined(__AVX512F__)
    #define PARALLEL_THRESHOLD_R16 512   // ~8K complex
#elif defined(__AVX2__)
    #define PARALLEL_THRESHOLD_R16 1024  // ~16K complex
#elif defined(__SSE2__)
    #define PARALLEL_THRESHOLD_R16 2048  // ~32K complex
#endif
```

**What it does:** Minimum K for OpenMP parallelization

**How to tune:**
- Too low: Thread overhead > benefit
- Too high: Misses parallel opportunities

**Guidelines:**
- If you have many cores (16+): Can reduce by 50%
- If you have few cores (4-8): Keep as-is
- For small FFTs only: Increase by 2x

---

## 📊 Profiling Commands

### Cache Miss Analysis
```bash
# Measure cache misses
perf stat -e L1-dcache-loads,L1-dcache-load-misses,\
             LLC-loads,LLC-load-misses,\
             cache-references,cache-misses \
    ./your_benchmark 1048576

# Goal:
# - L1 miss rate < 5%
# - LLC miss rate < 2%
```

### IPC (Instructions Per Cycle)
```bash
# Measure ILP effectiveness
perf stat -e cycles,instructions,stalled-cycles-frontend,\
             stalled-cycles-backend \
    ./your_benchmark 1048576

# Goal:
# - IPC > 2.5 (instructions/cycles)
# - Stalls < 20% of cycles
```

### Memory Bandwidth
```bash
# Check if bandwidth saturated
perf stat -e cpu/event=0xb7,umask=0x1,cmask=0x1/,\  # Off-core requests
             cpu/event=0xb7,umask=0x2,cmask=0x1/ \  # All requests
    ./your_benchmark 1048576

# If ratio → 1, you're memory-bound (good target for prefetch tuning)
```

---

## 🎯 Optimization Priority by FFT Size

### Small FFTs (N < 64K)
**Bottleneck:** Register pressure, instruction latency
**Focus on:**
- Prefetch L1 distance (most impact)
- Unroll factor (if registers available)
- Skip cache blocking (not needed)

### Medium FFTs (64K ≤ N < 1M)
**Bottleneck:** L2/L3 cache misses
**Focus on:**
- Prefetch L2 distance (most impact)
- L2 cache blocking (if N > 256K)
- Tune CACHE_WORK_FACTOR

### Large FFTs (N ≥ 1M)
**Bottleneck:** Memory bandwidth, cache thrashing
**Focus on:**
- Prefetch L3 distance (most impact)
- L3 cache blocking (critical!)
- Streaming stores
- Parallel threshold

---

## 🔬 Quick Tuning Procedure

### Step 1: Baseline (5 minutes)
```bash
# Run unmodified version
./benchmark_original 4096 65536 1048576 > baseline.txt
```

### Step 2: Tune Prefetch (15 minutes)
```bash
# Try different L1 distances: 6, 8, 10, 12
for dist in 6 8 10 12; do
    # Edit PREFETCH_L1_DISTANCE_AVX512 = $dist
    make clean && make
    ./benchmark_optimized 4096 >> prefetch_tune.txt
done

# Pick the fastest, then tune L2 and L3 similarly
```

### Step 3: Tune Cache Blocking (10 minutes)
```bash
# Only for large FFTs
# Try CACHE_WORK_FACTOR: 0.5, 0.625, 0.75
for factor in 0.5 0.625 0.75; do
    # Edit CACHE_WORK_FACTOR = $factor
    make clean && make
    ./benchmark_optimized 1048576 >> cache_tune.txt
done
```

### Step 4: Validate (5 minutes)
```bash
# Run full benchmark suite with tuned parameters
./benchmark_optimized 256 4096 16384 65536 262144 1048576 4194304

# Compare vs baseline:
# - Small: Should be +10-20% faster
# - Medium: Should be +15-30% faster
# - Large: Should be +30-55% faster
```

---

## ⚠️ Common Issues

### Issue: Performance worse on small FFTs
**Cause:** Prefetch distance too aggressive, cache pollution
**Fix:** Reduce L1 distance to 6, disable L2/L3 prefetch for small N

### Issue: Performance worse on large FFTs
**Cause:** Cache blocking not working, wrong tile size
**Fix:** Check actual cache sizes, verify CACHE_WORK_FACTOR < 0.75

### Issue: High cache miss rate
**Cause:** Prefetch too late, blocking tiles too large
**Fix:** Increase prefetch distances, reduce tile sizes

### Issue: Low IPC (<2.0)
**Cause:** Register spills, insufficient unroll
**Fix:** Reduce unroll factor, check for register pressure

### Issue: No speedup from parallelization
**Cause:** Threshold too high, overhead dominates
**Fix:** Lower PARALLEL_THRESHOLD_R16 by 50%

---

## 🏁 Success Criteria

After tuning, you should see:
- ✅ L1 cache miss rate < 5%
- ✅ LLC miss rate < 2%
- ✅ IPC > 2.5 on AVX-512, >2.0 on AVX2
- ✅ Small FFTs: +10-20% vs original
- ✅ Large FFTs: +30-55% vs original
- ✅ Within 10% of FFTW (goal: within 5%)

---
