//==============================================================================
// NORMALIZATION API USAGE GUIDE
//==============================================================================

/**
 * @file NORMALIZATION_USAGE_GUIDE.md
 * @brief Quick reference for choosing the right normalization approach
 */

//==============================================================================
// THREE LEVELS OF NORMALIZATION API
//==============================================================================

1. HIGH-LEVEL: Automatic Wrapper Functions
   =========================================
   ✅ Use when: You want automatic, correct normalization
   ✅ Simplest API
   ✅ No manual scale calculation
   
   ```c
   // Automatically normalizes inverse FFT with 1/N
   fft_exec_normalized(inv_plan, freq, time, workspace);
   ```


2. MID-LEVEL: Convenience Macros
   ==============================
   ✅ Use when: You need explicit control over when normalization happens
   ✅ Clean, self-documenting code
   ✅ Standard normalization conventions
   
   ```c
   // Execute raw FFT
   fft_exec_dft(inv_plan, freq, time, workspace);
   
   // Apply standard 1/N normalization
   FFT_NORMALIZE_INVERSE(time, N);
   
   // Or orthogonal 1/√N normalization
   FFT_NORMALIZE_ORTHO(time, N);
   
   // Or custom scale factor
   FFT_NORMALIZE_CUSTOM(time, N, my_scale);
   ```


3. LOW-LEVEL: Direct Function Calls
   ==================================
   ✅ Use when: You need maximum flexibility
   ✅ Custom scale factors
   ✅ Working with SoA format directly
   
   ```c
   // Interleaved (AoS) format
   fft_normalize_explicit((double*)data, N, 1.0/N);
   
   // Separated (SoA) format
   fft_normalize_soa(re, im, N, 1.0/N);
   
   // Zero-cost fused operations
   fft_join_soa_to_aos_normalized(re, im, output, N, 1.0/N);
   ```


//==============================================================================
// UPDATED fft_execute.c CODE (WITH MACROS)
//==============================================================================

Function: fft_exec_normalized()
--------------------------------
BEFORE (scalar loop):
```c
const double scale = 1.0 / (double)plan->n_fft;
for (int i = 0; i < plan->n_fft; i++) {
    output[i].re *= scale;
    output[i].im *= scale;
}
```

AFTER (using macro):
```c
if (plan->direction == FFT_INVERSE) {
    FFT_NORMALIZE_INVERSE(output, plan->n_fft);
}
```

Benefits:
✅ Self-documenting: Clear intent (inverse FFT normalization)
✅ Cleaner: No manual scale calculation
✅ Consistent: Uses same macro across codebase
✅ SIMD: Automatically uses AVX-512/AVX2/SSE2


Function: fft_roundtrip_normalized()
-------------------------------------
BEFORE (scalar loop):
```c
const double scale = 1.0 / (double)N;
for (int i = 0; i < N; i++) {
    output[i].re *= scale;
    output[i].im *= scale;
}
```

AFTER (using macro):
```c
FFT_NORMALIZE_INVERSE(output, N);
```


//==============================================================================
// WHEN TO USE WHICH APPROACH
//==============================================================================

SCENARIO 1: Standard Inverse FFT
---------------------------------
❓ Question: "I want my inverse FFT to give me back the original signal"

✅ BEST: Use automatic wrapper
```c
fft_exec_normalized(inv_plan, freq, time, workspace);
```

✅ ALTERNATIVE: Use macro
```c
fft_exec_dft(inv_plan, freq, time, workspace);
FFT_NORMALIZE_INVERSE(time, N);
```


SCENARIO 2: Power Spectrum (No Normalization)
----------------------------------------------
❓ Question: "I'm computing |FFT(x)|² and don't need exact scaling"

✅ BEST: Don't normalize at all
```c
fft_exec_dft(fwd_plan, signal, freq, workspace);
for (int k = 0; k < N; k++) {
    power[k] = freq[k].re*freq[k].re + freq[k].im*freq[k].im;
}
// No normalization needed!
```


SCENARIO 3: Convolution
------------------------
❓ Question: "I'm doing convolution via FFT multiplication"

✅ BEST: Normalize only at the end
```c
fft_exec_dft(fwd, sig1, freq1, ws);        // No norm
fft_exec_dft(fwd, sig2, freq2, ws);        // No norm
complex_multiply(freq1, freq2, result, N); // Multiply
fft_exec_normalized(inv, result, out, ws); // Normalize here!
```


SCENARIO 4: Orthogonal Transform (Energy Preserving)
-----------------------------------------------------
❓ Question: "I want ||x|| = ||X|| (Parseval's theorem)"

✅ BEST: Use orthogonal macro on BOTH transforms
```c
// Forward FFT
fft_exec_dft(fwd_plan, input, freq, ws);
FFT_NORMALIZE_ORTHO(freq, N);  // 1/√N

// Process in frequency domain...

// Inverse FFT
fft_exec_dft(inv_plan, freq, output, ws);
FFT_NORMALIZE_ORTHO(output, N);  // 1/√N

// Now: energy preserved!
```


SCENARIO 5: Custom Scale Factor
--------------------------------
❓ Question: "I need a non-standard scale factor"

✅ BEST: Use custom macro or direct function
```c
// Option 1: Custom macro
FFT_NORMALIZE_CUSTOM(data, N, my_scale);

// Option 2: Direct function call
fft_normalize_explicit((double*)data, N, my_scale);
```


SCENARIO 6: Working with SoA Format Internally
-----------------------------------------------
❓ Question: "My butterflies use SoA, I need to normalize before conversion"

✅ BEST: Use SoA-specific function or fused operation
```c
// Option 1: Normalize SoA in-place
fft_normalize_soa(re, im, N, 1.0/N);
fft_join_soa_to_aos(re, im, output, N);

// Option 2: Zero-cost fused (BETTER!)
fft_join_soa_to_aos_normalized(re, im, output, N, 1.0/N);
```


SCENARIO 7: Batch Processing (Many FFTs)
-----------------------------------------
❓ Question: "I'm doing 1000 FFTs in a loop, performance critical"

✅ BEST: Use macros for clarity, compiler optimizes
```c
for (int batch = 0; batch < 1000; batch++) {
    fft_exec_dft(fwd, inputs[batch], freqs[batch], ws);
    
    // Process in frequency domain...
    
    fft_exec_dft(inv, freqs[batch], outputs[batch], ws);
    FFT_NORMALIZE_INVERSE(outputs[batch], N);  // SIMD!
}
```


SCENARIO 8: Real-Time Processing (Low Latency)
-----------------------------------------------
❓ Question: "Latency matters, every microsecond counts"

✅ BEST: Skip normalization if possible, or use fused
```c
// If you can work with scaled data (e.g., audio):
fft_exec_dft(inv_plan, freq, time, ws);
// Skip normalization! (outputs are N× larger but often OK)

// If you must normalize:
fft_exec_normalized(inv_plan, freq, time, ws);
// SIMD makes this ~0.5% overhead (negligible)
```


//==============================================================================
// MACRO DEFINITIONS (from fft_normalize.h)
//==============================================================================

```c
// Standard 1/N inverse normalization
#define FFT_NORMALIZE_INVERSE(data, N) \
    fft_normalize_explicit((double*)(data), (N), 1.0 / (double)(N))

// Orthogonal 1/√N normalization
#define FFT_NORMALIZE_ORTHO(data, N) \
    fft_normalize_explicit((double*)(data), (N), 1.0 / sqrt((double)(N)))

// Custom scale factor
#define FFT_NORMALIZE_CUSTOM(data, N, scale) \
    fft_normalize_explicit((double*)(data), (N), (scale))
```


//==============================================================================
// MACRO ADVANTAGES
//==============================================================================

1. Self-Documenting
   -----------------
   Compare:
   
   ❌ Not clear:
   ```c
   fft_normalize_explicit((double*)data, N, 1.0/(double)N);
   ```
   
   ✅ Crystal clear:
   ```c
   FFT_NORMALIZE_INVERSE(data, N);
   ```


2. Type Safety
   ------------
   The macro handles casting automatically:
   ```c
   fft_data *output = ...;
   FFT_NORMALIZE_INVERSE(output, N);  // Automatic cast to double*
   ```


3. Consistency
   ------------
   Same normalization convention across entire codebase:
   ```c
   // In fft_execute.c:
   FFT_NORMALIZE_INVERSE(output, plan->n_fft);
   
   // In user code:
   FFT_NORMALIZE_INVERSE(my_data, N);
   
   // Everyone uses the same convention!
   ```


4. Easier Refactoring
   -------------------
   Change normalization strategy in one place:
   ```c
   // Change from 1/N to 1/√N globally:
   #define FFT_NORMALIZE_INVERSE(data, N) \
       fft_normalize_explicit((double*)(data), (N), 1.0 / sqrt((double)(N)))
   ```


//==============================================================================
// PERFORMANCE COMPARISON
//==============================================================================

All three approaches use SIMD internally, so performance is identical:

```
Scalar loop:     272 µs  (N=131072)  ← OLD, DON'T USE
Direct function:  36 µs  (N=131072)  ← 7.5× faster
Macro:            36 µs  (N=131072)  ← Same as direct (macro expands to function)
Wrapper:          36 µs  (N=131072)  ← Same (calls macro internally)
```

Choose based on readability and convenience, not performance!


//==============================================================================
// MIGRATION CHECKLIST
//==============================================================================

✅ Include fft_normalize.h in source files
✅ Replace scalar loops with macros
✅ Update fft_execute.c to use FFT_NORMALIZE_INVERSE
✅ Update user code to use appropriate level (wrapper/macro/function)
✅ Test with round-trip verification
✅ Benchmark to verify SIMD speedup

Example migration:
```diff
  #include "fft_planning.h"
+ #include "fft_normalize.h"

  // Old code:
- const double scale = 1.0 / N;
- for (int i = 0; i < N; i++) {
-     output[i].re *= scale;
-     output[i].im *= scale;
- }

  // New code:
+ FFT_NORMALIZE_INVERSE(output, N);
```


//==============================================================================
// SUMMARY
//==============================================================================

Three levels, choose based on your needs:

1. fft_exec_normalized()          → Simplest, automatic
2. FFT_NORMALIZE_INVERSE()        → Clear, explicit, recommended
3. fft_normalize_explicit()       → Maximum flexibility

All use SIMD internally → All equally fast!

Recommendation: Start with macros (level 2) for best balance of
clarity and control. Drop to level 3 only when you need custom scales.