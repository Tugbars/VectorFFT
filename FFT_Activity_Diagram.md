graph TB
    Start([Program Start]) --> Init[System Initialization]
    
    Init --> InitPrimes[Initialize Prime System<br/>- Build divisibility lookup table 0-1024<br/>- Sieve of Eratosthenes for primes 59-10000<br/>- Store extended prime table]
    
    InitPrimes --> InitChirp[Initialize Bluestein Chirp Tables<br/>- Precompute chirp sequences for<br/>primes: 17,19,23,29,31,37,41,43,47,53<br/>59,61,67,71,73,79,83,89,97,101,103<br/>107,109,113,127<br/>- Single contiguous allocation<br/>- High-precision sin/cos 0.5 ULP<br/>- AVX2 vectorized computation]
    
    InitChirp --> UserCall[User Calls fft_init N, direction]
    
    UserCall --> ValidateInput{Valid Input?<br/>N > 0<br/>dir = ±1}
    ValidateInput -->|No| Error1[Return NULL]
    ValidateInput -->|Yes| CheckDivisible{dividebyN N<br/>Factorable by<br/>2,3,4,5,7,8,11,13<br/>16,32?}
    
    CheckDivisible -->|No| PlanBluestein[Plan Bluestein's Algorithm<br/>- Find M = next power of 2 ≥ 2N-1<br/>- Set lt = 1 non-factorable<br/>- Allocate 4*M scratch space]
    
    CheckDivisible -->|Yes| Factorize[Factorize N<br/>Phase 1: Small primes 2-53<br/>Phase 2: Extended primes 59-10000<br/>Phase 3: Wheel factorization 6k±1]
    
    Factorize --> OptimizeRadices[Optimize Execution Radices<br/>Priority order:<br/>1. Radix-32 2^5<br/>2. Radix-16 2^4<br/>3. Radix-8 2^3<br/>4. Radix-9 3^2 if available<br/>5. Radix-4 2^2<br/>6. Prime radices: 7,5,3,2<br/>7. Other primes up to 53]
    
    OptimizeRadices --> CheckSingleRadix{Single Radix<br/>Optimization?<br/>All radices same}
    
    CheckSingleRadix -->|Yes| SingleRadixPlan[Single-Radix Path<br/>- Precompute ALL twiddle factors<br/>- Store stage offsets<br/>- Sequential k-major layout<br/>- Special: Radix-7 Good-Thomas ordering]
    
    CheckSingleRadix -->|No| MixedRadixPlan[Mixed-Radix Path<br/>- Compute per-stage scratch needs<br/>- Dynamic twiddle generation<br/>- Set lt = 0 factorable]
    
    SingleRadixPlan --> AllocBuffers
    MixedRadixPlan --> AllocBuffers
    PlanBluestein --> AllocBuffers
    
    AllocBuffers[Allocate Buffers<br/>- twiddles: N complex aligned 32B<br/>- scratch: max_scratch aligned 32B<br/>- twiddle_factors: if single-radix]
    
    AllocBuffers --> BuildTwiddles[Build Global Twiddle Table<br/>- Exact cardinal points: 0, pi/2, pi, 3pi/2<br/>- High-precision minimax polynomials<br/>- AVX2 vectorized 4-way<br/>- Conjugate symmetry exploitation<br/>- Special N=8 exact values]
    
    BuildTwiddles --> PopulateStageTwiddles{Single-Radix?}
    
    PopulateStageTwiddles -->|Yes| BuildStageTwiddles[Populate Stage Twiddles<br/>For each stage N/r:<br/>- Index: offset + radix-1 * k + j-1<br/>- Radix-7: sequential w1 to w6<br/>- Others: standard DIT]
    
    PopulateStageTwiddles -->|No| SkipStage[Skip - compute dynamically]
    
    BuildStageTwiddles --> AdjustInverse
    SkipStage --> AdjustInverse
    
    AdjustInverse{Inverse FFT?<br/>dir = -1}
    AdjustInverse -->|Yes| ConjTwiddles[Conjugate All Twiddles<br/>- Negate imaginary parts<br/>- Both global and stage twiddles]
    AdjustInverse -->|No| ReturnObj
    ConjTwiddles --> ReturnObj[Return fft_object]
    
    ReturnObj --> UserExec[User Calls fft_exec obj, inp, out]
    
    UserExec --> DispatchAlgo{Algorithm Type<br/>lt}
    
    DispatchAlgo -->|0 Factorable| MixedRadixEntry[mixed_radix_dit_rec<br/>Initial: stride=1, factor_idx=0]
    
    DispatchAlgo -->|1 Bluestein| BluesteinEntry[bluestein_fft]
    
    MixedRadixEntry --> RecurseCheck{N == 1<br/>Base Case?}
    RecurseCheck -->|Yes| CopyData[Copy input to output]
    RecurseCheck -->|No| GetRadix[Get Current Radix<br/>r = factors_factor_idx<br/>sub_len = N/r]
    
    GetRadix --> AllocScratch[Allocate Scratch Frame<br/>- sub_outputs: r * sub_len<br/>- stage_tw: radix-1 * sub_len<br/>unless precomputed]
    
    AllocScratch --> RecurseLoop[For each lane i = 0 to r-1:<br/>Recurse on sub-FFT<br/>mixed_radix_dit_rec<br/>dest: sub_outputs + i*sub_len<br/>src: input + i*stride<br/>N-prime: sub_len, stride-prime: stride*r<br/>SERIAL execution]
    
    RecurseLoop --> PrepTwiddles{Twiddles<br/>Precomputed?}
    
    PrepTwiddles -->|No| GenTwiddles[Generate Stage Twiddles<br/>For k=0 to sub_len-1:<br/>For j=1 to r-1:<br/>p = j*k mod N<br/>idx = p * nfft/N mod nfft<br/>stage_tw_base+j-1 = twiddles_idx]
    
    PrepTwiddles -->|Yes| UsePrecTwiddles[Use Precomputed<br/>stage_tw = twiddle_factors<br/>+ stage_offset_factor_idx]
    
    GenTwiddles --> DispatchRadix
    UsePrecTwiddles --> DispatchRadix
    
    DispatchRadix{Radix Value}
    
    DispatchRadix -->|2| R2[fft_radix2_butterfly<br/>AVX2: 16x unroll + pipeline<br/>SSE2: 2x unroll]
    
    DispatchRadix -->|3| R3[fft_radix3_butterfly<br/>OpenMP: parallel blocks ge 512<br/>AVX2: 8x unroll + pipeline<br/>Prefetch distance 128<br/>Non-temporal stores ge 4096]
    
    DispatchRadix -->|4| R4[fft_radix4_butterfly<br/>AVX-512: 16x unroll 4 complex/reg<br/>OpenMP: parallel blocks ge 512<br/>AVX2: 8x unroll 2 complex/reg<br/>Software pipelining]
    
    DispatchRadix -->|5| R5[fft_radix5_butterfly<br/>OpenMP: parallel blocks ge 512<br/>AVX2: 8x unroll<br/>Rader DIT algorithm<br/>C5_1=cos72deg C5_2=cos144deg]
    
    DispatchRadix -->|7| R7[fft_radix7_butterfly<br/>Good-Thomas algorithm<br/>Multiplicative FFT<br/>No twiddle multiplies in core]
    
    DispatchRadix -->|8| R8[fft_radix8_butterfly<br/>3-stage radix-2 decomposition<br/>Optimized for cache]
    
    DispatchRadix -->|11,13,16,32| RSpecial[Specialized Radix Kernels<br/>AVX2/AVX-512 optimized]
    
    DispatchRadix -->|Other| RGeneral[General Radix Fallback<br/>Precompute W_r_m = exp 2pi i m/r<br/>Fast paths: m=0, m=r/2<br/>AVX2: 4x + 2x unrolled<br/>Scalar: phase accumulation]
    
    R2 --> ReturnRec
    R3 --> ReturnRec
    R4 --> ReturnRec
    R5 --> ReturnRec
    R7 --> ReturnRec
    R8 --> ReturnRec
    RSpecial --> ReturnRec
    RGeneral --> ReturnRec
    
    ReturnRec[Return from Recursion] --> CheckMore{More Factors?}
    CheckMore -->|Yes| RecurseCheck
    CheckMore -->|No| SpecialFix{N=15 and Inverse?}
    SpecialFix -->|Yes| SwapOut[Swap output_9 with output_14]
    SpecialFix -->|No| Done
    SwapOut --> Done
    CopyData --> Done
    
    BluesteinEntry --> FindChirpIdx[Find Precomputed Chirp<br/>Binary search in pre_sizes<br/>17,19,23,29,...,127]
    
    FindChirpIdx --> ChirpFound{Found?}
    
    ChirpFound -->|Yes| UsePrecChirp[Use Precomputed Chirp<br/>Fast memcpy from all_chirps]
    
    ChirpFound -->|No| ComputeChirp[Compute Chirp Dynamically<br/>theta = pi/N<br/>For n=0 to N-1:<br/>chirp_n = exp +pi i n squared /N<br/>Incremental: n+1 squared = n squared + 2n+1]
    
    UsePrecChirp --> BuildKernel
    ComputeChirp --> BuildKernel
    
    BuildKernel[Build Convolution Kernel<br/>B_time_0 = 1<br/>B_time_n = conj chirp_n n=1..N-1<br/>B_time_M-n = conj chirp_n mirror<br/>Pre-scale by 1/M]
    
    BuildKernel --> FFTKernel[FFT of Kernel<br/>fft_exec plan_fwd, B_time to B_fft]
    
    FFTKernel --> PrepInput[Prepare Input<br/>Forward: A_n = input_n * conj chirp_n<br/>Inverse: A_n = input_n * chirp_n<br/>AVX2 vectorized<br/>Zero-pad to M]
    
    PrepInput --> FFTInput[FFT of Input<br/>fft_exec plan_fwd, A to A_fft]
    
    FFTInput --> Convolve[Pointwise Multiply<br/>C_fft = A_fft * B_fft<br/>AVX2: 2-way unrolled<br/>SSE2: scalar tail]
    
    Convolve --> IFFTResult[Inverse FFT<br/>fft_exec plan_inv, C_fft to C_time<br/>Pre-scaled so no 1/M needed]
    
    IFFTResult --> FinalChirp[Final Chirp Multiply<br/>Forward: out_k = C_k * conj chirp_k<br/>Inverse: out_k = C_k * chirp_k<br/>AVX2 vectorized]
    
    FinalChirp --> CleanupPlans[Free temporary FFT plans]
    CleanupPlans --> Done
    
    Done([Execution Complete]) --> UserFree[User Calls free_fft obj]
    
    UserFree --> FreeBuffers[Free All Buffers<br/>- twiddles<br/>- scratch<br/>- twiddle_factors if exists]
    
    FreeBuffers --> ProgramEnd([Program End])
    
    ProgramEnd --> Cleanup[System Cleanup Destructors<br/>- Free extended_primes<br/>- Free all_chirps<br/>- Free bluestein_chirp descriptors]
    
    style Start fill:#e1f5e1
    style ProgramEnd fill:#ffe1e1
    style Cleanup fill:#ffe1e1
    style Error1 fill:#ffcccc
    style CheckDivisible fill:#fff4e1
    style DispatchAlgo fill:#fff4e1
    style DispatchRadix fill:#e1f0ff
    style R2 fill:#cce5ff
    style R3 fill:#cce5ff
    style R4 fill:#cce5ff
    style R5 fill:#cce5ff
    style R7 fill:#cce5ff
    style R8 fill:#cce5ff
    style RSpecial fill:#cce5ff
    style RGeneral fill:#d5e8d4
    style BluesteinEntry fill:#fff2cc
    style Done fill:#e1f5e1graph TB
    Start([Program Start]) --> Init[System Initialization]
    
    Init --> InitPrimes[Initialize Prime System<br/>- Build divisibility lookup table 0-1024<br/>- Sieve of Eratosthenes for primes 59-10000<br/>- Store extended prime table]
    
    InitPrimes --> InitChirp[Initialize Bluestein Chirp Tables<br/>- Precompute chirp sequences for<br/>primes: 17,19,23,29,31,37,41,43,47,53<br/>59,61,67,71,73,79,83,89,97,101,103<br/>107,109,113,127<br/>- Single contiguous allocation<br/>- High-precision sin/cos 0.5 ULP<br/>- AVX2 vectorized computation]
    
    InitChirp --> UserCall[User Calls fft_init N, direction]
    
    UserCall --> ValidateInput{Valid Input?<br/>N > 0<br/>dir = ±1}
    ValidateInput -->|No| Error1[Return NULL]
    ValidateInput -->|Yes| CheckDivisible{dividebyN N<br/>Factorable by<br/>2,3,4,5,7,8,11,13<br/>16,32?}
    
    CheckDivisible -->|No| PlanBluestein[Plan Bluestein's Algorithm<br/>- Find M = next power of 2 ≥ 2N-1<br/>- Set lt = 1 non-factorable<br/>- Allocate 4*M scratch space]
    
    CheckDivisible -->|Yes| Factorize[Factorize N<br/>Phase 1: Small primes 2-53<br/>Phase 2: Extended primes 59-10000<br/>Phase 3: Wheel factorization 6k±1]
    
    Factorize --> OptimizeRadices[Optimize Execution Radices<br/>Priority order:<br/>1. Radix-32 2^5<br/>2. Radix-16 2^4<br/>3. Radix-8 2^3<br/>4. Radix-9 3^2 if available<br/>5. Radix-4 2^2<br/>6. Prime radices: 7,5,3,2<br/>7. Other primes up to 53]
    
    OptimizeRadices --> CheckSingleRadix{Single Radix<br/>Optimization?<br/>All radices same}
    
    CheckSingleRadix -->|Yes| SingleRadixPlan[Single-Radix Path<br/>- Precompute ALL twiddle factors<br/>- Store stage offsets<br/>- Sequential k-major layout<br/>- Special: Radix-7 Good-Thomas ordering]
    
    CheckSingleRadix -->|No| MixedRadixPlan[Mixed-Radix Path<br/>- Compute per-stage scratch needs<br/>- Dynamic twiddle generation<br/>- Set lt = 0 factorable]
    
    SingleRadixPlan --> AllocBuffers
    MixedRadixPlan --> AllocBuffers
    PlanBluestein --> AllocBuffers
    
    AllocBuffers[Allocate Buffers<br/>- twiddles: N complex aligned 32B<br/>- scratch: max_scratch aligned 32B<br/>- twiddle_factors: if single-radix]
    
    AllocBuffers --> BuildTwiddles[Build Global Twiddle Table<br/>- Exact cardinal points: 0, π/2, π, 3π/2<br/>- High-precision minimax polynomials<br/>- AVX2 vectorized 4-way<br/>- Conjugate symmetry exploitation<br/>- Special N=8 exact values]
    
    BuildTwiddles --> PopulateStageTwiddles{Single-Radix?}
    
    PopulateStageTwiddles -->|Yes| BuildStageTwiddles[Populate Stage Twiddles<br/>For each stage N/r:<br/>- Index: offset + radix-1 * k + j-1<br/>- Radix-7: sequential w^1..w^6<br/>- Others: standard DIT]
    
    PopulateStageTwiddles -->|No| SkipStage[Skip - compute dynamically]
    
    BuildStageTwiddles --> AdjustInverse
    SkipStage --> AdjustInverse
    
    AdjustInverse{Inverse FFT?<br/>dir = -1}
    AdjustInverse -->|Yes| ConjTwiddles[Conjugate All Twiddles<br/>- Negate imaginary parts<br/>- Both global & stage twiddles]
    AdjustInverse -->|No| ReturnObj
    ConjTwiddles --> ReturnObj[Return fft_object]
    
    ReturnObj --> UserExec[User Calls fft_exec obj, inp, out]
    
    UserExec --> DispatchAlgo{Algorithm Type<br/>lt}
    
    DispatchAlgo -->|0 Factorable| MixedRadixEntry[mixed_radix_dit_rec<br/>Initial: stride=1, factor_idx=0]
    
    DispatchAlgo -->|1 Bluestein| BluesteinEntry[bluestein_fft]
    
    MixedRadixEntry --> RecurseCheck{N == 1<br/>Base Case?}
    RecurseCheck -->|Yes| CopyData[Copy input to output]
    RecurseCheck -->|No| GetRadix[Get Current Radix<br/>r = factors[factor_idx]<br/>sub_len = N/r]
    
    GetRadix --> AllocScratch[Allocate Scratch Frame<br/>- sub_outputs: r * sub_len<br/>- stage_tw: radix-1 * sub_len<br/>unless precomputed]
    
    AllocScratch --> RecurseLoop[For each lane i = 0 to r-1:<br/>Recurse on sub-FFT<br/>mixed_radix_dit_rec<br/>dest: sub_outputs + i*sub_len<br/>src: input + i*stride<br/>N': sub_len, stride': stride*r<br/>SERIAL execution]
    
    RecurseLoop --> PrepTwiddles{Twiddles<br/>Precomputed?}
    
    PrepTwiddles -->|No| GenTwiddles[Generate Stage Twiddles<br/>For k=0 to sub_len-1:<br/>For j=1 to r-1:<br/>p = j*k mod N<br/>idx = p * nfft/N mod nfft<br/>stage_tw[base+j-1] = twiddles[idx]]
    
    PrepTwiddles -->|Yes| UsePrecTwiddles[Use Precomputed<br/>stage_tw = twiddle_factors<br/>+ stage_offset[factor_idx]]
    
    GenTwiddles --> DispatchRadix
    UsePrecTwiddles --> DispatchRadix
    
    DispatchRadix{Radix Value}
    
    DispatchRadix -->|2| R2[fft_radix2_butterfly<br/>AVX2: 16x unroll + pipeline<br/>SSE2: 2x unroll]
    
    DispatchRadix -->|3| R3[fft_radix3_butterfly<br/>OpenMP: parallel blocks ≥512<br/>AVX2: 8x unroll + pipeline<br/>Prefetch distance 128<br/>Non-temporal stores ≥4096]
    
    DispatchRadix -->|4| R4[fft_radix4_butterfly<br/>AVX-512: 16x unroll 4 complex/reg<br/>OpenMP: parallel blocks ≥512<br/>AVX2: 8x unroll 2 complex/reg<br/>Software pipelining]
    
    DispatchRadix -->|5| R5[fft_radix5_butterfly<br/>OpenMP: parallel blocks ≥512<br/>AVX2: 8x unroll<br/>Rader's DIT algorithm<br/>C5_1=cos72° C5_2=cos144°]
    
    DispatchRadix -->|7| R7[fft_radix7_butterfly<br/>Good-Thomas algorithm<br/>Multiplicative FFT<br/>No twiddle multiplies in core]
    
    DispatchRadix -->|8| R8[fft_radix8_butterfly<br/>3-stage radix-2 decomposition<br/>Optimized for cache]
    
    DispatchRadix -->|11,13,16,32| RSpecial[Specialized Radix Kernels<br/>AVX2/AVX-512 optimized]
    
    DispatchRadix -->|Other| RGeneral[General Radix Fallback<br/>Precompute W_r^m = exp2πim/r<br/>Fast paths: m=0, m=r/2<br/>AVX2: 4x + 2x unrolled<br/>Scalar: phase accumulation]
    
    R2 --> ReturnRec
    R3 --> ReturnRec
    R4 --> ReturnRec
    R5 --> ReturnRec
    R7 --> ReturnRec
    R8 --> ReturnRec
    RSpecial --> ReturnRec
    RGeneral --> ReturnRec
    
    ReturnRec[Return from Recursion] --> CheckMore{More Factors?}
    CheckMore -->|Yes| RecurseCheck
    CheckMore -->|No| SpecialFix{N=15 & Inverse?}
    SpecialFix -->|Yes| SwapOut[Swap output[9] ↔ output[14]]
    SpecialFix -->|No| Done
    SwapOut --> Done
    CopyData --> Done
    
    BluesteinEntry --> FindChirpIdx[Find Precomputed Chirp<br/>Binary search in pre_sizes<br/>17,19,23,29,...,127]
    
    FindChirpIdx --> ChirpFound{Found?}
    
    ChirpFound -->|Yes| UsePrecChirp[Use Precomputed Chirp<br/>Fast memcpy from all_chirps]
    
    ChirpFound -->|No| ComputeChirp[Compute Chirp Dynamically<br/>θ = π/N<br/>For n=0 to N-1:<br/>chirp[n] = exp+πi*n²/N<br/>Incremental: n+1² = n² + 2n+1]
    
    UsePrecChirp --> BuildKernel
    ComputeChirp --> BuildKernel
    
    BuildKernel[Build Convolution Kernel<br/>B_time[0] = 1<br/>B_time[n] = conj chirp[n] n=1..N-1<br/>B_time[M-n] = conj chirp[n] mirror<br/>Pre-scale by 1/M]
    
    BuildKernel --> FFTKernel[FFT of Kernel<br/>fft_exec plan_fwd, B_time → B_fft]
    
    FFTKernel --> PrepInput[Prepare Input<br/>Forward: A[n] = input[n] * conj chirp[n]<br/>Inverse: A[n] = input[n] * chirp[n]<br/>AVX2 vectorized<br/>Zero-pad to M]
    
    PrepInput --> FFTInput[FFT of Input<br/>fft_exec plan_fwd, A → A_fft]
    
    FFTInput --> Convolve[Pointwise Multiply<br/>C_fft = A_fft * B_fft<br/>AVX2: 2-way unrolled<br/>SSE2: scalar tail]
    
    Convolve --> IFFTResult[Inverse FFT<br/>fft_exec plan_inv, C_fft → C_time<br/>Pre-scaled so no 1/M needed]
    
    IFFTResult --> FinalChirp[Final Chirp Multiply<br/>Forward: out[k] = C[k] * conj chirp[k]<br/>Inverse: out[k] = C[k] * chirp[k]<br/>AVX2 vectorized]
    
    FinalChirp --> CleanupPlans[Free temporary FFT plans]
    CleanupPlans --> Done
    
    Done([Execution Complete]) --> UserFree[User Calls free_fft obj]
    
    UserFree --> FreeBuffers[Free All Buffers<br/>- twiddles<br/>- scratch<br/>- twiddle_factors if exists]
    
    FreeBuffers --> ProgramEnd([Program End])
    
    ProgramEnd --> Cleanup[System Cleanup Destructors<br/>- Free extended_primes<br/>- Free all_chirps<br/>- Free bluestein_chirp descriptors]
    
    style Start fill:#e1f5e1
    style ProgramEnd fill:#ffe1e1
    style Cleanup fill:#ffe1e1
    style Error1 fill:#ffcccc
    style CheckDivisible fill:#fff4e1
    style DispatchAlgo fill:#fff4e1
    style DispatchRadix fill:#e1f0ff
    style R2 fill:#cce5ff
    style R3 fill:#cce5ff
    style R4 fill:#cce5ff
    style R5 fill:#cce5ff
    style R7 fill:#cce5ff
    style R8 fill:#cce5ff
    style RSpecial fill:#cce5ff
    style RGeneral fill:#d5e8d4
    style BluesteinEntry fill:#fff2cc
    style Done fill:#e1f5e1
