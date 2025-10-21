#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>

#define MAX_FACTORS 64
#define MAX_MEMO_SIZE 100000
#define BLUESTEIN_THRESHOLD 64

// Cache parameters (adjust for your target CPU)
#define L1_CACHE_SIZE (32 * 1024)      // 32 KB L1 cache
#define L2_CACHE_SIZE (256 * 1024)     // 256 KB L2 cache
#define L3_CACHE_SIZE (8 * 1024 * 1024) // 8 MB L3 cache
#define CACHE_LINE_SIZE 64
#define COMPLEX_SIZE 16  // sizeof(complex float) = 8, complex double = 16

// Vectorization widths
#define AVX2_WIDTH 4    // 4 complex doubles or 8 complex floats
#define AVX512_WIDTH 8  // 8 complex doubles or 16 complex floats

typedef struct {
    int factors[MAX_FACTORS];
    int count;
    double cost;  // Using double for more precise cost
    bool uses_bluestein;
    int cache_misses_estimate;
    double vectorization_efficiency;
} FFTPlan;

typedef struct {
    FFTPlan plan;
    bool valid;
} MemoEntry;

static MemoEntry* memo_table = NULL;

typedef struct {
    const int* radixes;
    const int* costs;
    int num_radixes;
} RadixConfig;

// Algorithm types for different radixes
typedef enum {
    ALGO_COOLEY_TUKEY,  // 2, 3, 4, 5, 8, 16, 32
    ALGO_RADERS,        // 7, 11, 13
    ALGO_BLUESTEIN      // primes > threshold
} AlgorithmType;

static AlgorithmType get_algorithm_type(int radix) {
    if (radix == 7 || radix == 11 || radix == 13) return ALGO_RADERS;
    return ALGO_COOLEY_TUKEY;
}

static bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

static int largest_prime_factor(int n) {
    int largest = -1;
    for (int d = 2; d * d <= n; d++) {
        while (n % d == 0) {
            largest = d;
            n /= d;
        }
    }
    if (n > 1) largest = n;
    return largest;
}

static int count_radix_power(int n, int radix) {
    int count = 0;
    while (n % radix == 0) {
        count++;
        n /= radix;
    }
    return count;
}

// Estimate cache miss penalty based on data size and stride
static double estimate_cache_penalty(int n, int radix, int stage_number, int total_stages) {
    // Data size in bytes (complex numbers)
    long long data_size = (long long)n * COMPLEX_SIZE;
    
    // Stride pattern: early stages have unit stride, later stages have larger strides
    // Stage i processes sub-FFTs of size n / (product of previous radixes)
    int current_fft_size = n;
    for (int i = 0; i < stage_number; i++) {
        current_fft_size /= radix;  // Approximate
    }
    
    double cache_penalty = 1.0;
    
    // L1 cache - if working set fits, very fast
    if (data_size <= L1_CACHE_SIZE) {
        cache_penalty = 1.0;  // No penalty, everything in L1
    }
    // L2 cache - moderate penalty
    else if (data_size <= L2_CACHE_SIZE) {
        cache_penalty = 1.3;  // ~30% slower
    }
    // L3 cache - larger penalty
    else if (data_size <= L3_CACHE_SIZE) {
        cache_penalty = 1.8;  // ~80% slower
    }
    // Main memory - significant penalty
    else {
        cache_penalty = 3.0;  // 3x slower due to memory bandwidth
    }
    
    // Stride penalty: later stages have worse locality
    // Early stages: unit stride (good)
    // Late stages: large strides (bad)
    double stride_penalty = 1.0 + 0.15 * stage_number / (double)total_stages;
    
    // Non-unit stride radixes have worse cache behavior
    if (radix > 8) {
        stride_penalty *= 1.2;
    }
    
    return cache_penalty * stride_penalty;
}

// Estimate vectorization efficiency for a radix
static double estimate_vectorization_efficiency(int radix, int n) {
    // Cooley-Tukey radixes (4, 8, 16, 32) vectorize well
    if (radix == 4) {
        return 0.85;  // AVX2: 4 complex doubles fit perfectly, but some overhead
    }
    else if (radix == 8) {
        return 0.90;  // AVX: good vectorization, 8 butterflies in parallel possible
    }
    else if (radix == 16) {
        return 0.95;  // AVX512: excellent vectorization, 16 values at once
    }
    else if (radix == 32) {
        return 0.93;  // AVX512: very good, but register pressure starts to matter
    }
    else if (radix == 2) {
        return 0.70;  // Can vectorize but sub-optimal (only 2 ops per butterfly)
    }
    else if (radix == 3 || radix == 5) {
        return 0.60;  // Harder to vectorize efficiently (odd radix)
    }
    else if (radix == 7 || radix == 11 || radix == 13) {
        // Rader's algorithm: involves sub-FFT, less efficient
        return 0.40;  // More complex, harder to vectorize
    }
    
    return 0.50;  // Default
}

// Compute arithmetic cost (flops) for a radix-r butterfly
static double arithmetic_cost(int radix) {
    // Complex multiply: 4 real muls + 2 real adds = ~6 ops
    // Complex add: 2 real adds = 2 ops
    
    AlgorithmType algo = get_algorithm_type(radix);
    
    if (algo == ALGO_COOLEY_TUKEY) {
        // Cooley-Tukey: ~5*r*log2(r) flops per radix-r butterfly
        return 5.0 * radix * log2(radix);
    }
    else if (algo == ALGO_RADERS) {
        // Rader's: converts radix-p to FFT of size p-1, plus setup
        // More expensive than direct Cooley-Tukey
        return 8.0 * radix * log2(radix) + 20.0;
    }
    
    return 5.0 * radix;
}

// Enhanced cost model accounting for cache, vectorization, and arithmetic
static double compute_stage_cost(int n, int radix, int stage_number, int total_stages) {
    // Base arithmetic cost
    int num_butterflies = n / radix;
    double arith_cost = num_butterflies * arithmetic_cost(radix);
    
    // Cache penalty
    double cache_penalty = estimate_cache_penalty(n, radix, stage_number, total_stages);
    
    // Vectorization efficiency (higher is better, so we divide by it)
    double vec_eff = estimate_vectorization_efficiency(radix, n);
    
    // Twiddle factor complexity: more complex for larger radixes and later stages
    double twiddle_cost = radix * 0.5 * (1.0 + stage_number * 0.1);
    
    // Total cost with all factors
    double total_cost = (arith_cost + twiddle_cost) * cache_penalty / vec_eff;
    
    return total_cost;
}

// Evaluate a complete plan's cost
static double evaluate_plan_cost(const int* factors, int count, int n) {
    double total_cost = 0.0;
    int current_n = n;
    
    for (int stage = 0; stage < count; stage++) {
        int radix = factors[stage];
        double stage_cost = compute_stage_cost(current_n, radix, stage, count);
        total_cost += stage_cost;
        current_n /= radix;
    }
    
    return total_cost;
}

static int radix_priority(int radix) {
    // Adjusted priorities based on enhanced cost model
    if (radix == 16) return 1;  // Best for AVX512
    if (radix == 32) return 2;  // Also excellent for AVX512, but higher register pressure
    if (radix == 8) return 3;   // Good for AVX
    if (radix == 4) return 4;   // Good for AVX2
    if (radix == 2) return 5;   // Baseline, but many stages
    if (radix == 5) return 6;   // Small prime, manageable
    if (radix == 3) return 7;   // Small prime
    if (radix == 7) return 8;   // Rader's
    if (radix == 11) return 9;  // Rader's, more expensive
    if (radix == 13) return 10; // Rader's, most expensive
    return 11;
}

static FFTPlan fft_plan_internal(int n, const RadixConfig* config, int depth, double parent_cost, int* current_factors, int current_count) {
    FFTPlan result;
    result.count = 0;
    result.cost = INFINITY;
    result.uses_bluestein = false;
    
    if (n == 1) {
        result.cost = 0;
        return result;
    }
    
    if (n < MAX_MEMO_SIZE && memo_table[n].valid) {
        return memo_table[n].plan;
    }
    
    if (n > BLUESTEIN_THRESHOLD && is_prime(n)) {
        result.uses_bluestein = true;
        int next_pow2 = 1;
        while (next_pow2 < 2 * n - 1) next_pow2 *= 2;
        int temp_factors[MAX_FACTORS];
        FFTPlan sub = fft_plan_internal(next_pow2, config, depth + 1, 0, temp_factors, 0);
        result.cost = sub.cost + n * 100.0;  // Bluestein overhead
        result.count = 1;
        result.factors[0] = n;
        if (n < MAX_MEMO_SIZE) {
            memo_table[n].plan = result;
            memo_table[n].valid = true;
        }
        return result;
    }
    
    int pow2_count = count_radix_power(n, 2);
    if (pow2_count > 0) {
        int remaining = n >> pow2_count;
        if (remaining == 1) {
            result.cost = 0;
            result.count = 0;
            int temp_n = n;
            int temp_factors[MAX_FACTORS];
            int temp_count = 0;
            
            while (pow2_count > 0) {
                if (pow2_count >= 5) {
                    temp_factors[temp_count++] = 32;
                    pow2_count -= 5;
                } else if (pow2_count >= 4) {
                    temp_factors[temp_count++] = 16;
                    pow2_count -= 4;
                } else if (pow2_count >= 3) {
                    temp_factors[temp_count++] = 8;
                    pow2_count -= 3;
                } else if (pow2_count >= 2) {
                    temp_factors[temp_count++] = 4;
                    pow2_count -= 2;
                } else {
                    temp_factors[temp_count++] = 2;
                    pow2_count--;
                }
            }
            
            result.cost = evaluate_plan_cost(temp_factors, temp_count, n);
            result.count = temp_count;
            memcpy(result.factors, temp_factors, temp_count * sizeof(int));
            
            if (n < MAX_MEMO_SIZE) {
                memo_table[n].plan = result;
                memo_table[n].valid = true;
            }
            return result;
        }
    }
    
    int lpf = largest_prime_factor(n);
    if (lpf > 13) {
        result.uses_bluestein = true;
        int next_pow2 = 1;
        while (next_pow2 < n) next_pow2 *= 2;
        int temp_factors[MAX_FACTORS];
        FFTPlan sub = fft_plan_internal(next_pow2, config, depth + 1, 0, temp_factors, 0);
        result.cost = sub.cost + n * 100.0;
        result.count = 1;
        result.factors[0] = n;
        if (n < MAX_MEMO_SIZE) {
            memo_table[n].plan = result;
            memo_table[n].valid = true;
        }
        return result;
    }
    
    int candidate_radixes[16];
    int candidate_costs[16];
    int num_candidates = 0;
    
    for (int i = 0; i < config->num_radixes; i++) {
        if (n % config->radixes[i] == 0) {
            candidate_radixes[num_candidates] = config->radixes[i];
            candidate_costs[num_candidates] = config->costs[i];
            num_candidates++;
        }
    }
    
    for (int i = 0; i < num_candidates - 1; i++) {
        for (int j = i + 1; j < num_candidates; j++) {
            if (radix_priority(candidate_radixes[j]) < radix_priority(candidate_radixes[i])) {
                int tmp = candidate_radixes[i];
                candidate_radixes[i] = candidate_radixes[j];
                candidate_radixes[j] = tmp;
                tmp = candidate_costs[i];
                candidate_costs[i] = candidate_costs[j];
                candidate_costs[j] = tmp;
            }
        }
    }
    
    double best_cost = INFINITY;
    FFTPlan best_plan;
    best_plan.count = 0;
    
    int max_candidates = (depth < 3) ? num_candidates : ((num_candidates + 1) / 2);
    
    for (int i = 0; i < max_candidates; i++) {
        int radix = candidate_radixes[i];
        int quotient = n / radix;
        
        int new_factors[MAX_FACTORS];
        memcpy(new_factors, current_factors, current_count * sizeof(int));
        new_factors[current_count] = radix;
        
        double stage_cost = compute_stage_cost(n, radix, current_count, current_count + 10);
        
        if (parent_cost + stage_cost > best_cost * 1.5) {
            continue;
        }
        
        FFTPlan sub_plan = fft_plan_internal(quotient, config, depth + 1, parent_cost + stage_cost, new_factors, current_count + 1);
        
        double total_cost = stage_cost + sub_plan.cost;
        
        if (total_cost >= best_cost) {
            continue;
        }
        
        best_cost = total_cost;
        best_plan = sub_plan;
        best_plan.factors[best_plan.count] = radix;
        best_plan.count++;
        best_plan.cost = total_cost;
        best_plan.uses_bluestein = sub_plan.uses_bluestein;
        
        if ((radix == 16 || radix == 32) && depth < 2) {
            break;
        }
    }
    
    result = best_plan;
    
    if (n < MAX_MEMO_SIZE) {
        memo_table[n].plan = result;
        memo_table[n].valid = true;
    }
    
    return result;
}

FFTPlan* fft_generate_plan(int n, const int* supported_radixes, const int* radix_costs, int num_radixes) {
    if (!memo_table) {
        memo_table = (MemoEntry*)calloc(MAX_MEMO_SIZE, sizeof(MemoEntry));
    }
    
    RadixConfig config;
    config.radixes = supported_radixes;
    config.costs = radix_costs;
    config.num_radixes = num_radixes;
    
    int temp_factors[MAX_FACTORS];
    FFTPlan plan = fft_plan_internal(n, &config, 0, 0, temp_factors, 0);
    
    if (plan.count == 0 && !plan.uses_bluestein) {
        return NULL;
    }
    
    for (int i = 0; i < plan.count / 2; i++) {
        int tmp = plan.factors[i];
        plan.factors[i] = plan.factors[plan.count - 1 - i];
        plan.factors[plan.count - 1 - i] = tmp;
    }
    
    FFTPlan* result = (FFTPlan*)malloc(sizeof(FFTPlan));
    *result = plan;
    return result;
}

void fft_clear_memo() {
    if (memo_table) {
        free(memo_table);
        memo_table = NULL;
    }
}

/*
int main() {
    static const int SUPPORTED_RADIXES[] = {2, 3, 4, 5, 7, 8, 11, 13, 16, 32};
    static const int RADIX_COSTS[] = {1, 3, 2, 4, 6, 3, 10, 12, 4, 5};
    static const int NUM_RADIXES = 10;
    
    int test_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 1000, 1001, 10000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("FFT Planner with Sophisticated Cost Model\n");
    printf("==========================================\n");
    printf("Cost factors: Cache effects, Vectorization, Stride patterns\n");
    printf("Algorithms: Cooley-Tukey (2,3,4,5,8,16,32), Rader's (7,11,13)\n\n");
    
    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        printf("N = %d:\n", n);
        
        FFTPlan* plan = fft_generate_plan(n, SUPPORTED_RADIXES, RADIX_COSTS, NUM_RADIXES);
        
        if (!plan) {
            printf("  ERROR: Cannot generate plan\n\n");
            continue;
        }
        
        if (plan->uses_bluestein) {
            printf("  Uses Bluestein algorithm\n");
        }
        
        printf("  Plan: [");
        for (int i = 0; i < plan->count; i++) {
            printf("%d", plan->factors[i]);
            if (i < plan->count - 1) printf(" × ");
        }
        printf("]\n");
        printf("  Cost: %.0f (includes cache/vectorization penalties)\n", plan->cost);
        printf("  Stages: %d\n", plan->count);
        
        int r2=0, r4=0, r8=0, r16=0, r32=0, raders=0, small=0;
        for (int i = 0; i < plan->count; i++) {
            int r = plan->factors[i];
            if (r == 2) r2++;
            else if (r == 4) r4++;
            else if (r == 8) r8++;
            else if (r == 16) r16++;
            else if (r == 32) r32++;
            else if (r == 7 || r == 11 || r == 13) raders++;
            else if (r == 3 || r == 5) small++;
        }
        
        printf("  Profile: ");
        if (r32) printf("R32(CT)×%d ", r32);
        if (r16) printf("R16(CT)×%d ", r16);
        if (r8) printf("R8(CT)×%d ", r8);
        if (r4) printf("R4(CT)×%d ", r4);
        if (r2) printf("R2(CT)×%d ", r2);
        if (small) printf("small-prime×%d ", small);
        if (raders) printf("Rader's×%d ", raders);
        printf("\n\n");
        
        free(plan);
    }
    
    fft_clear_memo();
    return 0;
}
*/
