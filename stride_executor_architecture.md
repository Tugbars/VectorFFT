# Stride-Based In-Place FFT Executor

## Core Idea

One data buffer. Multiple passes. Each pass reads and writes the same buffer at different strides. The output lands in natural order with no permutation.

## Data Layout

For N = R1 × R2 × R3 with K simultaneous transforms, the data lives in a flat array of N×K doubles:

```
data[n * K + k]    where n = 0..N-1, k = 0..K-1
```

The index n encodes a multi-digit address. For N = 3 × 4 × 5 = 60:

```
n = n1 * 20 + n2 * 5 + n3

where n1 = 0..2  (the "radix-3 digit")
      n2 = 0..3  (the "radix-4 digit")
      n3 = 0..4  (the "radix-5 digit")
```

Each stage operates on one digit while the other digits parameterize independent butterflies.

## The Three Stages for N = R1 × R2 × R3

### Stage 1: DFT along the R1 dimension (innermost, no twiddle)

The R1 digit has stride R2 × R3 × K = 20K between its elements. For each combination of (n2, n3), we do one DFT-R1 on R1 elements spaced 20K apart.

```
for n2 in 0..R2-1:
    for n3 in 0..R3-1:
        base = (n2 * 5 + n3) * K
        n1_R3(data + base, data + base, is = 20*K, os = 20*K, vl = K)
```

But the n2 and n3 loops are just iterating over R2 × R3 = 20 butterflies, each processing K SIMD lanes. So it collapses to:

```
for group in 0..19:
    n1_R3(data + group*K, data + group*K, is = 20*K, os = 20*K, vl = K)
```

Or if the codelet has a vectorization loop internally (which yours do — the k-loop):

```
for group in 0..19:
    radix3_n1_fwd(data + group*K, data + group*K, is=20*K, os=20*K, vl=K)
```

Each call does K independent DFT-3 butterflies. Each butterfly loads 3 elements from positions {base, base+20K, base+40K}. Within each position, the k-loop reads K contiguous doubles — a perfect SIMD sweep.

### Stage 2: DFT along the R2 dimension (with twiddles)

After stage 1, we transform the R2 digit. Its stride is R3 × K = 5K. For each (n1, n3) combination we do one DFT-R2.

```
for n1 in 0..R1-1:
    for n3 in 0..R3-1:
        base = (n1 * 20 + n3) * K
        t1_R4(data + base, W_stage2, ios = 5*K, vl = K)
```

This is R1 × R3 = 15 calls, each doing K independent DFT-4 butterflies with twiddles. The butterfly reads 4 elements at stride 5K: {base, base+5K, base+10K, base+15K}.

### Stage 3: DFT along the R3 dimension (with twiddles)

The R3 digit has stride K. For each (n1, n2):

```
for n1 in 0..R1-1:
    for n2 in 0..R2-1:
        base = (n1 * 20 + n2 * 5) * K
        t1_R5(data + base, W_stage3, ios = K, vl = K)
```

R1 × R2 = 12 calls. Each reads 5 elements at stride K: {base, base+K, base+2K, base+3K, base+4K}.

## Why It Works (No Permutation Needed)

The DIT Cooley-Tukey factorization for N = R1 × R2 × R3 computes:

```
X[k1 * R2*R3 + k2 * R3 + k3] = DFT of x[n1 * R2*R3 + n2 * R3 + n3]
```

When all stages operate in-place on the same strides, the output index mapping is:

```
Stage 1 transforms along n1 (stride R2*R3) → produces partial results at same positions
Stage 2 transforms along n2 (stride R3) with twiddles → produces partial results at same positions
Stage 3 transforms along n3 (stride 1) with twiddles → produces final results at same positions
```

The output X[k] ends up at position k in the array — natural order. No digit reversal. This is because each stage transforms one digit in-place without moving data to different positions.

## Twiddle Tables

Each stage (except the first) needs twiddle factors. For stage s transforming dimension d with radix R_d:

```
W[j * stride + k] = exp(-2πi * j * m / N_partial)
```

where N_partial is the product of radixes processed so far (including current), j is the twiddle index, and m is the position within the group.

### Concrete example: N = 60 = 3 × 4 × 5

Stage 2 (DFT-4, twiddles from N=12 partial transform):

```
For each of the 15 groups, the 4 DFT legs need twiddles:
  leg 0: W_12^0 = 1 (no twiddle)
  leg 1: W_12^(n3)     where n3 = 0..4
  leg 2: W_12^(2*n3)
  leg 3: W_12^(3*n3)
```

Stage 3 (DFT-5, twiddles from N=60 full transform):

```
For each of the 12 groups, the 5 DFT legs need twiddles:
  leg 0: W_60^0 = 1
  leg 1: W_60^(n2*5 + n3)     varies per group
  leg 2: W_60^(2*(n2*5 + n3))
  leg 3: W_60^(3*(n2*5 + n3))
  leg 4: W_60^(4*(n2*5 + n3))
```

The twiddle table for each stage is precomputed and stored as flat arrays. The t1 codelet loads them the same way it loads external twiddles today.

### Twiddle table layout

For stage s with radix R and G groups (G = N / R):

```
tw_re[leg * G*K + group * K + k]    (or simplified when K is the vectorization dim)
tw_im[leg * G*K + group * K + k]
```

For leg=0 the twiddle is always 1, so the codelet skips it (same as your existing t1 convention where input 0 is untwiddle).

Alternatively, use the log3 derivation: store only W_N^1 per group, derive higher powers on the fly. This trades arithmetic for memory bandwidth — same tradeoff as your existing log3 variants.

## Executor Pseudocode

```c
typedef struct {
    int radix;                // R for this stage
    int stride;               // distance between DFT legs (in doubles)
    int num_groups;            // how many independent butterflies
    int group_stride;          // distance between groups (in doubles)
    double *tw_re, *tw_im;    // twiddle table (NULL for stage 0)
    codelet_fn n1_fwd, n1_bwd;  // stage 0 only
    codelet_fn t1_fwd, t1_bwd;  // stages 1+ only
} fft_stage_t;

typedef struct {
    int N;
    int num_stages;
    int K;                     // number of simultaneous transforms
    fft_stage_t stages[MAX_STAGES];
} fft_plan_t;

void fft_execute(const fft_plan_t *plan,
                 double *data_re, double *data_im,   // in-place
                 int direction)
{
    int K = plan->K;

    for (int s = 0; s < plan->num_stages; s++) {
        const fft_stage_t *st = &plan->stages[s];

        for (int g = 0; g < st->num_groups; g++) {
            double *base_re = data_re + g * st->group_stride;
            double *base_im = data_im + g * st->group_stride;

            if (s == 0) {
                // First stage: no twiddles, possibly out-of-place for initial copy
                // but can be in-place if input == output
                st->n1_fwd(base_re, base_im,
                           base_re, base_im,
                           st->stride, st->stride, K);
            } else {
                // Subsequent stages: in-place with twiddles
                st->t1_fwd(base_re, base_im,
                           st->tw_re + g * K,
                           st->tw_im + g * K,
                           st->stride, K);
            }
        }
    }
}
```

## Planner Pseudocode

```c
fft_plan_t* fft_plan(int N, int K) {
    fft_plan_t *plan = allocate_plan();
    plan->N = N;
    plan->K = K;

    // Factor N into available codelets, largest first
    int factors[MAX_STAGES];
    int num_factors = factorize(N, available_codelets, factors);
    // Example: N=720 → factors = {12, 20, 3} or {36, 20} etc.

    plan->num_stages = num_factors;

    // Build stages from innermost (rightmost factor) to outermost (leftmost)
    // Stage 0 processes the leftmost factor (largest stride)
    // Last stage processes the rightmost factor (stride = K)

    int remaining = N;
    for (int s = 0; s < num_factors; s++) {
        int R = factors[s];
        remaining /= R;

        plan->stages[s].radix = R;
        plan->stages[s].stride = remaining * K;
        plan->stages[s].num_groups = N / R;           // not exactly — see note
        plan->stages[s].group_stride = ???;            // depends on dimension mapping

        // Look up codelet for this radix
        plan->stages[s].n1_fwd = lookup_n1(R);
        plan->stages[s].t1_fwd = lookup_t1(R);

        // Precompute twiddle table for stages 1+
        if (s > 0) {
            precompute_twiddles(plan, s);
        }
    }

    return plan;
}
```

## Group Iteration Pattern

The tricky part is computing base addresses for each group. For N = R1 × R2 × R3:

```
Stage 1 (transform R1 dimension, stride = R2*R3*K):
  Groups: all (n2, n3) combinations → R2*R3 groups
  Base address: (n2 * R3 + n3) * K
  Group stride between consecutive groups: K

Stage 2 (transform R2 dimension, stride = R3*K):
  Groups: all (n1, n3) combinations → R1*R3 groups
  These are NOT contiguous — n1 jumps by R2*R3*K, n3 jumps by K
  Base address: n1 * R2*R3*K + n3 * K

Stage 3 (transform R3 dimension, stride = K):
  Groups: all (n1, n2) combinations → R1*R2 groups
  Base address: (n1 * R2 + n2) * R3 * K
  Group stride between consecutive groups: R3 * K
```

For 2 stages (N = R1 × R2), it simplifies:

```
Stage 1 (transform R1, stride = R2*K):
  R2 groups, base = g * K, for g in 0..R2-1

Stage 2 (transform R2, stride = K):
  R1 groups, base = g * R2 * K, for g in 0..R1-1
```

## Concrete Walkthrough: N=12 = 3×4, K=8

Data is 96 doubles: `data[n*8 + k]` for n=0..11, k=0..7.

```
Address map (showing n index, each position holds 8 doubles):
  n=0  [0..7]     n=1  [8..15]    n=2  [16..23]   n=3  [24..31]
  n=4  [32..39]   n=5  [40..47]   n=6  [48..55]   n=7  [56..63]
  n=8  [64..71]   n=9  [72..79]   n=10 [80..87]   n=11 [88..95]
```

**Stage 1: Four DFT-3 butterflies, stride = 4×8 = 32**

```
Group 0: base=0    reads data[0..7], data[32..39], data[64..71]    → n={0,4,8}
Group 1: base=8    reads data[8..15], data[40..47], data[72..79]   → n={1,5,9}
Group 2: base=16   reads data[16..23], data[48..55], data[80..87]  → n={2,6,10}
Group 3: base=24   reads data[24..31], data[56..63], data[88..95]  → n={3,7,11}
```

Each group: `n1_R3(data+base, data+base, is=32, os=32, vl=8)`

The codelet loads `data[base + 0*32 + k]`, `data[base + 1*32 + k]`, `data[base + 2*32 + k]` for k=0..7. Three contiguous 8-wide SIMD loads. Butterflies in registers. Writes back to same addresses.

**Stage 2: Three DFT-4 butterflies with twiddles, stride = 8**

```
Group 0: base=0    reads data[0..7], data[8..15], data[16..23], data[24..31]     → n={0,1,2,3}
Group 1: base=32   reads data[32..39], data[40..47], data[48..55], data[56..63]  → n={4,5,6,7}
Group 2: base=64   reads data[64..71], data[72..79], data[80..87], data[88..95]  → n={8,9,10,11}
```

Each group: `t1_R4(data+base, W, ios=8, vl=8)`

The codelet loads `data[base + 0*8 + k]`, `data[base + 1*8 + k]`, `data[base + 2*8 + k]`, `data[base + 3*8 + k]`. Four contiguous SIMD loads. Apply twiddles from W table. Butterfly. Write back.

**Result: data[n*8 + k] now contains the n-th DFT coefficient for the k-th transform. Natural order.**

## SIMD Contiguity Guarantee

At every stage, every codelet load is:

```
data[base + leg * stride + k]    for k = 0..K-1
```

The `k` dimension is always the innermost, contiguous dimension. Each SIMD load reads VL consecutive doubles starting at `data[base + leg * stride + k_start]`. This is a single aligned (or unaligned) vector load regardless of what `stride` is.

The stride only affects which "row" each leg reads from. It never affects the contiguity of the SIMD read within a row.

## Twiddle Factor Computation

For a CT decomposition N = R1 × R2 × ... × Rm, stage s (0-indexed, s > 0) needs twiddles.

The twiddle for stage s, group g, leg j is:

```
W_Ns ^ (j * g_index)

where Ns = R1 × R2 × ... × Rs  (product of radixes up to and including stage s)
      g_index = the position within the Ns-sized sub-problem
```

For N = 60 = 3 × 4 × 5:

```
Stage 1 (R=3): no twiddles

Stage 2 (R=4, N_partial = 3×4 = 12):
  15 groups, 3 twiddle legs (leg 0 is always 1)
  For group with n3-index = n3:
    leg j: W_12^(j * n3) for j=1,2,3 and n3=0..4

Stage 3 (R=5, N_partial = 3×4×5 = 60):
  12 groups, 4 twiddle legs
  For group with (n1,n2)-index:
    position = n1*4 + n2   (NOT n1*20+n2*5, it's the digit position within N/R3)
    leg j: W_60^(j * position) for j=1,2,3,4
```

Precompute these once during planning. Store as flat arrays matching the codelet's expected tw layout.

## Comparison With Current VectorFFT Approaches

| Property | n1_ovs + t1 (current) | Stockham ping-pong | Stride in-place (this doc) |
|----------|----------------------|-------------------|--------------------------|
| Buffers | 2 (input + temp) | 2 (ping-pong) | 1 |
| Passes | 2 + transpose | 2-3 | 2-3 |
| Cache footprint | 2N | 2N | N |
| Write pattern | Contiguous (after transpose) | Contiguous | Contiguous (SIMD within K) |
| Transpose cost | 4×4 shuffle + scatter | None | None |
| Permutation | Implicit (broken for mixed) | Automatic | None needed |
| Depth support | 2 levels only | Any | Any |
| Mixed-radix | Broken | Works | Works |
| Implementation | Complex | Medium | Simple |
