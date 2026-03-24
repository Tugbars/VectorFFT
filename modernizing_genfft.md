# Modernizing genfft DAGs for AVX2/AVX-512

## The Core Insight

FFTW's genfft produces near-optimal arithmetic DAGs for any DFT-N. The symbolic optimizer — CSE, algebraic simplification, constant folding — finds the minimum-operation network. Nobody has beaten genfft's operation counts in 20 years.

But the SIMD mapping that wraps those DAGs was designed for SSE2 in 2003. On modern hardware (AVX2 with 16 YMM registers, AVX-512 with 32 ZMM registers, FMA units, dual execution ports), the mapping leaves 30-50% of available throughput on the table.

The math is timeless. The SIMD mapping is 20 years obsolete.

## What genfft Gets Right

The DAG optimizer applies global common subexpression elimination across the entire DFT computation. For a DFT-20, this produces 208 operations — roughly 17% fewer than the 250 operations required by a factored 4×5 Cooley-Tukey decomposition. The savings come from subexpressions that span what would be the stage boundary in CT: sum/difference patterns from inputs assigned to different sub-DFTs share intermediate values that CT's factored structure cannot see.

These savings grow with radix size. DFT-4 has no cross-boundary savings (the butterfly is already minimal). DFT-20 saves ~17%. DFT-32 saves more. The larger the radix, the more sharing opportunities exist across what CT would wall off as independent stages.

## Five Obsolete Assumptions in genfft's SIMD Layer

### 1. Interleaved Complex Layout

genfft packs `[re, im]` into one SIMD register. On SSE2 (2 doubles), this holds 1 complex number — the shuffle to extract components is free. On AVX-512 (8 doubles), it holds 4 complex numbers, but every multiply now requires `vpermilpd` to separate re/im, plus `vxorpd` for sign flips. These shuffles consume port 5 bandwidth that could be doing useful arithmetic.

Split layout (`re[0..7]` in one ZMM, `im[0..7]` in another) eliminates all shuffles. A complex multiply becomes two `vmulpd` + two `vfmaddpd` — pure port 0/1 throughput.

**Cost:** Every complex multiply in IL layout burns 2-3 extra shuffle instructions. For DFT-20 with 15 twiddle multiplies, that's ~40 wasted instructions per K-group.

### 2. No FMA Awareness

genfft's optimizer predates FMA (2012 for Intel). It generates:

```
tmp1 = a * b
tmp2 = c * d
result = tmp1 - tmp2     // 3 ops, 3 intermediates
```

With FMA:

```
result = fmsub(a, b, mul(c, d))    // 2 ops, 1 fewer live value
```

Each complex multiply collapses from 4 ops to 2 FMAs. Across a DFT-20 with 15 twiddle cmuls, that's ~30 fewer operations and ~15 fewer simultaneously live values — directly reducing register pressure and spill count.

### 3. No Register-Aware Scheduling

genfft emits the DAG in a fixed topological order without considering register pressure. It relies on the C compiler's register allocator, which works adequately at 16 registers but makes increasingly poor choices at 32 registers because it can't see the full DAG structure.

A register-aware scheduler emits operations in an order that minimizes peak live values: complete each output arm and store it before starting the next arm's intermediates. For DFT-20, this keeps peak pressure at ~24 ZMM instead of the ~40 a naive topological sort produces — the difference between 0 spills and 8 spills.

### 4. No ILP Pairing for Dual-Port Execution

Modern Intel cores have two FMA ports (ports 0 and 1). Maximum throughput requires issuing two independent FMAs per cycle. genfft emits operations in dependency order without considering execution port utilization.

An ILP-aware codelet generator interleaves operations from independent parts of the DAG: while sub-computation A waits on a data dependency, sub-computation B's independent FMA fills the second port. For AVX-512 with R=20, this means processing two 4-element sub-FFTs simultaneously, their operations interleaved to saturate both ports.

### 5. No Width-Aware Twiddle Strategy

genfft applies twiddles one complex element at a time within its IL layout. With 8-wide AVX-512 in split layout, twiddles apply to 8 real elements simultaneously — the twiddle load is amortized over 8× more useful work per instruction.

Furthermore, genfft doesn't distinguish between twiddle strategies based on K (the stride). At small K, a "DAG" approach (inline the twiddle loads into the butterfly) wins because the twiddle data fits in registers alongside the butterfly. At large K, a "CT" approach (loop over K with a fixed butterfly) wins because the twiddle table exceeds L1. genfft generates one codelet for all K.

## DAG vs CT: Why Spills Beat Barriers

A key architectural insight for modern out-of-order CPUs: register spills to stack are nearly free, while pipeline barriers are expensive.

**Spills:** When the compiler spills a ZMM register to the stack (`vmovapd [rsp+offset], zmm`), the value enters the store buffer. When reloaded, store-to-load forwarding returns it in ~5 cycles — the value never reaches L1d. The out-of-order engine (512-entry ROB on Golden Cove) hides this latency behind the 30+ independent operations that separate each spill from its reload. Net cost: zero exposed cycles.

**CT barriers:** A factored Cooley-Tukey decomposition (e.g., DFT-20 = DFT-4 × twiddle × DFT-5) creates hard serialization points. All DFT-4 outputs must complete before any DFT-5 input is available. This dependency chain has depth ~30 cycles that the OoO engine cannot fill because it's a true data dependency, not a resource conflict. These are exposed cycles — nothing can run in parallel.

The decision framework: if the DAG butterfly fits in registers with room for twiddles (~20 ZMM for AVX-512), CT is fine. Once the butterfly exceeds that and CT needs staging, DAG with spills wins.

| Radix | DAG spills | Spill cost (hidden) | CT barriers | Barrier cost (exposed) | Winner |
|-------|-----------|-------------------|-------------|----------------------|--------|
| 4     | 0         | 0                 | 0           | 0                    | Tie    |
| 8     | 0         | 0                 | 0           | 0                    | Tie    |
| 16    | ~5        | ~25 cycles        | 1           | ~25 cycles           | Close  |
| 20    | ~10       | ~50 cycles        | 1           | ~30 cycles + reduced ILP | DAG 15-30% |
| 32    | ~20       | ~100 cycles       | 1-2         | ~60 cycles + reduced ILP | DAG    |

## The Modernization Recipe

Take genfft's DAG (the math), then:

1. **Re-target to split re/im layout** — eliminate all shuffles, double the useful work per instruction.

2. **Pattern-match FMA opportunities** — collapse `a*b ± c*d` pairs into `vfmadd`/`vfmsub`, reducing operation count by ~15-20% and cutting register pressure proportionally.

3. **Schedule for register pressure** — emit stores early, complete each output arm before starting the next, minimize peak live values. Target ≤24 ZMM for AVX-512, ≤14 YMM for AVX2.

4. **Pair independent ops for ILP** — interleave operations from different branches of the DAG to saturate dual FMA ports on modern Intel cores.

5. **Emit aligned loads when K guarantees alignment** — `vmovapd` instead of `vmovupd` when K is a multiple of the SIMD width, avoiding the micro-op penalty.

6. **Provide K-aware dispatch** — DAG codelet for small K (twiddles fit in registers), CT codelet for large K (twiddle table too large for register-resident approach).

This is what VectorFFT's hand-tuned DAG codelets implement. The `gen_r20_*.py` generators automate the translation from genfft's symbolic DAG to modern AVX2/AVX-512 intrinsics with split layout, FMA fusion, register-aware scheduling, and ILP pairing.

## Measured Impact

On Intel i9-14900KF (Golden Cove P-cores), R=20 DIT tw codelet, AVX2:

| K   | DAG (ns) | CT (ns) | DAG/CT ratio |
|-----|----------|---------|-------------|
| 8   | 48       | 69      | 0.70×       |
| 32  | 180      | 213     | 0.85×       |
| 128 | 1080     | 1154    | 0.94×       |
| 512 | 8540     | 10776   | 0.79×       |
| 2048| 33033    | 38715   | 0.85×       |

DAG wins at virtually every K, by 6-30%. The advantage persists even at K=2048 where the twiddle table far exceeds L1 — because the per-element ILP advantage of the DAG's tree-shaped dependency graph over CT's staged pipeline is a constant factor, independent of cache behavior.
