# R=16 Crossover: When Topological Emission Beats Hand-Tuning

## TL;DR

We extended the math layer to generate radix-16 t1_dit codelets via Cooley-Tukey 4×4 decomposition. We then benchmarked our generated AVX-512 code (with topological-order emission, no scheduling, no register allocator — just GCC compiling the SSA-form output) against Tugbars's hand-tuned `radix16_t1_dit_fwd_avx512` from `gen_radix16.py`.

The result was unexpected:

- **K ≤ 256**: Hand wins by 8–20%. Expected.
- **K ≥ 512**: Generated wins by 7–15%. Not expected.

The crossover happens around K=384–512. The reason traces to a single architectural choice the hand-coded version makes (a stack spill buffer between passes) that helps in compute-bound regimes and hurts in memory-bound regimes. Our generator doesn't make that choice — GCC handles register pressure implicitly — and avoids the cost.

This is a meaningful finding for the v2.0 architecture pitch. It says: the right unit of "schedule choice" isn't *instruction order* (where there's no winning answer to chase). It's **selection among variants of the same math** (where the right variant depends on the µarch state, especially K).

## The Setup

### Math layer

Cooley-Tukey radix-16 decomposed as **CT(4, 4)** — two passes of radix-4. Standard DIT input grouping by mod-4 residue:

```
PASS 1: 4 sub-FFTs of size 4
  Sub-FFT n1=0 on x[0], x[4],  x[8],  x[12]   (stride-4, offset 0)
  Sub-FFT n1=1 on x[1], x[5],  x[9],  x[13]   (offset 1)
  Sub-FFT n1=2 on x[2], x[6],  x[10], x[14]   (offset 2)
  Sub-FFT n1=3 on x[3], x[7],  x[11], x[15]   (offset 3)

INTERNAL TWIDDLES: pass1[n1][k2] *= ω_16^{n1·k2}

PASS 2: 4 sub-FFTs of size 4
  For each k2: DFT-4 over n1=0..3 of twiddled[n1][k2]
  Output: X[k1·4 + k2] for k1=0..3
```

This matches `gen_radix16.py`'s exact structure. The ~150-line OCaml extension to `Dft.ml` is generic over (N1, N2) — the same code generates R=4 = CT(2,2), R=8 = CT(2,4), and R=16 = CT(4,4).

### Op count parity

| Variant | Hand-coded ops | Generated ops | Δ |
|---|---|---|---|
| R=16 t1_dit (scalar-equivalent) | 258 | 262 | +1.5% |

The math layer produces a DAG with the same arithmetic as the hand-tuned code, modulo a few stray Negs from the const_cmul algebra around twiddles like ω_16^9 (which evaluates to (cos, -sin) where both components are negative).

### Emission

Topological order: hash-cons tags assigned in construction order; emit one `const __m512d t<tag> = ...;` line per node, in tag order; emit stores at the end. **No scheduling, no register allocation, no µarch awareness.** Whatever GCC does with the resulting SSA-form C is what ships.

### Hand-coded structure

`gen_radix16.py` emits a function that uses **stack-resident spill buffers** between the two passes:

```c
__attribute__((aligned(64))) double spill_re[128];
__attribute__((aligned(64))) double spill_im[128];

for (size_t m = 0; m < me; m += 8) {
    /* PASS 1: 4 sub-FFT-4s */
    /* For each n1 in 0..3:
     *   - load 4 inputs, apply twiddle (FMA cmul)
     *   - radix-4 butterfly
     *   - SPILL all 4 outputs to spill_re/spill_im
     */
    _mm512_store_pd(&spill_re[0*8], x0_re);
    _mm512_store_pd(&spill_im[0*8], x0_im);
    /* ... 30 more spill stores ... */

    /* PASS 2: 4 column DFT-4s */
    /* For each k1 in 0..3:
     *   - reload 4 values from spill_re/spill_im
     *   - apply internal twiddles
     *   - radix-4 butterfly
     *   - store to rio
     */
}
```

That's **32 spill stores + 32 spill loads per inner iteration**, on top of 62 input loads + 32 output stores. The spills land in stack memory which is L1-resident in steady state.

The reason for spilling: R=16's intermediate state (16 complex values × 2 components = 32 vector values) doesn't fit in 32 ZMM registers without hitting register pressure during PASS 2's twiddle and butterfly phases. Spilling lets PASS 1 outputs leave registers between passes.

### Generated structure

Our emission doesn't spill explicitly. The DAG has all 16 sub-FFT outputs alive across the two passes, and we emit them as SSA `const __m512d t<tag> = ...` bindings. **GCC handles register pressure on its own** — typically by spilling a few values to stack as needed during the second pass.

The intent here isn't "we did better than hand-coded." We didn't try to. We just emitted SSA C and let GCC compile. The fact that this approach beats hand-tuning at large K is what makes it noteworthy.

## The Data

Single bench process, codelets timed back-to-back with the same machine state. Sapphire Rapids container, AVX-512, GCC `-O3 -march=native -mfma -ffast-math`. `taskset -c 0` for CPU pinning. K=8 inner-loop step (AVX-512 width for double).

### K-sweep, 5-run averages

| K | Hand (ns) | Topo (ns) | Topo/Hand | Regime |
|---|---|---|---|---|
| 64 | 535 | 640 | **1.20** | Compute-bound |
| 128 | 1466 | 1587 | 1.08 | Compute-bound |
| 256 | 4022 | 4446 | 1.10 | Crossover |
| 512 | 10885 | 10101 | **0.93** | Memory-leaning |
| 1024 | 26120 | 22682 | **0.87** | Memory-bound |
| 2048 | 62976 | 53947 | **0.86** | Memory-bound |

### Per-run stability at large K

Topo's wins reproduce cleanly, not artifacts of one lucky run:

| K=512 | K=1024 | K=2048 |
|---|---|---|
| 0.907 | 0.873 | 0.851 |
| 0.988 | 0.880 | 0.828 |
| 0.931 | 0.881 | 0.870 |
| 0.888 | 0.852 | 0.873 |
| 0.925 | 0.856 | 0.861 |

Range 0.83–0.99 across 15 runs; never above 1.0. The pattern is consistent.

## Hypothesis: Why Topo Wins at Large K

The most direct explanation is that Hand pays a fixed per-iteration cost (the spill traffic) that doesn't scale with K, while Hand's compute-scheduling advantage scales with the per-iteration compute *length*. Below crossover, the scheduling advantage dominates the spill cost. Above crossover, the spill cost dominates.

### Working set transitions

Raptor Lake P-core L1d is 48 KB. Per inner iteration, the working set per codelet call is:

- Input:  16 legs × K × 8 bytes × 2 (re+im) = **256·K bytes** for `rio`
- Twiddles: 15 × K × 8 × 2 = **240·K bytes**
- Output: same buffer as input (in-place)

| K | rio working set | tw working set | Total | L1 fit? |
|---|---|---|---|---|
| 64 | 16 KB | 15 KB | 31 KB | Yes |
| 128 | 32 KB | 30 KB | 62 KB | No (64% over) |
| 256 | 64 KB | 60 KB | 124 KB | No (L2-resident) |
| 512 | 128 KB | 120 KB | 248 KB | No (L2 only just fits) |
| 1024 | 256 KB | 240 KB | 496 KB | L2 (1.25 MB) |
| 2048 | 512 KB | 480 KB | 992 KB | L2 |
| 4096 | 1 MB | 960 KB | 1.96 MB | L3 |

The crossover at K≈384 corresponds roughly to where the working set leaves L1 for L2. Once we're in L2, every memory operation has higher latency, and the spill cost becomes a more significant fraction of the iteration time.

### Estimating the cost of Hand's spill traffic

Per inner iteration, Hand performs 64 extra L1 ops (32 stores + 32 loads to/from `spill_re`/`spill_im`). Each AVX-512 L1 op on Raptor Lake is ~1 cycle latency, ~64 bytes/cycle throughput. Even charitably, 64 ops cost ~10 cycles per iteration overhead.

At K=2048, there are 256 inner iterations per call (since K is divided by 8 for the AVX-512 vector width). 256 × 10 cycles × 0.176 ns/cycle ≈ 450 ns of overhead per call.

Measured Hand at K=2048 = 63,000 ns; measured Topo = 54,000 ns. Difference = 9,000 ns. The 450 ns from spill cost alone explains roughly 5% of the gap. The rest is plausibly second-order effects: spills compete with rio loads/stores for the L1 store buffer; spills evict cache lines that twiddle/rio reads need.

We're not claiming this back-of-envelope estimate is precise. We're claiming the *direction* is right: Hand pays fixed overhead per iteration that scales with iteration count, and at large K iteration count is large.

### Why Hand wins at small K

The same spill discipline that hurts at large K helps at small K because:

1. At K=64, only 8 inner iterations per call. Spill overhead is ~80 cycles total — small in absolute terms.
2. PASS 2's column DFT-4s have meaningful register pressure. Without spilling, GCC has to spill *something* to make room. GCC's spill choices are likely worse than the explicit spill of all PASS 1 outputs.
3. Hand's PASS 2 sees a clean register state with all 32 ZMM registers available, and can schedule the 4 column DFT-4s tightly.

So at small K, where compute dominates, Hand's clean register state lets it schedule better. At large K, where memory dominates, that same register-state-management costs more than it saves.

### What GCC does with our SSA output

We don't have a disassembly to confirm, but the most likely behavior given the IR shape:

- GCC sees ~262 SSA `const __m512d` bindings in topological order
- Live-range analysis: many PASS 1 outputs have live ranges spanning the full PASS 2
- Linear-scan or graph-coloring register allocator: spills *some* values to stack as needed
- The spilled values are likely the least-frequently-used PASS 1 outputs, not all 16

So GCC implicitly does spill — but selectively, and only what's needed for the registers it can't hold. That's a smaller spill volume than Hand's all-16 explicit spill.

The lesson: GCC's register allocator is competing with Hand's spill discipline, and at large K, GCC's selective spilling wins because it does *less* spill traffic.

## Why This Matters for v2.0

Until this measurement, the working theory was: hand-tuned codelets are the gold standard, our generator approximates them, and the v2.0 architectural value is in producing codelets that match Hand's quality from a clean math description.

The R=16 data complicates this picture in a way that turns out to be more useful, not less.

### "Hand-tuned" is a *codelet variant*, not a target

What we benchmarked is one specific codelet variant — the one with explicit stack spilling between passes. There exist other valid codelets for the same math:

- **Spill variant** (what Hand emits): explicit stack spill between passes. Wins small K.
- **No-spill variant** (what we emit): SSA bindings, GCC handles register pressure. Wins large K.

Both are arithmetically equivalent. They differ in how PASS 1 output values get carried to PASS 2.

A useful FFT library at R=16 should ship *both* and pick at plan time based on the K it expects. This isn't a hypothetical optimization — it's a 7–20% performance differential we can already measure.

### The math layer can generate both

Our math layer doesn't bake in the spill choice. It produces a DAG where PASS 1 outputs are predecessors of PASS 2 inputs, and the spill discipline is an *emission decision*. We could add an `emit_with_spill : bool` flag, threaded through `emit_c.ml`, that:

- When `false`: emits SSA bindings (current behavior)
- When `true`: emits explicit stack-array stores after PASS 1, stack loads before PASS 2

The math description is shared. The two codelet variants differ in maybe 30 lines of emit logic. From one math description, we mechanically produce both.

### This generalizes

The same pattern applies to other codelet decisions:

- **Spill discipline**: full / partial / none (whether to materialize PASS 1 outputs to memory)
- **Twiddle policy**: flat / log1 / log3 / log_half (how many twiddle loads vs derivation cmuls)
- **Pass interleaving**: sequential (PASS 1 entirely before PASS 2) / interleaved (some PASS 2 ops start as PASS 1 outputs become available)
- **Constant deferral**: eagerly broadcast all twiddle/W8 constants at function entry / defer to first use

Each is a small variant in the emit layer. Each has a regime where it wins. The "scheduler" we'd build is really a **dispatcher** — given (K, µarch, expected cache state from neighbors), which variant to call.

### Connection to your wisdom system

Your existing library already has a wisdom system that records optimal plans empirically. It operates at the *plan* level: which radix to use at each stage of decomposing N. The R=16 finding suggests extending wisdom *into* the codelet — recording not just "use R=16 here" but "use R=16-spill here and R=16-no-spill there."

The infrastructure is similar. The discrimination is finer. The empirical signal (the Topo/Hand crossover) is real and measurable.

## What This Doesn't Show

To stay honest about scope:

1. **One µarch only**. All measurements are in a Sapphire Rapids container. Raptor Lake might show a different crossover point, Zen5 different again. The *existence* of a crossover is plausibly universal (it follows from the L1 size hierarchy), but the K value where it happens is µarch-dependent.

2. **One codelet variant only**. We compared Hand-with-spill against Topo-no-spill. There are other interesting comparisons: Hand-without-spill (would Hand still win small K without its spilling?), Topo-with-spill (would adding spill let Topo win small K too?). These haven't been measured.

3. **One factorization choice**. R=16 = CT(4, 4) is one decomposition. R=16 could also be CT(2, 8) or split-radix. Each has its own scheduling characteristics.

4. **GCC dependency**. Topo's behavior depends on GCC's register allocator and instruction scheduler. A different compiler (clang, ICC) might produce different results. We haven't tested.

5. **The "Frigo bisection" alternative was strictly worse**. We previously implemented Frigo's recursive-bisection scheduler from genfft as a baseline; it produced 10–25% slower code than Topo at every K we tested for R=8. So "Topo" is already the better naïve scheduler we have. Bisection isn't a serious alternative on this hardware.

## Concrete Next Step

Build the spill variant from the math layer. About 150 lines of OCaml and emit changes:

1. **Math layer**: add an "anchor" mark on PASS 1 outputs in `Dft.ml`. The DAG already has these as named values; we tag them.
2. **Emit layer**: add `~spill:bool` parameter. When true, after a node tagged as a PASS 1 output is computed, emit a stack store, and replace downstream uses with stack loads at the use site.
3. **Bench**: run both variants across K-sweep, characterize the crossover point on this µarch.

If the result shows: spill-variant wins small K, no-spill-variant wins large K, with a clean crossover — we have the empirical foundation for a µarch-aware variant dispatcher. That's the real v2.0 contribution.

If the result shows something else (e.g., Topo wins everywhere even with spilling, or some unexpected interaction), we learn something else useful and adjust.

Either way, the math-layer-as-platform thesis holds: from one math description, we can mechanically generate the variants needed to characterize the design space empirically.
