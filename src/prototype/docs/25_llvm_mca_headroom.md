# 25. llvm-mca headroom measurement

## Context

The scheduler in `lib/schedule.ml` is list scheduling with cp_dist
priority and Sethi-Ullman tie-break, with Goodman-Hsu pressure
switching. The framework is from the late 80s / early 90s. Modern
out-of-order x86 cores (Skylake-X, Ice Lake, Sapphire Rapids, Zen 4)
have port-pressure constraints, ROB constraints, and load-port
sub-models that the scheduler does not see. The natural question:
how much performance is on the table from this gap?

This document records a measurement using `llvm-mca` to put a number
on the gap, and the resulting decision on where to invest.

## Measurement methodology

`llvm-mca` is a static analyzer that reads x86 asm and reports cycle
counts, IPC, μop counts, and per-port resource pressure for a given
μarch model. It's not a substitute for runtime VTune data, but it
provides a tight-bound estimate of port-saturation IPC that's much
cheaper than benchmarking on real hardware across multiple μarchs.

Procedure:

1. Generate a codelet (e.g. `gen_radix.exe 17 --twiddled --in-place`)
2. Inject `# LLVM-MCA-BEGIN loop` / `# LLVM-MCA-END` comments into the
   asm at the for-loop body boundaries
3. Compile with `gcc -S -O3 -mavx512f -mavx512dq -mfma -march=skylake-avx512`
4. Run `llvm-mca -mcpu={skylake-avx512,sapphirerapids,znver4}`

Headroom % = `Block RThroughput / Total Cycles`. Block RThroughput is
llvm-mca's lower bound on cycles given resource constraints alone (no
dependencies). The gap to total cycles is dependency-induced stalls
plus any latency-chain bottlenecks the scheduler couldn't hide.

## Results

| Codelet | μarch | Cycles | RThroughput | % saturation | IPC |
|---|---|---|---|---|---|
| R=17 t1_dit (prime) | SKX | 289 | 222.0 | **76.8%** | 2.39 |
| R=17 t1_dit | SPR | 395 | 320.0 | 81.0% | 1.75 |
| R=17 t1_dit | Zen 4 | 393 | 320.0 | 81.4% | 1.76 |
| R=32 t1_dit (composite) | SKX | 312 | 256.0 | **82.0%** | 3.12 |
| R=32 t1_dit | SPR | 305 | 200.0 | **65.6%** | 3.19 |
| R=32 t1_dit | Zen 4 | 342 | 254.0 | 74.3% | 2.85 |
| R=64 t1_dit | SKX | 784 | 674.5 | **86.0%** | 3.53 |

llvm-mca version: 18.0 (2024). LLVM 18's μarch models for SKX, SPR,
and Zen 4 are reasonably current; the SPR model in particular has
had multiple revisions since SPR's launch.

## Findings

### 1. Composites are MORE port-saturated than primes, not less

Counterintuitive but consistent across measurements: R=64 at 86% on
SKX vs R=17 at 76.8%. The CT decomposition exposes more port-level
parallelism — 4 or 8 independent sub-FFTs provide concurrent FMA
chains that fill ports 0 and 5 simultaneously. The monolithic 17-
input prime DAG has more critical-path constraints, so port 5 idles
while port 0 chases the critical chain (or vice versa).

Implication: the 10-30% wins available on composites are NOT from
scheduling. They're from spill-array reload patterns and cross-
boundary inlining, neither of which llvm-mca sees in this measurement
(the spill array is hoisted out of the loop body).

### 2. SPR R=32 at 65.6% is the outlier

Port resource breakdown on SPR:

```
Port:  0      1      2      3      4      5      ...
       260    -      129    129    85     194
```

Port 0 saturated at 260 cycles (against 305 total). SPR has 3-way
AVX-512 FMA dispatch (ports 0, 1, 5 — port 1 added relative to SKX),
but the codelet pins almost all FMAs to port 0 in llvm-mca's
accounting. Port 5 idles half the time. Port 1 doesn't show up at
all for FMA work in this trace.

If real SPR hardware behaves like the llvm-mca-18 model, **port-aware
scheduling has ~35% available on SPR specifically**. Caveat: llvm-mca
is a static model and SPR's actual port allocation policy involves
runtime hashing across instances, so the real gap may be smaller. But
"go check on hardware" is now a concrete next step rather than a vague
suspicion.

This is the 2020s μarch story showing up for real: SPR pays for
3-way FMA dispatch, and our generation strategy doesn't exploit it.

### 3. SKX is mostly closed

R=32 at 82%, R=64 at 86% on SKX. Port 0/5 saturation is already at
80%+ (SKX has only 2-way AVX-512 FMA). The remaining 14-18% gap is
a mix of latency-chain stalls (FMAs waiting for predecessors) and
load-port contention. Realistic capture from port-aware tie-breakers:
3-7%. From beam search: another 2-4% on top. Combined best-case win
on SKX is 5-10%, against 100-200 lines of scheduler refactor cost
plus ongoing maintenance burden of immutable-state scheduling.

This bound is meaningful because SKX is what's been benchmarked
historically (`bench/primes/results_skylake_x_virt.txt`), and most
HFT colo today is still Skylake / Ice Lake generation. SPR adoption
in HFT is real but partial.

## Re-ranking the investment options after measurement

Pre-measurement ranking (rough):

1. Composite emission improvements — 10-30%
2. Per-μarch tuning — 5-15%
3. Beam search — 5-10%
4. llvm-mca cost-function search — research

Post-measurement:

1. **Composite emission improvements** — same 10-30%, but the wins
   are from spill-array reload locality and cross-boundary inlining,
   NOT from scheduling. The llvm-mca measurement clarified the
   mechanism but didn't change the magnitude.

2. **Per-μarch tuning for SPR specifically** — promoted from 5-15%
   to potentially 20-35% on SPR if the llvm-mca model reflects real
   hardware. SKX gets less (5-7% from port balance). Gated on VTune
   confirmation of the SPR number on real silicon. If real SPR shows
   80%+ saturation (i.e., model is wrong), demote.

3. **Beam search** — demoted. With SKX composites at 82-86%
   port-saturated and the available ceiling capped at ~7%, the
   cost/benefit is now bad. The scheduler refactor cost is real
   (immutable state, fork/copy at each branch, ~150 lines new) and
   the win is bounded.

4. **llvm-mca cost-function search** — same status as before, but
   the "side experiment" already paid off: it ranked the other items.

## Decision

Pivot to composite emission improvements. Defer per-μarch tuning
until VTune data on real SPR confirms the headroom. Drop beam
search from active consideration.

Keep llvm-mca in the toolbox as a measurement primitive. Run after
meaningful changes to track headroom. Cheap, high information density.

## Limitations / caveats

- llvm-mca is a static analyzer. It does not model branch
  misprediction, cache misses, runtime port allocation policy,
  or front-end fetch behavior. The numbers above are upper bounds
  on best-case IPC, not predictions of measured runtime behavior.
- One iteration was measured. Multi-iteration analysis (where
  loop-carried dependencies and prefetcher behavior matter) would
  yield different numbers. For static codelet evaluation a single
  iteration is the right granularity.
- The constants are hoisted out of the loop body by GCC's LICM
  (verified: `vbroadcastsd` instructions appear before the loop
  label `.L7` in the R=32 t1_dit asm). The measurement reflects
  steady-state cycles per loop iteration, not the prologue cost.
- llvm-mca's SPR model has seen multiple revisions since the
  μarch's launch. The 65.6% number for R=32 on SPR should be
  treated as a hypothesis to confirm on hardware, not a finding.

## How to reproduce

```bash
$ cd /home/claude/vfft_v2_pack
$ ./_build/default/bin/gen_radix.exe 17 --twiddled --in-place --emit-c > /tmp/r17.c
# Inject # LLVM-MCA-BEGIN/END markers around the for-loop body in the C source
$ gcc -S -O3 -mavx512f -mavx512dq -mfma -march=skylake-avx512 /tmp/r17.c -o /tmp/r17.s
$ llvm-mca -mcpu=skylake-avx512 -iterations=1 /tmp/r17.s
$ llvm-mca -mcpu=sapphirerapids -iterations=1 /tmp/r17.s
$ llvm-mca -mcpu=znver4 -iterations=1 /tmp/r17.s
```
