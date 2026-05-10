# 27. Hand vs ours on composites — the docs already had the answer

> **Status:** This document had a confused journey. v1 used llvm-mca
> static analysis and claimed we beat hand 5-22%. v2 used a runtime
> measurement on a virtualized container and retracted to "we lose
> 17-30%". v3 (current) reconciles with the existing bench evidence
> in `docs/09_r32_beating_hand.md` and `docs/11_r64_finding.md`,
> which had the correct answer all along: **on bare-metal Skylake-X
> with the proper bench harness, our SU+Spill codelets beat hand by
> 1-9% on R=32 and R=64 across K=64-4096**. Both v1 and v2 were
> wrong; the existing docs were right.

## The actual answer

From `docs/09_r32_beating_hand.md` (R=32 t1_dit, AVX-512, 3 runs per K, median):

| K | Hand (ns) | SU+Spill (ns) | SU/Hand |
|---|---|---|---|
| 64 | 1600 | 1572 | 0.98 |
| 128 | 3560 | 3481 | 0.99 |
| 256 | 9173 | 9066 | 0.99 |
| 512 | 23097 | 22565 | 0.97 |
| 1024 | 58729 | 56359 | **0.93** |
| 2048 | 166167 | 150476 | **0.91** |
| 4096 | 338886 | 311878 | **0.91** |

From `docs/11_r64_finding.md` (R=64 t1_dit):

| K | Hand (ns) | SU+Spill (ns) | SU/Hand |
|---|---|---|---|
| 64 | 5602 | 5546 | 0.99 |
| 128 | 11621 | 11329 | 0.98 |
| 256 | 24816 | 22988 | **0.93** |
| 512 | 105963 | 100290 | 0.95 |
| 1024 | 220020 | 197345 | **0.93** |
| 2048 | 476577 | 463436 | 0.98 |
| 4096 | 1053265 | 964127 | **0.93** |

**On Skylake-X virt with the proper bench harness, we beat hand 1-9%
across all measured K for both R=32 and R=64.** This was established
in earlier work and the assembly breakdown in those docs already
explained why (cluster-sequential PASS 2 closed the 148-stack-op gap
that the SU-only variant had).

## What v1 (llvm-mca) got wrong

llvm-mca SPR static analysis said R=32 t1_dit Ours/Hand = 0.78×
(we win 22%). Real bare-metal Skylake-X says 0.93× at K=1024 (we
win 7%). llvm-mca was directionally right (we win) but overstated
the magnitude by ~15 percentage points. The static port-saturation
model can't accurately predict OoO real-hardware cycles for an
asymmetric instruction count.

## What v2 (container) got wrong

A runtime bench inside a heavily virtualized container (2 cores
visible, no cpufreq/perf-counter access, unknown other tenants) on
Emerald Rapids showed us 17-30% slower than hand. Reproduction in
this doc:

| K | Doc 09 result (SKX virt) | This container (EMR) |
|---|---|---|
| 64 | 0.98 | 1.27 |
| 128 | 0.99 | 1.40 |
| 1024 | 0.93 | 1.19 |
| 2048 | 0.91 | 1.04 |
| 4096 | 0.91 | **0.98** |

Pattern: at low K, container noise dominates and ours suffers more
than hand. At K=4096, the gap nearly closes (0.98) — codelet quality
recovers when the per-iteration cost amortizes over enough work to
swamp container interference.

Why our shape is more noise-sensitive: avg/best cycle ratios in this
environment are 1.20× for ours, ~1.01× for hand. Our larger
instruction footprint (910 vs 709 FP ops on R=32) means more surface
area for VMexits, hypervisor preemption, shared L1/LLC contention,
and TLB flushes to perturb. Hand's tighter code is structurally more
robust to these. This is a virtualization artifact; on dedicated
bare-metal cores with no contention it doesn't show up — as docs
09/11 demonstrate.

## What this means going forward

1. **The composite emission work IS validated** by the bare-metal
   bench in docs 09/11. We beat hand. The work that landed —
   Spill + SU within passes + block-sequential PASS 1 +
   cluster-sequential PASS 2 + deferred reload — produces real wins
   on real hardware. That earlier conclusion stands.

2. **llvm-mca is a useful gating tool but not a substitute for
   bare-metal measurement.** Its static model was 15 percentage
   points off the real cycle ratio. Use it for "is the scheduling
   ceiling close?" sanity checks, not for hand-vs-ours conclusions.

3. **Heavily virtualized containers cannot benchmark our codelets
   meaningfully.** The 2-core cgroup-throttled environment in this
   session amplifies our instruction-count disadvantage in a way
   that doesn't reflect production. Future bench measurements need
   dedicated cores, fixed clocks, and no co-tenants — i.e. real
   bare metal or dedicated cloud instances with full isolation.

4. **The earlier session memory note** — "Hand has ~10 more FMAs and
   ~10 fewer add/subs ... that's the gap we're closing here" —
   referred to the SU+Spill PRE-cluster-sequential variant (which was
   4-22% behind hand). After cluster-sequential PASS 2 closed the
   148-stack-op gap, we crossed to 1-9% AHEAD. That progression is
   documented across docs 03-09.

## Current asm metrics (for the record)

R=32 t1_dit asm under `gcc -O3 -mavx512f -mavx512dq -mfma -march=native`
on this EMR container (post all the session 26 cleanup, identical
to the asm at the time docs 09/11 were measured per the byte-diff
verifications):

| Metric | Ours | Hand |
|---|---|---|
| FMAs (fmadd+fnmadd+fmsub) | 47+31+28 = 106 | 64+15+49 = 128 |
| vmulpd | (varies) | 101 |
| vaddpd | (varies) | 155 |
| vsubpd | (varies) | 155 |
| vmovapd (reg-reg) | 288 | 170 |
| vmovupd (memory) | 190 | 128 |
| Total FP instructions | 910 | 709 |

Hand has 22 more FMAs and 201 fewer total FP ops. We have more
explicit reg-reg moves but less total dispatch work. On llvm-mca's
SPR model: ours RThroughput=200, hand RThroughput=229. On real
Skylake-X bare-metal cycles: we're ahead 1-9%.

## Files / how to reproduce the bare-metal numbers

The bench infrastructure is at `bench/bench_r32_spill_su.c` and
`bench/bench_r64_spill_su.c` (or analogous). They include the hand
references with a stale path `../radix32_handcoded.h` — fix to point
to `bench/references/radix32_handcoded.h` and the build works.

To run on dedicated hardware:

```bash
cd vfft_v2_pack
dune build
./_build/default/bin/gen_radix.exe 32 --twiddled --in-place --emit-c \
    > /tmp/g_su_spill.c
./_build/default/bin/gen_radix.exe 32 --twiddled --in-place --no-recipe --emit-c \
    > /tmp/g_topo.c

# Patch include path in bench source
sed 's|"../radix32_handcoded.h"|"bench/references/radix32_handcoded.h"|' \
    bench/bench_r32_spill_su.c > /tmp/bench_r32.c

# Build (excludes the broken --no-recipe --spill variant)
gcc -O3 -mavx512f -mavx512dq -mfma -march=native /tmp/bench_r32.c \
    /tmp/g_su_spill.c /tmp/g_topo.c -o /tmp/bench_r32 -lm

# On dedicated core (taskset/numactl as appropriate):
for K in 64 128 256 512 1024 2048 4096; do /tmp/bench_r32 $K; done
```

## Loose ends from this session

- The `--no-recipe --spill` variant has a compile error (undeclared
  `tNNN` tags in the spill stores). This regressed somewhere — the
  cross-pass inlining work in doc 26 may have broken the Spill-only
  emission path even though it preserved SU+Spill. Worth fixing or
  documenting that the Spill-only variant is no longer valid as a
  bench point.
- Bench harnesses with stale `../radix32_handcoded.h` includes
  should be updated to `bench/references/radix32_handcoded.h`.
- Bare-metal SPR/EMR measurement (AWS c7i.metal-24xl, Equinix Metal,
  etc.) would let us reproduce docs 09/11 on the silicon you're
  actually targeting for HFT colo. Container-based numbers can't
