# 03 — Dynamic CPE (`radix_cpe.h`)

What `tools/radix_profile/measure_cpe.c` does and what's in
`src/core/generated/radix_cpe.h`. This is the **primary** input to the
cost model — the one that captures real per-codelet performance on the
calibration host.

## What it measures

For each `(radix R, variant, ISA)` slot in the registry where a
codelet exists, `measure_cpe` times it under controlled conditions and
records:

| Field | Meaning |
|-------|---------|
| `cyc_n1` | cycles per butterfly call, no-twiddle codelet |
| `cyc_t1` | cycles per butterfly call, twiddle codelet |
| `cyc_t1s` | cycles per butterfly call, scalar-twiddle codelet |
| `cyc_log3` | cycles per butterfly call, log3 derivation codelet |

All four are timed at `K = 256`. The numbers are derived as
`ns_per_call × measured_freq_GHz / K`.

Slots that the registry doesn't populate (e.g. R=2 has no t1s) get
`0.0`. The cost model treats `0.0` as "not measured" and falls back.

## Methodology

### Per-codelet timing kernel

For each codelet:

1. **Allocate aligned buffers** — `R*K` doubles for re/im inputs (and
   for n1, separate outputs); twiddle table of `(R-1)*K` doubles for
   t1/t1s/log3.
2. **Calibrate iteration count.** Run 16 → 64 → 256 → … iterations
   until one batch hits ~100 µs. Compute how many iterations give a
   ~5 ms target batch (`BENCH_TARGET_NS = 5_000_000`).
3. **Run BENCH_N_RUNS = 51 batches** of `n_iters` calls each. Record
   `dt / n_iters` per batch as one sample.
4. **Median across the 51 samples** → reported `ns/call`.
5. **Coefficient of variation across the 51 samples** → reported as
   `CV%`. Variance check threshold is 5%.

The function pointer is dereferenced through the registry struct so
the compiler can't hoist or inline it — every iteration is a real call.

### Frequency calibration

```c
static double measure_freq_ghz(void) {
    (void)__rdtsc();          /* warm */
    double t0 = now_ns();
    uint64_t r0 = __rdtsc();
    while (now_ns() - t0 < 50e6) { /* spin ~50 ms */ }
    uint64_t r1 = __rdtsc();
    double t1 = now_ns();
    return (r1 - r0) / (t1 - t0);  /* GHz */
}
```

Measured RDTSC ticks divided by elapsed nanoseconds give the **effective**
GHz the CPU actually ran at — not the nominal turbo, not what
`/proc/cpuinfo` reports. On a noisy consumer PC at idle this can be 3.2 GHz
even on a CPU rated for 5.7 GHz turbo, because the 50 ms window catches
mixed P-states.

### Variance check

After all codelets are timed, the tool computes the worst CV across
all (R, variant) and refuses to overwrite `radix_cpe.h` if it exceeds
5%, unless `--force` is passed. The threshold is empirical — quiet
calibration hosts produce CV under 3%; consumer PCs under load can hit
30–90%.

When the tool refuses, it prints which codelet had the worst variance
so the operator can investigate. Common causes: thermal throttling,
background process load, SMT sibling activity, frequency scaling.

### Header fingerprint

Every emitted `radix_cpe.h` carries a comment block at the top:

```
 * Calibration fingerprint:
 * Host OS:    Windows 11 Home (build 26200)
 * Host CPU:   Intel(R) Core(TM) i9-14900KF
 * ISA tag:    avx2
 * Eff. freq:  5.690 GHz (RDTSC over 50ms wall)
 * Max CV:     2.34% (refuse threshold 5.00%)
 * Date (UTC): 2026-05-04 09:12
```

This makes a stale or wrong-platform commit obvious in code review. A
header from a noisy laptop will have `Max CV: 28%` and `Eff. freq:
3.2 GHz` — both red flags.

## Variants

```c
typedef struct {
    double cyc_n1;
    double cyc_t1;
    double cyc_t1s;
    double cyc_log3;
} stride_radix_cpe_t;

static const stride_radix_cpe_t stride_radix_cpe_avx2[]   = { /* ... */ };
static const stride_radix_cpe_t stride_radix_cpe_avx512[] = { /* ... */ };
static const stride_radix_cpe_t stride_radix_cpe_scalar[] = { /* ... */ };
```

Only the table for the ISA `measure_cpe` was built against gets
populated. The other ISA tables are zero-initialized — the cost model
falls back to the static profile when looking those up.

This means a single host can only produce one ISA's CPE table. Cross-ISA
calibration requires running on hardware that supports the target ISA.

## Per-radix bottlenecks captured

The CPE numbers reflect real per-radix bottlenecks the static op count
can't see. From VTune profiling at K=256:

| Radix | Retiring | Bottleneck |
|-------|---------|------------|
| R=4 | 86% | Compute-peak — port 0/1 saturated |
| R=8 | 72% | Dependency chains in DFT-8 critical path |
| R=10/11/12/13 | 50–65% | Winograd FMA dependency chains |
| R=16 | 25% | Store-bound + L1 latency (post-prefetch) |
| R=32 | 34% | L1 store-DTLB overflow (~80 pages > capacity) |
| R=64 | 27% | Load + store DTLB overflow (~160 pages) |

The CPE captures all of these in one number — the cost model doesn't
need to know *why* R=32 is expensive at K=256, just that it is.

## Variance and the consumer-PC reality

On a noisy consumer PC, CV can run 20–90% across a measurement run.
The variance check exists to refuse such measurements rather than
commit them as truth. Two paths to a clean header:

1. **Run on a calibration-grade host.** Single P-core, frequency
   locked, performance plan, no other load. CV < 5% reliably.
2. **Run via `orchestrator.py --phase cpe_measure --auto-performance`.**
   The orchestrator handles power-plan switching, affinity, and signal-
   handler restore. Same calibration discipline as wisdom calibration.

Without those, the bench-vs-wisdom ratio drifts upward as the model
gets noisier inputs — but the architecture still works, just at lower
fidelity.

## Why K=256 specifically

Three reasons:

1. **Matches the wisdom calibrator's primary K.** Most wisdom entries
   are at K ∈ {4, 32, 256}, with 256 being where most production loads
   live (image / audio batch sizes).
2. **Large enough that loop overhead amortizes.** At K < 64 codelet
   call overhead distorts the per-butterfly number.
3. **Small enough to fit in L1.** R*K complex doubles at R=64, K=256 =
   32 KB — under the L1 budget on Raptor Lake. Bigger K would put the
   working set in L2 and confound the codelet timing with cache effects.

The cost model adds its own cache-aware correction (see
[04_factorizer.md](04_factorizer.md)) for plans where the per-stage
working set differs from K=256 conditions, so the K=256 baseline is
fine.

## Regeneration

### Direct (development iteration)

```
python build_tuned/build.py --src tools/radix_profile/measure_cpe.c
tools/radix_profile/measure_cpe.exe                    # variance check ON
tools/radix_profile/measure_cpe.exe --force            # bypass variance
tools/radix_profile/measure_cpe.exe --no-emit          # print only
tools/radix_profile/measure_cpe.exe --verbose          # print + emit
```

### Via orchestrator (calibration-grade)

```
python src/vectorfft_tune/common/orchestrator.py --phase cpe_measure --auto-performance
```

Includes preflight check (governor / power plan), affinity pinning, and
signal-handler restore. See [07_regeneration_workflow.md](07_regeneration_workflow.md).

## See also

- [`tools/radix_profile/measure_cpe.c`](../../tools/radix_profile/measure_cpe.c) — the source
- [`src/core/generated/radix_cpe.h`](../../src/core/generated/radix_cpe.h) — the output
- [`docs/vtune-profiles/`](../vtune-profiles/) — per-radix VTune docs explaining the bottlenecks the CPE numbers capture
- [04_factorizer.md](04_factorizer.md) — how the cost model uses the table
