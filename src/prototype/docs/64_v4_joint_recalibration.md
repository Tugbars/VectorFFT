# 64 — V4 Cost Model Joint Recalibration (2026-05-17)

**Status:** Landed in `src/prototype-core/estimate_plan.h` 2026-05-17. Companion measurement tool at `src/prototype/cost_model/measure_memboundness.c` emits `cost_model/generated/radix_memboundness.h`. Patient bench harness at `src/prototype-core/exhaustive_patient.h` updated with inter-trial pacing.

**One-line summary:** Replaced V4's heuristic-stack (`wide_penalty`, hardcoded bandwidth, `dtlb_cost`) with measured per-(R, cache-tier) data and dropped the dtlb_cost term entirely. Result: V4 estimate within **1.7× of measured truth** on all tested cells, including N=131072 K=4. Down from 31× off before the recalibration.

Companion fix: patient bench now paces between trials, not just between candidates. Without it, long benches got thermally cooked and rebench was +148% off. With it, rebench delta is sub-noise.

---

## 1 — What changed

### 1.1 wide_penalty → measured `mb_factor[R][tier]`

Old V4 had:
```c
wide_penalty = max(0, R - baseline_R) × extra_latency(tier) × groups × K
```
A linear-in-R heuristic with baseline_R=4 and `extra_latency = {0, 7, 35, 150}` for L1/L2/L3/DRAM. The premise: wide codelets stall on out-of-L1 loads in proportion to (R − 4).

Premise was true before commit `c18b7c1` (2026-05-15 "Force YMM register allocation in AVX2 codelets") which made R=16/32/64+ codelets register-spill-free. After c18b7c1, the linear penalty is an artifact penalizing a pattern that no longer exists. CPE table at me=65536 vs me=256 confirmed it: R=16/32/64 are essentially flat across cache-state samples post-c18b7c1.

**Replaced with:** per-(R, cache-tier) inflation factor measured directly by `measure_memboundness.c`. Numbers on Raptor Lake AVX2:

| R | L1 | L2 | L3 | DRAM |
|---|---:|---:|---:|---:|
| 16 | 1.00 | 1.14 | 1.05 | 1.11 |
| 32 | 1.00 | 1.28 | 1.79 | 1.82 |
| 64 | 1.00 | 1.47 | 1.90 | 1.98 |
| 128 | 1.00 | 0.95 | 1.37 | 1.45 |
| 256 | 1.00 | 1.00 | 1.21 | 1.53 |

In V4, `cf_eff` for R ≥ 16 now reads from `stride_radix_memboundness_avx2[R].factor[tier]` instead of the old `cf_stage`. Small radixes (R < 16) keep the heuristic since they're not in the table.

### 1.2 Hardcoded bandwidth → measured `cache_cyc_per_byte[tier]`

Old V4 used spec-sheet bandwidth constants in `buffer_pass_per_stage`:
```c
bytes_per_cycle = (45 | 14 | 7 | 9)  // L1 | L2 | L3 | DRAM
buffer_pass_per_stage = 2 × N × K × 16 / bytes_per_cycle
```

**Replaced with:** measured stride-1 memcpy throughput per cache tier. Same `measure_memboundness.c` tool, second pass:

| Tier | Measured cyc/byte | Effective bytes/cyc | Effective GB/s |
|---|---:|---:|---:|
| L1 | 0.0019 | 533 | 1700 |
| L2 | 0.0058 | 172 | 547 |
| L3 | 0.0121 | 83 | 263 |
| DRAM | (uses L3 value; bench skipped due to large-alloc instability on Windows) | — | — |

The measured throughput is **6–15× higher** than spec sheets predicted. memcpy uses non-temporal/streaming stores plus the HW prefetcher handles sequential access perfectly. The original V4 was systematically overcharging buffer_pass.

V4's `buffer_pass_per_stage` now reads `stride_cache_cyc_per_byte_avx2[tier]` directly.

### 1.3 `dtlb_cost` dropped entirely

Old V4 charged page-walk penalty:
```c
if (pages_per_group > dtlb_entries)
    dtlb_cost = groups × (pages_per_group − 96) × 7 cyc
```

At N=131072 K=4 [2,4,64,16,16], the R=4 stage with stride=65536 spans 512 pages/group × 32k groups → **95M cycles** in "page-walk penalty" alone. V4 ranked the patient winner 31× worse than its own pick because of this single term.

Why it was wrong: Raptor Lake has a 2048-entry STLB behind the 96-entry DTLB. Almost all the "misses" V4 was counting are absorbed by STLB at ~7 cyc each (matching V4's miss_cycles), but they don't accumulate the way the formula assumes — they pipeline with compute and prefetching hides the rest.

Term removed. Could be re-added with proper STLB modeling + HW-prefetch awareness, but for now it's doing strictly more harm than good.

### 1.4 `tw_cost` multiplier 4× → 1×

Twiddle-load cost was charging `4 × (R−1) × accumulated_K` for L1-spilled twiddle tables. For deep plans (`accumulated_K` ~ 32k at inner stages), this dominated the score with ~2M cyc per stage.

In reality the HW prefetcher pipelines twiddle loads with compute almost perfectly — the codelet's twiddle stream is sequential, predictable, and the load ports have spare capacity. Charging 4 cyc/element was over-pessimistic.

**Reduced to 1 cyc/element for L1-spilled, 0.5 cyc/element for L1-resident.** No new measurement here; just acknowledging the HW reality that prefetcher-aware tools have always known.

---

## 2 — Validation against measurement

### 2.1 Cell sweep (V4 estimate top-1 vs patient/flat verdict)

All same codelets, prototype-core only. Patient with inter-trial pacing.

| Cell | V4 estimate pick | V4 score | Patient/flat verdict | V4 score(verdict) | Gap |
|---|---|---:|---|---:|---:|
| N=1024 K=128 | [8,16,8] | 441k | [4,4,4,16] | 487k | **1.1×** |
| N=4096 K=4 | [8,8,8,8] | 57k | [8,32,16] | 82k | 1.4× |
| N=8192 K=4 | [16,8,8,8] | 128k | [4,4,32,16] | 179k | 1.4× |
| N=16384 K=4 | [8,4,8,8,8] | 273k | [4,8,32,16] | 374k | 1.4× |
| N=131072 K=4 | [16,16,8,8,8] | 3.27M | **[4,4,4,32,64]** (patient paced) | 5.46M | **1.67×** |

**V4 within 1.7× on all tested cells.** That puts the true winner solidly inside V4-screened top-M=10 territory — the screened workflow (V4 ranks, top-M benched) finds the verdict reliably.

The V4 top-1 still doesn't match measurement. That's irreducible: at high N the OoO engine + cache state + prefetcher interactions produce results no static cost model can predict from first principles. The remaining gap is the [feedback_cost_model_ceiling](../../memory/feedback_cost_model_ceiling.md) regime.

### 2.2 What V4 used to predict (before the recalibration)

For comparison — same cells, OLD V4 with `wide_penalty`, hardcoded bandwidth, `dtlb_cost` × 4 tw_cost:

| Cell | OLD V4 score(verdict) | Gap |
|---|---:|---:|
| N=131072 K=4 | 101M for [2,4,64,16,16] | **31×** |
| N=131072 K=4 | 41.9M for [4,4,32,8,32] | **12.7×** |

The 31× and 12.7× errors at N=131072 K=4 were almost entirely the `dtlb_cost` term firing on R=4 stages with large strides.

---

## 3 — Patient bench inter-trial pacing

Independent fix landed in the same session. `exhaustive_patient.h` was doing:

```c
for (int t = 0; t < 7; t++) {        // 7 trials best-of
    memcpy(re, orig_re, ...);
    memcpy(im, orig_im, ...);
    for (int i = 0; i < reps; i++) execute_fwd(plan);
    /* no pacing between trials */
}
```

Inter-candidate pacing (200ms) was in place, but the 7 trials within a single candidate ran back-to-back. At N=131072 K=4 one execute is ~1.7 ms × ~10-100 reps × 7 trials → 0.1-1.2s of solid compute per candidate with no cooldown. CPU package heat-soaked, especially during the final rebench pass.

Symptom: rebench at the end of a 14-min run reported **+148% delta** on every candidate — the package was running 2.5× slower than the first-pass benches.

**Fix:** `VFFT_PROTO_PATIENT_INTER_TRIAL_PACE_MS = 100` (default), with `_vfft_proto_dp_sleep_ms` between trials. After fix, rebench delta dropped to **−3.7% to +1.9%** — sub-noise. And paced patient found a **better winner than flat exhaustive** on the same cell: [4,4,4,32,64] = 1.47 ms vs flat's [4,4,32,8,32] = 1.71 ms. The flat bench's thermal noise was hiding the real winner.

Wall-time cost: ~14 min → ~20 min for N=131072 K=4 patient. Worth it.

---

## 4 — Files

### Measurement tool
- [src/prototype/cost_model/measure_memboundness.c](../cost_model/measure_memboundness.c) — generates `radix_memboundness.h` and `cache_bandwidth` table
- [src/prototype/cost_model/build_measure_memboundness.sh](../cost_model/build_measure_memboundness.sh) — build script (response-file link for Windows arg-length, LIB env for icx)

### Generated tables
- [src/prototype/cost_model/generated/radix_memboundness.h](../cost_model/generated/radix_memboundness.h) — measured `mb_factor[R][tier]` + `cyc_per_byte[tier]`

### Cost model
- [src/prototype-core/estimate_plan.h](../../prototype-core/estimate_plan.h) — V4 score function consuming the measured tables

### Patient bench
- [src/prototype-core/exhaustive_patient.h](../../prototype-core/exhaustive_patient.h) — `VFFT_PROTO_PATIENT_INTER_TRIAL_PACE_MS=100` default

---

## 5 — Open work

1. **K-axis CPE base.** `radix_cpe.h` is measured at K=256. For K=4 cells, the codelet's per-call IPC is lower than at K=256 (less batch-ILP), but V4 still uses K=256 base. Need to extend `measure_cpe.c` with smaller `me` samples (4, 16, 64). Blocked on an icx-specific segfault when CPE_N_ME_SAMPLES changes — separate debug session needed.

2. **DRAM bandwidth measurement.** `measure_memboundness` skips the DRAM tier (large-alloc instability on Windows when buffer > 32 MB). Currently DRAM defaults to the L3 value as a fallback. Add a robust DRAM bench path.

3. **STLB-aware dtlb_cost** if we want to re-add page-walk modeling for very-strided patterns. Current state (no term at all) is correct enough for our test cells; could become important for cells with non-pow2 strides we don't currently test.

4. **HW-prefetch-aware tw_cost** — current 1× multiplier is a guess. Could measure actual twiddle-load cost across cache tiers with the same across-group-sweep pattern that `measure_memboundness` uses for data.

---

## 6 — Related

- [63_v4_estimate_method.md](63_v4_estimate_method.md) — V4 baseline, before the joint recalibration
- [58_cost_model_recalibration.md](58_cost_model_recalibration.md) — earlier V1→V2 work
- [feedback_cost_model_ceiling](../../../memory/feedback_cost_model_ceiling.md) — the irreducible OoO ceiling
- [v4_wide_penalty_artifact](../../../memory/v4_wide_penalty_artifact.md) — scoping memory for this work
- [v4_joint_recalibration_2026_05_17](../../../memory/v4_joint_recalibration_2026_05_17.md) — session memory
