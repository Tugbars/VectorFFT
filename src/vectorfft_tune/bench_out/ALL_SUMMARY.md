# VectorFFT portfolio tuning summary

Generated: 2026-04-21 13:43:12
Host: Windows AMD64 (pinned to CPU 2)

## Portfolio

- **Radixes tuned** (with measurement data on disk): 17
- **Total measurements** (fwd + bwd, across all radixes): 1604
- **This invocation**: 17 passed, 0 failed, 17 total
- **Elapsed**: 556.9s (9.3 min)

## This invocation

| Radix | Status | Measurements | Validated | Failed | Elapsed | Reason |
|---|---|---|---|---|---|---|
| r3 | success | 192 | 288 | 0 | 32.6s |  |
| r4 | success | 192 | 84 | 0 | 30.1s |  |
| r5 | success | 144 | 108 | 0 | 20.4s |  |
| r6 | success | 144 | 84 | 0 | 22.1s |  |
| r7 | success | 144 | 108 | 0 | 24.8s |  |
| r8 | success | 228 | 120 | 0 | 37.8s |  |
| r10 | success | 144 | 108 | 0 | 24.3s |  |
| r11 | success | 144 | 108 | 0 | 22.1s |  |
| r12 | success | 144 | 108 | 0 | 23.2s |  |
| r13 | success | 144 | 84 | 0 | 23.5s |  |
| r16 | success | 300 | 336 | 0 | 46.8s |  |
| r17 | success | 144 | 84 | 0 | 22.2s |  |
| r19 | success | 144 | 84 | 0 | 22.6s |  |
| r20 | success | 144 | 108 | 0 | 24.6s |  |
| r25 | success | 256 | 192 | 0 | 39.0s |  |
| r32 | success | 300 | 336 | 0 | 48.9s |  |
| r64 | success | 300 | 336 | 0 | 59.7s |  |

## Dispatcher winners by radix (fwd direction)

For each (radix, ISA), count of sweep points each protocol wins. "Total" is the number of (me, ios) grid points where at least one protocol was measured.

### AVX2

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 16 | 15 | 1 | 32 | me≥768 |
| r4 | 12 | 1 | 5 | 18 | — |
| r5 | 12 | 12 | 0 | 24 | me≥640 |
| r6 | 8 | 16 | 0 | 24 | me≥8 |
| r7 | 4 | 20 | 0 | 24 | me≥56 |
| r8 | 12 | 1 | 5 | 18 | — |
| r10 | 14 | 9 | 1 | 24 | me≥256 |
| r11 | 15 | 5 | 4 | 24 | me≥256 |
| r12 | 8 | 14 | 2 | 24 | me≥16 |
| r13 | 11 | 10 | 3 | 24 | me≥128 |
| r16 | 10 | 3 | 5 | 18 | me≥128 |
| r17 | 10 | 10 | 4 | 24 | me≥64 |
| r19 | 6 | 11 | 7 | 24 | me≥32 |
| r20 | 2 | 17 | 5 | 24 | me≥8 |
| r25 | 11 | 11 | 2 | 24 | me≥64 |
| r32 | 6 | 5 | 7 | 18 | me≥64 |
| r64 | 4 | 6 | 8 | 18 | me≥64 |

### AVX-512

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|

## Last run per radix

| Radix | Last measurement written |
|---|---|
| r3 | 2026-04-21 13:34 |
| r4 | 2026-04-21 13:35 |
| r5 | 2026-04-21 13:35 |
| r6 | 2026-04-21 13:35 |
| r7 | 2026-04-21 13:36 |
| r8 | 2026-04-21 13:36 |
| r10 | 2026-04-21 13:37 |
| r11 | 2026-04-21 13:37 |
| r12 | 2026-04-21 13:38 |
| r13 | 2026-04-21 13:38 |
| r16 | 2026-04-21 13:39 |
| r17 | 2026-04-21 13:39 |
| r19 | 2026-04-21 13:40 |
| r20 | 2026-04-21 13:40 |
| r25 | 2026-04-21 13:41 |
| r32 | 2026-04-21 13:42 |
| r64 | 2026-04-21 13:43 |

---

Per-radix reports: `generated/rN/vfft_rN_report.md`. Raw measurements: `bench_out/rN/measurements.jsonl`. Orchestrator logs: `bench_out/rN/orchestrator.log`.
