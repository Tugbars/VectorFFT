# VectorFFT portfolio tuning summary

Generated: 2026-04-22 21:22:57
Host: Windows AMD64 (pinned to CPU 2)

## Portfolio

- **Radixes tuned** (with measurement data on disk): 17
- **Total measurements** (fwd + bwd, across all radixes): 1487
- **This invocation**: 17 passed, 0 failed, 17 total
- **Elapsed**: 518.3s (8.6 min)

## This invocation

| Radix | Status | Measurements | Validated | Failed | Elapsed | Reason |
|---|---|---|---|---|---|---|
| r3 | success | 192 | 288 | 0 | 32.0s |  |
| r4 | success | 192 | 84 | 0 | 29.7s |  |
| r5 | success | 144 | 108 | 0 | 21.8s |  |
| r6 | success | 144 | 84 | 0 | 22.4s |  |
| r7 | success | 144 | 108 | 0 | 24.7s |  |
| r8 | success | 228 | 120 | 0 | 37.2s |  |
| r10 | success | 144 | 108 | 0 | 24.4s |  |
| r11 | success | 144 | 108 | 0 | 21.8s |  |
| r12 | success | 144 | 108 | 0 | 22.6s |  |
| r13 | success | 144 | 84 | 0 | 23.1s |  |
| r16 | success | 222 | 258 | 0 | 36.0s |  |
| r17 | success | 144 | 84 | 0 | 22.7s |  |
| r19 | success | 144 | 84 | 0 | 23.3s |  |
| r20 | success | 144 | 108 | 0 | 24.4s |  |
| r25 | success | 256 | 192 | 0 | 40.5s |  |
| r32 | success | 222 | 258 | 0 | 36.9s |  |
| r64 | success | 222 | 258 | 0 | 42.9s |  |

## Dispatcher winners by radix (fwd direction)

For each (radix, ISA), count of sweep points each protocol wins. "Total" is the number of (me, ios) grid points where at least one protocol was measured.

### AVX2

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 11 | 20 | 1 | 32 | me≥24 |
| r4 | 8 | 4 | 6 | 18 | me≥64 |
| r5 | 5 | 19 | 0 | 24 | me≥40 |
| r6 | 6 | 18 | 0 | 24 | me≥8 |
| r7 | 0 | 22 | 2 | 24 | me≥56 |
| r8 | 6 | 3 | 9 | 18 | me≥128 |
| r10 | 15 | 9 | 0 | 24 | me≥256 |
| r11 | 12 | 9 | 3 | 24 | me≥256 |
| r12 | 2 | 18 | 4 | 24 | me≥16 |
| r13 | 9 | 7 | 8 | 24 | me≥32 |
| r16 | 6 | 6 | 6 | 18 | me≥64 |
| r17 | 18 | 4 | 2 | 24 | — |
| r19 | 10 | 8 | 6 | 24 | me≥128 |
| r20 | 2 | 20 | 2 | 24 | me≥8 |
| r25 | 11 | 12 | 1 | 24 | me≥64 |
| r32 | 5 | 4 | 9 | 18 | me≥64 |
| r64 | 2 | 3 | 13 | 18 | me≥64 |

### AVX-512

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|

## Last run per radix

| Radix | Last measurement written |
|---|---|
| r3 | 2026-04-22 21:14 |
| r4 | 2026-04-22 21:15 |
| r5 | 2026-04-22 21:15 |
| r6 | 2026-04-22 21:16 |
| r7 | 2026-04-22 21:16 |
| r8 | 2026-04-22 21:17 |
| r10 | 2026-04-22 21:17 |
| r11 | 2026-04-22 21:18 |
| r12 | 2026-04-22 21:18 |
| r13 | 2026-04-22 21:18 |
| r16 | 2026-04-22 21:19 |
| r17 | 2026-04-22 21:19 |
| r19 | 2026-04-22 21:20 |
| r20 | 2026-04-22 21:20 |
| r25 | 2026-04-22 21:21 |
| r32 | 2026-04-22 21:22 |
| r64 | 2026-04-22 21:22 |

---

Per-radix reports: `generated/rN/vfft_rN_report.md`. Raw measurements: `bench_out/rN/measurements.jsonl`. Orchestrator logs: `bench_out/rN/orchestrator.log`.
