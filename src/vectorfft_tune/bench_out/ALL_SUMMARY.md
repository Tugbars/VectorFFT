# VectorFFT portfolio tuning summary

Generated: 2026-04-21 12:00:29
Host: Windows AMD64 (pinned to CPU 2)

## Portfolio

- **Radixes tuned** (with measurement data on disk): 13
- **Total measurements** (fwd + bwd, across all radixes): 1316
- **This invocation**: 13 passed, 0 failed, 13 total
- **Elapsed**: 456.3s (7.6 min)

## This invocation

| Radix | Status | Measurements | Validated | Failed | Elapsed | Reason |
|---|---|---|---|---|---|---|
| r3 | success | 192 | 288 | 0 | 32.0s |  |
| r4 | success | 192 | 84 | 0 | 30.5s |  |
| r5 | success | 144 | 108 | 0 | 21.0s |  |
| r7 | success | 144 | 108 | 0 | 25.1s |  |
| r8 | success | 228 | 120 | 0 | 38.0s |  |
| r10 | success | 144 | 108 | 0 | 24.7s |  |
| r11 | success | 144 | 108 | 0 | 21.8s |  |
| r12 | success | 144 | 108 | 0 | 22.0s |  |
| r16 | success | 300 | 336 | 0 | 46.3s |  |
| r20 | success | 144 | 108 | 0 | 23.9s |  |
| r25 | success | 256 | 192 | 0 | 39.2s |  |
| r32 | success | 300 | 336 | 0 | 50.1s |  |
| r64 | success | 300 | 336 | 0 | 57.5s |  |

## Dispatcher winners by radix (fwd direction)

For each (radix, ISA), count of sweep points each protocol wins. "Total" is the number of (me, ios) grid points where at least one protocol was measured.

### AVX2

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 12 | 20 | 0 | 32 | me≥768 |
| r4 | 8 | 6 | 4 | 18 | me≥64 |
| r5 | 8 | 16 | 0 | 24 | me≥320 |
| r7 | 1 | 23 | 0 | 24 | me≥56 |
| r8 | 11 | 2 | 5 | 18 | me≥128 |
| r10 | 15 | 9 | 0 | 24 | me≥256 |
| r11 | 14 | 6 | 4 | 24 | — |
| r12 | 7 | 14 | 3 | 24 | me≥16 |
| r16 | 10 | 4 | 4 | 18 | me≥128 |
| r20 | 6 | 14 | 4 | 24 | me≥32 |
| r25 | 8 | 15 | 1 | 24 | me≥64 |
| r32 | 3 | 5 | 10 | 18 | me≥64 |
| r64 | 3 | 6 | 9 | 18 | me≥64 |

### AVX-512

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|

## Last run per radix

| Radix | Last measurement written |
|---|---|
| r3 | 2026-04-21 11:53 |
| r4 | 2026-04-21 11:53 |
| r5 | 2026-04-21 11:54 |
| r7 | 2026-04-21 11:54 |
| r8 | 2026-04-21 11:55 |
| r10 | 2026-04-21 11:55 |
| r11 | 2026-04-21 11:56 |
| r12 | 2026-04-21 11:56 |
| r16 | 2026-04-21 11:57 |
| r20 | 2026-04-21 11:57 |
| r25 | 2026-04-21 11:58 |
| r32 | 2026-04-21 11:59 |
| r64 | 2026-04-21 12:00 |

---

Per-radix reports: `generated/rN/vfft_rN_report.md`. Raw measurements: `bench_out/rN/measurements.jsonl`. Orchestrator logs: `bench_out/rN/orchestrator.log`.
