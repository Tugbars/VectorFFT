# VectorFFT portfolio tuning summary

Generated: 2026-04-25 23:07:58
Host: Windows AMD64 (pinned to CPU 2)

## Portfolio

- **Radixes tuned** (with measurement data on disk): 17
- **Total measurements** (fwd + bwd, across all radixes): 3551
- **This invocation**: 17 passed, 0 failed, 17 total
- **Elapsed**: 307.3s (5.1 min)

## This invocation

| Radix | Status | Measurements | Validated | Failed | Elapsed | Reason |
|---|---|---|---|---|---|---|
| r3 | success | 768 | 288 | 0 | 4.1s |  |
| r4 | success | 348 | 84 | 0 | 51.2s |  |
| r5 | success | 288 | 108 | 0 | 2.1s |  |
| r6 | success | 408 | 84 | 0 | 2.1s |  |
| r7 | success | 288 | 108 | 0 | 2.1s |  |
| r8 | success | 414 | 120 | 0 | 60.6s |  |
| r10 | success | 448 | 108 | 0 | 2.1s |  |
| r11 | success | 288 | 108 | 0 | 2.1s |  |
| r12 | success | 448 | 108 | 0 | 2.1s |  |
| r13 | success | 288 | 84 | 0 | 2.3s |  |
| r16 | success | 936 | 258 | 0 | 3.0s |  |
| r17 | success | 288 | 84 | 0 | 2.4s |  |
| r19 | success | 288 | 84 | 0 | 2.7s |  |
| r20 | success | 288 | 108 | 0 | 2.4s |  |
| r25 | success | 512 | 192 | 0 | 4.9s |  |
| r32 | success | 402 | 258 | 0 | 62.7s |  |
| r64 | success | 402 | 258 | 0 | 66.1s |  |

## Dispatcher winners by radix (fwd direction)

For each (radix, ISA), count of sweep points each protocol wins. "Total" is the number of (me, ios) grid points where at least one protocol was measured.

### AVX2

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 5 | 24 | 3 | 32 | me≥24 |
| r4 | 15 | 8 | 10 | 33 | me≥64 |
| r5 | 0 | 24 | 0 | 24 | me≥40 |
| r6 | 2 | 22 | 0 | 24 | me≥16 |
| r7 | 1 | 23 | 0 | 24 | me≥56 |
| r8 | 23 | 6 | 4 | 33 | me≥64 |
| r10 | 0 | 23 | 1 | 24 | me≥8 |
| r11 | 1 | 19 | 4 | 24 | me≥8 |
| r12 | 2 | 17 | 5 | 24 | me≥8 |
| r13 | 0 | 22 | 2 | 24 | me≥8 |
| r16 | 0 | 0 | 33 | 33 | — |
| r17 | 1 | 17 | 6 | 24 | me≥8 |
| r19 | 0 | 23 | 1 | 24 | me≥8 |
| r20 | 0 | 24 | 0 | 24 | me≥8 |
| r25 | 5 | 18 | 1 | 24 | me≥8 |
| r32 | 8 | 7 | 18 | 33 | me≥64 |
| r64 | 1 | 6 | 26 | 33 | me≥64 |

### AVX-512

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 0 | 32 | 0 | 32 | me≥24 |
| r5 | 0 | 24 | 0 | 24 | me≥40 |
| r6 | 1 | 22 | 1 | 24 | me≥8 |
| r7 | 1 | 20 | 3 | 24 | me≥56 |
| r10 | 0 | 21 | 3 | 24 | me≥8 |
| r11 | 0 | 16 | 8 | 24 | me≥8 |
| r12 | 1 | 12 | 11 | 24 | me≥8 |
| r13 | 0 | 18 | 6 | 24 | me≥8 |
| r16 | 0 | 0 | 33 | 33 | — |
| r17 | 0 | 19 | 5 | 24 | me≥8 |
| r19 | 1 | 15 | 8 | 24 | me≥8 |
| r20 | 0 | 20 | 4 | 24 | me≥8 |
| r25 | 0 | 15 | 9 | 24 | me≥8 |

## Last run per radix

| Radix | Last measurement written |
|---|---|
| r3 | 2026-04-25 01:13 |
| r4 | 2026-04-25 23:03 |
| r5 | 2026-04-25 01:14 |
| r6 | 2026-04-25 01:13 |
| r7 | 2026-04-25 01:13 |
| r8 | 2026-04-25 23:05 |
| r10 | 2026-04-25 01:13 |
| r11 | 2026-04-25 01:13 |
| r12 | 2026-04-25 01:13 |
| r13 | 2026-04-25 01:13 |
| r16 | 2026-04-25 01:13 |
| r17 | 2026-04-25 01:13 |
| r19 | 2026-04-25 01:14 |
| r20 | 2026-04-25 01:13 |
| r25 | 2026-04-25 01:13 |
| r32 | 2026-04-25 23:06 |
| r64 | 2026-04-25 23:07 |

---

Per-radix reports: `generated/rN/vfft_rN_report.md`. Raw measurements: `bench_out/rN/measurements.jsonl`. Orchestrator logs: `bench_out/rN/orchestrator.log`.
