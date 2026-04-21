# VectorFFT portfolio tuning summary

Generated: 2026-04-21 09:53:45
Host: Linux x86_64 (pinned to CPU 2)

## Portfolio

- **Radixes tuned** (with measurement data on disk): 13
- **Total measurements** (fwd + bwd, across all radixes): 3002

## Dispatcher winners by radix (fwd direction)

For each (radix, ISA), count of sweep points each protocol wins. "Total" is the number of (me, ios) grid points where at least one protocol was measured.

### AVX2

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 5 | 24 | 3 | 32 | me≥24 |
| r4 | 5 | 6 | 7 | 18 | me≥64 |
| r5 | 0 | 24 | 0 | 24 | me≥40 |
| r7 | 1 | 23 | 0 | 24 | me≥56 |
| r8 | 0 | 3 | 15 | 18 | me≥64 |
| r10 | 0 | 23 | 1 | 24 | me≥8 |
| r11 | 1 | 19 | 4 | 24 | me≥8 |
| r12 | 2 | 17 | 5 | 24 | me≥8 |
| r16 | 7 | 2 | 9 | 18 | me≥64 |
| r20 | 0 | 24 | 0 | 24 | me≥8 |
| r25 | 5 | 18 | 1 | 24 | me≥8 |
| r32 | 0 | 6 | 12 | 18 | me≥64 |
| r64 | 5 | 4 | 9 | 18 | me≥64 |

### AVX-512

| Radix | flat | t1s | log3 | Total | Transition me |
|---|---|---|---|---|---|
| r3 | 0 | 32 | 0 | 32 | me≥24 |
| r4 | 6 | 6 | 6 | 18 | me≥64 |
| r5 | 0 | 24 | 0 | 24 | me≥40 |
| r7 | 1 | 20 | 3 | 24 | me≥56 |
| r8 | 2 | 2 | 14 | 18 | me≥64 |
| r10 | 0 | 21 | 3 | 24 | me≥8 |
| r11 | 0 | 16 | 8 | 24 | me≥8 |
| r12 | 1 | 12 | 11 | 24 | me≥8 |
| r16 | 2 | 1 | 15 | 18 | me≥128 |
| r20 | 0 | 20 | 4 | 24 | me≥8 |
| r25 | 0 | 15 | 9 | 24 | me≥8 |
| r32 | 0 | 3 | 15 | 18 | me≥64 |
| r64 | 1 | 5 | 12 | 18 | me≥64 |

## Last run per radix

| Radix | Last measurement written |
|---|---|
| r3 | 2026-04-20 23:29 |
| r4 | 2026-04-21 09:34 |
| r5 | 2026-04-20 12:10 |
| r7 | 2026-04-20 11:56 |
| r8 | 2026-04-19 21:43 |
| r10 | 2026-04-20 20:48 |
| r11 | 2026-04-20 21:27 |
| r12 | 2026-04-20 21:05 |
| r16 | 2026-04-19 22:13 |
| r20 | 2026-04-20 15:10 |
| r25 | 2026-04-20 20:13 |
| r32 | 2026-04-19 22:28 |
| r64 | 2026-04-19 22:59 |

---

Per-radix reports: `generated/rN/vfft_rN_report.md`. Raw measurements: `bench_out/rN/measurements.jsonl`. Orchestrator logs: `bench_out/rN/orchestrator.log`.
