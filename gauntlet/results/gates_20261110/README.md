# The gates after the day's changes (2026-11-10 evening)

Branch head b0f52cf9, i9-14900KF (AVX2). The gates that cover what the day changed: FP64 power-of-two 1D C2C and the
FP32 library. Each gate rebuilt first and run on its own scratch store.

| gate | covers | result | file |
| --- | --- | --- | --- |
| `k1_fourstep_gate` | the FP64 four-step at 2^19, 2^20, 2^21: natural and scrambled order, out of place and in place, 8 threads, replay | ALL PASS | `k1_fourstep_gate.md` |
| `k1_pow2_gate` | the FP64 K=1 interleaved tier at 128..4096 through the front door | ALL PASS | `k1_pow2_gate.md` |
| `ztt_gate` | ZTURN-T at FP64: every registry cell, and the front door at every power of two of its band | ALL PASS | `ztt_gate.md` |
| `vfftf_gate --max 16777216` | the FP32 library at 2..2^24: one transform and a batch of 3, 1 and 4 threads, out of place and in place, forward and backward, the two one-buffer and two-buffer laws | ALL PASS, 96 served cells, 5 refusals | `vfftf_gate.md` |

- The FP32 gate's largest error: 2.27e-07 forward (N = 2^24), 2.25e-07 backward (N = 2^21); limit 1e-05. Its default
  stops at 2^20; `--max 16777216` is what reaches the FP32 four-step (2^21..2^24).
- `ztt_gate` took 872 s alone, against the runner's 900 s budget for it.
- Six more gates passed on the same head in a full-suite run that was stopped after them: `api_matrix_gate` (88 cells),
  `blocked_tail_gate`, `caller_pin_gate`, `dfti_shim_gate`, `estimate_gate`, `fftw_bind_gate`.
- `il_solo_gate` and `vfft_tcbatch_gate` passed earlier the same day (`../gap_fixes_20261110/`); their code did not
  change after that.

No other gate was run. The four-step also serves large FP64 lengths that are not powers of two, large real transforms
and the inner transform of large primes: no gate of this set covers those.
