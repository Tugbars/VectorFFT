# core/ — C executive + planning layer

This is VectorFFT's existing C planning/execution layer, included for
completeness. **It was not modified in the session that produced this bundle.**
That session studied `stride_executor.h` to diagnose why the flat executor
trails FFTW, and built the recursive blocked engine in `../engine/` as the
intended replacement for its composition approach. These files are the prior
state on disk (prototype-core dated 2026-05-19; `stride_executor.h` is the
2026-05-30 uploaded version, which carries earlier executor work).

## Contents
- `stride_executor.h`  — the full 1881-line stride-based in-place executor
  ("Method C": single buffer, multi-pass, DIT+DIF roundtrip, baked twiddles).
  This is the flat executor the recursive engine replaces. Its own header notes
  the R=64 t1_dit regression at K>=256 from strided access pressure — the exact
  memory-orchestration problem the blocked engine addresses.
- `executor.h`         — 1D C2C dispatch entry (`vfft_proto_execute_fwd`):
  specialized fast path from `plan_executors.h` else `executor_generic.h`.
- `executor_generic.h` — the generic plan-shape execution loop.
- `dp_planner.h`, `planner.h`, `plan.h` — DP / greedy planners + plan struct.
- `estimate_plan.h`, `exhaustive_plan.h`, `exhaustive_patient.h`,
  `exhaustive_screened.h` — the cost-model / search planners (FFTW-style
  ESTIMATE vs MEASURE/PATIENT analogues).
- `twiddle.h`          — twiddle table construction.
- `r2c.h`, `rader.h`   — real-to-complex and prime-size (Rader) paths.
- `threads.h`, `wisdom_reader.h`, `compat.h` — threading, wisdom, portability.
- `demo/`              — drivers exercising each planner/executor path
  (dp_planner, estimate, exhaustive, patient, roundtrip, n1024_k128,
  tier1_vs_mkl, ...), with their build scripts.

## Relationship to the new engine
The flat executor makes one full memory pass per stage over an element-major
layout `re[leg*K + k]`, scattering each radix leg by subFFT*K across the whole
working set. FFTW's cache-oblivious recursion (and the blocked recursive engine
in `../engine/`) touch the data far fewer times and far more locally. That
composition difference, not codelet quality, is the whole performance story.
