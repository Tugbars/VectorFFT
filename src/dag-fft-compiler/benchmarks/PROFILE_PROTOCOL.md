# Profiling protocol (mandatory, learned from the slice-3b thread-second bug)

Two rules, enforced like the byte-gate:

1. PROFILE SINGLE-THREADED. Phase accumulators summed from worker functions
   measure THREAD-seconds when T>1 (each thread adds to shared globals).
   Sigma(phases) = T x wall-time. Force stride_set_num_threads(1) around any
   phase-profiled run, OR use per-thread accumulators. (The VFFT_R2C_PROFILE
   accumulators in core/r2c.h are non-atomic file statics — single-thread only.)

2. NO PHASE TABLE WITHOUT ITS CONSERVATION CHECK. Assert
   Sigma(phases) / wall in [0.9, 1.1]. A profiler that fails conservation is
   measuring itself (instrument overhead, thread-second summation, or races).
   Abort/flag the run otherwise. This check, run on day one, catches the
   thread-second bug before any verdict is built on inflated shares.

ZERO-INSTRUMENT CROSS-CHECK (the ground truth): ablation. Stub the phase
(e.g. -DVFFT_R2C_STUB_POST early-returns from _r2c_postprocess), clean build,
T=1; the full-vs-stubbed delta is the phase's true cost with no timers anywhere.
Slice 3b result: postprocess true cost ~32.6us (~35%) at N=256 K=256 T=1 —
confirmed real after the thread-second timer had inflated it to ~70%.
