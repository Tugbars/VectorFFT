# 61 — VFFT vs MKL r2c-256: kernel op-count comparison (disassembly)

Follows doc 60 (rfft path beats MKL on r2c-256 by ~1.2-1.4x). That result is a TIMING
ratio. This doc asks the mechanistic question behind it: how do the two engines' kernels
compare in instruction count? Op count is one of the few things this container measures
FAITHFULLY — instruction counts are deterministic, not subject to the thermal/JIT noise
that makes absolute timings only directional here. So this is a high-confidence measurement.

Method:
- VFFT: count vector intrinsics directly in the emitted codelet C (exact, we own the source).
- MKL: closed-source + JIT, BUT the r2c-256 kernel it dispatches turned out to be a
  PRECOMPILED function in libmkl_avx512.so.3 (not JIT'd into anon mmap). Found via
  MKL_VERBOSE (which names the descriptor) + gdb PC-sampling inside one DftiComputeForward
  to land on the actual function, then objdump of that function. Callgrind does NOT work
  here (valgrind's VEX cannot decode the EVEX/AVX-512 byte stream — SIGILL), which is why
  the disassembly route is the one that yields MKL numbers.


## 1. What MKL actually runs for r2c-256

MKL_VERBOSE reports the descriptor as `drfo256*256` (double real-FFT-out, N=256, batch 256,
output_distance 129 = N/2+1 — interleaved CCE). gdb PC-sampling inside the compute lands in
TWO precompiled kernels in libmkl_avx512.so.3:

| MKL kernel | role | radix |
|---|---|---|
| mkl_dft_avx512_mg_rowbatch_twidl_fwd_008_d | twiddled row-batch stage | 8 |
| mkl_dft_avx512_mg_colbatch_plain_fwd_16_d  | plain (no-twiddle) col-batch stage | 16 |

So MKL factors the N/2=128 complex core as an (8,16)-class plan — strikingly close to VFFT's
own (16,8)/(8,16) choices (doc 59 §6). Both engines independently land on radix-8 and
radix-16 stages for this size. The win is not a smarter factorization; it is the per-kernel
efficiency and the fused real handling.


## 2. Per-kernel-body vector-op counts (exact, from disassembly)

This is the density comparison: how many vector instructions one execution of the kernel
body contains. All AVX-512 (zmm). VFFT counts are vector intrinsics in the emitted C; MKL
counts are zmm-touching instructions in the disassembled function body.

| kernel | radix | body vec ops | fma | load/store/bcast |
|---|---|---|---|---|
| VFFT radix32_r2cf (leaf)   | 32 | 250 | 83 | 64 |
| VFFT radix8_hc2hc (stage)  | 8  | 144 | 22 | 64 |
| MKL rowbatch_twidl_008     | 8  | 339 | 54 | 121 |
| MKL colbatch_plain_16      | 16 | 264 | 52 | 98 |

(VFFT radix32_r2cf: 103 arith + 83 fma + 64 load/store = 250. radix8_hc2hc: 58 + 22 + 64 =
144. MKL bodies counted as zmm-touching instructions incl. addressing.)


## 3. What the op count shows (high confidence — container measures this faithfully)

1. VFFT codelets are DENSER in useful butterfly work per instruction.
   - VFFT does a full RADIX-32 leaf in 250 vec ops.
   - MKL's RADIX-16 body is 264 vec ops — more instructions for HALF the radix.
   - VFFT packs roughly 2x the radix per instruction at the leaf. This is the genfft-style
     DAG scheduler + the fused real leaf (r2cf) paying off: bigger radix, fewer instructions.

2. MKL spends a larger fraction on addressing / twiddle streaming.
   - MKL rowbatch radix-8: 339 vec ops, 121 (~36%) are load/store/broadcast.
   - VFFT radix8_hc2hc: 144 vec ops, 64 (~44% of a much smaller total) are load/store.
   - In absolute terms MKL moves far more per body (121 vs 64 mem ops) — consistent with
     doc 59/60's "MKL streams more twiddles/planes per stage" diagnosis, and with why log3
     (which cuts VFFT's twiddle loads 7→3 slots) is the right lever.

3. The fused real handling shows up structurally.
   - VFFT's last stage IS the hc2c/hc2hc codelet (twiddle + butterfly + Hermitian fold in
     one body, no separate terminator — the FFTW design from doc 59 §8). MKL likewise fuses
     (it has no separate fold kernel in the sampled set). Both avoid a standalone fold pass;
     the difference is VFFT's bodies are leaner per radix.

Mechanistically, leaner kernels (more radix/instruction, less addressing) is a plausible
and consistent cause of the ~1.2-1.4x timing win in doc 60. The op count and the timing
agree in direction.


## 4. What this does NOT establish (honest limits)

- NOT a per-transform total. Both engines' kernels loop internally (MKL rowbatch over the
  K columns, colbatch over groups; VFFT codelets over vl in vec-width steps). A full
  "ops per r2c-256 transform" number needs the executor TRIP COUNTS on both sides. VFFT's
  are exactly extractable from the rfft stage loop; MKL's are only APPROXIMATE here
  (PC-sampling gave ~34 rowbatch : ~20 colbatch calls — a ratio, not a verified count).
  So §2/§3 are per-body density, which is the trustworthy comparison. The per-transform
  total is deferred (see §5).
- Op count is not cycle count. Instruction count ignores ILP, port pressure, and latency
  chains. Denser-per-instruction does not automatically mean faster wall-clock; it is
  consistent with the measured win but not a proof of it. (The container cannot measure
  cycles — no PMU.)
- MKL's body counts include instructions the AVX-512 disassembly attributes to addressing
  that a cycle-accurate view might fuse or hide. Treat the MKL bodies as upper-ish bounds
  on useful-work density, not exact FLOP counts.


## 5. Net + next step

Net: at the kernel-body level, VFFT's emitted codelets are LEANER per unit of radix work
than MKL's r2c kernels — more radix per instruction at the leaf (radix-32 in 250 ops vs
MKL radix-16 in 264), and less absolute addressing/streaming overhead. This is the
mechanistic story consistent with the doc-60 timing win, and it is measured on the one axis
this container is faithful on (deterministic instruction count, not noisy timing).

NEXT STEP (deferred, agreed): turn per-body density into a HARD per-transform total. Extract
VFFT's exact executor trip counts from the rfft stage loop, and tighten MKL's call counts by
instrumenting them under gdb (breakpoint-count, not PC-sample). Then the comparison becomes
a single verified "vector ops per r2c-256" number per engine rather than per-body density.
