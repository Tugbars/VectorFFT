# VTune: R32 t1 AVX2 codelet, L1-resident (gcc-15) — moves are free, codelet at the floor

**Date:** 2026-06-15 · i9-14900KF (Raptorlake-DT), gcc-15.2 -O3 -march=native, P-core pinned.
**Harness:** `src/dag-fft-compiler/benchmarks/{vtune_codelet.c, run_vtune.ps1}` — single-codelet,
L1-resident (me=8/ios=8 → 256 doubles), fixed-wall-time so the codelet is ~100% of the process.
HW event-based sampling needs an **Administrator** shell (the VTune sampling driver).

## Question
The in-place codelets emit ~87 reg→reg `vmovapd` per call — **none are in our C** (pure SSA
intrinsics), all gcc-inserted. Were they recoverable headroom (coalescing / alignment), or free?

## Top-down (uarch-exploration)
- **Retiring 68.2%** · **CPI 0.242** (~4.1 IPC) · 5.70 GHz · FP Vector 50.8% of uOps (256-bit).
- **Back-End Bound 28.5%:**
  - **Core Bound 13.9%** → Port Utilization 14.3%. Busy ports = **Port 0 (60%) + Port 1 (60%)** =
    the FMAs (real work). Port 6 8.9%. **Port 5 absent. Shuffles_256b 0.0%.** 3+ ports active in
    80.5% of cycles (OOO already extracting good ILP).
  - **Memory Bound 14.6%:** Store Bound 10.1% (> L1 Bound 4.1%); L1-latency-dependency is 55.6%
    **of** the 4.1% L1 bucket ≈ ~2.3% of total clockticks.

## Findings — every codelet-micro lever is a no-op on gcc-15/AVX2
1. **reg→reg `vmovapd` are move-eliminated in rename** (Port 5 absent, 0 execution-port cost).
   They are Source A (destructive-FMA operand-form), and **register-count-invariant** — R32 = 87
   on AVX2 *and* AVX-512 — so structural (DAG-driven), not a coalescing failure.
2. **Neck barrier** (`asm volatile("" ::: "memory")` at the pass-1→pass-2 seam): **inert** across
   R32/R64 × AVX2/AVX-512 × gcc-13/15 (bit-identical or within noise). The blocked two-pass
   construction (doc 58) already prevents the pass-merge spill blowup; the historical "280-spill"
   figure is **not reproducible** on the shipping codelets with either compiler (different
   toolchain/era or a pre-blocking codelet).
3. **Aligned scratch** (`_mm256_store_pd`/`load_pd` vs `storeu`/`loadu` on the 32-byte-aligned
   `spill_re/spill_im`): **byte-identical asm** (0 differing instruction lines). gcc emits
   `vmovupd` either way (== `vmovapd` on an aligned address, zero penalty). The Store-Bound /
   split-store metric is **not** the scratch.
4. **3-way independent liveness quant** of the 87 copies: ~17/72 are "avoidable in principle,"
   but all reduce to gcc **FMA-form selection** (132/213/231) — unreachable via C intrinsics, and
   move-eliminated anyway.

## Conclusion
**R32 t1 on gcc-15/AVX2 is at the move/codegen floor.** Do not micro-optimize moves, the
neck barrier, or scratch alignment. The residual ~14% memory cost is the **design-forced L1
seam** (64-value round-trip — forced by 16 YMM at radix-32, and the right trade vs extra
full-array passes), mostly OOO-hidden. The only structural lever that attacks the seam is
**AVX-512 (32 ZMM halves it)** — off-limits on the 14900KF; revisit on the EPYC/SPR port.

Method trail: neck-barrier A/B (`c:/tmp/neck_test`), 3-way liveness quant, this profile.
