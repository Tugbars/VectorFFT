# Z STAGED CASCADE — execution plan (checklist item 16; the high-N answer)

**Status**: designed 2026-07-24, ready to build; MKL high-N census **confirmed the architecture**
(2026-07-24 — see §0.5). This is the handoff doc — everything needed to execute is here or pointed
to. Companion docs: [il_native_design.md](il_native_design.md) (the z family design + checklist),
[../research/high_n_loss_analysis.md](../research/high_n_loss_analysis.md) (root cause),
[../research/mkl_il_512_anatomy.md](../research/mkl_il_512_anatomy.md) (mid-N MKL census),
[../research/mkl_highN_cascade_anatomy.md](../research/mkl_highN_cascade_anatomy.md)
(**≥2048 MKL census — this plan's architectural confirmation**).

## 0. Why (one paragraph)

The z family wins ≤128 outright and holds ~0.82× MKL-IL through 1024, then collapses
(0.54 @2048, 0.46 @4096). Root cause (measured, not argued): the flat two-pass four-step
FORCES ~N-wide strides in both passes — our 4096 pass 1 *alone* (4148 ns) costs as much as
MKL's entire transform (4087 ns). MKL switches implementation families between 1024 and
2048: a multi-stage contiguous-streaming cascade (ping-pong scratch, ~5 passes, small looped
stage kernels, user memory touched once each way — gdb census in
`../research/mkl_il_512_disasm/mkl4096_*.log/asm`). That cascade is structurally OUR STRIDE
ENGINE (the scrambled in-place K≥8 executor) — we simply never gave the z family its
stride-engine half. Three banked negatives (blocked-v1 12190 ns, n1t corner-turn, single-level
r64 blocking) all reduce to one principle: **every pass must stream contiguously; the
exchange amortizes across log-many stages; store locality decides everything.**

## 0.5. MKL high-N census CONFIRMS this architecture (2026-07-24)

Reverse-engineered MKL's own ≥2048 path (gdb+objdump at N=4096/8192/16384, 4-lens cross-checked —
[../research/mkl_highN_cascade_anatomy.md](../research/mkl_highN_cascade_anatomy.md)). MKL runs
**exactly the shape this plan proposes**, which retires the design risk:

- **One parameterized stage-kernel suite reused across all N** (pass-1 kernel VMA 0x1825c28ce
  identical at 4096/8192/16384) → validates "one z kernel per radix, stride/table-parameterized."
- **OOP transpose ingest (plane A→B, corner-turn) → in-place contiguous middle passes → final
  unload** → this IS our MODEB trick, verbatim, in MKL's binary. User memory read-once/written-twice;
  every log-N intermediate pass stays in scratch (two ping-pong planes, linear in N).
- **Mixed-radix schedule chosen per factorization** (radix-4 for 2¹²/2¹⁴; a radix-8 body absorbs the
  odd power at 2¹³) → validates `chain-per-cell = calibrator axis` (§4).

Two refinements this census forces into the build:

1. **Race radix-4 stages too, not just radix-8/16.** MKL's default *deep* ymm stage is **radix-4**
   (VL=2, FMA, in-place, contiguous), with radix-8 mixed in for odd powers. Our construction law
   favored radix-8; add radix-4 chains to the step-2 race (e.g. 4096 = 4·4·4·4·4·4, 8·8·radix-... ).
2. **We can go WIDE everywhere — our structural edge.** MKL is forced to a **128-bit** ingest
   because it *gathers* (index table `movsxd r11,[rbp+r11*2]`; scattered access won't vectorize
   wide) and only the contiguous cascade goes **256-bit ymm**. Our SCR terminator + stage-0 use
   **sequential streams** (measured 0.40× vs scatter 0.10×), so our z ingest stays **ymm VL=2** —
   a width advantage over MKL's narrow gather ingest. (Answers the earlier open question "does MKL
   use a precomputed offset table?" — yes, but for a narrow direct-DFT we already out-measure with
   streams; keep the sequential SCR terminator.)

## 1. The architecture — three shipped mechanisms, re-expressed in z

1. **Stage structure = the stride engine** (`stride_plan_t` shape, planner.h): factor chain
   [f0..f_{nf-1}], one pass per factor; per stage: radix-f kernel, groups, stage stride,
   per-stage twiddles. In z: a stage = a loop over group bases calling the EXISTING gated
   z kernels — `z_n1` (stage 0, twiddle-free) / `z_t2` (later stages, streamed VTW2) —
   with per-stage `(Ls = stage stride, count = group width)`. Consecutive butterflies are
   memory-adjacent within a group = the kernels' count axis = wide contiguous loads. NO new
   kernels needed for v1.
2. **OOP = the MODEB trick** (oop_execute.h): stage 0 reads user z, writes scratch/dst;
   stages 1.. run in-place on it (z kernels take separate zin/zout and are in-place-safe:
   all legs loaded before any store). Ping-pong A/B planes optional; in-place stages match
   the split engine and stream identically.
3. **Natural order = the SCR terminator** (natorder_scatter.h — shipped, won the K=4 band
   +16-27%): fuse the digit-reversal into the LAST stage. Plan-time GROUP-granularity
   tables (natural-q-ordered group bases — the src[P]/twg[P] pattern), terminator kernel =
   pre-twiddled radix-R DFT whose leg-j output lands at natural row q + j·P — i.e. R
   SEQUENTIAL STREAMS (measured 0.40× vs scatter's 0.10× on the split side). The z-t2
   kernel's store form (`leg l → l·OLs + column`, OLs = P) IS this pattern already.
   NOT MKL's per-element gather tables (we have none in any codelet, and sequential streams
   measured better anyway).

## 2. Composition spec (v1, N = 4096 first)

Chain: 4096 = 8·8·8·8 (or 16·16·16 — race both; construction law says favor radix-8 stages).
Scratch: one N-complex z plane (+ the dst); MODEB-style stage-0 redirect then in-place.

- **Stage 0**: `radix8_z_n1` per group; DIT stage-0 groups: legs at Ls = N/8 (512 complex).
  ⚠ 8 legs at 8 KB stride = 8 parallel sequential streams — fine (few streams, prefetchable);
  the 64-stream flat leaf was the killer, not stream count ≤ 8–16.
- **Stages 1..nf-2**: `radix8_z_t2` per group with per-stage VTW2 streams (small: stage s
  needs (R-1)·group-width records ORDERED per group — total per stage ≤ N/..; KBs each, not
  126 KB) filled at plan time in consumption order.
- **Terminator (stage nf-1)**: z-SCR — per natural-group table walk, `z_t2` (pre-twiddle =
  the last stage's twiddles, VTW2) with `OLs = P` natural-comb stores (R sequential streams).
- **Scrambled-z bonus route**: skip the terminator table walk (plain last stage) → scrambled
  output for `VFFT_ORDER_SCRAMBLED` K=1 callers — currently UNSERVED by the K=1 front door
  (it refuses scrambled and falls to the classic path). Free win, zero extra work.

Twiddle math per stage (DIT, natural input): stage s twiddle for leg l, butterfly b within
its group = W_{N_s}^{l·b} with the stage's sub-transform length — mirror
`vfft_proto_compute_twiddles_dit` (twiddle.h) for the values; layout = VTW2 cos-first
(il_native_design.md §1.3), one cursor per stage.

## 3. Build steps (numbered, each gated)

1. **Driver** `zil_cascade.c` (bench-level first, like every spike): plan-time = chain,
   per-stage group tables, per-stage VTW2 fills, terminator natural-q table; execute =
   stage-0 redirect + in-place stages + terminator. Gate vs naive-4096 (~1e-15 band).
2. **Race @4096**: cascade (both chains) vs flat 64×64 (8849 ns baseline) vs MKL-IL live
   (4087). Also the scrambled-z variant timed (its ceiling).
3. **Extend to 2048/8192/16384** (2048 = 8·16·16 etc.; 8192/16384 currently unserved or
   0.45×) — same driver, chain per cell. Gate + race each.
4. **Crossover calibration**: two-pass vs cascade per cell (expect ≈1024/2048 boundary,
   like MKL); enters the same per-cell wisdom discipline as everything else.
5. **If cascade wins**: productionize (plan struct + create/execute in the zil runtime
   layer), then registry/calibrator wiring per checklist items 9/11.
6. **Only if measurements demand**: emitter upgrades (per-stage specialized strides via
   JIT-style baking; blocked stage kernels; the 6b compact table knob per stage).

## 3.5. The in-place-machinery reuse seam — PORT, not drop-in (verified 2026-07-24)

Verified against the code (stride_executor.h + a z codelet) whether "just apply our in-place
machinery to IL" means reusing the stride engine directly. **Verdict: reuse the architecture, but
it is a z-twin executor port, NOT a function-pointer swap** — the stride engine is split-native and
batched, the z codelets are interleaved single-transform. Three seams block a literal reuse:

1. **Codelet ABI.** `stride_n1_fn(in_re, in_im, out_re, out_im, is, os, vl)` /
   `stride_t1_fn(rio_re, rio_im, W_re, W_im, ios, me)` (stride_executor.h:254-263, **two split
   planes**) vs z `(zin, zin_unused, zout, zout_unused, tw_re, tw_im, Ls, Gs, OLs, OGs, count)`
   (**one interleaved buffer**, `zin + 2*(l*Ls+k)`).
2. **Data addressing / layout.** The engine walks a **[N×K] batched SPLIT** buffer (`base_re =
   re + group_base[g]`, `base_im = im + group_base[g]`; leg stride = remaining-radixes·K —
   stride_executor.h:396-402). The z cascade walks **one interleaved buffer**, count axis =
   intra-transform columns (K=1). Different `group_base` arithmetic (complex units), no 2nd plane.
3. **Twiddle format.** Engine = split `(W_re,W_im)` Method-C tables + separate `cf0` on leg 0
   (stride_executor.h:434-436). z = fused **VTW2 cos-first BYTW2 stream** — cf0 folds into the
   stream at plan time, so the executor's whole scalar cf0/tw_scalar preprocessing block DISAPPEARS.

**What carries over as-is (the ~80%)** vs **what needs a z adaptation:**

| reusable template | z adaptation |
|---|---|
| `stride_plan_t` (factors, stages, group_base, needs_tw) | codelet-call sites → z ABI |
| stage-sweep loop `for s: for g: n1/t1` | re/im planes → one interleaved buffer |
| `plan_compute_groups` (dim_stride, num_groups=N/R) | group_base in complex units |
| ping-pong in-place discipline; K-split/group-par threading; DIT/DIF; bwd-via-conj | split (W_re,W_im) → VTW2 stream builder; drop split cf0/tw_scalar preprocessing |

The z codelets were **deliberately given the split-shaped 6-pointer ABI** (2nd/4th vestigial) and are
**in-place-safe** (all legs loaded before any store) — built to be wireable into staged machinery.
So step-1's driver is: **a ~200-line z-twin executor** (stage loop, interleaved addressing, z-ABI
calls) **+ a per-stage VTW2 builder** (mirror of the Method-C twiddle computation, emitting cos-first
streams). Do NOT reach for `stride_executor.h`'s function pointers expecting them to fit — mirror the
structure, not the split-plane call sites. (Residual: the VTW2-builder port scope wants a read of the
stride engine's `plan_compute_twiddles` back half, lines ~1303-1942, not yet done.)

## 4. Open details (resolve during build, all small)

- Exact DIT group/stride bookkeeping for z at K=1 — mirror `vfft_proto_compute_groups`
  (the split engine's) with complex units; verify with the naive gate per stage.
- Terminator pre-twiddle = last stage's combined twiddle incl. cf0 leg-0 factor (see
  natorder_scatter.h's terminator notes; twiddle.h:114-118 on the split side).
- Stage-0's 8 KB-stride streams: if measurably hot, swap stage order (DIF-style outer
  small-radix first — `t2d` exists) — a race variant, not a redesign.
- Chain choice per cell = calibrator axis (like ccol's cc_chain).

## 5. Current standings this plan attacks (interim ladder, band-corrected)

64: 1.02 WIN · 128: 1.02 WIN · 256–1024: ~0.81–0.83 · 2048: 0.54 · 4096: 0.46 · 8192+:
unserved/0.45 (split ccol only). Target: cascade brings ≥2048 into the ~0.8+ band and
extends the z family to 8192/16384.
