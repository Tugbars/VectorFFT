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

- **One parameterized cascade reused across all N** (pass-1 kernel VMA 0x1825c38ce identical at
  4096/8192/16384 — corrected +0x1000 per the anatomy doc's §0.5 erratum; it is ONE fused ymm
  function, fn 0x3800, end-to-end) → validates "one z kernel per radix, stride/table-parameterized."
- **OOP transpose ingest (plane A→B, corner-turn) → in-place contiguous middle passes → final
  unload** → this IS our MODEB trick, verbatim, in MKL's binary. User memory read-once/written-twice;
  every log-N intermediate pass stays in scratch (two ping-pong planes, linear in N).
- **Mixed-radix schedule chosen per factorization** (radix-4 for 2¹²/2¹⁴; a radix-8 body absorbs the
  odd power at 2¹³) → validates `chain-per-cell = calibrator axis` (§4).

Two refinements this census forces into the build:

1. **Race radix-4 stages too, not just radix-8/16.** MKL's default *deep* ymm stage is **radix-4**
   (VL=2, FMA, in-place, contiguous), with radix-8 mixed in for odd powers. Our construction law
   favored radix-8; add radix-4 chains to the step-2 race (e.g. 4096 = 4·4·4·4·4·4, 8·8·radix-... ).
2. **Width is PARITY, not an edge (corrected 2026-07-24 — anatomy §0.5 erratum).** Live
   re-verification proved MKL's 2^k ingest is **ymm-wide radix-4 with strided leg loads** (same
   shape as our stage-0 leaf, radix-4 vs our radix-8) — the earlier "MKL is stuck at a 128-bit
   gather ingest" claim came from a +0x1000 VMA mapping error and is retracted; the offset-table
   answer is "not on the 2^k path." What MKL's finisher actually does: size-specialized bodies
   (dispatch on remaining {4,8,16}) writing user memory through a **register corner-turn store
   lattice** (vunpck+vperm2f128) — no per-column VTW2 gather pass. Our sequential-stream SCR
   advice stands on our OWN measurement (0.40× vs 0.10×), not on an MKL contrast.

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

## 4.9. FIRST SPIKE BUILT + MEASURED (2026-07-24) — thesis validated

`build_tuned/benches/zil_cascade.c` (spike, per §3 step 1). N=4096, radix-8 (8^4),
both arms **bit-exact vs naive (3.195e-15)**. Index math derived + runnable-gated by a
3-way independent derivation workflow (scalar protos gated <1e-15 at 64/512/4096 before
wiring). Stable measured (pinned P-core, best-of-9, cachebust; MKL the stable reference):

| arm | ns | vs MKL | structure |
|---|--:|--:|---|
| flat 64×64 two-pass | 8600–10700 | 0.38–0.48× | four-step radix-64, whole-buffer strided |
| cascade A (natural) | ~8110 | 0.50× | recursive four-step ×4 radix-8, **scatter** middle (OLs=512) + 1 corner-turn (folded into a t2s strided read) |
| **cascade B (in-place)** | **~7210** | **0.56×** | grid-preserving DIT: middle stages **in-place small-stride** (Ls=OLs=64 then 8) = **L1-hot blocks**; t2s gather terminator; digit-reversed out |
| MKL-IL | ~4080 | 1.0 | in-place contiguous cascade |

**Thesis validated in-session**: in-place contiguous stages (B) > scatter four-step (A) >
flat. Localization is the lever — B's L1-hot in-place blocks are the MKL-shaped win; A's
OLs=512 scatter strides the whole 64 KB like the flat two-pass and gains little. Both cascades
compose the EXISTING gated z kernels (radix8 n1/t2/t2s) through their stride params — **zero new
codelets**, exactly as the plan predicted. Arm B maps to the census's OOP-ingest→in-place-middle
shape.

**Remaining gap to MKL (0.56→1.0), asm-verified 2026-07-24** (our codelet .o disasm + MKL live
x/i — anatomy §0.5 erratum applied):
- **S0 leaf is NOT the differentiator** (earlier claim corrected): our S0 = 8 ymm streams at
  16·Ls B apart (verified: 8 scaled `vmovupd ymm` loads/stores per iter); MKL's ingest is the
  SAME wide-strided shape (ymm radix-4, legs at N/4·{0..3}, live-verified at 0x38c5).
- **S3 terminator IS a real differentiator** (verified): our t2s = xmm+vinsertf128 pair-loads +
  `add rax,0x1c0` twiddle cursor = **448 B/col-pair → 112 KiB** VTW2 stream (flat pass-2:
  126 KiB — "as heavy as" confirmed). MKL instead finishes with **size-specialized bodies**
  (dispatcher `cmp r11,{4,8,16}` at 0x4ef8) writing user memory through a **register
  corner-turn store lattice** (vunpck+vperm2f128, site 0x5f05) — no 100 KiB-class per-column
  twiddle gather pass.
- **Execution shape**: MKL's whole cascade is ONE fused function (no per-group call loop);
  ours is 74 codelet calls.
Next levers (measure each): a corner-turn-lattice terminator (n1t-style stores instead of the
t2s gather + heavy VTW2); radix mixes (radix-4 deep stages / 16^3 fewer passes, §0.5 refinement
1); tighter S1/S2 col-const twiddle (re-reads identical records — cache-hot but wasteful);
fewer/fused calls. Extend to 2048/8192/16384 (§3 step 3) once the 4096 structure is tuned.

## 4.95. CHAIN PLANNER RUN (2026-07-24) — exhaustive per-cell search, steps 3+4 delivered

User directive: "we can't just arbitrarily choose the stages." Correct — and measured. Built
`build_tuned/benches/zil_chain_dp.c`: generalizes the arm-B executor to ANY factor chain
(mixed radices; grid-preserving DIT; TRUE in-place middles on one plane — itself worth ~2% over
ping-pong; t2s gather terminator; mixed-radix digit-reversed out, gate permutation
`m = drev(g·Rt+l)`), enumerates ALL chains (S0 n1∈{4..64}, mids t2∈{8..64}, term t2s∈{8..64};
nf=2 = the flat two-pass, subsumed), gates each vs naive, races all + MKL per cell. At these Ns
the space is tiny (13/18/24/34 chains at 2048/4096/8192/16384) so **exhaustive measurement IS the
planner** (per the cost-model-ceiling lesson); DP/beam only if the space ever explodes.
**All 89 chains gate <1e-12 across the 4 cells.**

| N | winner | ns | vs MKL | prior z standing | flat rank in field |
|---|---|--:|--:|--:|---|
| 2048  | 32.8.8   | 2966  | **0.73×** | 0.54× | 32.64 #6 (0.65×) — crossover cell |
| 4096  | 4.8.16.8 | 6762  | **0.64×** | 0.46× | 64.64 #10 (0.47×) |
| 8192  | 8.8.16.8 | 15268 | **0.62×** | unserved | — |
| 16384 | 4.8.64.8 | 37175 | **0.56×** | unserved | — |

Findings: (1) **every winner terminates radix-8** (t2s gather cheapest at 8 contiguous legs);
(2) interiors differ per cell — hand-picking fails: my 8.8.8.8 ranked #3 @4096, and the plan's
own 16.16.16 candidate ranked **#14, below flat** — the planner was not optional; (3) winners
lead with radix-4/small leaves where available (matches MKL's radix-4 census bias — and we only
have radix-4 as n1: **emit radix4_z_t2/t2s** is now a measured-priority follow-up); (4) crossover
flat↔cascade sits at ~2048 exactly where 2N·16 crosses 48 KB L1 — matching MKL's own family
boundary. Remaining gap (0.73→0.56 decaying with N) = the §4.9 verified differentiators
(terminator VTW2-stream weight; fused-function vs 74-call execution) — next levers unchanged.
Production path: per-cell wisdom stores the winning chain (cc_chain-style token), then bake/JIT
the winner to kill the call tax (MKL's fused function is AOT-baked — census-verified, not JIT).

## 4.96. LEVERS 1+2 MEASURED (2026-07-24) — t2c + t2sp kernels; high-N collapse CLOSED

Emitted two z-kernel kinds (codelet_zil.ml, flags `--z-t2c`/`--z-t2sp`): **t2c** = t2 with
group-constant VTW2 record set (frozen cursor, L1-hot — the z-analog of split t1s);
**t2sp** = t2s terminator streaming ONE w¹ record/col-pair, legs 2..R-1 built in-register
(VTW2 sign-folded form is closed under elementwise cmul: c'=c·c1−s·s1, s'=s·c1+c·s1 — 4 ops/leg).
Raced as variant axes in zil_chain_dp.c: **294/294 chain×variant gates pass**.

| N | base→both (ns) | lever gain | vs MKL (was at session start) | winner |
|---|--:|--:|--:|---|
| 2048  | 2889→2761   | −4.4%  | **0.73×** (0.54) | 16.16.8 t2c/t2sp |
| 4096  | 6494→6342   | −2.3%  | **0.67×** (0.46) | 4.8.16.8 t2c/t2sp |
| 8192  | 15255→13933 | −8.7%  | **0.67×** (—)    | 4.16.16.8 t2c/t2sp |
| 16384 | 36716→28773 | **−21.6%** | **0.70×** (—) | 4.8.8.8.8 t2c/t2sp |

Findings: (1) the twiddle-bandwidth lever is **N-dependent** — negligible at 4096 (tables
L2-resident, prefetch hides the stream; my 10%+ byte-count projection was WRONG there),
decisive at 16384 (FLAT tables 1.4 MiB ≈ L2 capacity; t2c alone −18.8%); (2) **t2c unlocks
deep chains** — the 16384 winner became the 5-stage 4.8.8.8.8 (previously mid-field) because
group-constant twiddles remove the deep-chain table penalty; 3 of 4 cells changed winning
chains → chain and kernel-variant must be calibrated JOINTLY, per cell; (3) **the high-N
collapse is closed**: z family now 0.67–0.73× MKL across ALL ≥2048, no decay with N.
Remaining gap ≈ 1.4×: per-stage kernel quality (MKL's lean radix-4 ymm bodies — emit
radix4 z_t2/t2c mids to test MKL-aligned all-radix-4 chains), lever 3 (n1t-load terminator),
lever 4 (AOT bake via emit_executor_h.ml pattern / JIT — kills the per-group call tax).

## 4.97. RADIX-4 FIELD + LEVERS 3/4 MEASURED (2026-07-24) — 16384 at 0.78×

**Radix-4 family** (emitted radix4_z_{t2,t2c,t2s,t2sp}; planner minpart=2, LEAN mode, 333 gates
PASS): competitive — radix-4-led chains fill the top ranks (2048: 4.4.16.8; 8192: 4.4.4.16.8) —
but NO step change; winners shuffle within the ±2-5% run band. MKL's radix-4 preference does not
transfer to our kernel family. **Chain space is saturated** (175 chains @16384 bunch within ~15%).

**Lever 3 — tiled terminator loads** (t2st/t2spt: wide ymm + vperm2f128 repack, the MKL-finisher
load shape; radix {4,8,16}; planner NV=8, 526 gates PASS): real but marginal — t2spt wins 2048
(+0.7%) and 16384 (+1.4%), ties 4096. The t2s→t2sp powers-stream remains the terminator lever;
load shape is a trim. (Noted: 8192's t2c arms swung ~20% between runs — allocation-layout
sensitivity, not a kernel property; cell best stable ~13.9µs.)

**Lever 4 — fusion/bake** (`zil_cascade_baked.c`: baked = codelets #include-renamed into one TU,
gcc-inlined into constant-trip loops = the MKL AOT shape; three arms isolate the components; all
gates bit-identical):

| arm | 4096 (8.8.8.8 t2c/t2sp) | 16384 (4.8.8.8.8 t2c/t2spt) |
|---|--:|--:|
| drv (base_of hot) | 6818 (0.63×) | 30157 (0.70×) |
| drvT (tabled bases) | 6439 (**0.67×**) | 27887 (0.76×) |
| baked (fused+inline) | 6455 (0.66×) | **27210 (0.78×)** |

Attribution: the bulk is **precomputed group-base tables** (−5.6%/−7.5% — free at plan time in
production); true inlining adds ~2.4pt only at high call counts (293 calls @16384; nothing at 74).
Production: plan-time base tables mandatory; the emit_executor_h.ml-pattern z backend (or JIT)
worth it for ≥16384-class cells.

**Standing after levers 1–4**: 2048 ~0.76×, 4096 ~0.67×, 8192 ~0.66×, 16384 **0.78×** (session
start: 0.54/0.46/—/—). Remaining gap ≈1.3–1.5× is now **kernel-body quality** (per-pass cost
0.176 ns/el MKL vs ~0.30 ours: scheduling/port balance of the z bodies, MKL's size-specialized
finishers) — codelet-level work, not composition. Next: productionize (wisdom chains + plan-time
tables + registry/calibrator wiring, checklist items 9/11) and/or M1-style body-quality race.

## 4.98. KERNEL CENSUS + LEVER 5 DEFINED (2026-07-24) — the block-split interior

Instruction-class census of our compiled loop bodies vs MKL fn 0x3800's loops (full table +
evidence: [../research/mkl_highN_cascade_anatomy.md](../research/mkl_highN_cascade_anatomy.md)
§4.5). Verdict on "are our codelets worse": **NO on spills (0 both sides), NO on op count
(whole-transform ≈88K insns BOTH at 4096)** — the 1.5× gap is IPC/port mix. Root cause
discovered: **MKL's high-N cascade runs a BLOCK-SPLIT scratch interior** (re-block+im-block;
z only at the API boundary; conversion fused into ingest stores + finisher store lattice).
Their mids: ZERO shuffles/xors (elementwise split cmul, splat-re/im twiddle pairs). Our z mids:
17 port-5 ops per 16 complex. Mid-N stays truly interleaved (2 passes can't amortize the
conversion); high-N converts (log-many passes can) — the layout switches WITH the family.

**Lever 5 — block-split interior (approved direction pending spike):** keep the z boundary
(user directive intact — no user-visible split), run the cascade scratch block-split:
- S0 leaf: z loads → radix butterfly → **split-converting corner-turn stores** (we own this
  machinery: n1t store lattice + Design-C repack patterns);
- mids: new **split-mid kind** (shuffle-free elementwise cmul, splat-pair twiddles — the split
  engine's arithmetic in cascade clothes; group-constant tables like t2c);
- terminator: gather + last twiddle + **split→z re-interleave fused in the store lattice**
  (the t2s* family grows the conversion role — completing its MKL-finisher shape).
Expected win bound: eliminate ~17 port-5 ops/16cx in every mid pass + IPC headroom; race vs
the t2c/t2sp champion per cell, same gates.

## 4.99. ⭐ LEVER 5 MEASURED (2026-07-24) — BLOCK-SPLIT INTERIOR IS THE BREAKTHROUGH: 16384 at 0.85×

`build_tuned/benches/zil_split_interior.c` — hand-kernel spike of the §4.98 design (S0
z→split converting leaf; shuffle-free split mids, splat-pair group-const twiddles, in-place;
last mid re-interleaves split→z in its stores; EXISTING z t2sp terminator unchanged). The z
API boundary is fully preserved — split exists only inside the scratch. All gates bit-exact.

**Layout granularity was decisive, exactly as MKL chose it**: the first cut used FULL split
planes (re[], im[] + anti-alias pad) — won at 4096 (−3%) but LOST at 16384 (+7.7%: two
streams per leg row double the stream/TLB pressure). Retrofit to **BLOCK-split** (64 B
`[re×4][im×4]` blocks — same bytes as 4 z-complex, addressing = z + `+4` for im, ONE stream
per leg row) swung 16384 by +29%:

| cell | split-interior | z-interior ctl (same chain) | lever | prior best (any arm) |
|---|--:|--:|--:|--:|
| 4096 (8.8.8.8) | **5609 = 0.73×** | 6157 = 0.66× | **−8.9%** | 6342 (0.67×) |
| 16384 (4.8.8.8.8) | **23086 = 0.85×** | 26410 = 0.74× | **−12.6%** | 27210 baked (0.78×) |

Even all-radix-4 split (4⁶ = 5894 @4096) beats every z-interior arm. The census's port-5
prediction held: mids run ZERO shuffles/xors; conversions live only in S0 (paid once) and the
last mid's stores. Spike arms are UN-baked (tabled bases, extern calls) — the §4.97 bake lever
stacks on top.

**Follow-ups to productionize lever 5**: run 2048/8192 + full chain×interior joint calibration
in the planner; t2spt tiled terminator for 16384 (spike used t2sp); bake the split winners;
promote hand kernels to emitter kinds (codelet_zil.ml split-mid backend — the split bodies are
plane-pair macros, mechanical); natural-order terminator; bwd twins. The hand kernels live in
the spike only — emitter promotion is the production path.

## 4.995. JOINT chain×interior PLANNER RE-RUN (2026-07-24) — split wins every cell; chains shifted again

Ported the §4.99 split-interior kernels into zil_chain_dp.c as variants v8 (SPL/t2sp) and v9
(SPL/t2spt); split arms eligible when nf≥3, R0∈{4,8}, mids∈{4,8}. Both fields run per cell
(minpart=3 all-10-variant attribution + minpart=2 LEAN {t2c/t2sp, split} big field).
**1256/1256 gates PASS.** Per-cell champions:

| cell | winner | ns | vs MKL | prior |
|---|---|--:|--:|--:|
| 2048  | **4.8.8.8 SPL/t2spt** | 2609 | **0.83×** | 0.76× (16.16.8 t2c — chain CHANGED) |
| 4096  | 4.4.4.8.8 ≈ 8.8.8.8 SPL | 5923 | 0.73× | = spike |
| 8192  | **4.4.8.8.8 SPL/t2sp** | 12477 | **0.75×** | 0.65× (4.16.16.8 t2c — chain CHANGED, cell unlocked) |
| 16384 | 4.8.8.8.8 SPL | 25821 | 0.80× (spike focused: 0.85×) | = spike chain |

Findings: (1) **split-interior wins EVERY cell** — 16384 attribution: SPL 25.8µs vs best z-arm
29.1µs (−11%); (2) **eligibility reshapes the chain space**: split mids exist only for r4/r8,
so the planner abandoned r16/r32 interiors — 8192's 0.65→0.75 jump came from the minpart=2
field unlocking 4.4.8.8.8; chain × interior must be searched JOINTLY (third time the axiom
holds); (3) **long-sweep drift confirmed** (~5-11% at 16384 vs the focused spike) — exactly
the thermal-ranking-drift the split DP planner's pacing exists for; wisdom-grade numbers need
paced, focused finals (see z_chain_planner_notes.md — the dp_planner.h study, same date:
adopt pacing + adaptive-reps + PATIENT re-measure of finalists into zil_chain_dp).
**Band after lever 5, best-observed: 0.73–0.85× across all ≥2048** (from 0.38–0.56 at session
start). Wisdom chains to bake/productionize = the 4 winners above.

## 4.996. PACED FINALS + THE BAKE VERDICT (2026-07-24) — bake the TABLES, share the KERNELS

`zil_split_baked.c`: the four §4.995 winners, three execution shapes each (all gates
bit-identical), paced trials (Sleep 150ms — the dp_planner lesson), same-run MKL:

| cell | drv (runtime bases) | **drvT (tabled bases, called kernels)** | baked (fused/inlined code) | drvT vs MKL |
|---|--:|--:|--:|--:|
| 2048 (4.8.8.8/t2spt)   | 2508  | **2456**  | 2898  | **0.83×** |
| 4096 (4.4.4.8.8/t2spt) | 5712  | **5337**  | 6171  | **0.74×** |
| 8192 (4.4.8.8.8/t2sp)  | 12022 | **11177** | 13162 | **0.78×** |
| 16384 (4.8.8.8.8/t2spt)| 24552 | **23375** | 26642 | **0.81×** |

**Verdict — the production execution shape is settled**: plan-time base TABLES (+2–8%) with
COMPACT SHARED kernels; full code-fusion is a measured NEGATIVE on the split interior (−7 to
−13%: the big split bodies inlined into constant-trip loops get unrolled into tens of KB and
blow L1i, while shared kernels stay resident). This is MKL's own architecture — the census's
"small looped stage functions driven from an outer driver" — now measured from our side. (The
z-interior lever-4 result decomposes consistently: its win was ~¾ tables, inlining only helped
at 293 tiny-body calls.) JIT/AOT emitters should therefore emit PLANS (tables + chain dispatch),
not fused kernel blobs, for this family.

**WISDOM-GRADE STANDINGS (paced, same-run MKL): 0.74–0.83× across all ≥2048** — session start
was 0.38–0.56 with 8192/16384 unserved. Remaining gap = kernel scheduling residue + MKL's
size-specialized finishers; next = emitter promotion of the split kernels + productionization
(registry/calibrator/wisdom, items 9/11) at these locked shapes.

## 4.997. EMITTER PROMOTION DONE (2026-07-24, end of day) — split family is generator-owned

`codelet_zil.ml` gained `emit_z_split` (template promotion of the gated hand kernels — a
plane-pair IR backend is not warranted for 2 radices): kinds **s0s** (z→split leaf,
deinterleaving loads), **ms** (split mid, in-place, shuffle-free, splat-pair group-const
twiddles), **msz** (split→z last mid, re-interleaving stores); flags `--z-s0s/--z-ms/--z-msz`;
standard 11-arg z ABI (registry-uniform; kernels use Ls + count). Emitted
`radix{4,8}_z_{s0s,ms,msz}_avx2.c` → codelets/zil/avx2 (lib now 675 objects).
**PROMOTION BIT-GATE PASS**: finals bench drv/drvT arms re-pointed at the EMITTED kernels —
all 12 gates identical relerr to the last digit vs the hand-kernel run; performance within
noise. Nuance banked honestly: drv vs drvT near-tied on split arms (drvT clearly +5% only at
4096; the big split bodies amortize base_of) — tables are free at plan time, keep them;
fused-code remains consistently worst (−8..−13% both runs).
Per-cell wisdom recorded: `build_tuned/k1_wis/zsplit_wisdom.txt` (4 locked shapes +
methodology). **QUEUED NEXT SESSION**: front-door productionization (registry resolvers +
calibrator + wisdom loader + vfft.c route — items 9/11) at these locked shapes; then the
gap-closing campaign (terminator weight, kernel scheduling residue, MKL-style size-specialized
finishers).

## 4.998. TERMINATOR-WEIGHT ATTACK (2026-07-24 night) — sterm wins: band 0.76–0.89×

Two fixes raced (`zil_term_opt.c`, paced, all gates PASS at every cell, both runs):

1. **Tree-powers** (emitted kinds t2sq/t2sqt: squaring tree w²=w₁², w₄=w₂², … — critical
   path 6→3 links): **marginal** (+0–3%, even −2% once at 2048). The OOO engine was already
   hiding most of the sequential chain. Kinds kept in the arsenal; wisdom decides.
2. **sterm — the split-input terminator** (hand-derived, then PROMOTED to
   `emit_z_split ~kind:"sterm"`, flag `--z-sterm`, emitted radix8_z_sterm; promotion bit-gate
   PASS with relerr identical to the last digit): reads the block-split plane directly (ALL
   mids stay plain `ms` — the msz pass is GONE), 4 columns/iteration via 4×4 register
   transposes on load, shuffle-free split butterfly + twiddles, **packed per-column w¹ table
   (16 B/col — half of w¹-VTW2)**, tree powers, re-interleave fused in the stores.
   **WINNER: +4–8% at every cell, both runs.**

| cell | champ (t2sp/t2spt) | **sterm** | vs MKL |
|---|--:|--:|--:|
| 2048  | 2603  | **2440**  | **0.89×** |
| 4096  | 5456  | **5205**  | **0.76×** |
| 8192  | 11478 | **10826** | **0.81×** |
| 16384 | 24762 | **23028** | **0.82×** |

Attribution lesson (banked): I ranked the latency-chain fix first and the structural rework
second — measurement inverted it. The terminator's weight was structural (z-form shuffles,
2-col width, stream bytes, the msz chore-pass), not the dependency chain.
**Band after the attack: 0.76–0.89× (wisdom-grade, paced, same-run MKL)**; wisdom file
updated (all-ms mids + sterm, packed twq table). Execution shape now: s0s → ms× → sterm —
three kernel kinds total, all generator-owned, all shuffle-work confined to the two
boundary-touching passes. Remaining vs MKL ≈ scheduling residue + their size-specialized
finisher bodies — VTune attribution is the queued next probe.

## 4.999. PRODUCTIONIZED (2026-07-25) — the cascade is live behind vfft.h

**bwd twins first** (the scrambled contract requires a matched-permutation roundtrip; split
routes get bwd free via the pointer-swap identity, z-interleaved does not): emitted kinds
`s0sb`/`msb`/`stermb` (flags `--z-s0sb/--z-msb/--z-stermb`) — INV butterflies derived by the
conj rule (every CROSS-PLANE term flips sign; verified line-by-line), twiddle conjugation is
TABLE-side (kernels unchanged), geometries mirror (stermb's comb loads are cheaper than fwd's
transposes). **Roundtrip gates 1.1–1.4e-15 at all 4 cells, first build.** bwd timing:
0.72–0.80× MKL-bwd; **2048 bwd = 2376 ns = 1.15× — an outright WIN over MKL.**

**Production wiring**: `src/core/oop/zsplit.h` (plan struct, create builds every table incl.
conj twins + plan-time base tables per the §4.996 verdict, execute_fwd/_bwd, destroy,
`vfft_zsplit_default_chain` = the calibrated winners — the ccol default-chain precedent);
`vfft.c`: create-time attach for K=1 OOP SCRAMBLED at covered N (classic oplan still built —
split-contract + uncovered N lose nothing), execute-time interleaved z→z dispatch ahead of
k1/classic, destroy. **Public-API gate `zsplit_api_gate.c` (--vfft --mkl): OVERALL PASS** —
drev-compare (doubles as ROUTING proof: the classic scramble would fail it) + roundtrip
through vfft.h at all 4 cells; front-door bench 0.79–0.95× both directions (same-process;
wisdom-grade numbers remain the §4.996/§4.998 paced finals).

**Remaining for full items 9/11**: calibrator emitting zsplit wisdom lines (defaults are the
compiled-in winners today; cc_chain-codec-compatible), in-place placement route, natural-order
terminator route, non-pow2/uncovered-N coverage, MT. The K=1 SCRAMBLED OOP interleaved
contract at 2048–16384 is DONE: served, gated, bidirectional, generator-owned end to end.

## 4.9991. VTUNE ATTRIBUTION (2026-07-25) — the cascade is CORE-bound; mids are near-ceiling; terminator + leaf carry the stalls

Harness: `dev/bench_vtune/vtune_zsplit.c` (production zsplit.h execute in a pinned loop; SW
hotspots unelevated + uarch-exploration from the user's admin shell). N=16384 fwd, 10 s,
core 2, MUX 0.995, 5.7 GHz.

**Process top-down**: Retiring 54.5% · **Back-End 43.5% (Core Bound 33.2%, Memory Bound only
10.3%)** · Front-End 2.2% (DSB 84.7% — the compact-shared-kernel shape vindicated) · Bad-spec
~0. Ports: **Port 1 = 66.3% vs Port 0 = 44.6%** (add/sub-heavy split bodies pile on the
FADD-capable port), 3+-ports-utilized 64.7%, shuffles only 5.5% of slots. **The twiddle/
bandwidth story is CLOSED** (L2-bound 4.5%, DRAM ~0) — levers 1–5 worked; what remains is
scheduling/ports/latency, the confirmed "kernel scheduling residue."

**Per-pass** (clockticks · instructions · CPI · retiring):

| pass | ticks/pass | insns/pass | CPI | retiring |
|---|--:|--:|--:|--:|
| r8_ms (×3) | 8.1e9 | 34.1e9 | **0.238 (4.2 IPC)** | **71%** — near the machine ceiling |
| sterm (×1) | 17.3e9 | 46.5e9 | 0.371 (2.7 IPC) | 44% |
| r4_s0s (×1) | 8.3e9 | 20.8e9 | 0.40 | — |

The terminator's 2.1× cost over a mid = 1.37× instructions × 1.56× worse CPI; the leaf's CPI
0.40 = the 64 KB-stride z loads' L2 latency + DEINT. **Mids are effectively done.**

**Next levers, in profile order**: (1) terminator scheduling — software-pipeline 2
column-quads/iter (hide TR4→bfly→REINT chains; store-latency submetric 30.6%); (2) leaf
latency — prefetch/wider blocks against the strided z loads; (3) port-1 rebalance experiment
(convert some adds to FMA-×1.0, shifting port 1/5 → port 0; raced emitter variant); (4) the
elephant for another day: 3.8% core utilization = single-thread by contract — K=1 MT
(group-parallel stages) is the largest untapped multiplier.

### §4.9991 addendum — the 4096 uarch profile (the weakest cell, explained)

| metric | 4096 | 16384 |
|---|--:|--:|
| CPI | **0.336** | 0.307 |
| Retiring | 49.5% | 54.5% |
| Core Bound | **38.0%** | 33.2% |
| Memory Bound | 9.7% | 10.3% |
| Port 1 / Port 0 | 62.8 / 42.8 | 66.3 / 44.6 |
| Mispredict resteers | **7.0% of clockticks** | ~0 |
| DSB misses | **13.1%** | 6.9% |

4096 — our worst vs-MKL cell — is the MOST core-bound and the least efficient (CPI 0.336),
with two 4096-specific signals: **7% mispredict-resteer clockticks and 2× the DSB misses**.
Cause: the 4.4.4.8.8 chain's deep stages run MANY SHORT LOOPS — the D=8 last mid is 64
separate calls of a 2-iteration loop; trip-count-2 loops mispredict their exits and churn the
DSB. 16384 amortizes the same shape over more work. Memory stays ~10% at both cells —
scheduling is the whole story.

**NEW TOP LEVER — "group-looped stage kernels" (msg)**: move the per-stage GROUP loop INSIDE
the kernel (one call per stage, internal loop over `ngroups` with constant group stride +
per-group table advance). This is MKL's exact shape (census: their stage functions are called
once per stage and loop internally); it kills per-group call overhead AND turns 64–256
short-loop exits into one long predictable loop — directly targeting the 7% resteers and the
DSB churn. Compact body preserved (no fusion bloat — the §4.996 lesson respected). Composes
with sterm software-pipelining (the other confirmed stall mass).

## 4.9992. ⭐ MSG LEVER LANDED + CONFIRMED (2026-07-24) — resteers 7.0% → 0.3%; the front end is CLOSED

**What shipped**: `msg`/`msgb` kinds in `codelet_zil.ml` — the per-stage GROUP loop moved
INSIDE the kernel. One call per stage; an `always_inline` static `_zsg{r}{f|b}_body` holds the
compact split body (§4.996 fusion lesson respected), and the exported wrapper loops groups:
`_body(...); bp += 2*R*Ls; twg += (R-1)*8;`. This hinges on a structural fact worth keeping:
**group bases are contiguous** — grid-preserving stages tile the plane, `base(g) = g·R·D` —
so the wrapper is a span bump, no index table. Emitted `radix{4,8}_z_msg_{fwd,bwd}_avx2.c`;
`src/core/oop/zsplit.h` execute_fwd/bwd swapped to ONE msg call per stage (5 calls total at
16384, down from ~74 in the spike era).

**Clean A/B** (interleaved vs frozen pre-msg build `vtune_zsplit_nog.exe`, msg won every round):

| N | pre-msg best | msg best | Δ | vs MKL |
|---|--:|--:|--:|--:|
| 4096 | 5350 ns | **5094 ns** | **−4.8%** | ≈0.78× |
| 16384 | 23472 ns | **21963 ns** | **−6.4%** | ≈0.86× (best ever) |

**uarch CONFIRMATION** (admin uarch-exploration, `vtu4096msg`, N=4096 — the cell with the
pathology). The §4.9991-addendum prediction verified line by line:

| metric | pre-msg | msg |
|---|--:|--:|
| Mispredict resteers | **7.0% of clockticks** | **0.3%** |
| Branch resteers total | — | 0.7% |
| Bad speculation | — | 0.5% of slots |
| Front-End Bound | — | **2.1% of slots** (DSB coverage 85.4%) |
| DSB misses | 13.1% | 10.9% (residual costs only 3.5% FE-bandwidth slots) |

The trip-count-2 loop-exit thesis was exactly right: one long internal loop per stage and the
predictor locks on. Note the profile *shifted*, not just shrank — CPI ticked 0.336→0.347 while
wall time fell, because msg **retires fewer instructions per transform** (call/prologue/
table-setup gone) and the freed slots now expose back-end memory latency: Memory Bound
9.7%→15.9% (FB Full 12.9% of clockticks, Store Latency the biggest sub-bucket). That is the
next lever's territory, not a regression: with the front end closed, 4096's remaining stalls
are **fill-buffer/store scheduling** (sterm software-pipelining, prefetch) and the unchanged
**port skew** (Port 1 62.3% vs Port 0 43.1% — the FMA-ify race is still live).

**Public-API gate w/ msg live (front door, `benches/zsplit_api_gate.exe`): OVERALL PASS** —
all 8 gates (drev routing proof + roundtrip) at 1e-15, and the best front-door band to date:

| N | fwd ns | vsMKL | bwd ns | vsMKL-bwd |
|---|--:|--:|--:|--:|
| 2048 | 2404.8 | 0.95 | 2248.9 | 0.93 |
| 4096 | 5064.5 | 0.78 | 5142.8 | 0.77 |
| 8192 | 10212.9 | **0.86** | 10231.4 | 0.86 |
| 16384 | 21920.1 | **0.87** | 22268.9 | 0.85 |

(8192 jumped 0.81→0.86 and 16384 0.82→0.87 vs the pre-msg wisdom finals; 2048 fwd ratio
run-varies 0.88–0.95 — the dedicated bwd bench had shown 1.15× at 2048. Run note: the gate
exe needs BOTH `C:\mingw152\mingw64\bin` (libwinpthread) and the MKL `bin` dir on PATH, else
it dies 0xC0000135 before main.)

## 4.9993. STERM SOFTWARE-PIPELINING CAMPAIGN (2026-07-24) — uj2 wins the kernel, placement luck eats the cascade; resolved as a MEASURED per-cell pick

**Setup**: VTune (§4.9991) named sterm the stall mass (CPI 0.371, 44% retiring, store
latency + FB-full). Five scheduling-only variants were built and adversarially verified
(bit-identity mandatory, several verifiers compiled + empirically bit-checked), then raced
inside the real cascade in `benches/zil_sterm_pipe.c` with two controls: `emit` (linked
emitted kernel) and `copy` (same source pasted in the bench TU — the code-placement
yardstick). All arms gated bit-identical at all 4 cells before timing.

**Arm verdicts** (vs the copy control):

- **uj2 — 2-quad unroll-and-jam: the kernel-level WINNER.** Two independent 4-column
  bodies per trip (A/B), phases [loads A+B (16 store lines in flight)] [TR4 A] [TR4 B]
  [twiddle trees alternated op-by-op] [BFLY A, stores A] [BFLY B, stores B]. In the
  first build: kernel-only −6..−18%, full-cascade −2..−5%, consistent across 2 process
  runs. The register-wall bet pays *at the kernel level*: doubled store-stream MLP + two
  independent squaring-tree chains outweigh the spills.
- **nt — non-temporal full-line stores: REFUTED** (+3..+24% kernel). Killing the RFO is
  not worth evicting the output to DRAM at these sizes.
- **rot — cross-iteration twiddle rotation: REFUTED** (+2..+8% kernel at most cells).
  The rotated c1/c2/c4 pairs deepen the live set exactly where spills already bite.
- **phase / pfw** (live-range phasing, PREFETCHW on the 8 output streams): noise-level.
- **bwd twin (sterm_bwd_uj2): cleanly REFUTED** (+29..+36% kernel-only at every cell —
  far beyond noise). bwd's 8 distant streams are LOAD streams the hardware prefetchers
  already handle; doubled MLP buys nothing, doubled spill pressure is pure cost.
  **bwd keeps the single-quad schedule.**

**The placement-luck discovery (the real finding).** Rebuilding the bench TU with the bwd
arms added reshuffled every function's address — and the fwd verdict FLIPPED at 2048/4096
(uj2 from −5% to +4..+7% vs copy, in BOTH its TU and linked placements), while a later lib
relayout flipped 2048 back. Code placement (DSB packing / JCC-erratum alignment) moves
these kernels by ±5% — the same order as the schedule delta. The first build's clean win
and the second's low-N loss are both "true" for their binaries. Corollary banked: at
effect sizes ≤5%, a single-binary A/B is evidence about THAT binary, not about the code.

**Resolution — both schedules are generator-owned, the pick is MEASURED per cell:**

- `codelet_zil.ml`: kind `sterm` = original single-quad schedule (§4.998, restored);
  kind `sterm2` = the uj2 schedule (`--z-sterm2`; emitted `radix8_z_sterm2_fwd_avx2`,
  fwd-only). Both heads/bodies are raw-string templates; emitted sterm2 body is
  byte-identical to the raced arm.
- `zsplit.h`: plan field `t2q` selects the fwd terminator
  (`t2q ? sterm2 : sterm`); create() sets a per-cell default from the production-path
  pick race (`benches/zil_sterm_pick.c`, 3 passes, isolated cells): **4096 → sterm2**
  (2/3 picks, largest deltas −2.3/−6.8%), 2048/8192 → sterm, 16384 coin-flip → sterm.
  TODO(calibrator): promote t2q to a measured zsplit-wisdom field per install.

**API gate after the campaign: OVERALL PASS** (drev + roundtrip 1e-15 at all 4 cells),
front door 2048 0.87/0.91 · 4096 0.78/0.73 · 8192 0.81/0.77 · 16384 0.87/0.85 (fwd/bwd
vs MKL) — within the msg-era band. Net: correctness-guaranteed dual-schedule machinery +
measured picks; the honest headline is that sterm scheduling alone cannot beat placement
noise at cascade granularity. The remaining profile-ordered levers (port-1→0 FMA-ify —
which changes the OP MIX, not just the schedule — and MT) are the ones with headroom
beyond the noise floor.

## 4.9994. CALIBRATE→WISDOM→CREATE WIRED (2026-07-24) — the t2q pick is now measured per install

The §4.9993 resolution is complete: the terminator pick no longer lives in a compiled
table — it is measured on the installed binary at first create and banked as wisdom.

**Persistence — kind-4 line in `oop_wisdom.txt`** (the K=1 tier's existing home, next to
the kind-3 engine lines; per the reuse-wisdom-files directive, no new file):

    N 1 4 zs_t2q cc_chain ns        e.g.  4096 1 4 1 22233 5005.5

`zs_t2q` = 0 sterm / 1 sterm2; `cc_chain` = the cascade chain via the kind-3 codec
(decimal digits = log2 factors: 22233 = 4.4.4.8.8) so the line also carries the chain
for future chain calibration. Isolation from the split-layout solvers that share the
table is by kind: `lookup_ord` skips kind-4 (as it skips kind-3), `lookup_k1` requires
kind-3, the K%8 guard blocks classic builds, and `_oop_wisdom_put_and_save` got a proper
4-class dedup (MODEB / native / K1 / zsplit) so banking a cascade line can never evict
the kind-3 or MODEB champion at the same (N,1) cell. Old binaries reading a new file
degrade safely (kind-4 falls through every build branch → NULL → fallback).

**The race — `_calibrate_zsplit_t2q` (vfft.c, next to `_calibrate_pad`)**: ~10 ms,
`_il_ab_race` shape — one bit-identity memcmp sanity check, burst size from one estimated
exec (~0.3 ms/burst), 9 rounds (MEASURE; 21 at higher rigor) with alternating arm order,
median per arm, **3% hysteresis toward the compiled default**. Runs inside the create
hook on wisdom miss or `recalibrate=1`; the verdict + chain + ns are banked immediately
via `_oop_wisdom_put_and_save`. Wisdom hit → pure read (chain decoded from cc_chain,
falls back to the default chain if a stale line no longer validates), zero measurement.

**Gates** (`benches/zsplit_wis_gate.c`, scratch `VFFT_WISDOM_DIR`, all 4 cells):
OVERALL PASS — roundtrip 1e-16 through vfft.h on the miss path; kind-4 line banked with
correct codec; wisdom-hit create bit-identical to the miss create; `recalibrate=1`
re-races and re-banks. API gate re-run: OVERALL PASS, best front-door band to date —
2048 0.92 fwd/**1.01 bwd (API-level MKL win)** · 4096 0.81/0.85 · 8192 0.86/0.85 ·
16384 0.93/0.92.

**The finding, live**: the wis-gate binary banked sterm2 at all 4 cells; the api-gate
binary (same lib, different link) banked 0/1/1/0 — per-binary placement truth, measured
per binary. This closes §4.9993's TODO; remaining zsplit roadmap: chain calibration into
the same kind-4 line, in-place route, natural-order terminator, MT.

## 4.9995. ⭐ WHERE THE LAST ~15% IS (2026-07-24) — the terminator is SHUFFLE-PORT bound, and every excess shuffle is the load-side transpose

Re-read of [mkl_highN_cascade_anatomy.md](../research/mkl_highN_cascade_anatomy.md) against our
own post-msg profile, plus two new measurements. Headline: **the remaining gap is not memory,
not op count, and not arithmetic — it is port 5.**

### (a) Shuffle census, ours vs MKL (per COMPLEX, normalized across radices)

Counted by reading the emitted kernels (macro-expanded: `TR4` = 8 port-5 ops, `DEINT`/`REINT` =
4 each; loop bodies × trip count) against anatomy §4.5's MKL census:

| pass | ours | MKL | time share (16384) |
|---|--:|--:|--:|
| leaf (r4 s0s) | 1.00 | 1.25 (r4 ingest) | 17% |
| mids (r8 msg) | **0** | **0** | 49% |
| terminator (r8 sterm) | **2.00** | **1.00** (r4 finisher) | 35% |

The mids at 0 is lever 5 banked — that fight is **won and at parity**. The leaf is *better* than
MKL's ingest. **The entire remaining shuffle excess is one pass and one cause**: our terminator
issues **64 port-5 ops per iteration** = 4×`TR4` on load (32) + 8×`REINT` on store (32), per 32
complex. Decompose it and the diagnosis is exact:

- **store-side re-interleave: ours 1.00/complex, MKL 1.00/complex — DEAD EQUAL.**
- **load-side transpose: ours 1.00/complex, MKL 0.** ← 100% of the excess.

MKL's finisher needs no load-side transpose because **its preceding stage hands the finisher data
already oriented** (anatomy §4: "corner-turn baked into the stores"). Ours does not, so the
terminator transposes on the way in and re-interleaves on the way out.

### (b) The port arithmetic (why this costs so much)

Shuffles (`vperm2f128`, `vpermute4x64`, `vunpck*`) issue on **port 5 only**; FP add/sub take
ports 1/5, mul/FMA take ports 0/1. Per terminator iteration: 80 FP ops (44 add/sub, 14 mul, 22
FMA) balance to ≈27 cycles across three ports, but the 64 shuffles need **64 cycles on port 5
alone** → the terminator is **≈2.4× oversubscribed on one port**. That is the whole story of its
CPI 0.371 / 44% retiring vs the mids' 0.238 / 71%.

**Independent cross-check (it holds).** Predicted whole-transform shuffle rate at 4096 (chain
4.4.4.8.8): leaf 1.0×4096 + mids 0 + term 2.0×4096 = 12 288 shuffles / ~29 000 cycles =
**0.42 shuffle-uops/cycle**; VTune measured `Shuffles_256b` 5.8% of pipeline slots ≈ **0.35/cycle**.
Within the terminator alone that is ~81% of port-5 capacity — saturated.

### (c) Stream-aliasing diagnostic — real but smaller than it first looked

`benches/zil_stream_diag.c` / `zil_stream_diag2.c`. Motivation: leg streams sit `16*Ls` bytes
apart with `Ls` a power of two, so the stride is always a multiple of 4096 = (64 L1 sets × 64 B)
— **every leg stream shares one L1D set**, and VTune shows FB-Full 12.9% + Store-Latency 35.1%.

⚠ **v1 over-claimed (banked as a measurement lesson).** v1 reported the leaf −25% at 16384, but
it allocated a *fresh buffer per arm*, so it varied stride AND base together — the same
placement confound as §4.9993. v2 varies them independently inside ONE 4 KB-aligned allocation:

| N | base shift (64 B…2 KB) | stride pad (+4c) | stride pad (+32c) |
|---|--:|--:|--:|
| 2048 | +0.3 … +11.1% (noise, both signs) | −5.8% | **−18.9%** |
| 4096 | −3.4 … +1.1% | −1.2% | −4.5% |
| 8192 | −8.2 … +0.8% | −9.0% | **−11.3%** |
| 16384 | −0.7 … +6.3% | −5.3% | −7.6% |

**Verdict: base decorrelation does nothing; stride decorrelation is real** (−4.5…−18.9% on the
leaf, −0…−6% on the terminator). v1's 16384 leaf "contract" number was 4805 ns vs v2's 3496 ns —
**27% from allocation alignment alone**, a fresh reminder that at these effect sizes the arm
must vary exactly one thing inside one allocation.

### (d) The levers left, ranked

1. **Move the terminator's load-side transpose into the last mid's stores.** Deletes 32 of its
   64 shuffles — the entire measured excess over MKL. The mid pays them, but the mid is
   port-balanced with headroom while the terminator is 2.4× over. Port arithmetic: mid ≈ +26%,
   terminator ≈ −44% → **net −5…−11% overall**. Best-evidenced lever; MKL's finisher is the
   existence proof. (Cost: the last mid becomes orientation-aware — a new emitter kind.)
2. **Padded plane pitch** — measured in (c): **~2–4% overall**. Needs a slice-aware group loop in
   `msg` (nested: bump by group span within a slice, by span+pad across slices) plus slice-aware
   leaf/terminator addressing. Contained, no new math.
3. **Radix-16 terminator** — MKL dispatches finishers on `cmp r11,{4,8,16}`, so it has one.
   Absorbs a whole mid pass (16% of time at 16384). Blocked today by registers: a radix-16 split
   butterfly wants 32 live ymm on a 16-register machine (the emitter's own radix cap). High
   risk / high reward.
4. **⭐ The finisher census (do this FIRST).** Anatomy §8 flagged characterizing MKL's three
   finishers (0x5c2e / 0x5831 / 0x5579) as "the now-sharpest remaining-gap comparison" and it was
   never done. It is objdump-only, no code churn, and it **decides both #1 and #3**: (i) do the
   r8/r16 finishers also have zero load-side transpose (confirms #1's premise), (ii) how does a
   radix-16 finisher survive 32 live values on 16 registers (confirms or kills #3).
   **Also note what the dispatch itself reveals: MKL specializes on the TERMINAL RADIX, not on N**
   — one suite for all N (§1), three terminal choices. **Our terminator is radix-8 ONLY**
   (`zsplit.h` rejects any chain whose last factor ≠ 8), so our planner races chains but is forced
   to end every one in 8. Terminal radix is a **planner axis we have closed off**; since the chain
   provably differs per cell, the terminal radix plausibly should too.
5. **MT** — unchanged: 3.9% core utilization, the 4–8× multiplier on a different axis.

**Queued structural item (user directive 2026-07-24): a terminator for EVERY power-of-two
radix** (2/4/8/16, plausibly 32), so the chain planner can search the terminal radix per cell the
way MKL's `cmp r11,{4,8,16}` dispatch does. Today `sterm` is radix-8 only and `vfft_zsplit_create`
rejects any chain whose last factor ≠ 8, so the axis is closed. ⚠ **Cost check before building
it by hand — see the generator note below**: each new terminal radix is currently a hand-written
`SPLIT_BFLY{R}` macro body + its INV twin + fwd/bwd kinds + bit-gates, i.e. the exact
combinatorial hand-writing the DAG compiler exists to eliminate.

### (f) ⚠ Generator note — the zil family does NOT go through the DAG pipeline

Verified by module census of `generator/lib/`:

| | `codelet_zil.ml` (z / block-split family) | `codelet_oop.ml` (split-layout family) |
|---|---|---|
| modules referenced | **stdlib only** (Printf, Buffer, Array, Hashtbl, List, String) | Algsimp · Dft · Expr · Pipeline · **Schedule** · **Emit_c** · Isa · Uarch |
| body production | **116 `Printf.sprintf`/`Buffer` sites** — raw C text + macros | DAG → simplify → schedule → regalloc → render |
| gets | nothing | algebraic simplification (`dedup_sub_pairs`, `collect_m`, `deep_collect`, `factor_common_muls`, `share_subsums`), FMA passes, `Schedule.su_schedule_subset`, `Emit_c.cluster_split_schedule`, `spill_info`, sched_wisdom |

So **every zil kernel is hand-authored C whose scheduling and register allocation belong entirely
to gcc.** That is why this campaign's wins came from restructuring *source* (block-split, msg,
uj2) and why placement/compiler luck (§4.9993) has been worth ±5% — we are negotiating with
gcc's scheduler, not driving our own.

**It isn't only the scheduler it opted out of — it re-implements EMISSION too.** `codelet_zil.ml`
bypasses the shared renderer (`emit_c.ml` 4167 + `emit_render.ml` 1426 + `regalloc.ml` 1428 +
`schedule.ml` 1713 + `uarch.ml` 151 + `isa.ml` 317 ≈ **9.2 K lines of shared machinery**) and
prints intrinsics as literal text. Measured consequences:

| symptom | zil | oop (pipeline) |
|---|--:|--:|
| literal `_mm256_` in the emitter | **319** | 48 (uses `Isa.loadu_pd` / `Isa.vec_type`) |
| literal `_mm512_` | **0** | — (ISA is a parameter) |
| ISA argument to the emit entry point | **`emit_z_split` takes none** (`emit_z_t2`/`_n1` take `~vec_width` only) | full `Isa.t` |
| re-pasted macro boilerplate in the emitted tree | **1546 lines = 9% of 16 387**, 129 copies of the same ~9 macros across 60 files | none (rendered per node) |

Two things follow that the roadmap cares about. (1) **The block-split production family is
AVX2-only *by construction*** — not by choice of target, but because the butterflies are literal
`_mm256_*` strings and `emit_z_split` has no ISA parameter at all. The AVX-512 phase-2 and EPYC
port items cannot reach the z cascade without a rewrite. (2) The DAG compiler's whole thesis is
*describe the butterfly once, let the pipeline specialize per ISA / uarch / schedule*; **zil opted
out of that thesis.** That was the correct call for a spike — it is how the band went 0.38 → 0.93
in days — but the family is now PRODUCTION behind `vfft.h`, and it is accruing exactly the debts
the pipeline exists to prevent.

**Consequences for the two items on the table**: (i) per-radix terminators as text templates
multiply hand-written macro bodies × {fwd,bwd} × gates; through the pipeline, radix is a
*parameter*. (ii) The radix-16 terminator's blocker — 32 live values on 16 registers — is
precisely what `regalloc`/`spill_info`/the SU scheduler exist to reason about; as a text template
we would be flying blind. **Counterweight (why it was hand-written in the first place)**: the
block-split addressing, splat-pair twiddles, and conversions fused into loads/stores are *layout*
tricks `Emit_c` does not currently express. Porting zil onto the pipeline means teaching Emit_c
the block-split form — a real project, and the honest prerequisite to a per-radix terminator
family rather than a fifth hand-written kernel.

### (e) The gap is not uniform — aim at 4096

Front door after §4.9994: 2048 **0.92** · 4096 **0.81** · 8192 **0.86** · 16384 **0.93**. So
16384 is 7% off and **4096 is 19% off**. 4096's chain (4.4.4.8.8) carries three cheap radix-4
mids against one expensive radix-8 terminator — the highest terminator-to-mid ratio of any cell,
i.e. exactly the shape lever #1 attacks hardest.

## 5. Current standings this plan attacks (interim ladder, band-corrected)

64: 1.02 WIN · 128: 1.02 WIN · 256–1024: ~0.81–0.83 · 2048: 0.54 · 4096: 0.46 · 8192+:
unserved/0.45 (split ccol only). Target: cascade brings ≥2048 into the ~0.8+ band and
extends the z family to 8192/16384.
