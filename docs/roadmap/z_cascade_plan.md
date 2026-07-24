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

## 5. Current standings this plan attacks (interim ladder, band-corrected)

64: 1.02 WIN · 128: 1.02 WIN · 256–1024: ~0.81–0.83 · 2048: 0.54 · 4096: 0.46 · 8192+:
unserved/0.45 (split ccol only). Target: cascade brings ≥2048 into the ~0.8+ band and
extends the z family to 8192/16384.
