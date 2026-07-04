# Natural-order in-place 1D C2C — design decision

**Date:** 2026-07-04 · **Status:** EMPIRICAL PROGRAM COMPLETE — **see §2e for the FINAL verdict map (supersedes §0 table + §2d)** · **Branch:** dev/arbitraryTail
**Provenance:** 16-agent design workflow — 5 code readers → 4 independent designs (fused-scatter /
perm-pass / reuse-OOP / self-sorting) + 3 *measured* Phase-0 gates → 3-lens judge panel (unanimous) +
completeness critic. Gate harnesses: `build_tuned/test/natorder_g{1,2,3}_*.c` (untracked).

> ⚠️ **Supersedes §"The plan" of [docs/natural_order_inplace.md](../natural_order_inplace.md).**
> That doc (committed d0eebbc3) designates a *fused strictly-in-place transpose codelet* as the chosen
> direction. §3 below shows that is impossible in the general case — but its performance goal (no
> separate plane pass) IS achievable via a plan-owned shadow plane, and its mechanism survives intact
> at small N where the whole transform fits one monolithic codelet.

---

## 0. TL;DR — the verdict

Natural order becomes a **per-cell measured verdict** (like everything else in this engine): a new
`order` axis on `vfft_config_t`, and the calibrator picks one of four modes per (N,K), persisted in
wisdom. No single mechanism wins everywhere; the hybrid does.

| Mode | Mechanism | Extra passes | Where it wins | Status |
|---|---|---|---|---|
| **FREE** | nf==1 plans + prime-N Rader/Bluestein overrides are already natural | 0 | only 9/230 wisdom cells (4%) | expose immediately |
| **LEAF-IP** | `n1_oop` monolithic codelet called with dst==src (load-all-then-store, natural writes free) | 0 — measured **1.3× FASTER** than OOP | N≤128 — the latency WIN | G1-proven; needs no-`__restrict__` regen |
| **SCRATCH** (headline) | shadow-plane ping-pong: stage-0 user→scratch (MODEB dataflow), middle stages in-place on scratch, **scatter terminator** scratch→user via existing `t1_oop`/`n1_oop` codelets | 0 (passes stay nf); **MEASURED §2b: ~+22% @4096/K=4** (stage-0 free in L2), ~+50% honest @K=32 (L3 double-footprint tax) | small-K / L2-resident cells | Phase-0 DONE — GO for small K; loses to PURE at L3-band |
| **PURE** | unchanged scrambled FFT + cycle-following K-row permutation pass | +1 pass; **MEASURED §2b: +24% @4096/K=32 (cycle)**, +56% @K=4 (gather) — plus G2's +17–25% @K≥64 | K≥32 / L3-band cells | zero codegen, land first |

**Backward direction is free**: natural-in bwd = natural-out fwd with re/im swapped on both sides
(swap identity, valid because both ends are natural order). Zero new code per mode.

~~Contender kept warm: **Stockham ping-pong** (D4)~~ — **REFUTED BY MEASUREMENT (T5, 2026-07-04,
`natorder_t5_stockham.c`)**: compute-matched probe (one generic AVX2 radix-4 kernel driving both a
generic in-place DIF and a generic Stockham ping-pong, identical flops, correctness ~1e-13 both) shows
Stockham at **1.86–2.40× the in-place time even fully L2-resident** (1024/4: 2.40×, 4096/4: 2.33×,
1024/32: 1.86×; 4M control: 3.18×). The "~0% in cache" hypothesis confused bandwidth with memory-op
throughput: in-cache the FFT is load/store-port-bound, and Stockham doubles line traffic per stage
(separate src+dst streams vs read-modify-write of the same lines) while its early stages do strided
small-granularity writes (the distributed transpose work). A 2× structural pattern deficit will not
be inverted by tuning. **DROPPED for all regions** — the cache-resident region belongs to the SCRATCH
scatter terminator (+22% measured, COBRA tiling = the remaining upgrade path toward ~+10%).

Refuted and closed (do not re-propose):
- **OOP + copy-back** (G3): loses everywhere — 1.26× (best, N=64 K=8) to 2.9× (K=256) vs scrambled
  in-place. Even *without* the copy-back, OOP never ties in-place on the measured cells.
- **Standalone perm pass as the universal answer** (G2): never <10% of FFT time in any of 12 cells;
  fails its own ≤1.3×-of-model criterion at K=8 (per-row overhead-bound, ~4.7–5× a streaming pass).
- **Strictly single-buffer fused scatter** (the old doc's plan): impossible in general — see §3.

---

## 1. Requirements — pin these before Phase 2

Two different things called "in-place" (critic finding; affects which modes are admissible):

1. **Pointer contract** (`vfft_execute(p, FWD, re, im, re, im)` works): ALL four modes satisfy this.
2. **Memory footprint** (no second N×K plane pair — the usual reason at 4096×256 = 16 MB/plane):
   only **PURE** and **LEAF-IP** satisfy this. **SCRATCH** allocates a plan-owned shadow plane
   (N·K·16 B) — footprint equals OOP; only the API is in-place.

Also unpinned: is **natural-in backward** (natural spectrum → natural signal) required, or only
natural-out forward? The swap identity gives both for free, but calibration/validation cost doubles.

Use-case escape hatch (**"Mode −1"**, zero cost, worth documenting in the API): consumers that only
convolve/filter/power-spectrum don't need natural order at all — for convolution, pre-permute the
*filter* coefficients once at plan time and stay scrambled end-to-end. Cheapest correct product for
the dominant use case.

## 2. What the gates measured (this host, i9-14900KF, AVX2)

### G1′ — LEAF-IP theory VALIDATED end-to-end (2026-07-04, T1–T4; supersedes parts of G1 below)

- **T1 reproduce** (`natorder_g1_leaf_alias.c`): 21/22 pass aliased, N=9 fails all K (err ~1–2). ✔
- **T2 positive control** (`natorder_t2_norestrict.c`): the *same generated N=9 source* recompiled with
  `__restrict__` neutralized (symbol renamed, nothing else) is **bit-exact aliased at every K (0.0)**
  while the library build fails every K. Restrict-UB confirmed as the *sole* cause. ✔
- **T3 contract gate** (`natorder_t3_gate.c` — the permanent per-build gate): **all 22 leaves**
  compiled no-restrict × K∈{1,2,3,4,5,8,12,23,64} (incl. the scalar/SSE2 tail lanes G1 skipped):
  aliased bit-exact + naive-DFT natural-order check + **aliased swap-identity roundtrip** (the bwd
  mode) — **22/22 PASS**. No-restrict is alias-safe by contract across the registry, fwd+bwd. ✔
- **T4 perf triangle** (`natorder_t4_perf.c`, 6 wisdom-covered cells, no new calibration):
  - **Option-A tax ≈ zero**: no-restrict vs restrict on the pure separate-dst OOP path = 0.98–1.03×.
    → **DECISION: Option A** — regen all `n1_oop` leaves without `restrict` (one emitter line);
    no separate `_ip` family needed. Keep T3 as the per-build gate forever.
  - Aliased-vs-OOP-call: ≈1.0× at most cells; the G1 1.3× win **reproduced at N=64 K=64 (0.74×)** —
    real but cell-specific (footprint halving pays where the working set is big enough to matter).
  - **vs the real scrambled in-place baseline** (the comparison G1 never ran): LEAF-IP **wins N=64 K=4
    outright (0.66× — natural order 1.5× FASTER than the current path)**, loses 1.37–1.76× at
    {128/4, 64/64, 128/64} and at tiny nf=1 cells (16/4, 32/4) where FREE mode already covers natural.
    → LEAF-IP is a per-cell **verdict candidate**, not a universal N≤128 winner — exactly what the
    mode architecture assumes. (Caveat: LEAF-IP timed as a bare call vs public-API dispatch for the
    baseline — small-N ratios carry some dispatch bias; re-measure inside the real dispatch in Phase 1.)

### G1 — aliased LEAF (`natorder_g1_leaf_alias.c`)
- dst==src bit-exact for **21/22** leaf codelets, K ∈ {4,8,12,23,64}.
- **Aliased call is 0.77× the separate-dst OOP call at N=64 K=64 — 1.3× faster. The only measured
  outright latency win for natural order.** (Monolithic load-all-then-store: the permutation happens
  in registers, exactly the old doc's "read block → permute in registers → write back" — it works
  precisely when the *whole transform* is the block, i.e. N≤128.)
- ⚠️ The single failure (N=9) is diagnosed: gcc-15 exploits `__restrict__` to reload inputs after
  output stores when register pressure exceeds 16 ymm (R=9..15 monolithic band). The 21 passes are
  therefore **compiler luck, not contract**. Productionization = regenerate `n1_oop` leaves without
  `restrict` (source dataflow is alias-safe by construction) **or** emit dedicated `_ip` variants +
  keep a per-build bit-exact alias gate. The stack-buffer copy fix measured 2.11× — refuted as
  "nearly free"; route failures to another mode instead.

### G2 — permutation-pass cost (`natorder_g2_perm.c`)
- 12 cells, N ∈ {256..16384} × K ∈ {8,64,256}: never <10% of FFT time. K=256: 17–24%; K=64: 21–34%;
  **K=8: 34–218%** (64 B rows are per-row-overhead-bound at ~30 GB/s vs 150+ GB/s streaming).
- Mechanism data: in-place cycle-following = 0.72–1.0× of a same-order memcpy pass at K≥64 (truly
  "+1 pass"); gather+scratch = 2.0–3.7× baseline, wins only at K=8/N≤4096. Wire both, verdict picks.
- Concretely: appending the pass at high K surrenders ~⅕–¼ of the MKL lead (16384/256: 2.17×→~1.74×).
- ⚠️ Caveat: the 256/256 headline (16.6%) uses the **anomalous 73.7 µs baseline**; on the corroborated
  ~45 µs it is ~27%. Re-baseline that cell before trusting per-cell verdicts near the boundary.

### G3 — OOP-natural + copy-back (`natorder_g3_oopnat.c`)
- Refuted everywhere measured (best 1.26×, worst 2.9×; degrades monotonically with K). Note: the
  calibrator picked BAILEY2 (not LEAF) for 4/5 small cells; outputs verified natural vs naive DFT.
- ⚠️ The sweep died at N=256 K=8 (8/18 cells unmeasured) — irrelevant to the verdict (trend is
  monotone against, and in-place stays cache-resident while OOP touches 2× planes + a third pass).

## 2b. Phase-0 probe results (2026-07-04, MEASURED — user-directed protocol)

Protocol (user): **real calibrated chains from spike_wisdom.txt, N=4096, no new calibration** — cells
K=4 (chain 4·4·8·32, wisdom `use_dif_forward=1`) and K=32 (chain 4·4·4·8·8, DIT). Probe =
`build_tuned/test/natorder_p0_scatter.c` (pinned core 0, QPC best-of-5, public-API baseline with
JIT-resolved calibrated plans, wisdom copy `natorder_wis_p0/`).

**Validated:** (a) order probe — `natural[k] = scrambled[perm[k]]` matched vs naive DFT at 1e-10 on
both cells with the **forward-order** digit reversal, *including the DIF-calibrated K=4 cell* (the
"DIF ⇒ reversed factor order" claim did NOT hold through the public-API path — implementation must
auto-detect orientation per plan, as the probe does); (b) comb algebra — sources of natural comb
{q+j·P} are contiguous R-row blocks on both real chains (design's §3 structure confirmed).

**Kernel results** (ns, best-of-5; baseline FFT: K=4 ≈ 22.2–22.9 µs, K=32 ≈ 155–189 µs across runs —
unlocked-machine variance; wisdom anchors 21.0/142.1 µs):

| kernel | K=4 (32B rows, 512KB ws = L2) | K=32 (256B rows, 4MB ws = L3) |
|---|---|---|
| same-order copy pass | 3 353 (156 GB/s) | 69 472 (60 GB/s) |
| in-place pass (last-stage proxy) | 3 428 (153 GB/s) | 36 228 (116 GB/s) |
| scatter-q (q-outer) | 32 833 (0.10×) | **81 753 (0.85×)** |
| scatter-b (seq reads) | 32 026 (0.10×) | 110 969 (0.63×) |
| scatter-s (**j-outer**, true sequential streams) | **8 382 (0.40×)** | 90 275 (0.77×) |
| PURE cycle-following | 22 316 | **36 538** |
| PURE gather+copyback | **12 787** | 155 178 |

⚠️ Loop order is decisive: q-outer round-robins tiny writes across R streams 4 KB apart (fill-buffer
thrash, 0.10× at K=4); j-outer writes each stream's *contiguous* P·K region sequentially (0.40×).
The naive kernel would have falsely killed the design at K=4.

**Verdicts (overhead vs scrambled in-place):**
- **K=4: SCRATCH wins — ~+22%** (terminator +21.7%; stage-0 redirect ≈ FREE in L2: two-buffer copy
  runs at in-place speed, 156 vs 153 GB/s). PURE best is +56% (gather) — the small-row catastrophe.
- **K=32: PURE-cycle wins — ~+24%.** SCRATCH honest total ≈ +50% (terminator +29% AND stage-0 +21% —
  the 4MB double-footprint evicts to L3 on both two-buffer passes). The design's "+5–20%, scatter-side
  degradation only" model was falsified: the scatter *bandwidth ratio* held (0.85× ≥ the assumed
  0.60–0.85×), but the model ignored the working-set doubling at L3-band cells.
- **The mechanism flips with K** — per-cell measured verdict (this design's architecture) is confirmed
  as necessary, not optional. Both winners cost ~20–25%: natural order at N=4096 ≈ a fifth of the MKL
  lead (1.5–2.1× → ~1.2–1.7×, natural-vs-natural).
- **OOP-machinery route re-confirmed dead at N>128** (user asked): our OOP loses to our scrambled
  in-place 1.16–2.49× *before* copy-back (G3) — same double-footprint tax measured here. "OOP beats
  MKL" holds, but PURE/SCRATCH deliver the identical feature 2–3× cheaper. N≤128 aliased LEAF-IP
  (OOP codelet, dst==src) remains the exception and the best result (G1: 1.3× faster).

## 2c. FAILED EXPERIMENT — Stockham ping-pong (T5, 2026-07-04). Do not re-propose without new evidence.

**Hypothesis (as argued, verbatim):** "Inside L2, two-buffer streaming runs at in-place speed
(measured 156 vs 153 GB/s at K=4), so spreading the reorder across all nf stages as regular-stride
sequential streams (Stockham autosort) delivers natural order at ~0% overhead for cache-resident
cells." Stockham was the theoretical ceiling for region 1: same pass count as today, no scatter,
natural order as a *property of the dataflow*.

**Method (`build_tuned/test/natorder_t5_stockham.c`, untracked):** compute-matched A/B — a naive
hand-Stockham vs our *tuned* codelets would measure codelet quality, not the theory, so the probe
implements ONE generic AVX2 radix-4 butterfly (identical DFT4 + 3 twiddle cmuls per group, identical
flop count, radix-4-only 4^n chains to stay memory-bound) and drives it two ways:
(a) generic in-place DIF — natural in → digit-reversed out, zero data movement, the engine's
pattern class; (b) generic mixed-radix Stockham — natural in → natural out, ping-pong A↔B, stage s
reads `src[j+L(k+Mr)]`, twiddles `W_{4L}^{jr}`, writes `dst[j+L(q+4k)]`. Correctness gated per cell:
(b) vs naive DFT directly (natural), (a) vs naive through the base-4 digit reversal — both ~1e-13.
QPC best-of-5, pinned core 0, rescale-chunked timing, wisdom-copy dir.

**Result — REFUTED in its own best region** (ratio = Stockham / in-place, same kernel):

| cell | ping-pong ws | ratio | note |
|---|---|---|---|
| 1024×4 | 128 KB (deep L2) | **2.40×** | the cell the hypothesis was strongest for |
| 4096×4 | 512 KB (L2) | **2.33×** | |
| 1024×32 | 1 MB (L2) | **1.86×** | |
| 4096×32 | 4 MB (L3 control) | 3.18× | out-of-region control behaved as predicted |

**Root cause — the hypothesis confused bandwidth with memory-op throughput.** The 156≈153 GB/s
copy measurement was a *once-through streaming pass*; the FFT re-touches its working set nf times,
and in-cache it is bound by load/store ports and cache lines per cycle, not GB/s. Two structural
taxes: (1) Stockham runs separate src and dst streams every stage — double the line traffic of
in-place's read-modify-write of the same lines; (2) its early stages (small L) write at stride
4L·K in row granularity — the transpose work is distributed across stages, not eliminated, and at
K=4 those are 32 B strided writes. A ~2× structural deficit; kernel tuning does not invert sign.
Consistent with practice: no major CPU FFT ships pure Stockham (it is a GPU pattern — different
memory system; see the EPYC/GPU roadmap note for where it may become relevant again).

**Scope of the falsification:** row-granular mixed-radix Stockham on this split-plane K-lane layout,
on this CPU class. Revisit only if one of these materially changes the premise: fused multi-stage
variants (radix-16 as two fused radix-4 halves the stage count — helps both sides though), a
lane-blocked data layout that turns the early-stage strided writes into full-line writes, or a GPU
port. Absent those, region 1 belongs to the SCRATCH scatter terminator (+22% measured; COBRA
L1-tiling is the upgrade path toward ~+10%).

**Side observation (context-quality only, dispatch overhead differs):** the *generic* radix-4 DIF
beat the tuned public-API path at both K=4 cells (1024/4: 3370 vs 3833 ns; 4096/4: 17 266 vs
20 361 ns). Possible K=4 calibration gap for the tuned chains (64·16 and 4·4·8·32-DIF) — breadcrumb,
not a conclusion.

**Probe-reuse gotcha:** the DFT4 macro originally used internal temporaries named `t0r..t3r`, which
silently shadowed caller variables of the same name passed as arguments (Stockham failed correctness
while DIF passed — the tell). Macro locals are now `_u`-prefixed. Check for capture before reusing
probe kernels.

## 3. Why the strictly-in-place fused scatter is impossible (and what survives)

Last stage, radix R, P=N/R groups: group q **reads** R *contiguous* rows [π(q)·R, π(q)·R+R) but its
natural-order targets are the **stride-P comb** {q + j·P}. Read units are contiguous R-row blocks;
write units are global combs. A wavefront argument over this block-vs-comb mismatch shows any schedule
must hold Θ(N·K) live rows before targets free up — there are no small closed cycle sets, so
pair/cycle-grouped butterfly scheduling and bounded row-rings both fail. (This is why OOP LEAF/BAILEY2
buy natural order with a second buffer, and why the old doc's single-buffer fused codelet cannot exist
for general mixed-radix N. The two survivors are the two ways to make the live set small: shrink N
until the codelet's registers/stack ARE the live set — LEAF-IP — or give the live set a home — the
SCRATCH shadow plane.)

## 4. Mode details

### SCRATCH — shadow-plane scatter terminator (headline; D1 "NAT-TERM Mode A")
1. **Stage 0** (always untwiddled n1, DIT): run OOP user→scratch — byte-identical to the existing
   MODEB stage-0 redirect dataflow; scratch = 2 plan-owned planes (rfft planeA/planeB precedent).
2. **Stages 1..nf−2**: unchanged in-place executors ON scratch (generic executor stage range now;
   MODEB-style `start_stage` JIT in Phase 3).
3. **Terminator** (stage nf−1) scratch→user, iterating groups in q-order: untwiddled groups via
   `radix{R}_n1_oop_fwd_avx2`, twiddled via `radix{R}_t1_oop_fwd_avx2` with `in_leg_stride=K`,
   `out_leg_stride=P*K` — natural-order scattered stores through the **existing 11-arg OOP codelets**
   (registry coverage verified: n1_oop {2..17,19,20,25,32,64,128}, t1_oop {4,7,8,13,16,32,64}).
   Twiddles: the K-replicated combined `cf_all` table that twiddle.h already bakes for the bwd generic
   path, baked for the terminator too. Per-group out-bases from the `_r2c_compute_perm` math restricted
   to the prefix chain, precomputed into the plan tape.
   In q-order the writes are R perfectly-sequential streams and reads are R·K·8 B contiguous chunks in
   scrambled block order; the b-order variant (sequential reads / comb writes) is the mirror — measure
   both, wisdom stores the winner.
- **Cost model**: passes stay **nf**. Overhead = scatter-side bandwidth degradation only, estimated
  0.6–0.85× normal stage bandwidth ⇒ **+5–20%** total (e.g. 1024/256: 1.86×→1.62–1.79× vs MKL-natural;
  4096/32: 2.57×→2.24–2.45×). ⚠️ **This 0.6–0.85× is the one load-bearing UNMEASURED number in the
  whole design** (critic's top finding) — Phase 0 microbenches it before any Phase-2 code.
- Constraint: plan is DIT-only initially (`order=natural` forces `use_dif_forward=0`); t1_oop radix
  coverage gaps route the cell to PURE.
- Risk cells: big-N K=256 where 2×16 MB scratch+data ≈ L3 — expect the verdict to flip those to PURE.

### PURE — permutation pass (D2; the always-correct floor)
Unchanged scrambled stages + one row-permutation pass from the plan-time cycle decomposition
(`_r2c_compute_perm` / `_perm_dif` — both DIT and DIF orders already computed in r2c.h). Rows are K
doubles ⇒ **K-agnostic: odd K and padded Kp are trivially correct** (row granularity doesn't care).
Both mechanisms wired (cycle-following K≥64, gather+scratch small-K), verdict picks. Never
auto-selected at K≤8.

### LEAF-IP (D3's survivor; N≤128)
Aliased `n1_oop` call, natural for free, 1 pass, measured faster than both scrambled-in-place-then-
permute and OOP. Prereqs: no-`restrict` regen (or `_ip` emission) + per-build alias gate + N=9-band
(R=9..15) routed to fallback. Covers the classic "single small FFT, natural bins" latency case.

### FREE (nf==1 + prime overrides)
Identity permutation — flag at plan create, zero execute cost. Tiny coverage (4%) but zero work.

## 5. Integration

- **API**: `order` field on `vfft_config_t` (VFFT_ORDER_DEFAULT=scrambled / VFFT_ORDER_NATURAL) —
  additive, zero impact on existing callers. Document Mode −1 (consumer-side reorder) beside it.
- **Wisdom**: one trailing `nat_mode` token on spike_wisdom lines. **Verified this session:** the
  OCaml reader (`emit_executor_h.ml parse_wisdom_line`) enforces only a *minimum* token count, so a
  trailing token is backward-compatible. ⚠️ Critic: three designs each invented a different v7 field,
  and the padding arc has its own pending format evolution (`exec_me` folds into spike_wisdom per
  [[wisdom-files-inventory]]) — **decide ONE v7 schema covering both padding and nat_mode before
  Phase 1 writes any wisdom.**
- **MT**: the FFT part K-splits as today. ⚠️ The perm pass must **row-range-split, not K-split** —
  a T=8 K-split of K=64 gives each worker 8-lane (64 B) sub-rows, exactly G2's catastrophic regime.
  SCRATCH terminator groups are independent ⇒ q-range split.
- **Odd K / padding**: all modes are row-granular or reuse rem-aware codelets (t1_oop/n1_oop carry
  the SSE2/scalar tail; perm rows are K-agnostic; LEAF-IP gate already tested K=23). Padded plans:
  perm/scatter run at exec_me=Kp or K — both correct, same honorable-verdict property as padding.
- **JIT**: Phase 1–2 run terminator/perm through direct codelet calls (no JIT dependency). Phase 3:
  `STAGE_NATTERM` in emit_executor_h.ml + emit_jit.py with `_nat` cache-key suffix +
  `VFFT_PROTO_JIT_VERSION` bump + baked-vs-JIT bit-exact smoke.
- **Kill switch**: `order` unset ⇒ byte-identical current behavior (scrambled path untouched).

## 6. Phase plan

- **Phase 0 (probes only, no src/ changes)** — (a) scatter-terminator microbench: mock comb-writes in
  q-order and b-order at K ∈ {4,8,32,64,256}, N ∈ {256,1024,4096} — pins the 0.6–0.85× estimate;
  (b) impulse-response order probe confirming `_r2c_compute_perm` direction on a mixed chain (N=24)
  and nf==1 identity; (c) re-baseline the anomalous 256/256 cell; (d) K ∈ {1,2,3} row-perm cost probe
  (classic single-FFT case; untested by G2). GO/NO-GO: scatter ≥0.6× ⇒ Phase 2 SCRATCH; else Stockham
  probe enters the bakeoff for K≤8 before committing.
- **Phase 1 (runtime-only, lands first)** — `order` axis; FREE detection; PURE mode (both mechanisms,
  fwd+bwd via swap identity, row-range MT, odd-K sweep + roundtrip + vs-FFTW-natural elementwise gate);
  LEAF-IP behind the per-build alias gate. Wisdom v7 schema decision (with padding arc).
- **Phase 2 (SCRATCH)** — scratch alloc, stage-0 redirect, terminator (q- and b-variants), cf_all fwd
  baking, `_calibrate_natorder` FREE/LEAF-IP/SCRATCH-q/SCRATCH-b/PURE verdict per cell.
- **Phase 3 (codegen, OCaml)** — no-`restrict`/`_ip` leaf regen; `t1s_oop` emission (kills cf_all
  traffic); STAGE_NATTERM baked + JIT parity; extend t1_oop radix coverage. Regen per
  [[gen_set-root-and-dune-cache-gotchas]].
- **Phase 4 (calibration + positioning)** — populate nat_mode across the 198-cell grid (isolated
  cell-per-process); **natural-vs-natural bench vs MKL** (MKL in-place c2c IS natural order — today's
  wins are our-scrambled-vs-their-natural; this mode finally enables the apples-to-apples column,
  modeled on bench_1d_vs_mkl.c per [[canonical-mkl-bench]]); dual (scrambled|natural) columns in
  docs/performance.

## 7. Open items (critic)

1. Scatter-bandwidth microbench = the load-bearing unmeasured number (Phase 0a).
2. K ∈ {1,2,3}: every mechanism's worst case; LEAF-IP covers N≤128 — probe the rest (Phase 0d).
3. No cell was measured under ALL candidate mechanisms (gates used disjoint grids) — the Phase-2
   calibrator IS that unified bakeoff; don't trust cross-gate ratio comparisons until then.
4. 256/256 baseline anomaly contaminates G2's best case (Phase 0c).
5. Requirements pin: footprint-in-place vs pointer-in-place; natural-in bwd needed? (§1)
6. FFTW prior art (buffered plans + in-place square transpose for N=m·p²): the palindromic-
   factorization special case makes digit-reversal an involution — a potential zero-scratch SCRATCH
   alternative for square-ish N; parked, revisit in Phase 2 if scratch memory is contested.

## 2d. T6 UNIFIED BAKEOFF RESULTS (2026-07-04) — the per-cell winner map

Full raw output: `natorder_t6_bakeoff_results.txt` (probe `build_tuned/test/natorder_t6_bakeoff.c`).
Baseline = wisdom chain+variants via LOW-LEVEL generic executor (no Tier-1/JIT; uniform across
candidates — matches wisdom best_ns within ~0-10% at most cells, but 128/4 +32% and **128/64 +60%**
off wisdom → those two cells' verdicts need a tuned-path recheck in Phase 1).

| cell | WINNER | overhead | runner-up |
|---|---|---|---|
| 16/4, 32/4 | FREE | 0% | PURE-cycle +15-18% |
| 64/4 | LEAF-IP | **−21%** | FREE 0% |
| 128/4 | SCR-t1 | +17.6% | SCR-t1s +23.7% |
| 64/64 | PURE-cycle | +36.2% | SCR-t1s +66% (worst cell) |
| 128/64 | **PSWAP (injected 4·8·4)** | **−9.4%** ⚠ vs generic baseline | SCR-t1s +20.7% |
| 1024/4 | SCR-t1s | +23.3% | SCR-t1 +27.0% |
| 1024/32 | SCR-t1s | +26.2% | PURE-cycle +30.2% |
| 256/256 | PURE-cycle | +14.8% | SCR-t1s +24.5% |
| 4096/4 | SCR-t1s | +15.9% | SCR-t1 +19.0% |
| 4096/32 | PURE-cycle | +23.4% | SCR-t1s +46.2% |
| 4096/256 | PURE-cycle | +16.4% | SCR-t1s +20.9% |

**Conclusions:** (1) FIVE different mechanisms win cells — per-cell verdict architecture CONFIRMED
required. (2) Shipped tax range: −21% to +36%, typical +15-26%. (3) **Chain injection validated**:
128/64's injected 4·8·4 runs 0.73× the wisdom chain here AND delivers natural order below scrambled
cost (⚠ recheck vs Tier-1 path — this cell's generic baseline is 60% off wisdom best_ns).
(4) **TILED scatter REFUTED as implemented** — lost to plain j-outer at every cell (staging copy
costs more than the pattern win); do not re-propose without a fused (no-extra-copy) design.
(5) **t1s_oop emission DEMOTED**: worth only 3-4 points at cells where scatter wins (K=4 band);
its big wins (15-20 pts at 4096/32+256) are at cells where PURE-cycle wins anyway. Phase-3 optional.
(6) PSWAP's pass ≈ cycle's pass in cost (involution gave no pass-side advantage); palindromic chains
matter only where the injected chain itself is competitive. (7) Scatter's honest table tax (t1 vs
t1s rows) = 3-20 points depending on cell; at tiny cells t1 is FASTER than t1s (L1-hot table).

---

## 2e. FINAL — complete empirical record (T7–T11) and shipped verdict map

**This section supersedes the §0 TL;DR table and §2d.** Methodology from T8 onward (user directive):
warm-up + 5 rounds with 150 ms cool-down pacing, **averaged** (not best-of), 400 ms between cells,
pinned core 0 — now the standard for natorder probes. Archive runs: t7_ub, t8_paced,
t9_celltrans, t10_ub2, **t11_final_clean** (game-noise-free, THE reference) in `natorder_t*_results.txt`.

### Final winner map (T11 clean run; stability judged across all clean runs)

| N | K | winner | tax | stability |
|---|---|---|---|---|
| 16/32/64 | 4 | FREE | 0% | solid (64/4: see LEAF-IP note) |
| 128 | 4 | SCR-t1s | +19% | stable 4/4 |
| 64 | 64 | PURE-cycle-UB | +30% | stable |
| 128 | 64 | **PSWAP (inj 4·8·4)** | **−7.3%** | negative 5/6 runs — real |
| 1024 | 4 | SCR | +27% | stable (t1↔t1s variant flips) |
| 1024 | 32 | PURE-cycle-UB | +22% | stable |
| 256 | 256 | PURE-cycle-UB | +30% | baseline drifts (old anomaly) |
| 4096 | 4 | PURE-cycle-UB | +27% | TIED w/ SCR |
| 4096 | 32 | PURE-cycle-UB | +18.5% | stable |
| 4096 | 256 | PURE-cycle-UB | +26% | TIED w/ SCR |

### Kernel-optimization campaign on the permutation pass (user-driven)

- **cycle-UB** (T7): plan-time flattened cycle lists + `_mm_prefetch` + AVX row moves instead of
  memcpy — shaved 2–13 pts off naive cycle; **owns the K≥32 band and is at its practical ceiling.**
- **cycle-UB2, 8-way interleaved cycles (T10): REFUTED** — uniformly slower at every cell. Root
  insight: the move-list has NO data-dependent addressing (next index comes from the list), so OoO
  already overlaps the loads; there was no serial chain to break. Remaining ~1.45× gap to the
  in-place-pass floor is TLB/cache-line-miss bound; huge pages already refuted for rfft → ceiling.
- **CELL-TRANSPOSE via transpose.h recursion (T9): REFUTED** — transpose.h API is scalar-element
  (can't take K-double cells); its cache-oblivious recursion ported to cell granularity loses every
  nf=2 cell (+45…+172%) because as a bolt-on it costs 2 passes vs cycle's 1 / scatter's ~0. Would
  only compete fused as the terminator — which IS the scatter mechanism.
- **Scatter table tax honestly priced** (T6): kernels stream a data-sized FLAT table (t1_oop reality)
  vs tiny scalar table (future t1s_oop). Delta = 3–4 pts at cells scatter wins → **t1s_oop OCaml
  emission DEMOTED to optional**.
- **L1-tiled (COBRA) scatter: REFUTED** (T6, every cell) — staging copy exceeds the pattern win.

### LEAF-IP: bimodal — calibrate, don't hardcode

Across identical clean runs, aliased-leaf at 64/4 measured **−30 / −24 / +48 / +57%** (same binary,
paced harness) — alignment/frequency-state sensitivity of the aliased monolithic call. The probe
cannot settle it; the per-machine plan-time race can (FREE at 0% is the safe default there).
Correctness is NOT in question (T2/T3: alias-safe by contract after no-restrict regen, 22/22 + gate).

### Phase-1 calibrator specification (from the data)

Candidate set: **FREE / LEAF-IP (T3-gated) / SCR-t1 (j-outer + cf_all; t1s if P3 ever lands) /
PURE-cycle-UB / PSWAP (INJECTED palindromic + single-leaf chains — the search must inject, wisdom
chains never contain them)**. Race at plan time under the T8 methodology; persist `nat_mode` (+
injected chain if PSWAP wins) in wisdom v7 (schema jointly with padding's exec_me). **Win-margin/
hysteresis rule required**: tied cells (4096/4, 4096/256, 1024/4-variant, 128/64 ~0%, 64/4
FREE↔LEAF-IP) must not flap on noise. Bands for priors: K=4 multi-stage → scatter; K≥32 → cycle-UB;
nf=1 → FREE; expect ~+16–30% typical, floor 0%, occasional negative via PSWAP/LEAF-IP.

Dead, measured, never re-propose: strict in-place fused scatter (§3), Stockham (§2c), OOP+copy-back,
standalone-perm-as-universal, COBRA-tiled scatter, transpose.h cell recursion, interleaved cycles.
