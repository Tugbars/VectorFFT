# Next-session brief — approved work queue state (2026-07-16)

Full pass given on items #5–#9 of the r2c/IL action list
(`docs/roadmap/r2c_c2r_il_design.md` §3 + geometry contracts §6a19–§6a21).
State and exact entry points:

| # | item | state | entry point |
|---|---|---|---|
| 9 | variant-bound DIF bwd emission | **DONE** (jit ver5, §6a21) — bit half complete, gate ALL PASS, special fold deleted. Perf residual: 3-stage DIF jit still ~8.7% behind core; suspect tw_buf block-broadcast hoisting difference between STAGE_DIF_BWD macro and the generic dif bwd range loop. Selection rule (bwd DIT→jit, DIF→core) RETAINED until closed. | compare `_vfft_il_bwd_dif_range` loop body vs the macro at (1000,4); micro-bench per-stage |
| 7 | wisdom free/save parity | **DONE** (§6a22) | `src/core/vfft.c`: `vfft_wisdom_free` / `vfft_wisdom_save` — neither touches fft2d tables (nor fft3d, inherited). Clone the c2c free/save lines per table; the 6a19 bundle-clone pattern in the create path shows the field names |
| 6 | DIF-inner + B-tail entry folds (r2c) | **B-tail DONE** (§6a23, gate_r2c_tail.c: stale guards removed, odd-B fused ≤3e-13). DIF fused = generator post-tw OOP spec (design doc §6); wiring deferred until family exists | `r2c.h` ~816: fused-first gate is `stages[0].n1_fwd && !use_dif_forward && (B&3)==0`. DIF case → wire the `t1_dif il_in` entry-fold family (6a16, gated); B-tail → tail-aware `me` path (gates already run 65/67). Gate vs explicit-pack BIT |
| 5 | z-store terminators + R2C/C2R NULL-halves contract | **DONE** (§6a24: gate_vfft_rz 12/12 BIT; public z −5.5%, store-variant −3.4%, post −13.7%; z/MKL 0.722×; rfft-native z + 2D z = follow-ups) | store-variant flag in `_r2c_postprocess` (+`_fused`); vfft.h rows `dim==NULL` (r2c out) / `sim==NULL` (c2r in), same convention as §6a19 C2C |
| 8 | Model-B fused last-stage | **CLOSED — measured negative** (§6a25: avx2 +24.5%; avx512 parity-at-best on (512,256) noise band, +2.3..+8.2% elsewhere; scaffold exonerated at 11.8µs, §6a20 diagnosis corrected; setter NOT wired, machinery dormant, both codelets + bench_ab_modelb.c in tree) | scalar specials loop at r2c.h ~1505–1521; do NOT wire the setter first (measured −17% as-is, §6a20). Prototype codelet `radix256_r2c_term_ls_r8_fwd_avx2` in archive; live-activation recipe in `benches/bench_r2c_tax.c` |

User-side in parallel: r2c inner-cell wisdom hygiene runbook
(`docs/wisdom/r2c_inner_cell_hygiene.md`) — audit half-N K-columns, dispatch-race
persistence check, rigor consistency.

Container rebuild recipes: `docs/il_oop_post_p1a_api.md` §7 (jit -D trio + PIC
rsp; note rsp now also carries `t1_dif_log3_bwd` avx2 objects for ver5 DIF-bwd
plans). Known debt list: rfft.h:523 UB (`vl` bound), public wisdom save/free
gap (=#7), model-B scaffold (=#8), 3-stage DIF residual (=#9 tail).

Canonical archive: `/mnt/user-data/outputs/VectorFFT-main-fftnd-accfix.tar.gz`
(pack2 mirror = source of the tar; working tree at /home/claude/vfft is
rebuilt per session from the tar).


## Queue empty — candidate next items (standing debt)
- 2D r2c/c2r z contract (§6a24 out-of-scope note)
- rfft.h:523 UB warning (vl bound, aggressive-loop-opt)
- 3-stage DIF bwd jit residual ~8.7% (broadcast hoisting suspect, §6a21)
- Gap-A post-tw OOP generator mode (spec ready, design doc §6)
- r2c.h:973 c2r n1_scaled_bwd tail guard (unverified family)
- n1 codelet profiling R=8/16/32 (standing TODO); TLB/hugetlbfs at large K

## §6a26 session close
- rfft native z terminator (fwd): SHIPPED — chunked-hcnr, 12/12 BIT, tax curve in §6a26.
- RESOLVED §6a27: rfft PIC set built (tools/build_rfft_pic_rsp.sh, rsp 94→209). Natural jit then MEASURED NET NEGATIVE across the rfft path's K domain → unbound (explicit NULLs at bind site); natural-z jit built, bit-exact, slow (jit-TU interleave 5× anomaly, unexplained) → closed. fwd_z route = native always. Packed rfft fwd jit: dead code (no PACKED-layout r2c create sites). c2r jit: newly active via rsp, measured PARITY (4 cells, ±3% band) — left bound, revisit flag. See §6a27 addendum.
- rfft.h:523 UB warning: RESOLVED (bycatch of mid zo restructure).

## §6a28 session close
- c2r native z-in: SHIPPED (chunked zscr deinterleave + same nat_init, 12/12 BIT first attempt). 1D z contract complete both directions. bwd z/MKL: parity at (2000,4), 1.23x at (1000,8), −10.9% structural win at 100K. Micro-cell floor noted (200,4).
- PACKED-input z entry: latent split-planes bug fixed (proper CCE->packed pack).

## §6a29 session close
- 2D z contract: SHIPPED v1 (convert-around; pre-existing SEGFAULT on the 2D z sentinel fixed). Gate 18/18 (2D cells added). Tax +13-18%.
- **CORRECTED (§6a29 addendum): no regression — c2c-2D still ties/beats MKL on the v1.0 config today (1.03×). Real debt, precisely: r2c-2D harvests only 1.31× real-vs-complex where MKL harvests 2.24×. Phase table in hand (wrapper 14.7%, pack 9%, inner-r2c 37.3%, core 0.79× vs MKL-real). Campaign order: wrapper elimination -> pack fusion -> inner-r2c.**
- Native 2D z (c2c z-store): recorded, parked behind the 2D perf campaign.

## §6a30 session close
- 2D wrapper elimination SHIPPED: copy-free OOP-native executors, split −15/−16% fwd, −9/−11% bwd; ratios 0.60→0.72× fwd vs MKL-real.
- Fused 2D z SHIPPED (the parked native-2D-z, free via the pack/unpack loops): **2D z tax ELIMINATED (parity with split)**; §6a29 z2tmp convert machinery removed same-day. Gate 18/18.
- Harvest ratio now 1.60× vs MKL's 2.24×. Next slice: inner-r2c (43.3% of total). f2dprof needs p2/p3 probes on the OOP path.

## §6a31 session close
- 2D row-pass inner: MEASURED engine selection shipped (rfft vs stride, A/B at create). 256² fwd 332→262µs cumulative (−21%), split/MKL 0.615→0.783×; 512² auto-keeps stride (+66% regression caught by the gate). Gate 18/18.
- B-sweep: eliminated (default 8 optimal). Harvest 1.72× vs MKL 2.24×.
- Next slices: bwd row inner (c2r mirror), col-c2c (22%), transposes (~24%). Open: (512,8) rfft-slower mechanism.

## §6a32 session close
- bwd row inner (c2r mirror): machinery + measured gate SHIPPED; adopts NOWHERE on this container (c2r natural slower than stride bwd inner at all tile shapes) — negative-for-now, self-enabling elsewhere. Gate 18/18.
- fwd engine win re-quantified same-process: −3.2% at 256² (not the −7% cross-run figure). A/B gates hardened (per-rep refill, inf hygiene).
- Weather-ghost #2 recorded (regime shift 23:00+; "bwd −31%" debunked by forced arms).
- 2D campaign remaining: col-c2c (~22%), transposes (~24%), (512,8) rfft-slower mechanism, bwd-vs-fwd stride-inner asymmetry.

## §6a33 session close — 2D campaign checkpoint
- Slicing campaign CLOSED at a principled boundary: every phase on best machinery (transposes = engineered SIMD kernels, col = jit-bound, inner = measured-selected, wrapper/pack = eliminated/fused).
- **Compute phases (~173µs) already beat MKL-real's total (205µs) at 256²; the ~101µs of mandatory movement (transposes+pack) IS the gap.**
- 2D v2 direction recorded: movement-free composition via strided-lane leaf/terminator codelet families (generator work, Gap-A-adjacent). Multi-session; parked designed.
- Queue: 2D v2 design, DIF bwd jit residual, Gap-A generator mode, c2r tail guard, n1 profiling, TLB/hugetlbfs, open mechanism studies.

## §6a34 session close
- v2 DESIGN SHIPPED: docs/roadmap/fft2d_v2_design.md (block-transposing codelet IO, spiked, staged, budgeted ~0.92-0.95x MKL-real). Stage 1 CORRECTED (see 6a34 addendum): v2 = extend the strided quadrant (strided r2c mono emission + DIF-front twiddle-stage codelets). Prior art: strided_rows_case_study.md governs. Stage-1 probe DONE (§6a35): mechanism validated (mono 31.9µs beats v1 composition 79.7µs at R=4096,N=16); wrapper composition rejected (copy + epilogue = bandwidth passes, the v2 disease in miniature). EMISSION SPEC FORCED: --strided-r2c mode = c2c strided body + OOP IO + in-register conj-split fused at store lattice (two-for-one internal). Projected −55%. Next: the generator mode itself (OCaml), then bwd, then twiddle-stage for N2>=128.
- Bycatch: skinny-transpose 8x4 fast path (standalone −35% at tile shape, guarded shape+size); adoption-gate hysteresis (>5% margin, weather-flip churn eliminated). Gate 18/18.
- Weather regime #3 logged; all §6a29-34 absolute µs are regime-stamped — profile SHARES and same-process deltas are the durable currency.

## §6a36 session close
- HAND-WRITTEN fused strided r2c codelet SHIPPED (r16_r2c_fwd_strided.c = the emission reference): −51.7%/−44.9% vs v1 tiled composition at R=256/4096, gate 2e-14. me=PAIRS, out_stride>=9.
- --strided-r2c OCaml mode now a transcription task (acceptance = emitted ≡ reference modulo naming). Then: bwd mirror, N coverage {4..64}, twiddle-stage family for N2>=128.

## §6a37 session close
- --strided-r2c OCaml mode SHIPPED: emitted BIT-IDENTICAL to the hand reference, −53.5% same-process at R=4096. Coverage {8,12,16,20,32,64} emitted + gated (r16 BIT, rest vs naive DFT ~4e-14). Hand reference superseded/removed.
- Next: bwd mirror (merge-prologue shape), 2D row-pass integration (§6a31 gates), twiddle-stage family for N2>=128.

## §6a38 session close
- --strided-r2c --bwd SHIPPED: merge-prologue emission, store lattice untouched, roundtrip 4.4e-16 (r16, R=256), coverage {8,12,16,20,32,64} installed. Family COMPLETE both directions.
- Next: 2D row-pass integration (both directions, §6a31 gates + hysteresis), then twiddle-stage for N2>=128.

## §6a39 session close
- Strided r2c/c2r INTEGRATED into the 2D row pass (fft2d_r2c.h-local: resolver + measured adoption w/ hysteresis + whole-pass replacement branches). Gate 21/21 (covered cell 64x64 added).
- **(256x32): split/MKL 0.801 -> 1.004x — FIRST 2D real cell at MKL parity.** (4096x64): fwd −29.8%, 0.632 -> 0.901x. z inherits automatically.
- CONTAINER: link TAIL now needs /tmp/osr2c/*.o (strided r2c objects) FIRST.
- v2 remaining: twiddle-stage family (N2>=128) — the last piece before the 256²/512² campaign cells.

## §6a40 session close (design study)
- Twiddle-stage composition law MEASURED: multi-sweep loses at DRAM scale (pass-count tax); row-blocked fusion (front+monos per 8-row block, one DRAM pass) wins both regimes (−9% @R=4096, −22% @R=256 vs tiled, c2c N=128, zero new emission). probe_dif_front.c.
- Emission target: fused strided twiddle-stage codelet (front + sub-FFTs one body per row-block). Open design: r2c edition (Sorensen real front + r2c/c2c mono leaves) + half-spectrum ordering/assembly map.

## §6a41 session close
- Twiddle-stage engine SHIPPED: strided_tw.h (r2/r4 fronts fwd+bwd, DIF map, mapped split/merge, row-blocked compositions N2 128/256), integrated in fft2d_r2c.h. Gate 24/24 (new cell 64x256).
- GATE-FIDELITY FIX: stw adoption A/B now runs FULL executors (isolated row-pass A/B misadopted at 256²: create said >5% better, execute said +15% worse). Correctly dormant now: ties everywhere (+1.3% @256², −0.4% @128²).
- VERDICT: hand composition ceiling reached; FUSED emitted tw codelet required for campaign cells. Spec sharpened: split-before-map (vector split on lane vectors, map in store addressing). Integration bed ready.
- GUIDE: docs/design/strided_codelet_families.md (4 families, ABIs, decision table).
- CONTAINER: /tmp/osr2c now also holds r64_c2c_{fwd,bwd}.o (needed by strided_tw.h links).

## §6a42 session close
- THE FUSED FAMILY ALREADY EXISTED: gen_radix {128,256} --strided-r2c emits monolithic large-N monos (ceiling was convention). Zero new OCaml. Gates: e-13 naive, e-16 roundtrip, 2D 24/24.
- 256²: bwd −7.3% ADOPTED (fwd +2.5% tie, declined). Campaign: split/MKL fwd 0.872x, bwd 0.974x. Files installed r{128,256}_n1_{fwd,bwd}_strided_r2c.c; /tmp/osr2c holds their .o.
- Remaining fwd gap = monolithic register pressure: try regalloc/pinning gate extension, GH pressure mode, or CT-blocked strided construction. N2=512 emission untested.
- Guide updated: family 4 = family 2's mode at N>=128.

## §6a43 session close
- M-knobs measured out at R=256: PIN +14.4%, FENCE +6.8% (retirement extended; provenance gate-string is stale text — the real policy is the 2026-06-09 M-note).
- N2=512 family SHIPPED: emitted, gated (2.9e-12 naive / 1.0e-15 rt), installed, resolver-wired, gate cell (64,512). Gate 27/27. Coverage now all power-of-2 campaign rows.
- NEXT: 512² whole-cell bench (long create — own timeout budget; expect bwd adoption per the 256² pattern); bench_f2d_profile refresh at 256² to locate the remaining fwd 0.87x deficit.

## §6a43 addendum
- GH measured out at R=256 (user-corrected): BIT-identical, −0.9% no-op. Knob ledger complete: PIN +14.4 / FENCE +6.8 / GH −0.9. fwd mono = schedule-knob-exhausted. VFFT_NO_GH env guard added (gen_main). Both provenance claims (regalloc gate string, GH +4-8%) documented stale for R>=128.

## §6a44: strided MT SHIPPED. Range-split wrappers (BIT-invariant, 4-pair-masked chunks), executors + adoption arms MT-faithful, stw stays ST. Gate: T∈{1,2,4} BIT + 27/27. Speed claim pending multicore host.

## §6a45 session close
- avx512 8x8 strided-r2c SHIPPED: {8,16,32,64,128,256,512} fwd+bwd, 14/14 gates, BIT-identical to avx2, r256 −9.0% isolated / −9.3% in context.
- FLAGS DOCTRINE: avx512 codelet objects need -mavx512f -mavx512dq -mbmi2 (target attrs are f-only tree-wide). il/avx512 has duplicate filename generations — per-symbol picks only, never bulk-compile.
- stw = fallback-only (gated on mono non-coverage) — avx512-build misadoption class closed structurally.
- Campaign avx512 build: (256,32) 1.097x BEATS MKL (fwd −45.6/bwd −31.2); 256² fwd −9.3 bwd −15.1 both MONO, 0.853x this regime. Gates: avx2 27/27 + avx512 27/27.
- Container: /tmp/ox512 (462 obj) + /tmp/osr512 (14) required in avx512-build link tails.
- NEXT: 512² bench (own budget); profile refresh 256² (fwd deficit now smaller — relocate); wisdom-persist; tails; queue unchanged.

## Q0 done: avx512 pairs%8 hazard (heap overrun at me=20, demo-proven) fixed via pairs-aware resolvers + 8-pair MT mask. Gate cell (40,32). Both builds 28/28.

## Q1 done: 3D REAL TRANSFORMS SHIPPED. Dark fftnd_r2c.h engine dispatched (dims==3 R2C/C2R, OOP, NL even); phases bridged in vfft_execute; z via il_out flip; strided engines + adoption inside; per-axis naturalization ported (empirical impulse detection, fail-safe) fixing the latent scrambled-axis bug (N=9 found by first gating). Gates: ND ALL PASS x2 builds; 2D 28/28 x2. Files: vfft.c, fftnd_r2c.h, benches/gate_fndr_q1.c.
## Q1 follow-ups queued: natorder sweeps MT-ization; dims>3 config cap lift; ND MKL campaign bench; wisdom for ND adoption.

## Q2 done: row tails (rows>=8, any count, both editions; Q0 constraint retired — resolvers always prefer avx512). Findings: prime-N1 2D col coverage broken pre-existing (cold-path WRONG VALUES = fail-safe violation, must-fix); STRIDE_ALIGNED_ALLOC rounds now.
## Q3 done: adopt_wisdom.h sidecar (VFFT_ADOPT_WISDOM_DIR — own env; VFFT_WISDOM_DIR empty-dir = 31-46s calibration trap documented). Warm creates skip A/Bs, decision-match gated. benches/gate_adopt_wisdom.c.
## Queue: Q4 batch verify; Q5 512² bench; prime-N1 col must-fix; natorder MT; profile refresh.

## Q2+Q3 done.
- Q2: tail-capable rows wrappers (odd rows via zero-partner two-for-one); resolvers -> (fn,blk); Q0 constraint retired; eligibility rows>=8 both 2D/ND; STRIDE_ALIGNED_ALLOC rounds size. FILED: prime-N1 2D col gap (warm create NULL / cold WRONG VALUES rt~1.0 — fail-safe violation, must fix). Gates: 4 matrices ALL PASS; new cells (27,32),(44,64),(9,128),(3,9,32).
- Q3: adopt_wisdom.h sidecar (VFFT_ADOPT_WISDOM_DIR — own env; VFFT_WISDOM_DIR empty-dir triggers 31-47s bundle calibration, decoupled). Cold records/warm skips; decision-match gate PASS; 2d+nd keys persist. benches/gate_adopt_wisdom.c.
- Remaining queue: Q4 batch verify; Q5 512² bench; prime-N1 col fix; natorder MT; profile refresh.

## Q4 done: dims==2 K!=1 rejected (hazard was live: create succeeded, executors K-blind). Gate line added; 2D avx2 ALL PASS. TODO: run remaining 3 gate matrices on this change (trivial early-return, all cells K=1); sequential-batch 2D/3D = designed feature, queued.

## Old-debt #1 DONE (§6a51): prime-N1 silent-wrong closed. Perm was blind digit-reversal from factors; create now impulse-verifies col+perm (production call, closed form) + row inner; mismatch => NULL + stderr tag. Cold gate (fresh-process 41,32 => NULL; 27,32 => e-15) both builds; 4 matrices ALL PASS. benches/gate_cold_prime.c. Real prime support (Bluestein/Rader) = designed feature. Next debt: DIF-bwd jit 8.7% + n1 profiling.

## Old-debt #2 DONE (§6a52): DIF-bwd jit residual INVERTED — jit now beats core (full −8.3%, ilin pair −3.6%, both BIT). No production gate excludes it; the §6a21 "rule retained" entry was stale. Zero code changes. Benches kept. Next debt: #3 n1 profiling (R=8/16/32) + the jit-TU interleave 5× mechanism.

## Gap-A DONE (§6a53): post-tw OOP family {5,10,20,25}×{avx2,avx512} emitted (+--post-tw generator mode, width-aware postamble), installed over the mislabeled pre-tw files, wired at both r2c fwd sites (log3 fuses free — variant-independent). Measured MIXED (−9.3%..+8.2%) ⇒ DEFAULT OFF, opt-in VFFT_DIF_FUSED=1; per-plan measured adoption = named follow-up. Gates: 16 standalone + 11-cell tail gate ×2 builds + 4 matrices ALL PASS. Files: codelet_oop.ml, gen_main.ml, 8 codelet files, r2c.h, gate_r2c_tail.c, bench_dif_fused_ab.c.

## §6a54 done: pad-to-8 port. K_pad roundup4→8 at 4 sites (2D r2c/c2r planners, wisdom, fftnd). avx512 col passes + ALL ND axis passes now full-width (Kc products include K_pad). 1D was already compliant (block_K | K, mult-8) — §6a53 codelets never tail in production. 4 matrices ALL PASS. Deferred: pad-width knob for a same-process perf A/B; strided-mono generator tails (pair-epilogue, no-mask) if direct-call safety ever needed.

## §6a55 done: IL padded arm (1D c2c z). Kp work buffers + aligned-chain cplan_il + jit tier + boundary helpers, first-execute-lazy, VFFT_IL_PAD forced arms. Gate ALL PASS (BIT where chains match, sortmag else; chain-defined z-order contract note). Perf mixed (-2.4%..+86% — the +86% = uncalibrated (1000,16) cold chain {25,20,2}); exec_me auto-engage removed (cross-context, §6a41). Follow-up: IL-specific verdict A/B that also calibrates the Kp cell. Files: vfft.c, benches/gate_il_pad.c.

## Target A CLOSED NEGATIVE (§6a56): compiler already vectorizes the IL converts (hand-avx2 only -3.4%, bandwidth-bound, BIT-proven). Convert cost is structural -> folded into Target C: MT convert-split + il2il MT gate lift. Bench kept: benches/bench_il_convert_vec.c.

## §6a57: Target A applied for compiler independence. _vfft_z_dein/inter intrinsic converts (avx512 permutex2var / avx2 unpack+perm / C floor, scalar epilogue, no masks), unified into fallback + §6a55 pad helpers. BIT residue sweep + nat-z fallback cells + 4 matrices ALL PASS both builds. IL remaining: Target C (MT convert-split + il2il MT gate).

## §6a58 done: Target C. il2il MT gate LIFTED (lane-slab dispatch, _c2c_mt pattern, pre-flighted resolves, mt_unsafe->fallback); fallback converts slabbed (C1, barriered, NK>=4096). MT-vs-ST BIT=0 fwd+bwd at 6 cells incl ragged + nat fallback, both builds; 4 matrices ALL PASS. No speedup claims (1 vCPU). IL arc COMPLETE: A(§6a56/57) B(§6a55) C(§6a58). Files: vfft.c, benches/gate_il_mt.c.

## §6a59 done: IL per-cell verdict. v7 wisdom field il_me; _il_ab_race at first execute (both arms, private scratch, alternating med9, 3% hysteresis to fused, winner roundtrip-gated, stamp+reuse proven, (1000,12) hazard self-resolves to K). Gate + 4 matrices ALL PASS both builds. IL tail arc COMPLETE and structurally identical to split's lifecycle. Files: wisdom_reader.h, vfft.c, gate_il_pad.c, il_padding_tail_handling.md.

## §6a60: ND partial-tile port. Measurement INVERTED the premise (fullB +61..+819% at small this_B, wins only at B-1) — fftnd was already right; shipped the measured guard (B-this_B<=1 -> fullB) at fftnd.h+fft3d.h tile sites; fftnd:173 fused-group = S3-by-necessity (in-place, no slack). fft2d is the actual laggard (always-fullB waste, ~+4% row-pass worst) — width-plumbing follow-up filed. 4 matrices ALL PASS. Bench: benches/bench_tile_partial.c.

## §6a61: featureset parity sweep (benches/sweep_featureset.c, 80 cells) — zero wrong-number cells; the ONE crash class (c2c z dims>=2, unwired NULL-im) FIXED via convert-around (§6a57 primitives + split engines + natorder on halves); header :91 un-staled. Filed: dims==4 exposure (engine is rank-general, dispatch stops at 3); real-side natural contract; ND howmany>1. 4 matrices ALL PASS.

## §6a62: dims==4 exposed. Two guards widened + dims==4 create block (3D contracts mirrored; c2c via stride_plan_nd(4), real via stride_plan_nd_r2c(4)); h->N4 + plane extension makes §6a61 z convert-around 4D-correct free. Proof: 4D r2c vs naive 3.2e-14 (20 bins), c2c Parseval+rt e-15/16, sweep d=4 all OK incl T4, 4 matrices ALL PASS. Remaining parity gaps (uniform, filed): real natural order; howmany>1 beyond 1D. Files: vfft.c, benches/gate_4d.c.

## K=1 finding (measured, no code): lane-padding K=1->8 is DEAD (+206..+287%, = the full K8 batch cost — zeros compute like anything); the scalar tier is only ~2.7x off the batched ceiling. The REAL K=1 answer = four-step intra-transform vectorization: col pass strided at K=N2 internal lanes + twiddle + row pass at K=N1 — the §6a41 DORMANT twiddle-stage engine + deployed strided rows are the pieces; K=1 (scalar baseline) is its redemption context vs the 2D row-pass context that declined it. Prize: ~2.7x at N=1024/4096. Bench: benches/bench_k1_answer.c. Full design session.

## K=1 doc: docs/performance/k1_single_transform.md — the full record (scalar-tier gap 2.7x, padding dead +206..287%, BAILEY2=the remembered fused2 already answers K=1 at -33% on nat-OOP, capturing ~55% of the gap). Routing session queued: default-K1 + inplace-K1 -> BAILEY2 via a §6a59-style per-cell verdict; deeper: scrambled variant / 2-phase MT for the ceiling residue. Benches: bench_k1_answer.c, bench_k1_bailey.c.
