# VectorFFT — Feature Ledger (2026-07-13 → 2026-07-18)

Every feature shipped across the campaign, in order — with what was edited and why.

## Pre-ledger era (July 13–15)

1. **3D/4D/rank-general FFT support (fftnd c2c)** — new fftnd engine (gather → axis c2c passes → scatter) so one machine covers any rank instead of per-rank code.
2. **Convolution API** — FFT-based conv entry points on top of the transforms; the first "consumer" proving the public surface.
3. **Accuracy harness + 29-ulp twiddle-constant fix** — systematic ULP comparison vs reference; found the OCaml generator quantizing twiddle constants, fixed at the emitter.
4. **Calibrator/wisdom system (T-keyed)** — measured plan selection persisted per (shape, thread-count) so repeat creates skip calibration.
5. **Full multithreading wiring** — pool-dispatch range splits across row/axis passes; ST fallbacks kept for BIT determinism.
6. **Natural-order maps** — natorder perm/cycle machinery so scrambled dag-convention outputs can be served in natural order when the API demands it.
7. **Rank-general r2c/c2r** — real-transform variants of the ND engine (pads, half-spectrum, pack/unpack).
8. **MKL benchmark-methodology correction** — split-complex vs CCE-interleaved storage was being conflated in comparisons; benches corrected so wins/losses are layout-honest.
9. **AVX-512 matched-ISA campaign + width-gain decomposition** — MKL pinned to matching ISA, gains decomposed into width vs scheduling so claims survive scrutiny.
10. **Strided row-pass deployment (2D/3D/4D)** — c2c row passes replaced gather/scatter tiling with single strided sweeps; adversarial tail shapes handled.
11. **Natural-order strided semantics** — the strided engines taught to honor natural-order contracts, not just scrambled.
12. **JIT wiring completion** — per-plan baked/compiled executors resolved at create; generic loop kept as fallback.
13. **IL P1a wrapper + derived IL codelets (il_derive.py)** — interleaved-layout boundary handled first by wrappers, then by mechanically source-transformed codelet twins.
14. **Inplace-donor re-derivation + chained-DIT adapter (il_execute.h)** — wrong donor (3× slower OOP) replaced with inplace donors; adapter chains DIT stages both directions.
15. **Emitter --strided-il-out + NT-store ls_mode** — OCaml emitter (emit_c/gen_main/emit_state) folds interleave into the strided back-transpose, with non-temporal store variants.
16. **Four IL-tax items** — per-slab fusion, inplace boundary fold, baked/JIT resume, strided il_in fold: each a measured elimination of one interleave-boundary cost.
17. **1D-vs-MKL native-interleaved benches + fat-plan falsification** — canonical comparison harness; the "fat plan" hypothesis tested and killed by measurement.
18. **DIF/twiddled IL codelet matrix (432 gates)** — the full variant×radix×direction IL codelet set generated and bit-gated.
19. **Geometry-contracts docs + port/merge guide** — the measurement doctrine and layout contracts written down so later work argues from record, not memory.

## Ledger era (§6a16–§6a53)

20. **6a16 — Adapter wiring: boundary folds + il2il orchestrators** — il_execute.h wires every gated boundary-fold codelet (adapters were absent); intent: variant-aware IL execution end to end.
21. **6a17 — jit stop-gate + fused t1s bwd entry** — emit_jit.py emits two symbols per plan TU (ver 3→4); runtime jit proven in-container with a stop-gate for partial ranges.
22. **6a18 — bwd_oop_jit** — oop_execute.h gains the pointer-swap identity IDFT(re,im)=swap(DFT(im,re)) over the JIT forward; restores OOP split→split bwd symmetry.
23. **6a19 — Public API convergence** — interleaved z contract + fft3d wisdom + dims=3, all by extending existing entry points' domains rather than adding functions.
24. **6a20 — r2c packing-tax bench** — attribution bench pinning where the r2c pack costs live; produced the design doc that later became Gap-A's spec.
25. **6a21 — Variant-bound DIF bwd emission (jit ver5)** — emit_jit maps DIF bwd stages per variant (LOG3 twin macros); intent: jit coverage for DIF plans.
26. **6a22 — Wisdom free/save parity** — vfft_wisdom_free/save extended to the full bundle set (+2D/3D tables, +bluestein); no leaks, no silent partial saves.
27. **6a23 — Fused-first coverage: B-tail closed** — stale (B&3)==0 guards removed at both fused call sites; DIF fusion formally spec'd for later.
28. **6a24 — Interleaved-z public contract + native z terminators** — NULL-halves convention (dim==NULL ⇒ CCE spectrum in dre) made the public rule; native terminators begun.
29. **6a25 — Model-B terminator: closed negative** — fused last-stage machinery built, measured, verdict negative; left dormant in-tree with the record.
30. **6a26 — rfft native z terminator (fwd)** — RFFT z path runs a native interleaved stage-0 terminator (chunked hcnr) instead of convert-around; BIT-exact, +19–23% tax gone.
31. **6a27 — rfft PIC set + natural-jit verdict** — 115 rfft/c2r codelets compiled -fPIC into the rsp (94→209 lines); natural-jit idea measured and closed negative.
32. **6a28 — c2r native z-in initiator** — the §6a26 bwd mirror: CCE input fed natively into stage-0; the 1D z contract complete both directions.
33. **6a29 — 2D z contract (v1)** — fixed a SEGFAULT (z sentinel dereferenced as split) with a convert-around v1; surfaced the 2D-vs-MKL gap that launched the campaign.
34. **6a30 — 2D wrapper elimination + fused z** — copy-free OOP entries; the full-plane z conversion tax eliminated at the 2D layer.
35. **6a31 — 2D row pass: measured inner selection** — rfft vs stride inner raced per-plan off the phase table (inner was 40–44% of runtime); winner adopted per shape.
36. **6a32 — bwd row inner (c2r mirror)** — same injection for C2R; machinery ships even though it adopts nowhere measured — honesty over pruning.
37. **6a33 — Campaign checkpoint** — analysis only: compute already beats MKL, data MOVEMENT is the whole remaining gap; set v2's thesis.
38. **6a34 — v2 spike + two bycatches** — fused-IO feasibility spike validated; skinny-transpose and gate-hysteresis v1 bugs caught and fixed along the way.
39. **6a35 — v2 stage-1 probe** — strided r2c composed from the existing c2c strided mono via the two-for-one trick; mechanism validated, emission spec forced.
40. **6a36 — Hand-written fused strided r2c codelet** — r16_r2c_fwd_strided.c as the emission reference; proves the spec before automating it.
41. **6a37 — --strided-r2c generator mode** — OCaml mode (emit_state ref, gen_main flag, emit_c split-postamble) emitting codelets BIT-identical to the hand reference.
42. **6a38 — --strided-r2c --bwd (c2r mirror)** — store lattice needed zero changes (c2c bwd Re/Im lanes ARE the even/odd rows); roundtrip at machine epsilon.
43. **6a39 — Strided family integrated into the 2D row pass** — fft2d_r2c.h resolver + measured adoption arms; first MKL-parity at a covered cell (256,32).
44. **6a40 — Twiddle-stage design study** — hand DIF front probe measuring the composition law before committing to an engine.
45. **6a41 — Twiddle-stage engine + gate-fidelity lesson** — strided_tw.h shipped correct but dormant (adoption declined it); gates must mirror production context — the lesson that reshaped later A/Bs.
46. **6a42 — Large-N monos already existed** — test-before-build: gen_radix at 128/256 just worked; bwd −7.3% adopted at 256².
47. **6a43 — M-knobs measured out; N2=512** — PIN/FENCE/GH knobs all measured dead at R=256; family extended to 512 with the knob ledger recorded.
48. **6a44 — Strided MT** — _run wrappers mirror the pool-dispatch pattern with block-aligned chunks; MT BIT-identical to ST by construction.
49. **6a45 — avx512 8×8 editions** — emit_c width-8 split/merge branches from the tree's own lattice vocabulary; r256 −9.0% BIT-identical, (256,32) beats MKL 1.097×.
50. **6a46/Q0 — avx512 pairs%8 hazard** — heap overrun demo-proven at me=20; pairs-aware resolvers + 8-pair MT mask; gate cell (40,32).
51. **6a47/Q1 — 3D real transforms lit up** — dark fftnd_r2c.h engine dispatched (vfft.c dims==3 R2C/C2R), phases bridged in execute, §6a24 z via il_out flip, strided engines + adoption inside from birth.
52. **6a47b — Empirical per-axis naturalization** — first-ever gating caught the scrambled-axis bug (N=9); impulse-probe detection + cycle passes, fail-safe build-NULL on anomaly.
53. **6a48/Q2 — Row tails** — rows-based tail wrappers (staged remainder; odd rows free via zero-partner two-for-one); resolvers → (fn, blk); Q0 constraint retired.
54. **6a48 — STRIDE_ALIGNED_ALLOC rounding** — macro rounds size to alignment; tree-wide ASAN-strict conformance found during the tail debugging.
55. **6a49/Q3 — Adoption wisdom** — adopt_wisdom.h sidecar under its OWN env (VFFT_ADOPT_WISDOM_DIR — sharing VFFT_WISDOM_DIR triggered 31–47 s bundle calibration); warm creates skip the A/Bs.
56. **6a50/Q4 — 2D howmany hole** — dims==2 stored K but executors were K-blind (silent-wrong, demo-proven); create now rejects K≠1, matching 3D.
57. **6a51 — Prime-N1 fail-safe** — root cause: pack perm computed blind from factors; create now impulse-verifies col+perm through the production call (+ row-inner probe); mismatch ⇒ NULL, never wrong values.
58. **6a52 — DIF-bwd jit residual closed** — measured today: jit BEATS core (full −8.3%, ilin pair −3.6%, BIT-identical); the §6a21 "rule retained" entry was stale — zero code changes, benches preserved.
59. **6a53 — Gap-A: post-tw OOP family + fused DIF entry** — --post-tw generator mode (pure-DFT body + width-aware cmul postamble), {5,10,20,25}×{avx2,avx512} installed over the mislabeled pre-tw files; wired at both r2c fwd sites (log3 fuses free — variant-independent); measured mixed (−9.3%…+8.2%) ⇒ default OFF, VFFT_DIF_FUSED opt-in, per-plan adoption queued.
