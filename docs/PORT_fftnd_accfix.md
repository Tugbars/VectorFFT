# PORT GUIDE — fftnd module + constant-quantization accuracy fix
(For transplanting into a pristine pre-session VectorFFT tree.)

## 1. The accuracy fix (constant quantization) — 3 modified files

Carriers (generator/lib/): **ir.ml, dft_recurse.ml, split_radix.ml**
(current copies also shipped standalone in the outputs folder; md5-verified
against the working tree).

Mechanism (full story: v1_0_results.md §2d): constants were quantized to 14
significant digits and the QUANTIZED value stored (`Ir.mk_const` %.13e
round-trip + `round_13` on the tan-factored FMA rotation inputs in
dft_recurse.ml / split_radix.ml). Fix: quantize only the dedup KEY, store the
first-seen full-precision value; the three math-layer sites feed full
precision through. Zero instruction changes — constants only.

Port procedure: drop the 3 files into generator/lib/, `dune build`, rerun your
standard gen_set regeneration (deterministic — regen diff was 0 on the
already-regenerated families). §2d regenerated 1074 codelets
(inplace/oop/strided/c2r x avx2/avx512); **rfft + trig trees were left for the
dev host** — regenerate them there and the fix flows through automatically.
Verification: accuracy_harness.c (benches/ + outputs) — expect the §2d table:
1.8–5.3 eps L2, MKL-parity class (1.3–1.7x), same O(sqrt(log N)) growth.

## 2. The fftnd module — 5 new headers, header-only

src/core/transforms/fftnd/: **fftnd.h, fftnd_planner.h, fftnd_wisdom.h,
fftnd_natorder.h, fftnd_r2c.h**

Companion headers it composes with (also new this cycle):
src/core/transforms/fft3d/**strided_rows.h** (the tiled B-row pass — required),
src/core/transforms/fft3d/**fft3d.h** (the 3-pass 3D; fftnd generalizes it),
src/core/transforms/conv/**conv.h** (optional — conv over any plan incl. fftnd),
src/core/transforms/conv/**il_layout.h** (only if you keep the IL converters).

How it works (documented in v1_0_results.md §2b–§2d + the header comments):
per-axis pass composition (axis-0 native at K=prod(trailing), middle axes as
per-plane native calls, last axis as the tiled strided-rows pass), sub-volume
fusion (all trailing passes execute while a block is cache-resident;
unfused axes become hierarchical work items), per-axis DP inner recipes, a
structural calibrator sweeping {split s x lane-block mode} roundtrip-gated
end-to-end, and T-keyed one-line wisdom:
  nd r=3 T=8 n=64,64,64 s=1 B=8 blk=0,0 ns=1.40e+06 ax0=T:4v0,4v2,4v2 ...
Natural order: transform stays scrambled (fusion contract); fftnd_natorder.h
provides per-axis maps via chain-free impulse phase probing + gather/scatter.
MT: windowed per-pass with a starved-grain mode (blocks < threads run
sequentially, full pool inside each block); wisdom is T-keyed because
s/blocking verdicts are thread-count-dependent.

Tests/benches to carry: test_fftnd_roundtrip.c, test_fftnd_mt.c,
test_fftnd_natorder.c, test_fftnd_r2c.c, test_fftnd_tails.c, bench_fftnd.c,
bench_fftnd_wisdom.c, test_conv.c (if conv), bench_fft3d_vs_mkl.c.
Known-good gates: roundtrips <=1.4e-14, sorted-|X| multiset vs MKL <=8.8e-15,
MT bit-EXACT T in {1,2,4,8}, conv 14/14 ~1e-14.

## 3. New .h files added to the project this development cycle

Core (the port set):
  src/core/transforms/fftnd/fftnd.h
  src/core/transforms/fftnd/fftnd_planner.h
  src/core/transforms/fftnd/fftnd_wisdom.h
  src/core/transforms/fftnd/fftnd_natorder.h
  src/core/transforms/fftnd/fftnd_r2c.h
  src/core/transforms/fft3d/fft3d.h
  src/core/transforms/fft3d/strided_rows.h
  src/core/transforms/conv/conv.h
IL campaign (skip unless wanted):
  src/core/transforms/conv/il_layout.h   (il2sp/sp2il SIMD converters)
  src/core/oop/il_execute.h              (stage-0 IL adapters, pow2 guard)

Modified files beyond the 3 accuracy carriers (all IL/emitter work — skip for
this port): generator/lib/{emit_state,emit_render,emit_c,gen_main}.ml,
codelets/il/** (new directory), tools/il_derive*.py,
generator/generated/spike_wisdom.txt (appended chain + @nat-era lines; your
pristine wisdom is authoritative), generator/generated/plan_executors.h
(regenerated — use your pristine copy).

Docs added: fft3d_design.md, fft4d_design.md, il_architecture.md,
interleaved_design.md, strided_rows_case_study.md, natorder_2d_status.md,
v1_0_results.md sections 2b/2c/2d (the 3D/4D/conv+accfix addenda, including
the 2026-07-14 methodology-correction notes — read those before trusting any
container-measured MKL ratio; the container free-dispatches MKL to AVX-512
against the AVX2 vfft build unless MKL_ENABLE_INSTRUCTIONS=AVX2 is forced).

## 4. The strided-codelet stack (used by 2D/3D/4D row passes)

Enabling modifications, by layer:
1. Generator `--strided` mode — gen_main.ml (flag), emit_c.ml (strided
   signature + in-register 8x8/4x4 transpose preamble/postamble, no scratch),
   emit_render.ml (Input(j) renders as preamble-bound lane_re_j/lane_im_j —
   the DAG/scheduler/regalloc pipeline unchanged). n1-only by design:
   multi-dim transforms carry no inter-axis twiddles.
2. Registry — bin/emit_strided_registry.ml ->
   generated/strided_registry_avx2.h / strided_registry_avx512.h
   (r8/16/32/64 x fwd/bwd dispatch).
3. Runtime — fft3d/strided_rows.h: tiled B-row pass over flattened last-axis
   rows (replaces gather/chain/scatter; ~19.7% structural transpose tax
   removed — see docs/performance/strided_rows_case_study.md), per-plan
   bit-exactness probe, native fallback for multi-stage row plans.
4. Wiring — fft3d.h native; fftnd.h rank-general (4D = r=4 same pass);
   fft2d.h RETROFITTED: opt-in include under VFFT_STRIDED_ROWS (line ~43) +
   pluggable _vfft_strided_fn srow_fwd/srow_bwd slots (NULL -> native).

Port routes: (a) USE ONLY — copy codelets/strided/** (already
accuracy-fix-regenerated per section 2d) + the two strided_registry headers +
strided_rows.h; no generator changes needed. (b) REGENERATE — additionally
port the gen_main/emit_c/emit_render strided modifications.

## 5. Merge recipe — integrating this tree into the pristine local master

Verdict (gated 2026-07-15): merge-clean. The default generator path is
byte-identical to shipped output with all session modifications in place
(r4/r16/r64 inplace both ISAs + r16 t1s: 7/7 diff-identical), so nothing
here perturbs what the master already trusts.

Steps:
1. ADDITIVE SET — copy wholesale (new files only): src/core/transforms/fftnd/,
   fft3d/{fft3d.h,strided_rows.h}, conv/{conv.h,il_layout.h},
   oop/il_execute.h, codelets/il/**, codelets/strided/**,
   generated/strided_registry_{avx2,avx512}.h, tools/il_derive*.py,
   generator/bin/emit_strided_registry.ml, build_tuned/benches/* (36 files),
   docs additions. (codelets/trig: verify against master — likely original.)
2. GENERATOR — replace 7 files in generator/lib/: ir.ml, dft_recurse.ml,
   split_radix.ml (accuracy fix) + gen_main.ml, emit_c.ml, emit_render.ml,
   emit_state.ml (strided + IL emitter modes; default path gated identical).
   dune build; optional sanity: regenerate any inplace codelet and diff vs
   shipped — expect byte-identical.
3. WISDOM — keep the master's spike_wisdom.txt. Optional appends (the full
   session delta, @nat untouched):
     64 4096 2 4 16 60000.0 0 0 0 0 0 2 0
     64 64 2 8 8 950.0 0 0 0 0 0 2 0
     64 4096 3 4 4 4 60000.0 0 0 0 0 0 2 2 0
     128 32 2 4 32 0.0 0 0 0 1 1 0 0
     128 64 2 8 16 0.0 0 0 0 0 0 2 0
     64 64 2 4 16 0.0 0 0 0 1 1 0 0
     (+ the (64,4096) [16,4] line if present in the tree copy)
4. plan_executors.h — keep the master's. To adopt new baked coverage later:
   emit_executor_h.exe --wisdom generated/spike_wisdom.txt >
   generated/plan_executors.h (deterministic; verify with the three-tier
   bit gate).
5. REGENERATE rfft + trig on the dev host (accuracy fix flows through), then
   run: accuracy_harness, test_conv (14/14), test_fftnd_{roundtrip,mt,
   natorder,r2c,tails}, test_fft3d_roundtrip, test_il_emit_composed +
   test_il_{n1x,twiddled}_composed if keeping IL.
6. bench_1d_vs_mkl.c exists only in the master (absent from this fork since
   upload; the CMake target expects it and will work post-merge).

Methodology reminder for any container benching: force
MKL_ENABLE_INSTRUCTIONS=AVX2 on AVX-512 hosts when the vfft build is AVX2,
and treat cross-session container ratios as unreliable (documented ~3x
session swings); same-process, same-window ratios only.

## 6. Reproduction proof (2026-07-15, canonical bench, this tree)

The master's bench_1d_vs_mkl.c (uploaded) built against this tree first-try
(zero missing symbols — full API surface intact) and, run with the master's
own spike_wisdom on the container (matched ISA, MKL_ENABLE_INSTRUCTIONS=AVX2,
isolated single-cell mode, both flip orders):

  (1024,4)  64x16/DIT    baked  14.1-14.2us  vs MKL 21us   -> 1.48-1.53x  err 8.4e-16
  (1024,32) 4x4x8x8/DIT  baked  93-100us     vs MKL 154-168us -> 1.65-1.69x  err 6.5e-16

Both inside the v1.0 pow2 win distribution (doc (1024,32) row: 1.33x on the
14900KF). Notable: the executors resolved BAKED — i.e. the regenerated
plan_executors.h served the canonical protocol at shipped-class ratios and
e-16 accuracy, which is its end-to-end validation. Wisdom compatibility:
the master's file drove cell selection and plans unmodified.
