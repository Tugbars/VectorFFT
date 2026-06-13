# c2r session handoff — what was done, where it stands

Goal of this work: build the **fast rfft-native cascade c2r** (the inverse real
FFT that mirrors the forward r2c path which beats MKL ~1.2-1.4x). A working but
SLOW c2r already exists (the stride r2c.h backward path); this effort is the
performance upgrade, not a correctness gap-fill.

Status in one line: **the new construction (r2cb leaf) and the executor
foundation are built and gated; the multi-stage cascade executor is NOT done**
— it fails its gate due to a now-understood mixed im-packing geometry. The
session ended with the root cause pinned and the right tool built, but the
cascade itself unfinished.

---

## What was DONE and GATED (solid, trust these)

1. **r2cb leaf codelet — the one genuinely new construction.**
   The backward real leaf (halfcomplex in -> real out), inverse of r2cf.
   - `dft_expand_r2cb` in `generator/lib/dft_r2c.ml`
   - `r2cb_signature` emit block in `generator/lib/emit_c.ml` (distinct 6-arg
     ABI: in_re,in_im,out_re,is,os_re,vl — NOT r2cf's 7-arg real-input one)
   - `--r2cb` flag wired in `generator/lib/gen_main.ml`
   - GATED: brute forward rdft -> r2cb == N*x, all radices pass:
     r2=0, r4=9e-16, r5=4e-14, r7=4e-14, r8=2e-14, r16=7e-14 (odd radices too).
   - Two bugs found+fixed by the gate: (a) it inherited r2cf's real-input
     signature (compile error, fixed with dedicated signature); (b) the
     backward DFT of the Hermitian spectrum came out TIME-REVERSED under the
     library's Bwd sign convention — fixed by mapping output k <- result[(n-k)%n].

2. **c2r executor foundation (nf=1 leaf-only) — gated.**
   - `core/c2r.h`: c2r_plan_t (reuses rfft_plan_create for geometry/twiddles,
     re-points leaf->r2cb, stages->hc2hc_dif_bwd), c2r_execute_packed.
   - Added to `core/rfft.h`: rfft_r2cb_fn typedef + r2cb / hc2hc_dif_bwd[_log3]
     slots in rfft_codelets_t.
   - GATED: c2r nf=1 (N=8) -> 1.95e-14 PASS.

3. **DIT/DIF question answered (your question).**
   Forward r2c uses DIT; backward c2r uses DIF. CONFIRMED in our code: forward
   rfft wires the DIT hc2hc slots, never reads hc2hc_dif. This mirrors FFTW
   exactly (R2HC=apply_dit, HC2R=apply_dif). It is a correct DUALITY, not an
   inconsistency: DIT = twiddle-then-DFT (raw input), DIF = DFT-then-twiddle
   (already-transformed input). r2c being DIT is correct; DIF belongs to the
   inverse direction only.

4. **FFTW execution structure — web-confirmed** (source not in container):
   hc2r = r2hc reversed, unnormalized (r2hc->hc2r = N*x). Backward twiddle
   stage = `--hc2hc --dif --bwd --t1s` (verified: generates + compiles). The
   --t1s scalar-twiddle requirement (doc 60 gotcha) still applies.

---

## What is NOT done (the open item)

**The multi-stage c2r cascade executor.** Two hand-derivations both failed the
nf=2 gate (N=16=(2,8): err ~17, structurally wrong, not a sign/scale tweak).

### Root cause (pinned by instrumenting the forward executor)
The forward executor uses **MIXED imaginary-part packing conventions** across op
types. Forward move map (N=16=(2,8), offsets in K-units, from the trace tool):

```
leaf g=0: x[0] is=2 -> planeA re[0]  im[16]  os=2   (im at +NK region)
leaf g=1: x[1] is=2 -> planeA re[1]  im[17]  os=2
k0:       planeA re[0]          -> out re[0] im[16]  (im at +NK)
hc k=1:   planeA re[2] im[14]   -> out re[1] im[7]   (OUT im at MIRROR row m-k)
hc k=2:   planeA re[4] im[12]   -> out re[2] im[6]
hc k=3:   planeA re[6] im[10]   -> out re[3] im[5]
```

So leaf + DC(k0) write im to the **+NK region**; interior hc writes im to the
**in-plane mirror row (m-k)**. The final output is single-plane halfcomplex.
Both my hand-derivations assumed ONE uniform convention -> structural failure.

### The remaining gap
Even with the move map, the leaf->stage row correspondence is incomplete: the
stage reads planeA rows {2,4,6}/{14,12,10} but the leaf wrote rows {0,1}/{16,17}.
That mismatch is the S-group / Q-fold interleaving across the S leaf groups,
which a single-(q) move trace doesn't fully expose.

### The plan (next session)
1. Capture the COMPLETE multi-group + q-loop move map (all groups, not one).
2. Build the backward executor as an **inverse REPLAY** of that map (swap
   in<->out, swap is<->os, reverse order) — correct BY CONSTRUCTION, not by
   re-derivation. This is the key method change: stop reasoning about the
   packing, replay the measured forward moves inverted.
3. Implement the mid (k=m/2) column inverse (deferred so far).
4. Gate nf=2, then nf>=3, vs `dft_c2r_direct` (the monolithic correctness
   oracle — already correct, already in dft_r2c.ml).
5. Once gated: c2r coverage quadrant + auto-registry (the proven 6x pattern),
   then the MKL perf race (ratios in one process, doc 60 epistemics).

DO NOT claim c2r works until nf>=2 gates pass.

---

## Files (what each is, and its state)

| file | what it is | state |
|---|---|---|
| `generator/lib/dft_r2c.ml` | `dft_expand_r2cb` (new) + `dft_c2r_direct` (oracle, pre-existing) | r2cb DONE+gated |
| `generator/lib/emit_c.ml` | `r2cb_signature` emit block (new) | DONE |
| `generator/lib/gen_main.ml` | `--r2cb` flag wiring (new) | DONE |
| `core/rfft.h` | added rfft_r2cb_fn typedef + r2cb/hc2hc_dif_bwd[_log3] slots | DONE |
| `core/c2r.h` | c2r plan + executor; nf=1 path gated, **nf>=2 cascade FAILS** | nf=1 DONE, cascade OPEN |
| `core/rfft_trace.h` | instrumented forward executor (VFFT_RFFT_TRACE dual-base move recorder) | TOOLING, reusable |
| `benchmarks/trace_rfft_fwd_moves.c` | runs the forward trace, dumps the move map | TOOLING |
| `benchmarks/gate_c2r_nf1.c` | nf=1 leaf-only gate | PASSES |
| `benchmarks/gate_c2r_nf2.c` | nf=2 cascade gate | FAILS (expected, cascade open) |
| `docs/62_c2r_scope_from_proven_rfft.md` | full running design log (updates 1-4) | the detailed record |

## How to resume (mechanics)
- Restore: `tar -xzf /mnt/user-data/outputs/vectorfft-oop-engine.tar.gz`
  -> `/home/claude/vectorfft-oop-engine/`
- Toolchain (wiped on reset): `apt-get install -y ocaml-nox ocaml-dune`
- Build generator: `cd generator && dune build`
- Regenerate a codelet: `_build/default/bin/gen_radix.exe <R> --r2cb --su --emit-c --isa avx512`
- The oracle for gating: `dft_c2r_direct` (dft_r2c.ml) — monolithic, correct.
- Start from docs/62 UPDATE 4 and the move map above.
