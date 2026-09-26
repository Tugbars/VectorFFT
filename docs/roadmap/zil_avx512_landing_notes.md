# zil AVX-512 landing notes (commit `efb8deb`, 2026-09-26)

Summary of what landed when the avx512 zil codelets entered the corpus. Details and
evidence: `zil_avx512_design.md` §11.8.

## What's in the tree

- `codelets/zil/avx512/`: **716 codelets** — shared 62, shared/col 110,
  shared/col/blocked 12, flat 198, flat/odd_mid 15, pair2p 161, pair2p/blocked 12,
  rows 82, chain3 27, ztt 35, mono 2.
- Registries: new `generated/il_registry_avx512.h` and `generated/ztt_registry_avx512.h`
  (159 cells).
- Fused ZTT drivers: the 30 avx512 drivers are in `generated/fused_codelets/avx512/`, a
  subfolder so the avx2 builds never compile them.
- Not generated:
  - t0tp and tld at radix 4 (4 cells) cannot exist at this width; the corpus records the
    reason (`Corpus.zil_isa_gap`).
  - The 20 kernels that exist only as env-knob or sed-rename recipes (pair2p/tangent and
    five pair2p/blocked kinds) are not in the corpus, so they have no avx512 version yet
    (decision D3).

## Generator

The investigators' prototypes are merged into `generator/`; the merge needed a few
conflict resolutions and one test update (the width-generic turn replaced the IR's
`CLo`/`CHi` with `CPart`). **AVX2 output is byte-identical**: `gen_set all` produces the
same 1,867 files from the old and the new generator. The only avx2 registry changes are
three added macros in `il_registry_avx2.h` (`VFFT_IL_ISA_NAME`, `VFFT_IL_VW`,
`VFFT_IL_SYM`) and the stale line 6 of `ztt_registry_avx2.h`.

## Verification on the AVX-512 host

- **535 widened kinds:** 516 match their AVX2 twins bit for bit; 19 blocked forms are
  within 2.3e-16, expected because they use a different construction in the tail.
  0 failures.
- **137 turned kinds:** identical to the set that passed the bitwise comparison with
  AVX2 (counts 1–13, 4 buffer offsets, guard page).
- **ZTT and msz:** identical to the set that passed 7,832 end-to-end checks.
- **Compilation:** all 716 compile with the build's flags (`-mavx512f -mavx512dq
  -mfma`), and the whole avx512 codelet library (1,269 codelets) builds. *Correction:*
  an earlier version said they compile with only their own target attribute; 116 do
  not (their static helper bodies carry no attribute — the same gap exists in the AVX2
  tree). See `docs/design/avx512_tail_handling.md` §6.

## Corpus ceremony

`full_corpus_gate.sh manifest`, `record`, then `verify` → **GATE PASS**.

- `recipes.tsv` gained `genset` rows for the 1,258 files that had none (the 716 new ones
  and 542 avx2 zil files that were never recorded); two stale mono rows were removed.
- 2,511 of 2,620 files byte-identical; every avx512 folder at 100%.
- **Please review:** 35 avx2 files went from IDENTICAL in the old baseline to
  BODY_DIFFERS. All 35 were already diverging under the *unchanged* generator — the
  radix-3/5 FMA-fold drift (D0). They are pinned as they are, not regenerated.

## Defaults chosen where decisions are still open

Each is a one-line change.

- **D1, tail:** *decided since:* `ladder_m3` (xmm for 1 leftover, ymm for 2, masked zmm
  for 3), written as L10 in `policy.h`; the study is `docs/design/avx512_tail_handling.md`.
  The turned kinds keep the per-column xmm tail with the lane fix.
- **D2, uarch:** `sapphire_rapids_avx512`.
- **D4, target attribute:** zil-only `avx512f,avx512dq,avx512vl,fma`.
- **D5, fused driver layout:** a per-ISA subfolder.

## Not done yet

A working `VFFT_ISA=avx512` library. That needs the runtime work (§10 stage 3):
per-ISA registry selection, 16-double twiddle tables, and the ZTT laws, permutation and
threading grain.

### Stage 3, step 1: done (per-ISA kernel names and registries)

- `src/core/support/build_isa.h` is the one ISA test in the core. `vfft_isa()` (via
  `env.h`) and the IL family both read it.
- `src/core/oop/il_isa.h` includes `il_registry_<isa>.h` and names the ZTT registry for
  the build's ISA; the runtime names kernels through `VFFT_IL_SYM(stem)`. The 20 kernels
  that exist only as avx2 recipes (D3) go through `VFFT_IL_AVX2_ONLY(stem)`, which is 0 at
  avx512: absent, never an avx2 fallback.
- The IL solo resolvers in `oop_leaf_registry.h` are no longer gated to avx2.
- The avx512 fused ZTT drivers are built (CMake and both `build.py`).
- The two 2D strided row paths that fell back to avx2 kernels (r2c N2 = 12/20, and the
  `strided_tw.h` tier) are absent at avx512.
- **The whole avx512 build links** (library and every gauntlet program), for the first time.
- **AVX2 is unchanged:** all 1,370 objects, both libraries and the four gauntlet
  executables are byte-identical to the pre-change build.
- **Results at avx512 are still wrong** on every route that uses twiddle tables (pair,
  chain3, flat, ZTT: relative error ~1). Twiddle-free routes are correct (the N = 8
  solo). That is step 2: the runtime still builds AVX2-shaped tables. Smoke test:
  `zil_avx512_prototypes/harness/runtime/api_smoke.c`.

## For the owner

- Generation here used OCaml 4.14, not your 5.2 (D12). Before relying on the recorded
  baseline, rerun `full_corpus_gate.sh verify` on WSL.
- `gauntlet/CMakeLists.txt` was committed earlier on this branch; move your untracked
  local copy aside before pulling, or git will refuse to overwrite it.
