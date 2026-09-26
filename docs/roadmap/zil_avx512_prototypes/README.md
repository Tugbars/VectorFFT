# zil AVX-512 prototypes (2026-09-26)

Reference patches from the multi-agent investigation recorded in
`../zil_avx512_design.md` §11. They were built and measured in scratch copies of
the generator on an AVX-512 host (Emerald Rapids VM) and are kept here so the
work survives the ephemeral container. **None of them is applied to the tree.**
Each applies cleanly, on its own, to commit `b8f91cc`; they overlap (several touch
`c2c_il.ml` / `cx_render.ml` / `isa.ml`), so they are inputs to the staged plan in
§10 of the design doc, not a patch series.

| file | apply from | what it proves |
|---|---|---|
| `turn_store.diff` | `generator/`, `patch -p1` | Width-generic corner-turn: the 4x4 complex transpose as two rounds of one ISA op (`permute2f128 0x20/0x31` at 256, `shuffle_f64x2 0x88/0xDD` at 512), `CTurn`/`CPart` in the cx IR, per-quarter stores, 4-group blocked pass-tuples, and the tail lane fix. AVX2 byte-identical over 732 recipes + 5 header-less files; all 141 turned recipes generate at AVX-512 and are bitwise equal to their AVX2 twins (counts 1..13, 3 strides, 4 base offsets, guard page). |
| `colstride_and_tail_lane_fix.patch` | `generator/`, `patch -p0` | The minimal correctness fix for the kinds that already emit at AVX-512: the column-stride gather/scatter at width 8 (`cx_render.ml`) and the narrow-tail twiddle lane offset (`c2c_il.ml`). 153/153 pair2p/rows twins correct, AVX2 150/150 byte-identical. |
| `tail_policy_*.patch` | `generator/`, `patch -p1` | Tail policy as a switch (`VFFT_TAIL512=ladder|masked|hyb2|narrowfix|narrow`), constant typing by lane count, and the width-parameterized k1 mono (`c2c_split.ml`: N=64 8x8 IL mono bitwise equal to AVX2, 45 ns vs 75 ns). AVX2 728/728 byte-identical; twin gate 535/535 under the ladder. |
| `zsplit_generator.diff` | `generator/`, `patch -p1` | Boundary-split / ZTURN-T at VW=8: `Isa` shuffle ops (deint/reint/transpose/const splat), the t0tp 8x8 lattice, the R%VW law, ISA-aware fused drivers and ZTT registry. 50 of 54 kernels emit (t0tp/tld at radix 4 cannot exist at VW=8); AVX2 kernels, 30 fused files and the registry body byte-identical. |
| `zsplit_ztt_h_runtime.diff` | repo root, `patch -p1` | `ztt.h` parameterized by VW (record fill, per-class laws, sigma_8 permutation). With it the AVX-512 kernels pass 404 plans / 7832 checks end to end, worst error 5e-16. |
| `corpus_registry_isa.diff` | `generator/`, `patch -p1` | ISA as a parameter of the typed corpus cells (`zil-*-avx512` quadrants), `zil_isa_gap` absence predicate, `emit_il_registry --isa`. `gen_set` emits 527 avx512 files byte-equal to the per-file replays; the avx2 quadrants are byte-neutral. |
| `twin_gate.py`, `twin_gate.c` | scratch | The table-driven twin gate: emits the avx2 and avx512 twin of each recipe with the same generator, builds per-width twiddle tables from the same logical twiddles, and compares outputs bitwise (`-ffp-contract=off`) and against a long-double DFT. |

`harness/` keeps the load-bearing test and measurement sources from the investigation
(turn A/B and 4x4 store proof, the column-stride/tail differential driver, the ZTT/msz
runtime tests incl. the MT-cut and permutation-inverse traps, the table-contract probes,
the tail and alignment benchmarks, the k1 mono test). Scripts inside still point at the
old scratch paths; adapt them before use.

Build any patched generator **outside the tree** (`dune build --root <copy>
--build-dir <scratch>`); a bare `dune build` in the repo promotes tracked headers.
