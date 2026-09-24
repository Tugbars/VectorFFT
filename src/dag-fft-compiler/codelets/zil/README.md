# THE IL CODELET TREE — READ THIS FIRST

The interleaved-complex (IL) kernels: everything a 1-D or N-D INTERLEAVED
c2c plan of the library runs on. Since 2026-09-24 the tree is one folder per
engine (`docs/roadmap/il_codelet_reorg.md`); every file keeps the name
`radixN_z_<kind>_avx2.c`, and its first two lines name its emitter and recipe.
Each folder's README says which route uses it, what each kind is, which wisdom
token selects it, and what in it is unused.

| folder | files | kinds | engine, wisdom |
| --- | ---: | --- | --- |
| [`avx2/shared/`](avx2/shared/) | 62 | `n1` | a whole small transform in registers: MONO forward, the pair's and chain3's backward leaf |
| [`avx2/shared/col/`](avx2/shared/col/) | 110 | `n1c`, `t2c` | the N-D column stage (2D, 3D) and the in-place 1D MONO path; `chain=` |
| [`avx2/shared/col/blocked/`](avx2/shared/col/blocked/) | 12 | `n1cb*`, `t2cb*` | the radix-32/64 column forms, raced: `forms=` |
| [`avx2/mono/`](avx2/mono/) | 2 | `mono64_il` | N = 64 as a fused 8x8, MONO form 1 (raced; the pair wins here) |
| [`avx2/pair2p/`](avx2/pair2p/) | 161 | `n1t`, `t2`, `t2t`, `*_ct` | the two-pass Bailey pair: `il_route=2p`, `il_kv` |
| [`avx2/pair2p/blocked/`](avx2/pair2p/blocked/) | 18 | `n1b*`, `n1tb*`, `t2b*`, `t2bt*` | radix 16/32/64 as two passes: an `il_kv` variant |
| [`avx2/pair2p/tangent/`](avx2/pair2p/tangent/) | 14 | `tan`, `bw32`, `t256`, `m128` | the tangent interior: `il_kv` 3 and 4 |
| [`avx2/rows/`](avx2/rows/) | 82 | `n1ccs`; `n1tr`, `t2r`, `t2tr`, `n1r` (+ `tan`) | the 2D row pass: `ro=2`, `ro=3` + `rbk=` |
| [`avx2/chain3/`](avx2/chain3/) | 27 | `t2tg_bwd` | the three-pass chain's backward scatter: `il_route=chain3` |
| [`avx2/flat/`](avx2/flat/) | 198 | `t2cp`, `t2cs`, `t2csg`, `t2csgn`, `t2csgt`, `t2csgnt` | the flat mixed-radix DIT for odd N: `il_route=flat` |
| [`avx2/flat/odd_mid/`](avx2/flat/odd_mid/) | 15 | `msz`, `mszt` | the flat engine's odd middle radices 3..15 |
| [`avx2/ztt/`](avx2/ztt/) | 39 | `t0tp`, `tmg`, `tlf`, `tlfi`, `t0d`, `tmgd`, `tld` | ZTURN-T, 2048..262144: `il_route=ztt`, `il_tw=` |

740 files. Two layouts and two emitters: every folder but `ztt/` and
`flat/odd_mid/` is **pure IL** (packed complex through all the arithmetic,
`generator/lib/gen/c2c_il.ml`, corpus rows `zil_pure_cells`); those two are
**boundary IL / split interior** (`cascade_z.ml`, rows `zil_boundary_cells`).
The fused pow2 drivers of ZTURN-T stay in `generator/generated/fused_codelets/`,
derived output with their own build rule.

Regeneration is by quadrant through `gen_set` to a temporary root, never in
place; the folder a file lands in follows its kind (`Corpus.dir_of_file`), and
the law is byte identity against the shipped file (the index is LF; a checkout
may carry CRLF). `generator/gates/full_corpus_gate.sh` is the tracked
acceptance gate for the whole corpus; the builds (`gauntlet/build.py`,
`build_tuned/build.py`, `CMakeLists.txt`) list the folders explicitly and fail
loudly on an empty one.

Deleted on 2026-09-24 as unused (the owner's ruling): the 52 `log3` kernels,
the 28 odd blocked `n1b`/`t2b`, the 6 tangent forms at radix 6, 10 and 12, and
the 4 `b416` column forms.

## Who runs what

| consumer | kinds | role |
| --- | --- | --- |
| `src/core/oop/il2p.h` | `n1`, `n1t`, `t2`, `t2t` and their blocked (`b*`), tangent (`tan`), wide (`w32`) forms; `msz`/`mszt` | the K=1 pair and chain3 engines: `n1t` leaf with the four-step transpose fused into its stores, `t2` twiddled mid, `t2t` the backward twin; the odd mids at 3/5/7/9/15 are `msz` |
| `src/core/oop/il_flatdit.h` | `n1c`, `t2c`, `t2cs`, `t2csg`, `t2csgn`, `t2csgt`, `t2csgnt`, `t2cp`; `msz`/`mszt` | the flat mixed-radix DIT (both order classes) over the column-form kinds |
| `src/core/transforms/fft2d/il2d_cols.h`, `fftnd/fftnd_il.h` | `n1c`, `t2c` (+ blocked `cb*`) | the N-D column passes: a lane is a column, vectorized ACROSS columns; rows are a K=1 plan through the front door |
| `src/core/oop/ztt.h` | `t0tp`, `tmg`, `tlf`, `tlfi`, `t0d`, `tmgd`, `tld` (+ `_bwd`) | ZTURN-T, natural and plain (scrambled) classes: pow2 cells as the fused codelets in `generator/generated/fused_codelets/`, 2^a·odd cells as these kernels called per stage |

Which engine serves a cell is a raced wisdom verdict, never a rule in this
tree; `include/vfft.h` declares the tiers and their bands.

## The two layouts

| flavour | what it means | where it is right |
| --- | --- | --- |
| **pure IL** | re/im adjacent in the register through all arithmetic; complex multiply = `cflip` + `mul` + `fma` | the L1-resident cells: solo kernels, the pairs, chain3, the flat DIT |
| **boundary IL / split interior** | packed at the buffer edges; deinterleave once at ingest, compute on split planes (a complex multiply is four real multiplies, no shuffle), reinterleave once at the terminator | the L2-and-above cells: ZTURN-T, whose stages amortize the two conversions over the whole transform |

🔴 A full-IL interior for the split-interior engine was refuted twice by
measurement under identical chains (-8.9% at 4096, -12.6% at 16384). Do not
propose it again.

## Rules

- **NEVER build an IL-boundary / split-interior *codelet*** — a kernel whose
  ABI mixes layouts (`in_re` + `in_im` beside a packed side) converts at
  every pass boundary; that shape (`il_in`/`il_out`) measured slow and was
  deleted. Every file here takes one interleaved `zin`/`zout`; the split
  engine converts twice per transform, in its own ingest and terminator.
- **`t2t` is the only backward flat kind.** Its rival `t2p` (pre-twiddle,
  R2-first inverse) lost the raced decision at every R1 <= 32 and covered
  fewer pairs; its kernels, registry and route are gone and `--cil-pretw`
  fails loudly in the emitter. Pre-twiddle semantics, if ever needed, are the
  `t2t` leg-stride store variant, not a revival.
- **Count contracts.** Pure-IL kinds loop `k += per` and drop a trailing
  remainder; `il2p_create` refuses odd R1/R2 up front. `msz` carries the
  narrow VEX-128 and scalar arms (any count >= 1). The ZTURN-T kinds are
  `count % 4 == 0`.
- **Symbol != filename.** A file's exported symbol can differ from its
  basename (blocked kinds tag the SYMBOL with `b`, so blocked and flat
  variants link side by side). Check by exported symbol before deleting.
- **`log3` was a `t2` twiddle policy only** (the emitter refuses `--cil-log3`
  on `n1`/`n1t`); its kernels were deleted on 2026-09-24, no consumer.
