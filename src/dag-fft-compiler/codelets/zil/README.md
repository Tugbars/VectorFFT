# THE IL CODELET TREE — READ THIS FIRST

The interleaved-complex (IL) kernels: everything a 1-D or N-D INTERLEAVED
c2c plan of the library runs on. Two directories, two layouts, two emitters;
every file's first two lines name its emitter and its recipe.

| directory | files | layout | emitter | corpus rows |
| --- | --- | --- | --- | --- |
| `avx2/pure_il/` | 499 (+20 in `tangent/`) | **pure IL** — packed complex through all the arithmetic | `generator/lib/gen/c2c_il.ml` | `corpus.ml` `zil_pure_cells` (`gen_set --root <tmp> zil-pure`) |
| `avx2/boundary_split/` | 54 | **boundary IL / split interior** — interleaved at the buffer edges, 64-B `[re x4][im x4]` planes inside | `generator/lib/gen/cascade_z.ml` | `corpus.ml` `zil_boundary_cells` (`gen_set --root <tmp> zil-boundary`); README there |

Regeneration is by quadrant through `gen_set` to a temporary root, never in
place, and the law is byte identity against the shipped file (the index is
LF; a checkout may carry CRLF). `generator/gates/full_corpus_gate.sh` is the
tracked acceptance gate for the whole corpus.

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
- **`log3` is a `t2` twiddle policy only**; the emitter refuses `--cil-log3`
  on `n1`/`n1t`.
