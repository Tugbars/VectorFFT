# ztt/ — the ZTURN-T stage kernels (boundary-IL / split-interior)

39 files: `t0tp`, `tmg`, `tlf`, `tlfi` (natural order) and `t0d`, `tmgd`, `tld`
(the plain, scrambled set), + `_bwd`. This folder is `ztt/` since 2026-09-24
(it was `boundary_split/`); the `msz`/`mszt` odd mids that shared the layout
moved to [`../flat/odd_mid/`](../flat/odd_mid/), the engine that runs them.
Run by `src/core/oop/ztt.h` under `il_route=ztt`, N = 2048 to 262144, with the
tile `il_tw=`; the pow2 cells run as the fused drivers in
`generator/generated/fused_codelets/`, which inline these stage kernels.

**These files are NOT pure IL, and that is deliberate. Do not "fix" them.**

Every other folder under `zil/avx2/` is **pure IL** — packed complex,
re/im adjacent in the register, through all the arithmetic. Everything *here*
is the opposite in the middle: **interleaved at the buffer boundary, separate
re/im planes in the interior** (64-byte `[re x4][im x4]` blocks, `z`
addressing `+4` for the imaginary half, one stream per leg row). A packed
complex multiply needs a shuffle per multiply; a split one is four real
multiplies with none. Converting twice per transform (ingest, terminator)
amortizes the boundary over every stage between. Emitted by
`generator/lib/gen/cascade_z.ml`; rows in `generator/lib/gen/corpus.ml`
(`zil_boundary_cells`); declared for the runtime by the generated
`il_registry_avx2.h`. Contract: **`count % 4 == 0`** (4 columns per
iteration).

## What's here

| kind | role | radices |
| --- | --- | --- |
| `t0tp` (+`_bwd`) | ZTURN-T NATURAL ingest: packed legs at stride N/R0, the turn lattice, run-contiguous stores at `rb[c]*R0` | 4, 8 |
| `tmg` (+`_bwd`) | ZTURN-T mid: in-place group-looped combine with the column-varying pre-twiddle; the backward is the plain class's mid too | 3, 4, 5, 7, 8, 9, 15 |
| `tlf` / `tlfi` (+`_bwd`) | ZTURN-T NATURAL terminator: REINT packed stores in natural order; `tlfi` = the in-place twin with its output streams prefetched | 4, 8 |
| `t0d` | ZTURN-T PLAIN (scrambled) ingest: de-interleave, radix-R0 DFT, post-twiddle, block-split leg-major stores, in place when `zin == zout` | 4, 8 |
| `tmgd` | ZTURN-T PLAIN mid: `tmg` with the twiddle after the butterfly | 3, 4, 5, 7, 8, 9, 15 |
| `tld` (+`_bwd`) | ZTURN-T PLAIN terminator, in place: TR4 loads, twiddle-free DFT, unpack-only interleaved stores | 4, 8 |
| `msz` / `mszt_bwd` | the odd mids of the pair and chain3 engines (`src/core/oop/il2p.h`): the split body between interleaved edges, unordered lanes, any count | 3, 5, 7, 9, 15 |

The odd radices exist for the MIDS only: the ingest and the terminators are
lane lattices (a 4x4 transpose), the mids' edges are radix-agnostic — the
odd radix is always a mid (`docs/design/ztt_odd_design.md`).

Two consumers: `src/core/oop/ztt.h` calls these kernels per stage and block
(the STAGED executor, every 2^a·odd cell), and the pow2 cells run the fused
codelets in `generator/generated/fused_codelets/`, which inline these same
bodies with literal trip counts (`ztt_gate` holds staged == fused bitwise).

The table above is the whole directory and the whole of what `cascade_z.ml`
emits: the corpus law (`gen_set --root <tmp> zil-boundary`, then byte
comparison) holds 54 of 54, and `gates/full_corpus_gate.sh` replays it.

🔴 **A full-IL split-interior replacement was refuted twice by independent measurement
(-8.9% at 4096, -12.6% at 16384 against this split interior under identical
chains). Do not propose it again.**
