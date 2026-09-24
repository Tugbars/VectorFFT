# mono/ — N = 64 as an 8x8 with both halves fused in registers

2 files, `vfft_k1_mono64_il_fwd_avx2.c` and `_bwd_`, emitted by the split
library's MONO emitter (`generator/lib/gen/c2c_split.ml`) in the interleaved
layout. The K=1 door races it as MONO form 1 at N = 64 against the two-pass
pair; on this host the pair wins, so no shipped row selects it. Moved here from
`codelets/oop/avx2/` on 2026-09-24: it is an interleaved kernel and belongs
with the interleaved tree.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../README.md`](../README.md).
