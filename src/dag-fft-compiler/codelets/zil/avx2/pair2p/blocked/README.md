# pair2p/blocked/ — the large power-of-two radices as two passes

18 files: `n1b`, `n1tb*`, `t2b*`, `t2bt*` at radix 16, 32 and 64, each radix run
as two passes through a small parked buffer (`b44`, `b48`, `b88`, `b216`,
`b416` spell the split), because the one-pass form spills at these radices.
The rule behind it: splitting a radix pays exactly when the one-pass kernel runs
out of AVX2's 16 registers; radix 8 and below is never blocked, 16 is raced,
32 and 64 are always blocked. Selected through the pair's `il_kv` variant
(`src/core/oop/il2p.h`). Six of these files are emitted by their own recorded
`gen_radix` recipes (`generator/gates/recipes.tsv`), not by `gen_set`. The odd
blocked radices 9..49 (28 files) were deleted on 2026-09-24: no resolver reached
them.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../README.md`](../../README.md).
