# flat/odd_mid/ — the odd middle radices of the flat engine

15 files: `msz`, `msz_bwd`, `mszt_bwd` at radix 3, 5, 7, 9 and 15: a split-body
stage with interleaved edges (the boundary-split layout of [`../../ztt/`](../../ztt/),
emitted by the same emitter and listed with the ZTURN-T rows in
`corpus.ml`'s `zil_boundary_cells`), carrying the narrow VEX-128 and scalar
arms so any count >= 1 runs. Selected per stage by the flat engine
(`src/core/oop/il_flatdit.h`, form `m`), and used by the pair engine's odd
cells through `il2p.h`. Moved here from `boundary_split/` on 2026-09-24: the
engine that runs them is the flat one.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../README.md`](../../README.md).
