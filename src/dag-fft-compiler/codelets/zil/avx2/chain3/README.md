# chain3/ — the backward scatter of the three-pass chain

27 files, `t2tg_bwd`. A three-pass chain exists so odd factors can be used at
all: a two-pass pair needs both pass counts even. Its forward passes are the
pair's kinds (`n1t`, `t2` from [`../pair2p/`](../pair2p/)) and the shared leaf;
`t2tg_bwd` is the backward pass's strided scatter that the chain needs and the
pair does not. Run by `src/core/oop/il2p.h` (`_vfft_il3p_*`) under
`il_route=chain3`; 2,070 shipped 1D rows on 2026-09-17. Nothing here is unused.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../README.md`](../README.md).
