# Codelet tree

One folder per library, one folder per family, one per ISA:

    split/ztt/avx2                                            the SPLIT library's codelets
    zil/{avx2,avx512}/<engine folder>                          the IL library's (zil/README.md)

Every file carries a PROVENANCE header (command line + each auto-rule decision
with reasons); the family a file belongs to is decided by that header, not by
its name. The folder a file lands in follows its quadrant (`Corpus.dir_of_quadrant`)
or, for zil, its kind (`Corpus.zil_folder`). The builds (`gauntlet/build.py`,
`CMakeLists.txt`) list the folders explicitly and fail loudly on an empty one.

Regenerate to a temporary root, never in place:

    generator/_build/default/bin/gen_set.exe --root <tmp> all

and compare with `generator/gates/full_corpus_gate.sh` (LF-canonical bytes).
