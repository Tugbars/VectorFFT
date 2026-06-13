# Codelet tree
Layout: {inplace,oop,strided}/{avx2,avx512}. Every file carries a PROVENANCE
header (command line + each auto-rule decision with reasons).
Regenerate any or all of it with:
    bash generator/scripts/regen_codelets.sh [all + per-quadrant targets incl. strided-avx2/strided-avx512]
