# ZTURN-T fused drivers, avx512

The 30 whole-transform drivers for VW=8 (159 cells: `../../ztt_registry_avx512.h`),
emitted by `emit_ztt_drivers.exe --isa avx512 --uarch sapphire_rapids_avx512 --split .`.
They live in their own folder so the avx2 builds, which glob `fused_codelets/*.c`,
never compile them. No build links them yet: the avx512 runtime is §10 stage 3 of
`docs/roadmap/zil_avx512_design.md`. N=16 and N=32 have no cells at VW=8.
