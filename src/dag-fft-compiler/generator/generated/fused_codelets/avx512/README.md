# ZTURN-T fused drivers, avx512

The 30 whole-transform drivers for VW=8 (159 cells: `../../ztt_registry_avx512.h`),
emitted by `emit_ztt_drivers.exe --isa avx512 --uarch sapphire_rapids_avx512 --split .`.
They live in their own folder so the avx2 builds, which glob `fused_codelets/*.c`,
never compile them; an avx512 build (CMake or `gauntlet/build.py` with `VFFT_ISA=avx512`)
compiles this folder in their place. N=16 and N=32 have no cells at VW=8.
