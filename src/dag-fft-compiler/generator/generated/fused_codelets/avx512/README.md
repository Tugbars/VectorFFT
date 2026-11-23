# ZTURN-T fused drivers, avx512

The 30 whole-transform drivers for VW=8 (159 cells: `../../ztt_registry_avx512.h`),
emitted by `emit_ztt_drivers.exe --isa avx512 --uarch sapphire_rapids_avx512 --split .`.
They live in their own folder so the avx2 builds, which glob `fused_codelets/*.c`,
never compile them; an avx512 build (CMake or `gauntlet/build.py` with `VFFT_ISA=avx512`)
compiles this folder in their place. N=16 and N=32 have no cells at VW=8.

## The split library's drivers (zturn-ts)

`zts_drivers_avx512_<N>.c` (one transform and contiguous batches, split planes) and
`ztv_drivers_avx512_<N>.c` (lane-major batches), N = 16..262144: 30 files, the cells of
`../../zts_registry_avx512.h` (224 rows), emitted by
`emit_zts_drivers.exe --isa avx512 --uarch sapphire_rapids_avx512 --split .`.
The stage kernels they inline are `codelets/split/ztt/avx512`.

The lane-major family has every {4,8} chain of the avx2 list (223 cells): a column is one
vector per plane, no lane law. The one-transform family writes each column's run as whole
8-lane blocks, so its ingest is radix 8 (R0 % 8 = 0): the chains that start with 8, and
N = 16 as 8.2 on the radix-2 terminator (105 cells). Its ingest runs the 1..7 columns past
the last whole vector with masked loads, so N / R0 need not be a multiple of 8 (N = 16, 32).
A registry row names the drivers of the families that have the cell; the other entries are 0.
