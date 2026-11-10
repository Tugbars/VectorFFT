# ZTURN-T fused drivers, avx512_f32 (FP32)

STOPGAP: the same 8-lane (ymm) drivers as `../avx2_f32/`, emitted under the
`avx512_f32` name by
`emit_ztt_drivers.exe --isa avx512_f32 --uarch sapphire_rapids_avx512 --split .`
and compiled for `avx512f,avx512vl,fma`, so the compiler has the 32 ymm registers
(the stack spills of the 16-register build mostly disappear). Same 186 cells (N = 16..1048576, 34 files)
(`../../ztt_registry_avx512_f32.h`), same tables, results bitwise those of avx2_f32.

Why not 16 lanes: the ingest writes each column's R0-run as whole VW-column blocks
and the plain terminator reads R_last-runs the same way, so 16 lanes need
R0 % 16 = 0 (and R_last % 16 = 0); no {4,8} chain qualifies. A 16-lane form needs
radix-16 ends (a radix-16 ingest lattice with 16 x 16 transposes, a radix-16 tld)
and the core's chain grammar at 16: the follow-up, not built.

Own folder, so no AVX2 or FP64 build compiles it.
