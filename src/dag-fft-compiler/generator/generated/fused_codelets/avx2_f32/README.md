# ZTURN-T fused drivers, avx2_f32 (FP32)

The 32 whole-transform drivers (N = 16..524288: FP32's ceiling is the same 4 MB plane as FP64's 262144) for interleaved complex float at 8 lanes per ymm
(179 cells: `../../ztt_registry_avx2_f32.h`), emitted by
`emit_ztt_drivers.exe --isa avx2_f32 --uarch raptor_lake_avx2 --split .`.
8 float lanes is FP64 AVX-512's lane count, so the cells, laws and table layout are
those of `../avx512/`, in floats: per stage, for each 8-column block and leg
r = 1..R-1, one record `[c x8][s x8]` (64 B), 2*(R-1)*L floats per stage; the plane
holds `[re x8][im x8]` blocks; `rb[]` is `size_t`. N = 16 and N = 32 have no cells.

They live in their own folder so the FP64 builds, which glob `fused_codelets/*.c`,
never compile them; an FP32 AVX2 build compiles this folder (target `avx2,fma`).
Registry: `emit_ztt_registry.exe --isa avx2_f32 > ../../ztt_registry_avx2_f32.h`.
