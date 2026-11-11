# ZTURN-T fused drivers, avx512_f32 (FP32)

The 13 whole-transform drivers (N = 256..1048576) for interleaved complex float at
16 lanes per zmm (117 cells: `../../ztt_registry_avx512_f32.h`), emitted by
`emit_ztt_drivers.exe --isa avx512_f32 --uarch sapphire_rapids_avx512 --split .`
and compiled for `avx512f,avx512dq`.

The 16-lane form is the natural order class only. A chain is a radix-16 ingest
followed by radix-4 and radix-8 stages: the ingest writes each column's R0-run as
whole 16-column blocks, so R0 % 16 = 0 and (N / R0) % 16 = 0, which makes 256 the
smallest length. The ingest turns 16 digits per column through a 16 x 16 float
transpose; the mids and the terminators are the radix-4 and radix-8 bodies at 16
lanes. Per stage, for each 16-column block and leg r = 1..R-1, one record
`[c x16][s x16]` (128 B); the plane holds `[re x16][im x16]` blocks; `rb[]` is
`size_t`. The plain (scrambled) class has no cell here: its last stage would need
R_last % 16 = 0.

They live in their own folder so no AVX2 or FP64 build compiles them; an FP32
AVX-512 build compiles this folder.
Registry: `emit_ztt_registry.exe --isa avx512_f32 > ../../ztt_registry_avx512_f32.h`.
