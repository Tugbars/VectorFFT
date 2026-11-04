# The split layout in v1.0: contract, what goes, what is built

Owner's decision, 2026-11-05: the interleaved library is the main library; the split library is the
side helper for low-N batched work, kept small. In v1.0 the split side serves C2C at rank 1, 2 and 3
and nothing else: no R2C, no C2R, no real-to-real. This page is the declaration of that contract and
of the work it implies. The measurements behind it: docs/research/split_layout_scope_2026_11_04.md.

## The contract

| request under the split layout | v1.0 |
|---|---|
| 1D C2C, power-of-two N in zturn-ts's registry, natural order, both placements | served: K = 1; a contiguous batch of any K (serial, or slabs over the pool at T > 1); a lane-major batch of any K that is a multiple of the vector width |
| 2D C2C, both extents in the registry, natural order on both axes, both placements | served: the row pass as a contiguous batch of N1 rows, the column pass as a lane-major batch of N2 lanes (one vector per column, the column twiddles broadcast) |
| 3D C2C, all extents in the registry, natural order, both placements | served: the plane passes as 2D, the third axis as a lane-major batch of N2 x N3 lanes |
| any extent outside the registry (odd, prime, 2^a x odd, above the ceiling) | refused at create; the message names the interleaved layout |
| scrambled order | refused at create (beta on the interleaved side; the split engine is natural by construction) |
| R2C, C2R, DCT, DST, DHT | refused at create: not a split transform in v1.0 |
| rank 4, padded batches (VFFT_FLAG_OWN_PITCH), the estimate flag | refused at create |

One engine serves the whole table: zturn-ts (split/rank1/zts.h), its two families (the K=1 driver with
its contiguous twin, the lane-major driver) and its two threaded forms (slabs of whole transforms;
lane groups over the pool's slots, to be built). One wisdom grammar: the split natural cell rows
(eng=zts chain= tile=, the threading rows eng=tcb tcmt=) and, for rank 2 and 3, the per-axis cells.

## What goes

Everything in the split tree that is the stride engine or stands on it:

- the engine (`split/engine`: planner, executor, the generated executor table, twiddles), its
  planning (`split/planning`: the DP, exhaustive, measure, padding), the natural-order tapes
  (`split/natorder`), the JIT, the padded batch (`split/rank1/vfft_batch.h`);
- the classic out-of-place kinds and the kind-3 K=1 routes (`split/oop`, the K=1 create's route
  table), the split primes (`split/primes`), the split real engines and routes (`split/real`), the
  trig family (`split/trig`), the current rank-2/3 creates on the stride engine (`split/rank2`,
  `split/rank3`);
- the wisdom codecs of those families (`wisdom2_stride_reader.h`, `wisdom2_oop_split.h`,
  `wisdom2_2d_split_reader.h`, `wisdom2_fftnd.h`), the scrambled store shard and the stride, classic
  and kind-3 split rows of the natural shard, their migration paths;
- the generated kernels of those families (`codelets/split/inplace`, `oop`, `strided`, `rfft`,
  `c2r`, both ISAs: about 800 files), their registries, their gate rows and gates, the bench modes
  that drive them;
- the split arms of the real bridge (`bridge/`): the split real create and execute paths.

What stays: `split/rank1/zts.h`, `zts_race.h`, `split/wisdom/wisdom2_zts_split.h`, the 1D creates
reduced to the zturn-ts paths, `split_create.h` / `split_execute.h` reduced to the three ranks,
`codelets/split/ztt`, the fused zts/ztv drivers and their registry, `common/math/qw_fill.h` with the
baked quarter wave.

## What is built, in this order

1. **The AVX-512 emission of zturn-ts** (the split library must not be empty on that ISA): the 32
   stage kernels, the fused drivers and the registry at eight lanes; compile-checked with the Intel
   tools (no AVX-512 machine here to time or verify numerically).
2. **The interleaved real lane-major batch loses its engine.** Today an interleaved R2C/C2R batch at
   K > 1 in the lane-major geometry runs on the split real engines through the bridge (owner
   decision D1, "the one crossing left"). With those engines gone, that request needs an
   interleaved-native engine, or it is refused at create until one exists. **Owner's call, open.**
3. **2D C2C on zturn-ts**: the row pass as the contiguous batch (one plan at N2, K = N1), the column
   pass as the lane-major batch (one plan at N1, K = N2; the lane cap lifted from 64, since the
   group planes do not grow with K), natural order on both axes, both placements; the threaded forms
   already in the engine (slabs) plus the lane groups over the pool's slots; raced and banked per
   axis cell. The column pass's edges take a block of lane groups per row visit (B groups into B
   group planes resident in L2; B a raced axis of the cell, 1 where the plane fits L2): the pass
   that walks every row once per group thrashes the second-level TLB from 512x512 up, and the
   blocked form, emulated on 2026-11-05, is worth 1.4x at 512x512, 1.5x at 1024x1024 and 2.1x at
   2048x2048 on the whole transform, ahead of the stride engine at every cell and at best parity
   with the interleaved tier at the big ones (two sweeps of the plane is the floor there).
4. **3D C2C** the same way: planes, then the third axis as a lane-major batch of N2 x N3 lanes.
5. **The cut**, one subsystem per commit, each with the front door's refusals, the docs
   (vfft_front_door.md, the user guide, the performance pages), the gates, the benches and the store
   updated in the same commit, and the build (CMake, build.py) dropping the families as they go.
6. **The store**: the scrambled shard retired; the natural shard trimmed to the zturn-ts rows, the
   interleaved rows and the shared recipes.

## Open decisions

- The N ceiling for the split side: the registry spans 16 to 262144 (one fused file per family per
  N). "Low N" may mean a lower ceiling; each N dropped removes two fused files and shrinks the race
  pools.
- The interleaved real lane-major batch (item 2 above): build its native engine inside this
  overhaul, or refuse that request until it exists.
- Whether a 2D or 3D cell may keep the natural order on only one axis (today's split 2D serves a
  scrambled comb at DEFAULT); the proposal above is natural on every axis, the interleaved tier's
  own contract.
