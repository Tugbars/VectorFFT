# The split layout in v1.0: contract, what goes, what is built

Owner's decision, 2026-11-05: the interleaved library is the main library; the split library is the
side helper for low-N batched work, kept small. In v1.0 the split side serves C2C at rank 1, 2 and 3
and nothing else: no R2C, no C2R, no real-to-real. This page is the declaration of that contract and
of the work it implies. The measurements behind it: docs/research/split_layout_scope_2026_11_04.md.

## The contract

| request under the split layout | v1.0 |
|---|---|
| 1D C2C, power-of-two N in zturn-ts's registry, natural order, both placements | served: K = 1; a contiguous batch of any K (serial, or slabs over the pool at T > 1); a lane-major batch of any K (2026-11-07: the lanes past the last whole vector group run as one zero-padded group on the plan's own scratch and are copied back, the tail; internal padding, the caller's tight layout untouched) |
| 1D C2C at N = 2^a·m, a >= 4, m over 3, 5, 7, 9, 15 (the odd band, 2026-11-06) | served in every form above, staged (one stage kernel call per stage and block; the chain grammar is the race's pool) |
| a lane-major 1D C2C batch, or a column axis of a plane or volume, at any N that is a product of 2 to 7 radices over 3, 4, 5, 7, 8, 9, 15 (the lane-major band, 2026-11-06) | served, staged: the lane-major family's kinds carry no lane lattice, so its ends take the odd radices (the split side's lane-major flat DIT) |
| any other N up to 131072 (a prime, 2·odd, a pure-odd K = 1 or contiguous batch) | refused at create, naming the interleaved layout. The split Bluestein that served these lengths (`split/rank1/zts_bluestein.h`, 2026-11-06; its threaded forms `zts_blu_mt.h`; the TURNED lane-major form and the threading RULE of 2026-11-07) is OUT OF THE BUILD since 2026-11-07 — the owner: too weak to refine and release unless customers ask for it; the files stay, compiled only under `VFFT_SPLIT_BLUESTEIN`, which no build defines |
| 2D C2C, both extents in the registry, natural order on both axes, both placements | served: the row pass as a contiguous batch of N1 rows, the column pass as a lane-major batch of N2 lanes (one vector per column, the column twiddles broadcast) |
| 3D C2C, all extents in the registry, natural order, both placements | served: the plane passes as 2D, the third axis as a lane-major batch of N2 x N3 lanes |
| an extent above 131072 | refused at create; the message names the interleaved layout. (A column pass whose lane count is off the vector width was refused until 2026-11-07; it runs the tail now.) |
| scrambled order | refused at create (beta on the interleaved side; the split engine is natural by construction) |
| R2C, C2R, DCT, DST, DHT | refused at create: not a split transform in v1.0 |
| rank 4, padded batches (VFFT_FLAG_OWN_PITCH), the estimate flag | refused at create |

One engine serves the whole table: zturn-ts (split/rank1/zts.h), its two families (the K=1 driver with
its contiguous twin, the lane-major driver; the staged executor of both for the cells without a fused
driver; under VFFT_USE_JIT a deployed cell of that kind gets its fused driver compiled at create,
`src/dag-fft-compiler/jit/zts_jit_runtime.h`, 2026-11-06: the build-time emitter's driver written for
the plan's chain with the stage kernels' sources inlined, cached on disk, the staged walk kept on any
failure) and its threaded forms, on a zturn-ts cell every one a verdict raced at the plan's thread count against serial
and against each other and banked on the cell's threading row (on a Bluestein cell, out of the build since 2026-11-07, chosen by rule)
(2026-11-06; the row's value names the
form: 0 serial, 1 slabs or lane block ranges, 3 and 4 the sectioned walk's BLOCKS and TILES, 5 lane
group ranges): slabs of whole transforms for a contiguous batch of K >= T; lane BLOCK ranges over
the pool's slots for a lane-major batch of two or more blocks and lane GROUP ranges for two or more
groups (the same slots at group granularity, each thread's row visits holding fewer groups than the
banked block); the SECTIONED walk (split/rank1/zts_mt.h, ZTURN-T's on the split layout) for one
transform, a contiguous batch of K < T (every stage's dispatch covering all K transforms with T / K
ranges each) and any lane-major batch, block by block — every stage's units cut into ranges, one fork-join per stage, bitwise the serial
walk, with its BLOCKS and TILES arms. The row's value names the form with the arm (0 serial, 1 slabs or
lane blocks, 3 sectioned blocks, 4 sectioned tiles), so a row banked under one form is never read under
the other. One wisdom grammar: the split natural cell rows (eng=zts chain= tile= blk=, the lane-major
cells at lay=slm, the threading rows eng=tcb tcmt=) and, for rank 2 and 3, the per-axis cells plus a
rank-3 plan's own threading row (its middle axis's planes over the slots).

## Out of the build: the split Bluestein (owner, 2026-11-07)

"The Bluestein part of the split engine is too weak, and unless the customers want it we won't give it
full effort, refine it and release it. Make it out of the compilation and production code; keep the
files, behind a build guard that is not defined. Requests for prime-count FFTs will be refused."
So: `split/rank1/zts_bluestein.h` and `split/rank1/zts_blu_mt.h` stay in the tree, every call site
(the 1D create's dispatch past the bands, the rank-N admission, the execute branch, the destroy, the
plan field, the threading setters) compiles only under `VFFT_SPLIT_BLUESTEIN`, which no build file defines.
Without it the split layout serves its bands alone — the registry's powers of two, the odd band, the
lane-major band — and refuses any other length at create, a prime among them, naming the interleaved
layout. The API matrix gate asserts the refusal (1D C2C SPLIT 127, both placements).

## What the split side is for (owner, 2026-11-07)

Batched pow2 and odd-band work at mid N is where the split side wins and what it is for (the
owner, 2026-11-07: good for odd and pow2): at K = 4 on the 14900KF it is 1.1x to 1.8x ahead of the
interleaved library at 256 to 4096 (lane-major 1.6x to 1.8x at 1024 to 4096) and 1.0x to 1.6x ahead
across the odd band (docs/research/split_zts_jit_2026_11_06.md); what it could serve only by Bluestein against a direct engine (the pure-odd contiguous
batch, the primes where Rader wins) it refuses since 2026-11-07, the Bluestein out of the build. The owner's words: good enough; a prototype for those who want
extreme pow2 mid-N performance. The remaining levers (odd ends for the S family, a split Rader) are
reported, not pursued.

## What went (deleted 2026-11-06)

Everything in the split tree that is the stride engine or stands on it, deleted on 2026-11-06
(the owner: the old machinery's C code and codelets are discovered and deleted); the list as it
was drawn:

- the engine (`split/engine`: planner, executor, the generated executor table, twiddles), its
  planning (`split/planning`: the DP, exhaustive, measure, padding), the natural-order tapes
  (`split/natorder`), the padded batch (`split/rank1/vfft_batch.h`). The JIT stays (owner,
  2026-11-05): `src/dag-fft-compiler/jit` and the VFFT_USE_JIT paths are not part of the cut, nor
  is `common/move/transpose.h` (a tool for users, kept though no engine includes it);
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
- the real bridge (`bridge/`) whole: its interleaved K = 1 door moved to `il/real/real_create_il.h`
  (`_vfft_create_real_door`), its engine walk to `il/il_execute.h` (`_vfft_real_execute_il`);
  the old JIT runtimes of `src/dag-fft-compiler/jit` (the stride executors', the K=1 routes', the
  rfft and c2r ones) and the trig tools of the baseline harness went with it.

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
   interleaved-native engine, or it is refused at create until one exists. **Decided 2026-11-06:
   refused at create with a warning until the native engine exists; the bridge is deleted.** (The
   front door already served the even-N out-of-place r2c lane-major batch natively,
   `il/rank1/lm_batch_il.h`, and refused the rest before the bridge was reached; the lane
   Bluestein's create, the odd-N piece of that engine, sits in `il/real/real_create_il.h`
   unreached.)
3. **2D C2C on zturn-ts** (built 2026-11-06, `split/rankn/c2c_nd_create_split.h`): the row pass as
   the contiguous batch (the K = 1 cell at N2 run N1 times, in slabs over the pool under the batch's
   threading verdict), the column pass as the lane-major batch (the cell at N1 with N2 lanes, in
   place where the rows landed; its lane blocks over the pool's slots under its own verdict), natural
   order on both axes, both placements; raced and banked per axis cell, no rank-2 row of its own. The
   lane-major cells are keyed `lay=slm` (the law of the interleaved `lay=lm`): their threading row
   would otherwise collide with the contiguous batch's at the same q. The column pass's edges take a block of lane groups per row visit (B groups into B
   group planes resident in L2; B a raced axis of the cell, 1 where the plane fits L2): the pass
   that walks every row once per group thrashes the second-level TLB from 512x512 up, and the
   blocked form, emulated on 2026-11-05, is worth 1.4x at 512x512, 1.5x at 1024x1024 and 2.1x at
   2048x2048 on the whole transform, ahead of the stride engine at every cell and at best parity
   with the interleaved tier at the big ones (two sweeps of the plane is the floor there).
4. **3D C2C** the same way (built 2026-11-06): the rows as the contiguous batch of N1 x N2, the middle
   axis as the lane-major cell at N2 with N3 lanes run per plane (at T > 1 the planes over the pool's
   slots, the 3D plan's one verdict of its own, banked on its rank-3 row), the first axis as the
   lane-major cell at N1 with N2 x N3 lanes.
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
