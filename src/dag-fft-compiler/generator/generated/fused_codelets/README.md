# Fused codelets

Owner's ruling, 2026-09-14. This directory holds the **fused codelets** of the
ZTURN-T engine, and nothing else: one whole-transform function per pow2 cell
`(N, chain, direction, buffer mode)`, generated, with the engine's stage
kernels inlined and every trip count a literal.

## What they are, and what they are not

The library has two kinds of generated code for this engine, and both stay:

| | stage kernels | fused codelets |
| --- | --- | --- |
| where | `codelets/zil/avx2/boundary_split/radix{4,8}_z_{t0tp,tmg,tlf,tlfi,t0d,tmgd,tld}*.c` | here |
| what | one stage of the transform, radix 4 or 8, forward or backward | the whole transform of one pow2 cell, stages inlined |
| composes | yes: the same stages build 2^a*odd chains, the 2D/3D pow2 passes, and both order classes' backwards | no: one pow2 chain, frozen |
| how many | 12 natural + 8 plain kernels | 223 cells x (4 natural + 2 plain) functions |
| how used | called through the stage table of any solution | bound whole by the ZTURN-T plan through `ztt_registry_avx2.h` |
| why | the product: every other solution is built from them | the pow2 solution's executable form: no calls, literal trip counts, 4-12% at N <= 2048, 0-3% above |

The stage kernels are the product. A fused codelet is a copy of those bodies
stapled together for one cell, so that the pow2 solution runs without a call
and with literal loop bounds. It is strictly the pow2 ZTURN-T solution, natural
and scrambled order, and only that; nothing else in the library may bind one.

## Files

- `ztt_drivers_avx2_<N>.c` — natural order: `{fwd, bwd} x {dest, plane}` per
  cell. ABI `(zin, zout, plane, tw, rb, tile)`.
- `zttp_drivers_avx2_<N>.c` — plain schedule = scrambled order: `{fwd, bwd}`
  per cell. ABI `(zin, zout, tw, tile)`; no plane, no run-base table;
  `zin == zout` is the same function.
- One file per (family, N) so that a change to one family's kernels recompiles
  only that family's files, in parallel (the single TU took 47 minutes).

## Where the core binds them

- `src/core/oop/ztt.h` — `vfft_ztt_create_chain` looks the cell up in
  `ztt_registry_avx2.h` and binds `fwd_dest` / `fwd_plane` / `bwd_dest` /
  `bwd_plane` (natural) and `fwd_scr` / `bwd_scr` (plain).
- `src/core/planning/dp_planner_il.h` — enumerates the registry's chains as the
  ZTURN-T race pool.

## Regeneration

Derived from the corpus cells (`lib/gen/ztt_drivers.ml`, `Ztt_drivers.cells`),
emitted by `bin/emit_ztt_drivers.exe --split .` under the promote rule in this
directory's `dune` (WSL: `eval "$(opam env --switch=5.2.0 --set-switch)";
export DUNE_CACHE=disabled; dune build`). The registry
`../ztt_registry_avx2.h` is derived from the same cell list, so a cell either
has every driver or is refused at create. The inlined bodies are byte-identical
to the standalone kernels (`Cascade_z.emit_codelet ~body_only:true`), which is
what the fused-equals-unfused gate checks.
