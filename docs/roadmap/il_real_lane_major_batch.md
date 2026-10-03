# IL real batch, lane-major: the native engine (planned)

**Status (2026-10-03): decided, not started.** Owner: "we will add lane-major
support to all K > 1 real handling; we will do this later."

## What it is

An interleaved real batch (r2c / c2r, `howmany = K > 1`) in the LANE-MAJOR
geometry: element e of transform t at `x[e*K + t]`, bin f of transform t at
`z[2*(f*K + t)]`. The caller chooses the geometry through
`vfft_config_t.batch_geom` (`VFFT_BATCH_LANE_MAJOR` /
`VFFT_BATCH_TRANSFORM_CONTIGUOUS`), as MKL's distance/stride and FFTW's
`idist`/`istride` make the caller state it; the library serves the geometry
asked for with that geometry's own IL engine. There is no race between the
two geometries inside an ordinary create -- the caller's buffers fix the
layout. Comparing them, like comparing the split and IL plans, is the
planning model's max-performance mode, which tells the caller which layout
to bring.

## Why a native engine

Today the lane-major real batch runs on the SPLIT real engines through their
`_z` doors (`bridge/real_bridge.h` -> `_vfft_create_real_split`): the last
crossing of the layout separation (D1), except odd N without a chain, which
the lane Bluestein `il/real/zrb_lanes.h` serves natively. The crossing goes
when this engine serves.

Measured 2026-10-03 (one thread, pinned, paced, memcpy control): against the
transform-contiguous wrapper (`tcb` over the K=1 IL engine), the split
lane-major path wins r2c at N <= 64 with K >= 16 by 1.3-3x (N=16: 2-3x), is a
wash at N = 128-256, and loses everywhere else (c2r 1.5-3.8x, r2c N >= 512
1.3-2.8x). The win is lane-major vectorization across K at small N, where the
wrapper pays a per-transform dispatch cost; at large N the wrapper keeps one
transform L1-resident while lane-major streams all K lanes through every
stage. The IL lane-major row kernel measured 2.1 ns/row at N=16 on lane-major
input (10-01) against the split path's 4.4 and the wrapper's 9.2: the native
engine beats both at small N.

## The build, in two slices

1. **The real leaf over lanes.** A real kind `r2zl`: the `r2z` leaf's
   lane-major loads (sample j of lanes k..k+3 = one vector at
   `zin[j*Ls + k]`), the `r2zr` row kind's CCE slots (DC and Nyquist as
   (x, 0)), and a lane-major store edge (bin p of lanes k..k+3 zipped into one
   store at `zout[2*(p*OLs + k)]`), forward and backward, at the real leaf's
   radices {4, 6, 8, 10, 12, 16, 32, 64}. The engine `zrl` (il/real/) is one
   kernel call per execute: `Ls = OLs = K`, `count = K`, out of place, both
   directions. Gate as `r2zr` was gated: generator emission reproduced by
   `gen_set`, a kernel gate against a naive real DFT, the door's probe against
   FFTW. The 64 leaf may spill as it did for rows; the gate decides.
2. **Every other even N, and odd N with a chain.** A two-stage lane-major
   form: the real leaf at `Ls = R1*K` over all R1*K real columns, then a top
   stage per transform with a lane-major store edge of its own (the pair's
   `t2m` writes transform-contiguous today), or the 2D column-pass machinery
   over K lanes (the shape `zrb_lanes.h` already uses for its complex inner).
   Odd N with a chain has no lane-major IL form at all today (the split rfft
   serves it through the crossing).

The crossing is cut when both slices serve; until then the split engines
serve lane-major real batches behind the bridge, and the transform-contiguous
geometry is available to every caller through `batch_geom` with nothing new
built.

Related: `docs/roadmap/layout_separation_plan.md` (D1),
`docs/roadmap/fft2d_real_il_design.md`, `docs/roadmap/il_real_engine_research.md`.
