# The strided families: a field guide

*docs/design/strided_codelet_families.md — the disambiguation reference.
There are now several distinct "strided" things in this tree; this guide is
the map. When in doubt, the function NAME and the SIGNATURE identify the
family uniquely.*

## Why "strided" at all

All families share one idea (Design C / the §6a34-corrected v2 thesis):
process rows **in their natural row-major layout** — no transpose passes,
no scratch round-trips — by doing whatever marshalling is needed **in
registers at the memory boundary**. They differ in transform domain, IO
contract, and mechanism.

## The families

### 1. c2c strided monos — the original quadrant

| | |
|---|---|
| files | `codelets/strided/{avx2,avx512}/rN_n1_{fwd,bwd}_strided.c` |
| symbol | `radixN_n1_{fwd,bwd}_{avx2,avx512}_strided` |
| ABI | `(rio_re, rio_im, tw_re, tw_im, row_stride, me)` — **in-place**, 6-arg; tw unused (n1) |
| semantics | full N-point **complex** FFT per row; VW rows per block via the 4×4/8×8 in-register transpose lattice; output NATURAL order (one-digit chain ⇒ digit reversal is identity) |
| coverage | N ∈ {4,8,12,16,20,32,64} avx2; {8,16,32,64} avx512 |
| generator | `--strided [--bwd] --isa …` |
| wired at | `strided_rows.h` (fft3d / fftnd / 2D **c2c** row passes, `-DVFFT_STRIDED_ROWS`), padded-tail machinery, natural-order fast path |
| record | docs/performance/strided_rows_case_study.md (1.72×/1.40× row-pass wins) |

Variants in the same emission path (postamble/preamble forks, NOT separate
families): `--strided-il-out[-nt]` (stores z-interleaved to `out_z`, ABI
gains the out_z pointer), `--strided-il-in` (loads from interleaved
`in_z`). Also `--oop-strided` (a distinct pack-fix ABI in codelet_oop.ml,
mutually exclusive with `--strided`).

### 2. r2c / c2r strided monos — the §6a35-39 family

| | |
|---|---|
| files | `codelets/strided/avx2/rN_n1_{fwd,bwd}_strided_r2c.c` |
| symbol | `radixN_n1_{fwd,bwd}_avx2_strided_r2c` |
| ABI fwd | `(rio, out_re, out_im, tw_re, tw_im, row_stride_in, out_stride, me)` — **OOP**, 8-arg |
| ABI bwd | `(in_re, in_im, out, tw_re, tw_im, in_stride, row_stride_in, me)` |
| semantics | **real** rows ↔ half-spectra via row-level TWO-FOR-ONE (even rows = re lanes, odd = im; `me` = PAIRS); the c2c DAG body is untouched — fwd fuses the conjugate SPLIT at the store lattice, bwd runs a MERGE prologue before the body. Output rows unnormalized on bwd (= N·x). `out_stride ≥ N/2+1`. |
| coverage | N ∈ {8,12,16,20,32,64} avx2; {8,16,32,64} avx512 (§6a45; 12/20 fall back to avx2); ANY row count ≥ 8 (§6a48: full blocks + staged tail; odd lone row via zero-partner two-for-one) (tails: future — the quadrant's padded-tail pattern applies) |
| generator | `--strided-r2c [--bwd] --isa avx2` |
| wired at | `fft2d_r2c.h` §6a39 + `fftnd_r2c.h` §6a47 (3D real row pass): resolver + create-time **measured adoption** (>5% hysteresis) + whole-row-pass replacement branches; z contract inherits automatically |
| record | mkl_geometry_contracts.md §6a35-39 (−45..52% row pass; **MKL-real parity at covered 2D cells**) |

**How to tell 1 from 2 at a glance:** the `_r2c` suffix; in-place 6-arg vs
OOP 8-arg; `me` = rows (c2c) vs `me` = PAIRS (r2c).

### 3. Twiddle-stage engine kernels — strided_tw.h (§6a40-41)

**Not codelets.** `src/core/transforms/fft2d/strided_tw.h` contains
transpose.h-class hand ENGINE KERNELS composing family-1 monos into longer
rows:

| piece | what |
|---|---|
| `_stw_front{2,4}_{fwd,bwd}` | DIF radix-2/4 front stages, vectorized ALONG the row (contiguous span pairing — no lattice, no batching needed) |
| `_stw_map(bin, r)` | the DIF ordering map: Z[r·k+j] at column j·64+k; nothing is reordered in memory |
| `_stw_split_row / _stw_merge_row` | the §6a36/38 conjugate formulas addressed THROUGH the map (scalar; see limits) |
| `_stw_r2c_fwd / _stw_c2r_bwd` | ROW-BLOCKED compositions (§6a40 law: one DRAM pass): per 8-row block, front → r64 c2c monos on sub-bands → mapped split (fwd; mirror for bwd) |

Coverage N2 ∈ {128, 256}; correctness proven exact-position vs naive DFT
(e-13) and roundtrip (e-16), gate cells (64,128)/(64,256). Wired with
full-executor measured adoption in fft2d_r2c.h.

**Status honesty (§6a41):** the composition TIES the production tiled row
pass (+1.3% at 256², −0.4% at 128²) — the gates correctly keep it dormant.
§6a40's −9% was vs a dispatch-heavy stand-in; against the real incumbent,
the per-block call overhead, the front→mono L1 reload, and the SCALAR
mapped split consume the margin (the map scatters f-adjacent bins 64
columns apart, blocking the §6a36 vector split at this layer).

### 4. Large-N strided r2c monos — the "fused" family (§6a42: it already existed)

The §6a41 verdict demanded a fused single-body codelet. §6a42 discovered
the generator could ALREADY emit it: the N=64 mono ceiling was convention,
not capability — `gen_radix.exe {128,256} --strided-r2c [--bwd]` emits
monolithic bodies (314/681 KB, 7-13 s compiles) where the "front stages"
are simply DAG depth and the §6a37 split postamble consumes
out_lane_0..N-1 unchanged. Zero new OCaml. Split-before-map holds by
construction: the mono IS natural order, there is no map.

| | |
|---|---|
| files | codelets/strided/{avx2,avx512}/r{128,256,512}_n1_{fwd,bwd}_strided_r2c.c |
| ABI / semantics | identical to family 2 (same mode, bigger N) |
| gates | naive-DFT 3.2e-13 / 7.6e-13; roundtrips 8.9e-16 / 7.8e-16; 2D gate 24/24 |
| measured (256 squared, forced arms) | bwd -7.3% vs tiled — ADOPTED; fwd +2.5% tie, correctly declined (16-reg monolithic spills tie the tuned incumbent) |
| campaign effect | 256 squared split/MKL-real: fwd 0.872x, bwd 0.974x |

The remaining fwd gap is the sharpened next item: regalloc/pinning gate
extension to this size class, GH pressure mode, or a CT-blocked strided
construction. Family 3 (the §6a41 compositions) is FALLBACK-ONLY as of §6a45: its
adoption gate requires the mono resolver to have no coverage at that N2
(currently never true). Execute order is mono -> stw -> tiled.

## Decision table for callers

| you have | you want | use |
|---|---|---|
| complex rows, N ≤ 64 | in-place row FFTs, natural order | family 1 via strided_rows.h |
| real rows, N ≤ 64, rows % 8 == 0 | half-spectra (or z) | family 2 via fft2d_r2c.h engines |
| real rows, N in {128,256} | same | family 4 monos via the same fft2d_r2c.h engines (bwd adopted at 256 squared; fwd gate decides per-cell) |
| real rows, N = 512 | same | family 4 (§6a43: emitted, gated, resolver-wired; 512² bench pending) |
| real rows, N > 512 | — | uncovered: emit family-4 (untested) or tiled |

## Provenance cheat-sheet

Every emitted file's header states its flags. `--strided` → family 1;
`--strided-il-*` → family-1 IL variants; `--strided-r2c` → family 2.
Family 3 is hand engine code (strided_tw.h header comment is its
provenance). Family 4 = family 2's mode at N >= 128; the provenance header is identical apart from N.
