# Strided codelets and the twiddle-variant axis — status + roadmap

*2026-07-22. Written while integrating the container campaign (see
[vectorfft_feature_ledger.md](../vectorfft_feature_ledger.md), §6a35–§6a45). Records why the
strided row engines have no FLAT/T1S/LOG3 variants, where the variant machinery actually
lives, and what a future `t1_strided` family would look like if strided composition is ever
revived.*

## 1. Present state (verified in-tree)

**The strided row engines are monolithic full-DFT codelets — `n1` only, no twiddle
variants exist.** `src/dag-fft-compiler/codelets/strided/avx2/`:

| family | sizes | form |
|---|---|---|
| `rN_n1_{fwd,bwd}_strided` (c2c) | N ∈ {4, 8, 12, 16, 20, 32, 64} | one straight-line N-point DFT per row |
| `rN_n1_{fwd,bwd}_strided_r2c` (two-for-one real) | additionally N ∈ {128, 256, 512} | two real rows per complex DFT (§6a38: c2c bwd Re/Im lanes ARE the even/odd rows) |

The provenance header states it outright (`r64_n1_fwd_strided.c`):
`Construction: MONOLITHIC (below blocking threshold, or prime/Direct: no CT pass boundary)`.
One loop in the whole file — the batch loop over rows (4 rows/pass AVX2, 8 on the §6a45
AVX-512 editions) — and the `tw_re/tw_im` parameters are `(void)`-discarded. Rows are
dispatched by *exact* N2 (`_f2d_sr2c_fwd_resolve`); sizes outside the mono set fall back to
the tiled composed row pass, with create-time adoption A/Bs (§6a49 wisdom sidecar) deciding
strided-vs-tiled per shape.

## 2. Why no variants — the axis doesn't apply to monos

FLAT / T1S / LOG3 answer one question: **how is the inter-stage twiddle applied at a CT
stage boundary?**

- FLAT — stream K-replicated per-leg table rows through the codelet (`W[(j-1)·me + m]`);
- T1S — per-leg *scalars*, broadcast inside the codelet;
- LOG3 — raw per-leg twiddles, executor pre-applies cf0 to all legs.

A monolithic full-N DFT **has no stage boundary**. Its twiddles are constants compiled into
the butterfly network by the DAG scheduler. There is no application *strategy* left to
choose, hence nothing to calibrate — the variant axis is undefined for this family, not
merely unimplemented. Natural bin order also falls out for free (no digit scramble, no
PURE/PSWAP reorder), which is part of why the row pass wants monos in the first place
(§6a33: compute already beat MKL; data movement was the whole remaining gap).

## 3. Where the variant machinery lives near strided data

1. **The IL codelet matrix carries the full variant axis** (ledger item 18, "432 gates"):
   `codelets/il/avx2/` holds `n1`, `t1_dit`, `t1_dit_log3`, `t1s_dit`, `t1_dif`,
   `t1_dif_log3` × fwd/bwd × `il_in`/`il_out` boundary folds. Every twiddle-application
   variant of the *composed* 1D path got an interleave-folded twin — per-stage measured
   selection survives across the layout boundary.

2. **`strided_tw.h` (§6a40/41, dormant) uses a fourth method of its own**: per-leg tables
   `[W^m | W^2m | W^3m]` indexed by **column position**, streamed in a dedicated vector
   pass. It shipped correct but adoption declined it everywhere ("gate-fidelity lesson":
   gates must mirror production context).

3. **The 2D transform as a whole is a combination**: mono strided rows (zero twiddle
   machinery) + a variant-composed **column** chain with full measured selection — 2D
   wisdom entries carry per-stage variant codes for the column, e.g. 256² col =
   4·4·4·4 `[FLAT, T1S, T1S, LOG3]`.

## 4. The geometric insight (publication-relevant)

The three-method menu is not universal — it is a consequence of **lane-constant twiddle
geometry**. In the 1D lane-batched layout, W is constant per (stage, group, leg) across all
K lanes, which is exactly what makes scalar-broadcast tricks (T1S) and raw-per-leg+cf0
(LOG3) profitable. In a strided-row composition the twiddle **varies along the vector axis**
(columns), so a table must be streamed — the menu collapses toward FLAT-like methods.
`strided_tw.h`'s column-indexed tables are the geometry-correct analogue. This extends the
"3 twiddle methods, per-stage measured" thesis: *the admissible method set is a function of
which axis the SIMD vector runs along relative to the twiddle index.*

## 5. Roadmap — the `t1_strided` family (build only on demonstrated demand)

If strided composition is ever revived (e.g. row lengths outside the mono set where the
tiled fallback measurably loses, or growing the r2c mono ceiling past 512), the natural
emission is a **twiddle-applying strided stage codelet** — `rN_t1_{fwd,bwd}_strided` — i.e.
fold `strided_tw.h`'s separate twiddle pass into the mono's load/store lattice, the same
fusion pattern as `--post-tw` (§6a53) and the IL boundary folds.

Its own variant axis would then be measurable:
- **streamed column-indexed table** (what `strided_tw.h` does today, unfused);
- **fused table-in-lattice** (the `t1_strided` default candidate);
- **recompute-in-register** (angle recurrence; trades table bandwidth for FLOPs — attractive
  exactly here because the strided regime is movement-bound);
- LOG3-style raw+cf0 does *not* transfer (needs lane-constant W).

Gating discipline (learned §6a41, §6a25): build the emitter mode only after a hand-written
reference proves the spec (§6a36 pattern), gate BIT-exact against
mono-row + separate-twiddle-pass, and let **per-shape adoption** decide — in-production
context, not synthetic A/Bs. If adoption declines it again, record the negative and leave it
dormant; the mono + tiled-fallback pair already covers production shapes.

**Explicit non-goals today**: no speculative `t1_strided` emission, no variant sweep for the
monos (§2 — the axis is undefined there), no strided_tw revival without a shape where the
tiled fallback demonstrably loses.
