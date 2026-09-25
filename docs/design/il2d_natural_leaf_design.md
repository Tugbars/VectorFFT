# The 2D natural leaf — staged rows, not scattered stores

Design, 2026-09-16. Basis: the 2D interleaved tier's NATURAL class against
its scrambled class on the same cells, `benches/il2d_mt_probe.c`
(`VFFT_PROBE_NAT`), quiet machine, 2026-09-15:

```
 plane       class    T=1        T=8       plane bytes
 2048x512    scr      2.42 ms    0.34 ms   16 MB
 2048x512    nat      3.12 ms    0.86 ms
 512x2048    scr      2.43 ms    0.29 ms   16 MB
 512x2048    nat      3.28 ms    0.53 ms
 2048x2048   scr     15.6 ms     4.8 ms    64 MB
 2048x2048   nat     52.3 ms     9.8 ms
```

1.3x at 16 MB, 3.4x serial and 2x threaded at 64 MB — and the 3D tier's
natural class stands on this pass (`3D_natural_il_design.md`,
`ilnd_natural_strip_design.md`).

## The mechanism

The natural pass runs the wide stages into a scratch plane and redirects
the LEAF: the n1c kernel of block b reads its R contiguous rows from the
scratch and writes its R output rows to their natural positions,
perm[b*R] + r * N1/R — R rows at a stride of N1/R rows (`_il2d_col_pass_nat`,
`_il2d_col_pass_nat_range`, `_il2d_nat_leaf_range`: OLs = (N1/R) * rn).
At 2048x2048 with R = 32 that stride is 64 rows x 32 KB = 2 MB, and every
power-of-two multiple of the set stride maps to the SAME L1 and L2 sets:
the kernel's 32 output streams contend for 12 and 16 ways and evict each
other line by line, every store a miss with a read-for-ownership. The
backward gather reads the same 32 streams. At 16 MB planes the stride is
512 KB and the conflict is partial; at 64 MB it is total. The scratch
plane is also plain `malloc` — 16-B aligned rows under kernels built for
64-B lines.

## Contract

The natural class's leaf never stores or loads its R natural rows at their
stride. Forward, the leaf writes its R rows into a per-worker STAGING
block (R rows x the leaf's column count, contiguous, L2-resident: 1 MB at
32 x 2048), the row plans run there when the walk fuses rows, and each
row leaves as ONE sequential stream to its natural position (streaming
stores when 32-B aligned, a plain copy otherwise). Backward, each natural
row is copied sequentially into the staging before the leaf reads it (and
the row plan runs there first when the walk fuses rows). The kernels see
the same values in the same order: bitwise the unstaged walk. The sweep
count does not change — the staging is the band's own L2 — and the block
arm of the threaded natural walk fuses its rows into the leaf phase (the
row phase it ran afterwards, a re-read of the plane, is gone for that
arm). A finished row leaves by streaming stores; a row the walk will
revisit (the strip arm, the unbanded walk: rows after the pass) leaves by
a cached copy, so the row phase finds it in L2. The staging is 64-B
aligned (`VFFT_ZS_ALLOC`).

The leaf's form is a RACED plan parameter at T > 1 (built 2026-09-16, the
raced-twin law): the serial walk is staged outright — staged wins every
serial cell — and every threaded arm of a natural cell runs staged and
strided ("block/str", "strips64/str"), the winner banked `nls=` beside
`cmt`/`mtarm`/`msw` on the row keyed by that T and replayed there. At the 16 MB planes the
strided block arm wins (the copy buys nothing where the stride's conflict
is partial); at the 64 MB planes the staged arms win by 27-40%.

## Where

- `il2d_cols.h`: `_il2d_nat_leaf_stage` (block -> staging, the leaf kernel
  with OLs = the staging pitch), `_il2d_nat_stage_out` (staging rows ->
  natural rows, sequential streams), `_il2d_nat_stage_in` (natural rows ->
  staging), `_il2d_nat_leaf_unstage` (staging -> the scratch comb, the
  backward leaf); `_il2d_nat_leaf_range`, `_il2d_col_pass_nat`,
  `_il2d_col_pass_nat_range` take the staging and call them.
- `il2d_col.h` / `fft2d_create.h`: `natstage` (T x R_last x rn complexes)
  beside `natscr`, both aligned; freed with the plan.
- `vfft_execute.h`: the serial banded natural walk fuses its rows in the
  staging per block (forward: leaf, rows, stream out; backward: stage in,
  rows, leaf), the unbanded walk stages the leaf.
- `il2d_tier.h`: the threaded natural walk's block arm (mode 4) stages per
  block with its rows fused (the caller's tid picks the staging and the row
  clone) and skips the row phase; the strip arm (mode 5) stages its leaf
  over its column range and keeps the row phase (a strip's rows are not
  whole).

## Gates

The tier's own gates and the 3D probes' bitwise passes; an A/B probe
switch (`VFFT_IL2D_NATLEAF_OLD=1`, the scattered leaf) exists only while
this is built, for the bitwise comparison and the timing, and is deleted
with the old path once the new one is proven. The four-step is untouched
(its child is the scrambled class).

## Measurement

`il2d_mt_probe` with `VFFT_PROBE_NAT`, cold-raced cells, quiet machine,
2026-09-16, natural against scrambled on the same cell:

```
 plane       nat T=1   scr T=1   nat/scr   nat T=8   scr T=8   nat/scr   threaded verdict
 2048x512    2.97 ms   2.37 ms   1.25      0.63 ms   0.34 ms   1.85      block, strided
 512x2048    2.81      2.32      1.21      0.52      0.34      1.53      block, strided
 1024x1024   2.92      2.55      1.15      0.71      0.34      2.09      strips64, staged
 2048x2048  15.9      14.0       1.13      7.09      4.96      1.43      block, staged
 1024x4096  15.7      13.9       1.13      5.96      6.73      0.89      block, staged
```

Before (the strided leaf, 2026-09-15): 1.3x serial at 16 MB, 3.4x serial
and 2x threaded at 64 MB. Bitwise the strided leaf in both directions at
every plane (the A/B switch, deleted once the strided path became the
raced arm). The 3D tier still calls the pass without a staging (its own
scratch, its own campaign).

A probe caveat, not a verdict defect: timing a 64 MB natural cell in the
SAME process right after its cold race read 40-53 ms serial on several
runs, while the banked verdicts replayed in fresh processes run 15.2-15.6
ms and every pinned variant of a "slow" verdict (8.8.8.4 with wl 64/256,
rows in place or out) runs 15-17 ms. The in-process reading after a race
is the artifact (the race's plane churn), and nothing timed that way is
quoted; the shipped store's rows stand.

## Checklist

- [x] 1. This design.
- [x] 2. The staging: allocation, the four leaf helpers, the three passes.
- [x] 3. The walks: serial banded (rows fused in the staging), unbanded,
      the threaded block arm (rows fused, row phase dropped) and strip arm;
      the leaf's form raced per threaded arm (`nls=`).
- [x] 4. Bitwise A/B against the scattered leaf at every probed plane,
      both directions, both thread counts; the real tier's gate and the
      four-step gate (the 3D tier's call is unchanged).
- [x] 5. Measure; the switch deleted; the shipped store's 2D natural rows
      at 4 MB and above dropped and the five probed planes re-raced into it
      (the 64 MB verdicts replay at 15.2-15.6 ms serial in fresh processes);
      records.
