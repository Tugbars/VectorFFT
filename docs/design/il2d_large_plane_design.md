# Large planes on the 2D interleaved tier — sweeps, not shapes

Design, 2026-09-15. Basis: the four-step's 4194304 cell at T=8 (parity
with MKL, `k1_fourstep_design.md`), whose 2D child alone runs 13.4 ms
serial and 4.4-4.9 ms at eight threads while the 32 MB planes of the 2M
cell scale 5x; the phase instrument on the threaded banded walk
(`VFFT_IL2D_PHASES`, `benches/il2d_mt_probe.c`), quiet machine:

```
 plane         T=8   wl  cut  prefix (wide stages)   bands (suffix+rows)   whole
 2048x2048      8    32   2         3.53 ms                1.69 ms         5.2 ms
 4096x1024      8   128   2         4.09 ms                1.39 ms         5.5 ms
 1024x4096      8   128   1         2.39 ms                4.40 ms         6.8 ms
  512x4096      8     8   2         0.70 ms                0.37 ms         1.1 ms
```

Every wide stage is a full-plane sweep (read + write 64 MB = 128 MB) and
runs at 73-75 GB/s, the DRAM roof; so are the bands. The chain and the band
width re-raced at T=8 move the whole by at most 3% (nine chains, five
widths, `il2d_mt_probe` on cold scratch stores). At 1024x4096 the serial
verdict's band (128 rows x 4096 x 16 B = 8 MB) fits L3 with one band live
and not with eight, so its band phase is four DRAM sweeps of the band. MKL
2D at these planes is 36 ms serial (one order class apart); the target is
MKL 1D at 4194304, 6.9 ms at T=8, 24 ms serial.

## Contract

A 2D interleaved c2c cell whose plane exceeds the last-level cache, in
either order class, in both placements, at every thread count, is served
by a walk whose sweep count is the minimum the memory hierarchy allows:
one sweep for the whole column pass and one for the rows. The four-step's
natural class, whose destination is the transposed plane, pays no third
sweep for the transpose. Nothing below the last-level cache changes: the
banded walk with fused rows remains the serial and threaded form wherever
the race keeps it.

## 1. The column strip walk with a raced width

The unbanded walk already runs the whole column chain over a column range
at once (`_il2d_col_pass_range`, every stage over [k_lo, k_hi)): a strip
of N1 rows x w columns is the closed unit of the column pass (columns are
independent across every stage), and with N1 x w x 16 B under L2 the
strip is read from DRAM once and written once — the column pass in ONE
sweep, whatever the chain's cut. Today the serial unbanded walk tiles by
`wc` only under `VFFT_IL2D_WC` (never raced, never banked) and the
threaded strips arm cuts the columns into T contiguous ranges of N2/T
columns each (2048 x 256 x 16 B = 8 MB per worker at 2048x2048 — eight
L2s of 2 MB), so neither is L2-resident at a large plane and the race
never sees the form that is.

The strip width `sw` becomes a RACED plan parameter, the 3D tier's `nsw`
in 2D form (`ilnd_natural_strip_design.md`): the ladder {16, 32, 64, 128,
256} gated by N1 x sw x 16 <= L2 (`vfft_cpu_l2_bytes`), every entry an arm.

- Serial: the axis race gains the arms (wl = 0, sw) beside its banded
  widths; the unbanded walk tiles by the banked `sw` (the `wc` field: the
  env pin stays as the probe override). Banked `sw=` on the row with
  `wl=0`.
- Threaded: the column-MT race gains the arms "strips sw" for EVERY class
  (today the natural class alone races a strip partition, and unsized);
  the strips tramp walks its column range in sub-strips of `sw`. The
  verdict banks `mtarm=` (0 = the serial form threaded: bands, or the
  natural block partition; 1 = strips) and `msw=` beside `cmt=`/`cmtt=`,
  read back only at the T they were raced at, like `cmt`. The threaded
  strips form then runs the rows as its own phase (row slabs across the
  pool) — the second sweep.

Raced 2026-09-15 (the four-step gate's children on a cold store, T=8,
`[il2d-c2c] threaded arms`): the sized strips LOSE to the bands at every
large plane — 2048x2048 bands 3.6 ms, strips64 7.3 ms; 4096x1024 3.7 vs
7.6; 1024x4096 4.9 vs 6.5 — and the serial axis race kept its bands too.
A strip of N1 rows at a 32-64 KB row pitch touches one page per row per
stage: its floor is the TLB, not DRAM, and the sweep it saves never
arrives. The arms stay in both races (the race is the law; they cost
milliseconds per cold create) and the width plumbing stays as their
plan parameter; the lever moves to the banded walk's own prefix.

## 1b. The prefix pair — the wide stages two per sweep

The banded walk's prefix runs each wide stage as a full-plane sweep (the
digit-split phase). Two consecutive wide stages s, s+1 close over a set
of R_s x R_{s+1} rows — stage s+1's digit e within a stage-s block (rows
m*L_{s+1} + e + k*D_{s+1}, m < R_s, k < R_{s+1}) is exactly stage s's
digits {e + k*D_{s+1}} across the block — so the pair runs as ONE sweep:
per digit e and per column chunk (cw columns), stage s on its R_{s+1}
digit ranges (src to dst), then stage s+1 on its R_s blocks in place, 64
rows x cw x 16 B L2-resident per unit. The arithmetic per element is the
two stages' own kernels on their own table entries in another traversal:
bitwise the two-stage walk (`_il2d_col_prefix_pair`, il2d_cols.h). The
pool cuts the digits e across the workers. Backward: the reversed pair
(s+1 first, then s) — the reversed prefix's own order. Legal when both
stages carry a digit axis and L_{s+1} = L_s / R_s (every DIT chain the
tier builds). A cut of 2 (one pair) saves one of three sweeps at
2048x2048; a cut of 1 has nothing to pair.

Probed 2026-09-15 (`VFFT_IL2D_PAIR=1`, `il2d_mt_probe`, bitwise at every
cell): the pair moves the prefix from 3.5 to 3.0 ms at 2048x2048 T=8 —
the whole 4.88 to 4.71 ms (3.6%), 4096x1024 5.50 to 4.96 (10%), serial
2-4%, 512x4096 T=8 2.5% behind. Column chunks 256..2048 change nothing.
The fused sweep streams at 43 GB/s against the plain stages' 75: its 64
short runs per unit (one per row of the closed set) do not stream like
the plain stage's contiguous digit ranges, and most of the saved sweep is
paid back in bandwidth. A real but small win; not dominant (one cell
behind), so a raced arm if kept, never the walk. Held as the probe switch
pending the owner's ruling.

## 2. The four-step's transpose folded into the row pass — REFUTED

The natural class pays a fourth sweep over the scrambled class's three:
`_k1fs_transpose` reads the child's plane and streams the k2-major
destination (2.6 ms serial, 1.5-2 ms at T=8 at 4194304). Two folds were
designed and one built; neither stands.

The band-local fold (the row pass storing its group of 16 plane rows as
16 output columns while the band is hot) is not available on a SCRAMBLED
child: 16 consecutive plane rows hold columns k1(p) spaced N1/16 apart,
so the block store writes 16-B pieces of 64 lines per output row — a 4x
write amplification. Whole-line stores need 16 consecutive k1, and those
rows sit in R0 different bands (k1 = R0*k + d0 puts them at plane stride
N1/R0).

The NATURAL-child fold was built (2026-09-15: a natural 2D child, always
out of place into the four-step's plane, its rows in k1 order so 16
consecutive rows are 16 consecutive columns; the row pass in groups of 16
through a per-worker scratch, the 16 x 16 block store straight to the
destination; backward mirrored, rows first; both placements; gate ALL
PASS at 1M, T=8 bitwise). It LOSES at every cell and both thread counts,
because the 2D tier's natural class costs far more than the transpose it
saves — the natural pass (prefix into a natural scratch plane, the leaf's
scattered row writes, rows after) against the scrambled banded walk,
`il2d_mt_probe` on the shipped store:

```
 plane       class    T=1        T=8
 2048x512    scr      2.42 ms    0.34 ms
 2048x512    nat      3.12 ms    0.86 ms
 512x2048    scr      2.43 ms    0.29 ms
 512x2048    nat      3.28 ms    0.53 ms
 2048x2048   scr     15.6 ms     4.8 ms
 2048x2048   nat     52.3 ms     9.8 ms
```

Folded four-step natural vs the shipped form: 524288 T=1 1.92 vs 1.41 ms,
T=8 348 vs 274 us; 1048576 T=1 5.19 vs 3.12 ms, T=8 1.21 vs 0.61 ms. The
fold is deleted; the natural class keeps the scrambled child and the
streaming transpose (`k1_fourstep_design.md`). Two findings stand:

- The 2D tier's NATURAL class collapses at large planes: 3.4x the
  scrambled class serial at 64 MB (52 ms), 2x at T=8; 1.3x at 16 MB. The
  leaf scatter writes R rows at stride N1/R (2 MB apart at 2048x2048 —
  the same L1 sets), and its own scratch plane is a sweep the scrambled
  walk does not pay. A 2D-tier item of its own, outside this design.
- The only fold that keeps the scrambled child is the SUPER-BAND: R0
  blocks of wl rows at plane stride N1/R0 (holding R0*wl consecutive k1),
  L2-resident at wl = 8, suffix per block, rows in k1 order, the block
  store to the destination — a new band form for the 2D child raced beside
  the contiguous band, with a chain whose last stage is small. Not built;
  the owner's call.

## Gates

- `il2d_real_gate`, the four-step gate (T=8 bitwise the plan's own serial,
  in place bitwise out of place, replay without a race, at 2^19..2^22),
  `k1_pow2_gate`, `ztt_gate`; the 3D probes' bitwise passes (the 3D tier
  shares the column pass).
- A strips arm is a pure loop restriction of the full pass and sub-strips
  of a strip are the same restriction: bitwise the serial walk by
  construction, proven by the gate at every cell it wins.

## Measurement

`il2d_mt_probe` at the 4M and 2M planes (both classes, T=1 and T=8) with
the phase instrument; then the canonical bench `--k1noop` and `--k1noop
--mt` at 2^19..2^22 against MKL, the records updated in place.

## Checklist

- [x] 1. This design.
- [x] 2. The strip width: the `sw` field and ladder, the sub-strip loop in
      the strips tramp and the serial unbanded walk, the arms in both races
      (serial axis race, column-MT race, every class), the `sw=` / `mtarm=`
      / `msw=` tokens banked and replayed at the raced T.
- [x] 2b. The strips arms raced at the large planes: REFUTED (above); the
      arms stay, the width plumbing stays.
- [ ] 2c. The prefix pair: the kernel walk, the serial and threaded prefix
      of the banded walk in both directions and both classes, the serial
      axis race and the threaded race each gaining the pair as an arm
      (`pp=` on the row with wl; `mpp=` with cmt/cmtt), replayed at the
      raced T. Gate: bitwise the two-stage walk at every cell.
- [ ] 3. Re-race the shipped store's 2D rows above 16 MB at T=8 (their
      `cmt`/`cmtt` tokens dropped so the cold race runs) and the four-step's
      per-T splits; gates ALL PASS; measure.
- [ ] 2d. The prefix pair as a raced arm (or deleted): the owner's ruling.
- [x] 2d. The prefix pair: DELETED (owner 2026-09-15: a 4-10% arm that loses a cell is not worth its race cost and its code); the strip-width arms stay.
- [x] 4. The folded transpose: BUILT (natural-child form), gated, REFUTED at every cell, DELETED. The super-band form is the owner's call. the `il2d_fs_out` / `il2d_fs_k1` hook, the
      group store in the four walks both directions, both placements;
      `_k1fs_transpose` demoted to the kernel; the four-step's natural
      execute drops its separate sweep. Gates ALL PASS; measure.
- [ ] 5. Records: `v1_0_results.md` (the upper-band table and the 2D
      section, in place), `k1_fourstep_design.md` (the natural class
      paragraph), `design_contracts.md`, memory.
