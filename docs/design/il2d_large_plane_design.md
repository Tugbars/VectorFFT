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

## 3. The SUPER-BAND — the four-step's natural class folds its transpose (owner 2026-09-15: "let's pursue this")

The natural class keeps the SCRAMBLED child (its chain, its row plans,
its twiddle records) and walks the child's plane itself, in the four-step
(`k1_fourstep.h`), so the row pass can store the k2-major output directly
and the separate transpose sweep disappears.

The column chain (R_0 ... R_{m-1}) digit-reverses the plane rows: column
k1(p) has p's MOST significant digit (radix R_0) as its LEAST significant
digit — k1 = m + R_0 * k1'(q) for p = m * N1/R_0 + q, m < R_0. So the R_0
rows {m * N1/R_0 + q : m < R_0} hold R_0 CONSECUTIVE columns, a run of
R_0 * 16 B = two whole lines at R_0 = 8. The SUPER-BAND j is R_0 blocks of
wl contiguous rows at plane stride N1/R_0 (block m = rows m * N1/R_0 +
j * wl + [0, wl)), wl = R_{m-1} = the last stage's span, so every block is
closed under the last column stage (the n1c leaf) and the super-band
holds wl runs of R_0 consecutive columns. Its walk, forward:

1. the wide prefix, stages 0 .. m-2, on the whole plane — the tier's
   digit-split stages, serial or across the pool (`_il2d_stage_digits_mt`),
   the caller's array to the four-step's plane;
2. per super-band, in a per-worker scratch of R_0 * wl rows: the blocks
   copied in, the last stage on each block, the twiddled row plans on every
   row (`_il2d_row_exec_t`, the position's own record), then for each row
   index i the R_0 rows {block m, row i} stored as columns R_0 * K(j, i) + m
   of every output row — an R_0 x 16 block transpose per 16 columns of k2
   (lane permutes, streaming stores when 32-B aligned). The plane is READ
   once here and never written: three sweeps, the scrambled class's count.
   Super-bands are disjoint in plane rows and in output columns: the
   pool's unit axis.

Backward mirrors and runs the rows first: per super-band the R_0 rows of
each i gathered from the k2-major source (the block transpose's inverse),
the row plans backward with the conjugate twiddle, the last stage
reversed on each block, the blocks written to the plane; then the reversed
prefix, plane to the destination. In place the caller's array is the
prefix's source (consumed before any run lands) and the reversed prefix's
destination (written after every run was read): one plane, both
placements. The map K(j, i) and the R_0-run law are COMPUTED at create
from `_il2d_nat_perm` of the chain and checked (every super-band's runs
R_0-aligned and complete); a chain that fails the check has no super-band
form.

The chain is the form's own axis: residency asks for a small R_0 * wl (64
rows x 32 KB = 2 MB at 2048x2048: R_0 = 8, wl = 8 — a chain like 8.32.8),
which the child's own race (8.8.32, wl = 32) does not choose, so the
super-band form carries its chain. The natural 1D cell races, per split,
form 0 (the child's banked chain + the streaming transpose) against form 1
with every chain of the tier's enumerator whose R_0 and R_{m-1} are in
{4, 8, 16} (the resolver and table builder are the tier's); the winner
banks `il_kv=1 il_sb=<chain>` on the kind-3 row beside `il_pair`, and the
per-T split race races the same arms at T, banking `il_mtsb=<chain|0>`
beside `il_mt`. Replay builds the form from the row. `VFFT_K1_FSSB=<chain>`
pins the form for a probe.

Expected at 4194304: T=8 from 7.2 to ~5.3 ms (1.3x MKL), serial from 17.6
to ~15 ms; at 2^19..2^21 the scrambled class's numbers. The race decides.

BUILT AND RACED 2026-09-16 (cold natural cells on a scratch store, the
gate ALL PASS at 1M, 2M and 4M with every arm inside 1.2e-15 of the
reference; T=8 bitwise the serial walk; in place bitwise out of place):

```
 cell            form 0 (best split)      form 1 (best split/chain)     verdict
 1048576  T=1    3.36 ms  512x2048         3.63 ms  256x4096/8.8.4       form 0
 1048576  T=8    670 us   512x2048         777 us   2048x512/8.32.8      form 0
 2097152  T=1    7.68 ms  512x4096         8.35 ms  512x4096/8.8.8       form 0
 2097152  T=8    1.99 ms  512x4096         3.04 ms  1024x2048/8.16.8     form 0
 4194304  T=1   18.3 ms   2048x2048       18.9 ms   2048x2048/8.8.8.4    form 0
 4194304  T=8    7.31 ms  4096x1024        6.46 ms  4096x1024/8.64.8     form 1 (-12%)
```

The fold does what it was built to do — at its own split and thread
count it beats form 0 (2048x2048 T=8: 7.28 against 9.50 ms) — but the
saved sweep comes back on the store: every k2-major run touches one page
per output row, the TLB floor the strips met, and below 64 MB the
transpose it replaces was never the cost. One cell, one thread count,
12%: the four-step's largest cell from parity to ~1.07x MKL.

RULING (owner 2026-09-16, "only race above where L3 can't cover the
transforms anymore"): the form stays, and it is an ARM only where the
plane outgrows the last-level cache — N x 16 B > `vfft_cpu_l3_bytes()`,
the hardware's own L3 (an L3-less or unknown part admits it and the race
decides), never a baked constant: on this host's 36 MB that is 4194304
alone, on a 16 MB part the 2M cell too, on a 100 MB server part nowhere.
Both races take the residency sub-ladder (R_0 = 8, last in {8, 16},
depth <= 3: the serial race found nothing outside it that wins). Below
the gate the natural class is form 0 without a race, the cells the
transpose was never the cost of.

RE-RACED 2026-09-16 on the quiet machine, the shipped store's 4194304
natural cell at T=8: form 0 2048x2048 7.36 ms, super-band 2048x2048/8.32.8
7.64, 4096x1024/8.64.8 7.92 — form 0 won and is banked (`il_mtsb=0`). The
12% of the first quiet run (6.46 against 7.31) did not reproduce; the two
forms sit inside the threaded race's own run-to-run spread at this size.
The canonical bench at 4M, T=8: 7.09-8.03 ms against MKL 7.41-8.13,
1.01-1.04x; serial 1.43-1.44x. The super-band has no cell it reliably
wins on this host. RULING (owner 2026-09-16): KEPT — "technically sound";
form 0 and the super-band race only in cells whose plane is above L3, as
built (`_k1fs_sb_admit` in both races); below it, form 0 without a race.

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
- [x] 4. The folded transpose: BUILT (natural-child form), gated, REFUTED at every cell, DELETED. The super-band form is the owner's call.
- [x] 5. The SUPER-BAND (§3): BUILT, gated ALL PASS at 1M/2M/4M, raced — wins 4M T=8 by 12%, loses every other cell; KEPT as an arm above L3 (the owner's ruling, reaffirmed 2026-09-16 after the quiet re-race: form 0 and the super-band race only above L3). form 1 in `k1_fourstep.h` (create: the chain's
      kernels, tables, spans, the K map and the run check, the per-worker
      scratch; execute both directions, both placements, serial and across
      the pool; destroy), the natural 1D race's form x chain arms, the
      `il_kv` / `il_sb` / `il_mtsb` tokens banked and replayed, the probe
      pin. Gate: `k1_fourstep_gate` ALL PASS at 2^19..2^22.
- [x] 6. Re-race the natural cells on the shipped store (T=1 and T=8),
      measure against MKL, records in place. 2026-09-16: the shipped store
      keeps its quiet-machine form-0 rows; the 4194304 natural cell's
      threaded verdict is dropped and re-races (form 0 against the super-band
      chains) at the next quiet window — the first attempt ran under a game's
      load (MKL's own 4M number moved 6.9 to 9.4 ms in the same run) and was
      discarded. Nothing measured under load is quoted. the `il2d_fs_out` / `il2d_fs_k1` hook, the
      group store in the four walks both directions, both placements;
      `_k1fs_transpose` demoted to the kernel; the four-step's natural
      execute drops its separate sweep. Gates ALL PASS; measure.
- [x] 6b. The ztt gate's law restored (2026-09-16): the scrambled pool inside
      ZTURN-T's band is ZTURN-T's alone; the four-step's scrambled arms at
      262144 are gone (a race under load had banked one).
- [x] 7. Records: `v1_0_results.md` (the upper-band table and the 2D
      section, in place), `k1_fourstep_design.md` (the natural class
      paragraph), `design_contracts.md`, memory.
