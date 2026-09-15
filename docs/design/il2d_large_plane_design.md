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

Expected at 2048x2048, T=8: 128 MB (strips) + 128 MB (rows) against
today's 384 MB — about 3.5 ms for 5.2. The race decides; a cell whose
banded walk stays ahead keeps it.

## 2. The four-step's transpose folded into the row pass

The natural class today: the 2D child writes its plane (rows in place on
the plane after the column pass), then `_k1fs_transpose` reads the plane
and streams the k2-major destination — a third sweep (read 64 + write 64
MB: 2.6 ms serial, 1.5-2 ms at T=8 at 4194304). Folded: the row pass
runs its rows in groups of G (16, or the band width when smaller) into a
per-worker scratch of G rows, and the group leaves as the 16-column block
store of `_k1fs_transpose_range` (lane permutes, streaming stores)
straight into the k2-major destination at columns k1(p) of the group's
plane rows — the plane row is read once (L2-hot after its column stage or
its band) and never written back with row output. The child carries the
hook as it carries the twiddle: `il2d_fs_out` (the destination) and
`il2d_fs_k1` (the column map k1(p)), set by the four-step plan for the
natural class before execute and absent otherwise; the walks — the serial
banded (rows per band) and unbanded (rows after the pass), the threaded
band arm (rows per band) and the row phase of the strips arm — take the
group store in place of the in-place row when the hook is present.
Backward mirrors: the group's rows are gathered from the k2-major source
by the block load (the transpose's backward kernel into the scratch), the
conjugate-twiddled row plan runs from the scratch into the plane, then the
column pass runs reversed — rows first, as the four-step's backward already
orders them. In place: the column pass consumes the caller's array before
any row is stored back into it (strips: the whole pass; bands: the wide
prefix), so the destination is the caller's array in both placements and
the four-step's own plane is the child's only plane.

The four-step's natural class then costs the scrambled class's two sweeps
plus the scattered k2-major write in place of the plane's row write —
no separate transpose in either direction. `_k1fs_transpose` stays as the
kernel the group store calls.

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

- [ ] 1. This design.
- [ ] 2. The strip width: the `sw` field and ladder, the sub-strip loop in
      the strips tramp and the serial unbanded walk, the arms in both races
      (serial axis race, column-MT race, every class), the `sw=` / `mtarm=`
      / `msw=` tokens banked and replayed at the raced T.
- [ ] 3. Re-race the shipped store's 2D rows above 16 MB at T=8 (their
      `cmt`/`cmtt` tokens dropped so the cold race runs) and the four-step's
      per-T splits; gates ALL PASS; measure.
- [ ] 4. The folded transpose: the `il2d_fs_out` / `il2d_fs_k1` hook, the
      group store in the four walks both directions, both placements;
      `_k1fs_transpose` demoted to the kernel; the four-step's natural
      execute drops its separate sweep. Gates ALL PASS; measure.
- [ ] 5. Records: `v1_0_results.md` (the upper-band table and the 2D
      section, in place), `k1_fourstep_design.md` (the natural class
      paragraph), `design_contracts.md`, memory.
