# The four-step above 262144 — the K=1 interleaved upper band on the 2D tier

Design, 2026-09-15. Basis: the owner's ruling (2026-09-15: "our Bailey engine
is the best solution for this and MKL's code also shows that they are using
Bailey for 256k and above"), the MKL dispatch RE (`mkl_il_split_tiering`:
`order >= 18` calls a different engine; the avx512 static RE: `>131072`
`cFftFwd_Large_64fc`, blocked), the K=1 door's ceiling (`k1_commit.h`
`VFFT_ZTT_MAX_N` = 262144: above it a pow2 interleaved request returns 0 from
the planner race and the bench's direct cell is silent), the 2D interleaved
tier (`il2d_tier.h`, `vfft_execute.h`: the banded column walk with the row
pass fused per band, both partitions threaded, per-row K=1 children through
the front door, the 2026-08-25 standing 11/11 over MKL CCE), and ZTURN-T's
standing at the top of its band (262144 natural out of place 1.15x over MKL
at one thread today, 1.7x on the day of record; 6.2x at T=8).

## Contract

K=1 interleaved c2c, both placements, both order classes, every power of
two from 524288 to 4194304 (2^19..2^22) is SERVED — today the door refuses.
262144 stays ZTURN-T's cell unless the four-step beats it in the cell's own
race. The 2^a·odd cells above 262144 are not in this design (the 2D child
can hold odd axes, so they are a later admission, not a new engine).

## The engine: N = N1 × N2 on the 2D interleaved tier

Bailey's four-step on a row-major N1 × N2 view of the signal:

```
x[n1*N2 + n2]                                       (N1 rows of N2 contiguous complexes)
step 1  columns:  Y[k1][n2] = sum_n1 x[n1][n2] W_N1^(n1 k1)     N2 transforms of length N1, stride N2
step 2  twiddle:  Y[k1][n2] *= W_N^(k1 n2)                        one multiply per point
step 3  rows:     Z[k1][k2] = sum_n2 Y[k1][n2] W_N2^(n2 k2)      N1 transforms of length N2, contiguous
        X[k1 + N1*k2] = Z[k1][k2]                                the output is the TRANSPOSE
```

Steps 1 and 3 ARE the 2D interleaved tier: the column-axis chain over the
N1 × N2 plane, walked in bands of `wl` rows with the row pass fused while the
band is L2-hot; the rows are K=1 plans of length N2 created through the front
door (ZTURN-T at 2048 and 4096, the pairs and solos below). Step 2 fuses
into that walk at the one seam every row passes through, `_il2d_row_exec`:
a per-row multiply just before the row plan (forward) and, mirrored, just
after it with the conjugate (backward). The band walk's order already
matches what the twiddle needs in both directions — forward `[suffix, rows]`
per band then nothing wide after; backward `[rows, suffix reversed]` per
band, then the reversed wide prefix — because the suffix stages of a band
touch only that band's rows, so `[rows⁻¹, conj twiddle, suffix⁻¹]` per
band is exact. Nothing in the column chain, the banding, the row children
or the two threaded partitions changes.

The twiddle. Row position p of the plane holds column output k1(p) (the
scrambled column chain leaves k1 digit-reversed; the plan knows the map,
`_il2d_nat_perm`). Per position p a two-level table, ZTURN-T's scheme:
n2 = a·B + b, w(k1, n2) = C[p][a] · F[p][b], B = 32 (or the largest power of
two ≤ √N2): (N2/B + B) complexes per row, N1·(N2/B + B)·16 bytes in all —
1 MB at 1024 × 1024, ~1 ulp. Two complex multiplies per point on an L2-hot
row, small against the row transform.

## The two order classes

SCRAMBLED (and DEFAULT): the plane as it stands after step 3 — position
p·N2 + k2 holds frequency k1(p) + N1·k2. That is a fixed permutation of
the plan (the 2D child's own column order times the four-step's
transpose), self-consistent, inverted by the matched backward: a legal
scrambled writer, no extra pass. In place: the 2D child in place, nothing
else.

NATURAL: the plane transposed with the row permutation folded in — row p
of the plane becomes column k1(p) of the output (k2-major), a blocked
transpose, one sweep of the array. Out of place it reads the 2D child's
result and writes the destination; in place it goes through a plane of
scratch (N complexes: 64 MB at 2^22, the price of natural order in place
at this size, the same class of cost as the 3D natural forms). Backward
mirrors: the (inverse) transpose first, then the 2D child backward with the
conjugate twiddle. The 2D child threads by its own verdict.

The transpose is the natural class's whole cost over the scrambled class,
and it is bandwidth: 16 x 16 complexes per block through a 4 KB local
buffer, each 2 x 2 turned as one 128-bit lane permute (AVX2), the output
rows leaving as whole lines by streaming stores when the destination is
32-B aligned (no read-for-ownership; plain stores otherwise, the bytes
identical). Measured against the scalar 16 x 16 walk at 1024 x 4096
(`benches/tp_probe.c`, block size, loop order and store kind raced): 10.3
to 2.9 ms at one thread, 4.2 to 1.5 ms at eight. The pool cuts the k1
blocks across the workers (disjoint spans both ways, bitwise the serial
walk), on the caller thread, never from a worker. A transposed-write twin
of the row terminator (the rows stored k2-major directly, the sweep
folded into the band walk while the band is L2-hot) remains the lever
beyond this; MKL's six-step pays its transposes too.

## The race, and what wisdom banks

The 1D cell (N, il, order, placement) races its SPLITS: every (N1, N2) with
N1·N2 = N and both in {256, 512, 1024, 2048, 4096} — at most five arms per
cell — each arm a 2D child on its own rank-2 cell (N1×N2, lay=il, ord=scr,
in place), whose column chain, band width, row route and threading are that
cell's own raced verdicts, banked on its own rows (one 2D cell serves both
1D classes). At 262144 the standing ZTURN-T plan is an arm of the same race.
The 1D verdict banks `il_route=fs il_R1=N1 il_R2=N2` on the kind-3 row (the
`il_R1/il_R2` fields exist; route 10). Threading: the plan's T is the 2D
child's T, its verdict banked on the 2D row; the natural transpose threads
by blocks with the same pool. The SPLIT is a per-T verdict: the children's
own threaded verdicts reorder the ladder (4194304 at T=8: the serial
winner 1024x4096 runs 7.1 ms, 2048x2048 4.9 ms), so a plan at T > 1 races
the splits AT T (each child a 2D cell created at T) in its own placement
and banks `il_mt=N1 il_mt_t=T` (`il_mt_ip` / `il_mt_ip_t` in place) on the
cell's row — ZTURN-T's tokens, each route reading them as its own arm;
replay rebuilds the banked split when it differs from the serial row's.
`VFFT_K1_FS=N1xN2` pins a split for a probe (never banks, skips the per-T
race). No fallback: a cell whose splits all refuse (a missing row
plan) refuses.

## Gates

- `k1_fourstep_gate`: at 2^19..2^22 (and 2^18 pinned), both classes, both
  placements: the forward against a long-double reference at spot bins
  (natural: read at the bin; scrambled: through the plan's own permutation,
  which the gate computes from (N1, N2) and the 2D child's column map), the
  matched roundtrip, in place bitwise out of place, T=8 bitwise T=1 with the
  engagement counter, replay from a warm store with no race.
- The 2D tier's own gates unchanged (the twiddle hook is inert when the
  table is absent: the 2D cells stay bitwise their standing).
- `bench_1d_vs_mkl` admits the upper band to its direct K=1 cell.

## Measurement (a quiet machine)

`bench_1d_vs_mkl --k1noop` (natural out of place) and the scrambled and
in-place cells through the front door at 2^18..2^22, one thread paced, T=8
unpaced with engagement, against MKL DFTI 1D; the create log naming the
split and the child's verdicts per cell. MKL's own one-thread times at these
sizes (`benches/mkl_1d_probe.c`) are the target datum.

## Checklist

- [x] 1. This design.
- [x] 2. The twiddle hook in the 2D tier: plan fields (per-position two-level
      tables, the plane base for the row index), the multiply in
      `_il2d_row_exec` / `_il2d_row_exec_t` (forward before, backward after
      with the conjugate); inert when absent. Gate: the 2D probes bitwise.
- [x] 3. `src/core/oop/k1_fourstep.h`: create (the 2D child through the
      internal 2D create at N1×N2, IL, scrambled, in place, the plan's T; the
      twiddle tables; the natural transpose scratch), execute both
      directions, both placements, both classes (the blocked permuting
      transpose), destroy.
- [x] 4. The door: route 10 in `oop_plan.h`; the split enumerator and the
      race (with ZTURN-T at 262144) in `k1_commit.h`; `il_route=fs il_R1
      il_R2` bank and replay; execute dispatch; destroy; the env pin.
- [x] 5. `k1_fourstep_gate.c`; the bench's admission. Gate ALL PASS.
- [x] 6. Race and bank at 2^18..2^22, both classes, both placements, T=1 and
      T=8, on a quiet machine; the shipped store gains the rows.
- [x] 7. Measure against MKL (quiet machine); rule what the numbers say at
      262144 (ZTURN-T or the four-step) and on the natural class's transpose.
- [x] 8. Records: `v1_0_results.md` (the 1D K=1 section's band extended, in
      place), `include/vfft.h` (the K=1 tiers paragraph: the four-step above
      262144), `design_contracts.md` section 4, memory.
