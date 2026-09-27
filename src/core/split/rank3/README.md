# `core/transforms/fftnd/` — rank-N (3D / 4D) transforms

Rank 3 and 4 on one split pair, as A/B/C(/D) passes; OOP is copy-then-in-place,
the same shape as 2D. The engines were rank-general all along
(`FFTND_MAX_RANK = 4`) — rank 4 was an exposure, not a new engine.

**Contracts.** `K == 1` (a batched rank≥3 call arrives as a K=1 override plan),
order DEFAULT or SCRAMBLED only, real transforms out-of-place. Rank-3 NATURAL is
the `fftnd_natorder.h` `nat_col_list` follow-up and is refused loudly until then.
2D or higher trig is refused: DCT/DST/DHT are 1D only.

**Wisdom.** A dedicated `(N1,N2,N3)` table. HIT → `vfft_fft3d_plan_from_entry`;
MISS → greedy per-axis exhaustive with the inners visible, banked through
`vw2_3d_bank_entry` when the result is expressible.

| file | role |
|---|---|
| `fftnd.h` | the rank-general builder (`stride_plan_nd`) and executor |
| `fftnd_r2c.h` | rank-N real transforms, including the interleaved-pair complex side |
| `fftnd_planner.h` | RETAINED, not yet wired into any TU (owner ruling 2026-09-01) — the planned rank-N planner; `allow_calibrate = 0` gives the heuristic-shaped fallback |
| `fftnd_wisdom.h` | RETAINED, not yet wired — the rank-N wisdom generation `fftnd_planner.h` consumes |
| `fftnd_natorder.h` | RETAINED, not yet wired — the rank-3 NATURAL `nat_col_list` follow-up |
| `fftnd_create.h` | the rank-3 / rank-4 CREATE tier — both dispatcher arms, lifted whole |
