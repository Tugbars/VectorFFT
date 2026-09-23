# gauntlet_2_4096_2026-09-23

The 1D interleaved K=1 natural out-of-place record of every length 2..4,096 after
the odd-radix kernel regeneration of 2026-09-23 (reduced-angle constants and the
blocked odd form). One csv, same contract and columns as every run:

- 2050 cells whose plans use a regenerated kernel (every non-prime route with an odd
  factor >= 7) are the rows of `oddblk_2026-09-23/gauntlet.csv`, benched on the
  regenerated tree the same day (control cell 4096 in `control.csv`);
- the other 2045 cells are the rows of `gauntlet_2026-09-20/gauntlet.csv` (the
  2..2048 full retime of 2026-09-22) and `gauntlet_2048_4096/gauntlet.csv`
  (2049..4096, 2026-09-22): powers of two and the prime cells (Bluestein at a
  power-of-two M), whose kernels the regeneration does not reach.

No wisdom store here: the verdicts are the shipped ones, replayed.
