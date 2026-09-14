# The cascade's threading method — kept as a record for ZTURN-T's MT arm

Written 2026-09-15 when the ZTURN-S cascade was deleted (owner: "only keep
the MT strategy of the old cascade alive"). The code (`zturn_mt.h`, the
`_zt_mt_*` racer in `k1_commit.h`, the `zt_mt*` wisdom tokens) is gone; this
is the method it implemented, verified on the emitted bodies, so ZTURN-T's
threaded arm starts from it rather than from nothing. Its own measurements
are in `docs/performance/v1_0_results.md` ("K=1 scrambled cascade —
intra-transform MT", 2026-08-27).

## The principle: sections are loop-range restrictions, never a new walk

A stage of a run-contiguous or block-split cascade already contains its
parallel axis. Threading it restricts the serving loop's range per thread;
nothing is recomputed, nothing is decomposed differently, so **the MT output
is bitwise the ST output** and the gate that holds it is a `memcmp`.

| phase | what the loop is | the range | pointer edits |
| --- | --- | --- | --- |
| ingest (the turn) | a pure map over ingest columns, linear in k, no tables | a column count range | two (in, out) |
| mids | one twiddle record per group, the stream walked linearly; group g owns its contiguous span | a group range | three (base, stream cursor, count) |
| terminator | linear in k over the plane, the output at 2k | a column range, 8-column aligned so both quad forms hold | two |

Stages stay ordered; one join per stage (~100 ns each on the pool). Ranges
are cut so that every thread's span holds whole groups of the stage (the
same law as tiling: a span is legal when the stage's group divides it).

## What declined to thread, and why

Two configurations opted out: the tiled driver (its per-tile stage order is
not a range restriction of one stage's loop) and natural order (its
rho-order table walks are not a simple range). Declining was the correct
outcome, not a gap: an arm that cannot be a pure restriction is not built.

For ZTURN-T this maps as: the natural class's ingest (`t0tp`, a map over
columns with `rb[]` per column) and its mids (`tmg`, one call per group with
a group-invariant stream) are pure ranges; the terminator (`tlf`/`tlfi`,
linear in k) is a range; the tiled loop threads per TILE (each tile is an
independent block of the tiled prefix); the plain class threads per block
for its suffix and per group range for its sweeps.

## The verdict: raced per thread count, banked on the served row

Threading is a raced plan parameter, never a rule. The arm race
(`_zt_mt_race`) timed the ST walk against the MT walk at the plan's T on
the plan's own buffers, aliased for in place and distinct for out of place
(two different measurements, two verdicts). The verdict was banked on the
recipe row that served the cell as `zt_mt_t=<T> zt_mt=<0|1>` (and the
`_ip` pair for in place): a T match replays, a mismatch re-races and
re-banks (validity-condition banking — "cores sharing one transform",
`measurement_arms.md`). A re-raced recipe row is rebuilt fresh, so it
drops the MT verdict with the recipe. An env pin (`VFFT_ZT_NO_MT`) beat
wisdom and was never banked (the tcut law). The engagement counter
(`vfft_zt_mt_passes`) proved the arm ran — MT results are vacuous without
an engagement proof.

For ZTURN-T: the same tokens on the `il_route=ztt` row (`il_mt_t`,
`il_mt`, as the flat DIT already banks its threading verdict), the same
aliased/distinct pair, the same engagement counter, and the pacing law that
MT arms need (the parking trap: threads park between paced races, so the
MT arm is measured unpaced inside its own race).
