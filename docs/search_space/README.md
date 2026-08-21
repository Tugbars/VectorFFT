# Search space

What each engine actually **races**: the axes, their legal values, the
measured candidate counts, and — equally recorded — what has no search at all.

One document per engine. These are inventories, not flows: for the sequence a
calibrator runs (enumerate → build → gate → bench → bank → replay) see
[docs/wisdom/](../wisdom/), which covers the pipelines. This folder answers a
narrower question: **given a cell, what are the arms?**

## Documents

- [consistency_il_c2c_vs_real.md](consistency_il_c2c_vs_real.md) — cross-engine
  consistency audit (2026-08-22): IL C2C vs IL r2c/c2r, with split as the
  maturity benchmark. 6 independent dimension audits, each adversarially
  verified. Headline: the composite is arithmetically sound; what is behind is
  everything downstream of the create-time decision. Also finds IL real AHEAD
  of both siblings on wisdom lookup discipline.
- [parity_plan.md](parity_plan.md) — the plan to bring the BACKWARD space up
  to the forward one, and r2c/c2r up to IL C2C. Phased, with the unmeasured
  assumption isolated into its own gating falsifier.
- [il_c2c.md](il_c2c.md) — interleaved K=1 C2C (`dp_planner_il.h`). Both order
  classes: radix pair and `il_kv` mid/leaf form pools on NATURAL; chain,
  engine, terminator and tile width on SCRAMBLED; plus the `dir=bwd` backward
  pass.

## Coverage, across engines

The asymmetry is the point of this table. It is not that some engines race
badly — several do not race at all.

| engine | axes searched | verdict stored |
|---|---|---|
| IL K=1 C2C, natural | route, pair, `il_kv` mid × leaf | pair + form code |
| IL K=1 C2C, scrambled | chain, engine, terminator, tile width | chain + `t2q` + route + width |
| IL K=1 C2C, backward | `il_bkv` mid × leaf | form code, `dir=bwd` sibling cell |
| split / stride | factorization, permutation, per-stage variants, orientation | chain + vars (see [docs/wisdom/05](../wisdom/05_calibrator_pipeline.md)) |
| split kernel forms | **none** — `sp_kv` is reserved and written nowhere | — |
| IL r2c / c2r route | 2 arms in principle; in the shipped store, unraced | one route token, no `ns=` |
| Hermitian fold | **none** — one fixed implementation | — |

## Rules that hold across every engine here

1. **Counts are MEASURED, never inferred from the loops.** The IL census
   probe exists because an envelope estimate over the loop nesting was wrong
   by ~2.4x.
2. **Legality is delegated to whatever builds the plan.** A validator
   re-implemented inside an enumerator is a second copy, and it drifts.
3. **Never filter a legal arm on a model.** If it is legal, time it. A width
   that is never timed leaves no trace, so a wrong filter is undetectable
   from its own output.
4. **A truncated pool is a BIASED pool.** Enumerators walk in a systematic
   order, so a cap that truncates silently drops a *class* of candidate, not
   a random sample. Refuse loudly instead.
5. **A direction-dependent axis is a `dir=` sibling CELL**, not more payload
   bits — wisdom2 keys direction and does not key kernel forms.
