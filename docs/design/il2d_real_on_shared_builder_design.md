# R6: the 2D real tier's column chain through the shared builder (design, 2026-09-17)

`_il2d_col_build` (il2d_tier.h) builds one column axis under a wisdom key:
env pin > banked row > the chain race > the column-axis Bluestein, its
per-stage forms, the natural-leaf redirection, the N-arm race, the stage
tables. The 2D c2c tier calls it (`fft2d_create.h:262`) and copies the
result into its locals; the 3D tier calls it for both axes. The 2D REAL tier
carries the same decision inline -- ~170 lines of `fft2d_create.h` (from
`rok = _il2d_env_chain(...)` to the end of its N1-arm race) that are a copy
of the builder's block with its own lookup and bank -- and three
`_il2d_blu_build` calls that are the builder's cold build, replay-rebuild and
N-arm race. Every divergence the builder fixed this week (the `nat_req`
race pass, the one-candidate race, the greedy's deletion, the blu
replay-rebuild) had to be found twice or was found once and left the copy
behind. This design makes the real tier the builder's third caller.

## What is genuinely different, and stays

Read from the code, not assumed:

- **The row.** The real tier's row IS the ilcol key with `real = 1`:
  `vw2__ilcol_key` builds `{t=r2c, rank 2, N1xN2, ord, lay=il}`, exactly
  `vw2_2d_rl_lookup/bank`'s key. `chain` and `blu` are direction-shared on
  it. So the builder's lookup and bank land on the real tier's row as they
  are.
- **The per-direction tokens.** `rw`, `wl`, `cmt`, `cmtt` are spelled by
  `vw2__rl_tok(is_c2r, i)`: plain for r2c, `_c2r`-suffixed for c2r, on the
  one row. The builder reads the PLAIN names, so its `bwl/bcmt/bcmtt`
  out-params are r2c's for a c2r create. THE REAL TIER IGNORES THEM and
  keeps `vw2_2d_rl_lookup` for its four direction tokens, as today. One
  extra read of a row already in memory.
- **The bank order.** The builder banks the chain row (fresh record, with
  the race time), forms update it, and `_il2d_real_rowrace` later banks
  `rw`/`wl` through `vw2_2d_rl_bank`, which UPDATES an existing row
  (`have = vw2_lookup ... vw2_update_field`), so `forms=` survives. Same as
  today's order.
- **The width.** `rn = N2/2 + 1` (hp1): a builder parameter.
- **The natural leaf.** The real tier's `_il2d_nat_perm` block (:636) is the
  builder's M4-lite leaf re-implemented; `nat_req =
  vfft_policy_rankn_axis_nat(2, 0, il2d_ord)` gives the builder the same
  answer, and its copy-out carries `nat/natperm/natscr`.
- **The N1-arm race** (chain vs Bluestein for an odd-radix chain): the real
  tier's own comment says "the same two arms as the c2c tier". The builder's
  version is gated the same way (`!c->blu && !getenv("VFFT_IL2D_CHAIN")`).

## The change

In the real branch, replace the inline block with the c2c branch's pattern:

    vw2_ilcol_key_t ck = { 2, N1, N2, 0, il2d_ord, 0, /*real=*/1 };
    vfft_ilcol_t col;  memset(&col, 0, sizeof col);
    rok = _il2d_col_build(W, cfg, &ck, N1, (size_t)N2 / 2 + 1,
                          vfft_policy_rankn_axis_nat(2, 0, il2d_ord), &col,
                          il2d_fm, sizeof il2d_fm,
                          &bwl_ignored, &btf_ignored, &bro_ignored,
                          &bcmt_ignored, &bcmtt_ignored, &il2d_bblu);
    /* the copy-out, verbatim from :262-292 */
    ... il2d_nst = col.nst; memcpy(il2d_R, col.R, ...); ... il2d_nat = col.nat; ...
    /* the four per-direction tokens, exactly as today */
    (void)vw2_2d_rl_lookup(&W->vw2, N1, N2, is_c2r, tmpR, &tmpn, &il2d_brw,
                           &il2d_bwl, &il2d_bcmt, &il2d_bcmtt, &tmpblu, il2d_ord);

Deleted: the real tier's env-pin call, its `vw2_2d_rl_lookup`-gated replay
of the chain, its enumerate-and-race, its `vw2_2d_rl_bank` of the chain,
its forms serve, its natural-leaf block, its cold `_il2d_blu_build`, its
replay-rebuild, its N1-arm race. Kept: everything from the row-route race
on. `hasodd`/`be`/`il2d_bblu` consumers after the block are re-read for what
they still need.

## Gates

- `il2d_real_gate` (r2c and c2r, cold then warm, bitwise-equal forwards).
- A before/after ARM CENSUS of real cells (`il2d_real_census.exe`, `[il2d-real]
  chain race ... arms`): R6 must leave every real cell's arm list IDENTICAL
  -- it is a move, the pool is the same pool. (R2, applied first and censused
  separately, is the one that changes arms.)
- The sweep.

## Checklist

- [x] 1. This design.
- [x] 2. R2 applied and censused.
- [x] 3. Done 2026-09-17: 298 lines -> 55.
- [x] 4. `il2d_real_gate` ALL PASS both directions; census arms identical to R2's after.
- [x] 5. Records.
