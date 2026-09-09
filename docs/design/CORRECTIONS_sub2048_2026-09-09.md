# Corrections to two tracked design notes (2026-09-09)

Both files below are tracked and under another session's edit, so the fixes
are given here as replacement text rather than applied. Source of the census:
`docs/research/sub2048_mkl_method/campaign_state/probes/ZT/census/` (shipped
`.obj/avx2` objects, tree flags, reproduced under gcc 15.2).

## Why

`radix8_z_sterm_fwd_avx2` is the **legacy zsplit engine's** terminator
(`src/core/oop/zsplit.h:365`). Every banked N >= 2048 pow2 cell in
`wisdom2_oop.txt` is `eng=zturn`, whose terminator is `stf_r4` (radix-8: 7
spill ops; radix-4: 0), and 5 of the 7 banked chains end in radix 4. Both notes
call `sterm` "the terminator the >= 2048 cascade actually runs". It is not.

Additionally, the live-range-shortened form of the worst served kernel
(`radix8_z_stf_r4_bwd_avx2`, 33 spill ops) was already built and raced in-situ
on 2026-07-27 (`docs/research/scheduler_store_sinking_b1.md` §5c): 6-spill
`stf_r4sk_bwd` washed, the 33-spill incumbent won, a 55-spill order regressed
up to 1.86x. Static spill counts did not predict time there either.

## `docs/design/sub2048_transfers_to_cascade.md`

Replace the summary-table row for item 1 with:

| 1 | Register-pressure relief on the radix-8 bodies | **mechanism real, value refuted twice** | measured, both sides |

Replace the body of §1 from its table through "...machinery that exists." with:

> Innermost loop, identical classifier both sides, `spill` = a memory operand
> based on `rsp`/`rbp`. Kernels **served by `eng=zturn`** at N >= 2048:
>
> | shipped kernel | insns | spill ops | ymm |
> |---|---|---|---|
> | `radix8_z_stf_r4_fwd_avx2` | 182 | **7** | 16 |
> | `radix4_z_stf_r4_fwd_avx2` | 74 | **0** | 15 |
> | `radix8_z_stf_r4_bwd_avx2` | 234 | **33** | 16 |
> | `radix8_z_msg_fwd_avx2` / `_bwd` | 145 / 154 | **6 / 16** | 16 |
> | `radix8_z_stfn_r4_fwd_avx2` (natural order) | 257 | **49** | 16 |
>
> Every radix-4 body is at zero. The split is the same one the sub-2048 probe
> showed, but on the served forward path it barely bites: 5 of the 7 banked
> chains end in radix 4. The spills that ship are in the **backward** radix-8
> kernels and in `stfn`, whose only source difference from `stf_r4` is one
> table indirection (`kn = 4*rho[k>>2]`).
>
> **Value: refuted, twice, on this exact kernel family.** The 33-spill
> `stf_r4_bwd` already has a 6-spill store-sunk twin in the tree
> (`radix8_z_stf_r4sk_bwd_avx2`), raced in-situ 2026-07-27: the incumbent won,
> the sunk form washed, and a scheduler-chosen 55-spill order regressed up to
> 1.86x (`scheduler_store_sinking_b1.md` §5c). On the sub-2048 probe the same
> treatment measured -0.08% isolated (a null), -1.74% in one fused context and
> ~0% in another. Do not re-race `sk`. `stfn` is un-raced but serves only the
> sub-2048 natural cascade, which the ZTURN-T admission is about to race; judge
> it afterwards on whatever cells ZTURN-S keeps.

Also in §3 item 4 and §4, leave as is.

Replace §2's last paragraph ("This is create-time work whose mechanism does not
depend on N...") with:

> The mechanism is an index shift into a quarter-wave table at M points,
> `idx = pw << (log2 M - log2 RL)`, and it **refuses when RL > M**. ZTURN-S's
> terminator angles are taken modulo N at every N the store serves, up to
> 262144 — a quarter-wave at that M is 512 KB, not bakeable. So the build
> transfers **as-is to ZTURN-T (RL <= 2048, 4 KB table, the accuracy gain
> intact)** and to the >= 2048 cascade only as a two-level product: the baked
> coarse table times a 128-entry per-N fine table (128 `cos`/`sin` at create in
> place of N), one complex multiply per twiddle, ~1 ulp. Create stays cheap at
> large N; the 4–6x accuracy edge does not survive the product. Item 2 is
> therefore lowest-risk for ZTURN-T and a design choice for the cascade.

And in the summary table, item 2's "transfers?" cell becomes
"**YES for ZTURN-T; two-level product for the cascade**".

## `docs/design/cascade_stage_fusion.md`

Search for `sterm` and for "the terminator the" — any sentence naming
`sterm` as the >= 2048 cascade's terminator should name `stf_r4` instead, with
the spill figures above. No other numbers in that note are affected.
