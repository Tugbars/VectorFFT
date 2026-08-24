# Cascade chain fence — every non-pow2 N ≥ 2048 falls off the tier

Status: **ROADMAP / not started.** Recorded 2026-08-24 from a coverage audit, not a race.
Scope: the N ≥ 2048 cascade tier. The Bailey band below it is `il_odd_chain.md`'s subject,
which defers here explicitly ("the cascade owns ≥2048").

## The fence

The cascade accepts chains of `{4, 8}` and nothing else:

| engine | accepted chain |
|---|---|
| `vfft_zsplit_create` | `chain[0] ∈ {4,8}`, mids `∈ {4,8}`, `last == 8`, `nf ∈ [3,6]` |
| `vfft_zturn2_create_chain` | same, plus `last == 4` (the radix-4 terminator) |

So `N = prod(chain)` must be a **pure product of 4s and 8s**. `2048`, `4096`, `8192` … are
served. `6144 = 2048·3`, `10240 = 2048·5`, `12288`, and every other non-pow2 N ≥ 2048 are
not cascade-legal at all and fall to a slower tier.

## What this is NOT

It is **not** an odd-count tail gap, which is how it first reads. Cascade codelets declare
`CONTRACT: count % 4 == 0` and carry no narrow arm, but that contract is **unreachable by
construction**, so adding a tail would be dead code:

* stage counts are `s0t: N/4`, `msg: D_s`, `stf: N/8` (or `N/4` at the r4 terminator)
* `D[i] = D[i+1] * chain[i+1]` with every factor in `{4,8}`, so every `D` is a multiple of 4
* `zturn.h:596` guards the tightest case outright — `if (p->D[nf-2] % 4) goto fail;`,
  commented *"asserted, not assumed (spec risk)"*

The `% 4` is the **block granularity**, not a SIMD width: the cascade interior is BLOCK-SPLIT
into 64-B `[re×4][im×4]` records, so four columns per iteration is the record shape. Audited
2026-08-24: 0 of 265 pure-IL codelets lack a narrow arm; the 32 cascade files are the only
IL codelets without one, and correctly so.

## What unlocking it would take

Odd-radix **cascade** codelets — `s0s`/`msg`/`sterm` (or `s0t`/`msg`/`stf`) at radix 3, 5, 7 —
so a chain like `4.8.8.3` becomes expressible.

🔴 The constraint is the split-layout register file, and `cascade_z.ml`'s tier gate states it:

> *"the split family is radix 4/8 ONLY and monolithic BY DESIGN (16 planes fit the ymm file;
> 'r16 split = 32 live planes, spills')"* — and it deliberately does **not** consult
> `Dft.should_spill`, because *"its n≥5 clause would put R=8 on the spill recipe and pay stack
> traffic the legacy kernels don't have."*

A split kernel holds re and im as separate planes, so live-vector count is `2R`:

| radix | live planes | fits 16 ymm? |
|---|---|---|
| 3 | 6 | yes |
| 5 | 10 | yes |
| 7 | 14 | yes |
| 9 | 18 | **no** |

So radices 3, 5 and 7 are arguably within reach and 9+ is not — which happens to be the same
boundary the `_ct` work found from the other side (radix 9 has 0% spill in IL and loses `_ct`;
in SPLIT it is the first radix that spills). That symmetry is suggestive, not evidence.

## Order of work, if picked up

1. **Quantify first.** `benches/nonpow2_tier_probe.c` (written, unrun) compares ns/point for
   pow2 cascade cells against their non-pow2 neighbours — 2048/3072, 4096/6144, 8192/10240,
   12288/16384. If the step is small the whole item is not worth doing.
2. Only then consider emitting radix 3/5/7 cascade kinds, and only after checking the tier
   gate's claim holds at those radices rather than assuming the `2R` arithmetic settles it.

## Related

* `il_odd_chain.md` — the same question below 2048, where the two-stage Bailey parity argument
  (`count % 2 == 0` on both stages ⇒ both factors even) forces a 3-stage shape.
* Non-pow2 **banking** is a separate blocker with the same symptom: a non-pow2 cell races fine
  and then reports `banked 0 verdict(s)` because the kind-3 line needs an encodable split
  `cc_chain`. Fixing the cascade fence would not fix banking, or vice versa.
