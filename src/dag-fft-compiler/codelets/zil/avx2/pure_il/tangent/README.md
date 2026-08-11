# Tangent-scaled butterfly codelets (pure-IL, AVX2)

The `tan` family: same transforms as the sibling classic pure-IL codelets,
different **interior arithmetic**. Rotations are factored

```text
e^(-iθ) = cos θ · (1 − i·tan θ)
```

so the shear runs **un-normalized** (one FMA, no standalone multiply) and the
`cos θ` normalization is **deferred into the downstream butterfly's FMA pair**.
Naked butterfly adds — which can only issue on the FADD ports — become FMAs,
which issue on the FMA ports. That rebalances execution-port pressure, which
is the binding limit for L1-resident transforms.

Full write-up: [`docs/performance/tangent_scaled_butterflies.md`](../../../../../../docs/performance/tangent_scaled_butterflies.md).
Emitter arc and gate ladder: [`docs/roadmap/tangent_emitter_plan.md`](../../../../../../docs/roadmap/tangent_emitter_plan.md).

## Contents

| file | kind | shape | status |
|---|---|---|---|
| `radix16_z_t2tan_avx2.c` | R16 mid (t2) | mono, full-kernel wing | **raced: parity with the hand kernel, −25% vs classic** |
| `radix16_z_n1ttan_avx2.c` | R16 leaf (n1t) | mono, corner-turn | gated |
| `radix16_z_t2btan28_avx2.c` | R16 mid | blocked halves 2.8 | gated |
| `radix16_z_n1tbtan28_avx2.c` | R16 leaf | blocked halves 2.8 | gated |
| `radix32_z_t2btan216_avx2.c` | R32 mid | blocked halves 2.16 | **raced: −3.2…−3.6% vs classic `t2b48`** |
| `radix32_z_n1tbtan216_avx2.c` | R32 leaf | blocked halves 2.16 | gated |
| `radix16_z_t2tan_bwd_avx2.c` | R16 mid, backward | mono | gated |

Every file carries a `PROVENANCE` header with its exact generator command,
tolerance-gate result, race result (or an explicit "not raced"), and its own
caveats. Read it before quoting a number.

## Hard-won facts (do not re-derive)

- **`tan` ≠ `tg`.** The `t2tg` symbols elsewhere in `pure_il/` are the
  *leg-strided turned store* kind, an unrelated family. This family is tagged
  `tan` precisely to avoid that collision.
- **Split strings are `m.p`.** `2.8` / `2.16` are the *halves* splits whose
  pass 2 runs through the tangent-aware `butterfly_pair` m=2 arm. `8.2` /
  `16.2` route pass 2 through the classic rotation path and stay classic.
- **Backward mids are POST-twiddled**: `Y[o] = e^{+2πi·o·k/N} · Σ_l e^{+2πi·o·l/R} x_l`
  — diagonal applied by *output* leg after the DFT, `+`-exponent records. This
  is not pre-twiddle with a flipped table sign; guessing that costs an hour.
- **R16 ≠ R32.** At R16 the tangent constant set is three scalars, they stay
  register-resident, and the kernel reaches hand parity. At R32 the ladder
  needs 14 distinct magnitudes (7 tan + 7 cos) where classic cos/sin needs 7,
  so they cannot be resident on a 16-register file — the R32 win is the port
  mix alone (LP bound 116.5 → 88.7), not constant hoisting.
- **Scope is L1-resident (N ≤ 512-class).** The advantage is port pressure,
  which stops binding once the working set leaves L1: at N=1024 a hand-built
  tangent route *inverted* (+16%) against classic blocked forms. Do not
  promote these into ≥1024 plans without racing that cell.
- These are **tolerance-gated, never bit-identity-gated** — the association
  differs from the classic forms by construction.

## Verifying

`build_tuned/benches/tangent_gate.c` re-gates all seven against golden DFTs:

```sh
gcc -O2 -static -o tangent_gate build_tuned/benches/tangent_gate.c \
    src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/*.c -lm && ./tangent_gate
```

Last run: all seven correct, worst error 2.9e-13.

## Regenerating

The generator knobs are **default-off**; every classic emission is
byte-identical without them. Each file's header carries its exact command.
The knobs are: `--cil-tangent` (the interior), `VFFT_CX_WING=1` (the
machine-translated R16 wing construction), `VFFT_CX_LAZYLOAD=1` /
`VFFT_CX_LAZYSTORE=1` (interleaved loads/stores, the peak-pressure fix), and
`VFFT_CX_SCHED=asis|cpl|cpl2` (scheduler choice; `asis` preserves the wing's
origin order and is what the R16 parity result used).
