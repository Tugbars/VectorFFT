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

## What is here — every file has a measured win

Nothing ships on static analysis or on principle. Each file below beat the
classic form that ships today, on hardware, under the banked paired protocol
(pinned core 2, warmup, 200 ms pacing, alternating order, control twin).

| file | slot | correctness | measured vs classic |
|---|---|--:|---|
| `radix8_z_t2tan_avx2.c` | R8 mid (t2) | **bit-identical** | **−3.1%** (62.8 vs 64.9 ns, 5 runs) |
| `radix8_z_n1ttan_avx2.c` | R8 leaf (n1t) | **bit-identical** | **−3.9%** (26.5 vs 27.7 ns, 5 runs) |
| `radix16_z_t2tan_avx2.c` | R16 mid (t2) | 1.5e-13 | **−25%** vs `t2b44`; **parity with the hand-built kernel** (5/5 runs) |
| `radix16_z_n1ttan_avx2.c` | R16 leaf (n1t) | 5.7e-14 | **−19.8%** vs `n1tb44` (57.7 vs 71.9 ns, 32/35, ctrl −0.03%) |
| `radix32_z_t2btan216_avx2.c` | R32 mid, blocked 2.16 | 2.9e-13 | **−3.2…−3.6%** vs `t2b48` (3/3 runs, ctrl ≤0.09%) |

**Radix 8 is the free case.** It has no general-twiddle sites — every angle is
a quarter turn or the π/4 class — so the construction reduces to one rewrite at
the two √½ folds (`x + rot(x)` → one FMA, 3 ops → 2). Because the fused
multiply is by exactly ±1, the result is **bit-for-bit identical** to the
classic codelet, verified by `memcmp` over full output planes. It is a drop-in
with no numerical risk and may be gated on bit-identity rather than tolerance.

Re-gate them any time:

```sh
gcc -O2 -static -o tangent_gate build_tuned/benches/tangent_gate.c \
    src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/*.c -lm && ./tangent_gate
```

## How they are wired

They ship as **`il_kv` variant 3**, alongside the existing blocked forms
(1 = blocked 2·16, 2 = blocked 4·8, 0xF = force monolithic):

| layer | file | what it does |
|---|---|---|
| registry | `src/core/oop/il2p.h` | `vfft_il2p_mid_v_fn` / `vfft_il2p_leaf_v_fn` return the tangent symbol for variant 3 |
| plan search | `src/core/planning/dp_planner_il.h` | variant 3 enters the candidate pool wherever a form exists, so the cell can measure it |
| apply | `src/core/vfft.c` | unchanged — `_k1_il2p_apply_kv` was already generic over the nibble |
| build | `build_tuned/build.py` | the `tangent/` dir is in the codelet source list |

Two details the registry encodes deliberately:

- The R8/R16 forms are **monolithic** emissions carrying the inline VEX-128
  odd-count tail, so variant 3 is resolved *before* the even-count gate and
  stays legal at odd counts (verified at N=240, pair 16×15). Only the R32 mid
  is blocked and needs the gate.
- **The R32 tangent leaf is absent on purpose** — it lost its race by 32%.
- Forward only; there are no backward tangent twins, the same scope the
  blocked forms already have.

**Wiring is not selection.** The kernels are in the pool and correct, but a
cell only uses one once the plan search measures it and banks the winning
`il_kv`. Until those cells are re-raced, plans keep their existing forms.

## Verifying

Two gates, and they check different things:

```sh
# 1. the codelets in isolation (own twiddle table, direct calls)
gcc -O2 -static -o tangent_gate build_tuned/benches/tangent_gate.c \
    src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/*.c -lm && ./tangent_gate

# 2. the WIRED path: il2p's own table + geometry + the il_kv plumbing
cd build_tuned && python build.py --src benches/il2p_tangent_gate.c
```

The second is the one that catches wiring bugs — a kernel can be correct
standalone and still be wrong when driven by il2p's table layout.

## Killed by measurement — do not regenerate

These were emitted, gated correct, and then **lost their race**. They are
deleted on purpose; re-deriving them wastes a session.

| variant | why it is gone |
|---|---|
| R32 leaf, tangent blocked 2.16 | **+32.4% SLOWER** than classic `n1tb48` (240.1 vs 182.0 ns, 3/35 wins). Note its static census predicted a *win* (LP bound 88.7 vs 116.5) — hardware disagreed by a third. Static bounds do not decide this. |
| R16 mid/leaf, blocked 2.8 | No slot: R16 mono already fits the register file and reached hand parity, so the blocked shape has nothing to buy. Unraced, no consumer — deleted rather than left to confuse a grep. |
| R16 mid, backward | Feature coverage only, and incomplete: there is no backward *leaf*, so it cannot form a backward route. Regenerate as a pair when the backward arc is actually built. |

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
  mix alone, and it is much smaller.
- **Scope is L1-resident (N ≤ 512-class).** The advantage is port pressure,
  which stops binding once the working set leaves L1: at N=1024 a hand-built
  tangent route *inverted* (+16%) against classic blocked forms. Do not
  promote these into ≥1024 plans without racing that cell.
- These are **tolerance-gated, never bit-identity-gated** — the association
  differs from the classic forms by construction.

## Regenerating

The generator knobs are **default-off**; every classic emission is
byte-identical without them. Each file's header carries its exact command.
The knobs: `--cil-tangent` (the interior), `VFFT_CX_WING=1` (the
machine-translated R16 wing construction), `VFFT_CX_LAZYLOAD=1` /
`VFFT_CX_LAZYSTORE=1` (interleaved loads/stores — the peak-pressure fix that
lets gcc keep the loop-invariant constants in registers), and
`VFFT_CX_SCHED=asis|cpl|cpl2` (scheduler; `asis` preserves the wing's origin
order and is what the R16 parity result used).
