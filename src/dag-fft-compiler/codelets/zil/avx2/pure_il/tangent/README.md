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
| `radix32_z_t2bw32_avx2.c` | R32 mid, blocked 2.16 **wing32** | 5.7e-13 (DFT-512 e2e) | **−3.3…−5.5%** vs `t2btan216` in-route (3 runs, ctrl ≤0.08%) — supersedes it |
| `radix32_z_n1tbw32_avx2.c` | R32 **LEAF**, blocked 2.16 **wing32**, TURNED-128 store | 5.7e-13 (DFT-512 e2e) | route (32,16) = **parity with the hand w32tgL champion** (301.5–305.4 ns @512, ±1.3%, 3 runs) |
| `radix32_z_n1tbw32t256_avx2.c` | R32 **LEAF**, wing32, **T256 store edge** (TURNED axis, leaf variant 4) | gate 2.2e-14/1.8e-13 | **dp-PROMOTED 2026-08-16 at BOTH raceable cells**: 128 → pair 4×32 kv 64 (63.6 ns; front door 65 ns = 1.04–1.05× vs MKL) and 512 → kv 67 (301.1 ns; front door 304 ns = 0.94–0.98×). Same interior as `n1tbw32`; only the edge differs (16 `vperm2f128`+16 `vinsertf128`+32 wide stores vs 32 extracts + 64 halves). |

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
| `radix32_z_t2btan216` (R32 mid, per-site angles) | **POOL-SUNSET 2026-08-13**: superseded by `t2bw32` (wing32 combine + ROTFMA, −3.3…−5.5% in both shapes, 3 runs) and the dp re-race banked kv 51 with the wing32 forms (`512 1 3 … 303.1 51`; canonical bench 0.96× vs MKL, was 0.87–0.88×). Race-arm copy preserved in `build_tuned/benches/tangent_hand/`. Its 23-literal ulp-twin constant table is the recorded counter-example the canonical-angle combine fixes. |
| R32 leaf, tangent blocked 2.16 (PAIRED-PERMUTE edge) | **+32.4% SLOWER** than classic `n1tb48` (240.1 vs 182.0 ns, 3/35 wins). Static census predicted a *win* — hardware disagreed by a third. **POSTMORTEM 2026-08-13: the tax was the store EDGE, not the interior** — 32 `permute2f128` (port-5-only) from the paired corner-turn. `radix32_z_n1tbw32` re-ships the slot with the TURNED-128 split-store edge and ties the hand champion. The kill stands for the *paired-edge* form only. |
| R16 mid/leaf, blocked 2.8 | No slot: R16 mono already fits the register file and reached hand parity, so the blocked shape has nothing to buy. Unraced, no consumer — deleted rather than left to confuse a grep. |
| M-128 mids (`t2tanm128`, `t2bw32m128`) — TURNED-axis mid edge | **LOST the 2026-08-16 dp race at every cell they can serve** (128/256/512; 1024 is owner-locked). The split stores they delete are cosmetic (`bound-on-stores` ≤0.037 — same verdict as the refuted S[]-align lever); the extra store uop bought nothing. Sunset per pool policy; mid variant 4 resolver returns 0 (degrades). Regenerate with `VFFT_CX_STORE128=1` if a future cell wants the arm — the knob stays in the emitter. |
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
- **R16 ≠ R32 — internally the cause is SPILLS, not port mix.** ⚠ This is an
  *internal* A/B (our tangent R32 vs our classic R32). It is **not** why MKL
  leads at R32: a like-for-like census against MKL's own 32-point column kernel
  shows **MKL spills 1.6× more per arithmetic op than we do and wins anyway**
  (78 stack ops vs our 48), buying 18% fewer instructions, 1.9× the FMAs and
  *zero* bare multiplies with that traffic. Against MKL the levers are
  instruction count and naked-add/bare-mul elimination — the direction this
  family already pushes — plus the interleaved-complex sign/lane tax
  (our 82 shuffles + 29 xors vs their 54 + 15). See `v1_0_results.md` §1.
  Measured census
  (`gcc -O2 -mavx2 -mfma -S`):

  | kernel | spills | movs | fma | naked add+sub | naked share |
  |---|--:|--:|--:|--:|--:|
  | R32 classic `t2b48` | 24 st / 21 ld | 11 | 67 | 152 | 56% |
  | R32 tangent | 33 st / 30 ld | 35 | 113 | 94 | **39%** |
  | R16 tangent | **6 st / 4 ld** | 9 | 70 | 90 | 47% |

  The lever *works* at R32 — it improves the mix **more** than at R16
  (56% → 39% naked; R32 tangent's mix is better than R16 tangent's 47%). It
  still returns only −3.2%, because those 58 add→FMA conversions cost +9 spill
  stores, +9 spill loads and +24 register moves that R16 never pays. The
  port-mix win is handed straight back to the load/store ports.

  Root cause: **the R32 form is not an R32 design.** It is the R16 kernel run
  twice through a memory plane (see its header: two half-DFTs park to `S[]`,
  then a combine pass), in a `split 2.16` geometry chosen for the *classic*
  kernel, which was already at the register limit. Tangent's constants pushed
  it over. Same story for the R32 tangent leaf losing by 32%.

  So a **native** R32 tangent is *open, not refuted* — but the lever there is
  blocking geometry co-designed with the tangent constant set (fewer live
  values per block), not more tangent.
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

wing32 additions (2026-08-13, A-1 — `docs/roadmap/r32_tangent_parity_plan.md`):
`VFFT_CX_W32TG=1` (radix-32 fwd pass-B combine with CANONICAL angles — kills
the ulp-twin constant table, 23 → 13 literals), `VFFT_CX_ROTFMA=1` (render
fold: butterflies over a −i rotation emit sign-folded `[c,−c]·cflip` FMAs —
bit-exact, zero xors), `VFFT_CX_TURN128=1` (blocked N1T corner-turn as
per-output split 128-bit stores — no `permute2f128`, lazy-store legal, no
even-p restriction). All three OFF-verified byte-identical on the whole
shipped set (8/8, incl. classic `t2b` which runs the edited m=2 arm).
