# Cost-Model Encoding + R=64 AVX2 — Closing the Loop

## Summary

Two changes wrap up the variant-dispatcher work:

1. **Cost-model rule encoded in `should_spill`.** The CLI now auto-applies the full recipe (`--spill --su`) when the rule says yes, with `--no-recipe` as an opt-out. The generator is now self-tuning across (R, ISA) without flags.

2. **R=64 AVX2 validated.** The biggest recipe wins we've measured anywhere — **18-47% faster than Topo** across K. As predicted, AVX2's 16 YMM × R=64's ~70 peak live made Topo catastrophically bad without the recipe.

## R=64 AVX2 results (3 runs each, median SU/T)

| K | Topo (ns) | SU+Spill (ns) | **SU/T** |
|---|-----------|---------------|----------|
| 64 | 12614 | 6673 | **0.53** |
| 128 | 27786 | 16600 | **0.60** |
| 256 | 65289 | 39057 | **0.60** |
| 512 | 166281 | 120135 | **0.75** |
| 1024 | 373807 | 287238 | **0.77** |
| 2048 | 764216 | 597848 | **0.79** |
| 4096 | 1639767 | 1340733 | **0.82** |

47% faster at small K. The recipe wins at large K too because AVX2's narrow vectors mean each saved spill load/store is proportionally bigger than at AVX-512.

## What this completes

The recipe is now validated across the full grid:

| | AVX-512 (vec_regs=32) | AVX2 (vec_regs=16) |
|---|-----------------------|---------------------|
| R=4 | recipe ON, ~tied (noise) | recipe OFF, ~tied (verified neutral) |
| R=8 | recipe ON, 2-10% better | recipe OFF (would regress 3-7%) |
| R=16 | recipe ON, beats Hand K≥128 | recipe ON, **12-20% better than Topo** |
| R=32 | recipe ON, **beats Hand all K** | recipe ON, **19-44% better than Topo** |
| R=64 | recipe ON, **beats Hand all K** | recipe ON, **18-47% better than Topo** |

The auto-detection chooses correctly for every cell. We never tested R=4 AVX2 directly, but the rule says "Topo" there (n+6=10 < 16, vec_regs=16 < 32) which is consistent with R=8 AVX2 regression.

## How the encoding works

In `dft.ml`:

```ocaml
let should_spill (n : int) (vec_regs : int) : bool =
  (n + 6 > vec_regs) || vec_regs >= 32
```

In `bin/gen_radix.ml`:

```ocaml
let recipe_applicable =
  !twiddled
  && not !bisect
  && not !annotate
  && not !no_recipe
  && Vfft_v2.Dft.should_spill n isa.vec_regs
in
if recipe_applicable then begin
  if not !spill then spill := true;
  if not !su then su := true
end;
```

Auto-on when applicable. `--no-recipe` forces off. Explicit `--spill` and `--su` always work.

## Behavior matrix

| Command | What happens |
|---------|--------------|
| `gen_radix 32 --twiddled --emit-c` | recipe AUTO-ON (was Topo before) |
| `gen_radix 8 --twiddled --emit-c --isa avx2` | recipe AUTO-OFF (correct) |
| `gen_radix 32 --twiddled --emit-c --no-recipe` | force Topo |
| `gen_radix 32 --twiddled --emit-c --spill --su` | same as default |
| `gen_radix 4 --twiddled --emit-c --isa avx2 --spill` | force on (override) |
| `gen_radix 32 --twiddled --emit-c --bisect` | use bisection (recipe stays off) |

## What this means for users

Before: had to know which flags to pass per (R, ISA). Wrong choice silently regressed.

After: just pass the radix. The generator picks the right variant.

```bash
# AVX-512, all radices: optimal
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place 16
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place 32
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place 64

# AVX2, R≥16: optimal; R≤8: Topo (correct)
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --isa avx2 8   # Topo
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --isa avx2 16  # full recipe
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --isa avx2 64  # full recipe
```

## What's left (genuinely small)

- ~~R=4 AVX2 spot-check~~. **Done** — ran sweep K=64..4096; SU/T is noise-tied (medians 0.98-1.00). Rule's prediction (Topo) is correct but neutral: forcing the recipe ON wouldn't hurt either, just adds code with no benefit. Rule errs toward simpler code, which is the right call.
- **K-threshold** for spill at R=16 K=64 on AVX-512. Would close the small ~5% regression in that one corner. Probably ~30 lines for a K-aware gate. Not blocking anything.
- **R=128**. No hand-coded reference, so we'd validate against fftw or naive. Would just measure recipe headroom at the next size class.

## What we learned, finally

The "variant dispatcher" framing held up across:
- 5 radices (R=4 through R=64, 16× DAG-size growth)
- 2 ISAs (AVX-512 and AVX2)
- 7 K values (64 through 4096)

Same OCaml code path, three orthogonal levers (Spill + SU + cluster-sequential PASS 2), one cost-model rule, beats hand-tuned codelets where applicable and ties or improves on Topo where it doesn't.

Math layer never moved during this work. All wins came from emission policy.
