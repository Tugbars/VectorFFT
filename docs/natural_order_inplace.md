# Natural-order in-place c2c — known limitation & planned feature

## Current state

The in-place complex FFT is **digit-reversed (scrambled) order** on the forward
pass. This is by design, not a bug: the engine runs **DIT forward / DIF backward**
in a single buffer with **no bit-reversal and no transpose** (see the
"Permutationless, transposeless" section of [README.md](README.md)). The two
scrambles cancel, so a forward→backward roundtrip is the original ×N with **zero
reorder passes** — which is the basis of the memory-bound win and the zero-copy
K-split MT.

The caveat: a *standalone* forward returns the spectrum in digit-reversed order.
Roundtrip / convolution / power-spectrum use cases don't care (the inverse
un-scrambles, or order is irrelevant). Only consumers that need **natural spectral
order** (bin-for-bin comparability, e.g. matching MKL elementwise) are affected.

This is uniform across the c2c path: **both in-place and the bulk OOP kind
(MODEB) are scrambled.** The OOP `LEAF`/`BAILEY2` kinds are the *only* natural-order
c2c outputs, and the 2D/r2c/trig paths absorb the reorder inside their own
pre/post-processing rather than paying a standalone permutation pass.

## Why it is not a codelet swap

The scramble is **not** produced by the `n1` (untwiddled) stage-0 codelet — it is
a *global* property of the transposeless DIT dataflow. Every stage operates at its
natural stride directly on the single buffer; the passes change *which elements a
codelet touches, never where they live*. `n1` vs `t1`/`t1s` differ only in whether
they apply twiddles — none of them move elements to natural positions. So
substituting stage-0's codelet changes nothing about output ordering.

The reason OOP `LEAF`/`BAILEY2` *can* emit natural order is that they have a
**separate `dst` buffer** and *scatter* outputs to natural positions — `BAILEY2`
fuses the transpose into its stores. That free reorder **requires `dst ≠ src`**;
with `dst == src` the same codelet would overwrite indices it still needs to read
(aliasing corruption). Natural order is bought by the second buffer, not by the
codelet kind.

## The plan: fused in-place transpose codelets

We already have all the surrounding machinery:

- The **digit-reversal permutation is known** — `transforms/real/r2c.h` computes
  `perm` / `iperm` (mixed-radix digit reversal, factor order reversed) for exactly
  this need.
- **Transpose-fused stores** are proven in the OOP path (`oop_plan.h` `BAILEY2`
  s1, the line-filling blocked transpose in `transforms/fft2d/transpose.h`).
- The **planner / wisdom / dispatch** infrastructure already selects per-cell
  execution variants and persists them.

The one missing piece is a **fused in-place transpose codelet**: a kernel that
lands the final stage's output at its natural index *within the same buffer*,
register-blocking the permutation so it never aliases (read a block → permute in
registers → write back, cycle-aware for the non-square strides). With that, a
natural-order in-place mode is forward-DIT + fused in-place digit-reversal — no
separate full-plane permutation pass, the reorder folded into the last stage's
stores the way OOP folds it into `dst`.

### Alternatives considered (and why the fused codelet wins)

| Approach | Natural order | Stays in-place | Cost |
|---|---|---|---|
| Standalone digit-reversal pass | ✓ | ✓ | extra full memory pass (~single-core pack-tax magnitude); in-place permutation needs cycle-following / temp |
| Self-sorting (Stockham autosort) | ✓ | ✗ | ping-pongs two buffers → becomes OOP, defeats the permutationless design |
| **Fused in-place transpose codelet** | ✓ | ✓ | reorder folded into the last stage's stores; no separate plane pass |
| Use OOP `LEAF`/`BAILEY2` | ✓ | ✗ (OOP) | already shipped — natural order needs the extra `dst` buffer |

The fused-codelet route keeps the single-buffer, single-pass character that the
whole engine is built on, which is why it's the chosen direction.

## Status

**Planned, not yet implemented.** Until it lands:
- Need natural-order c2c out-of-place → use OOP `LEAF`/`BAILEY2`.
- Need natural-order c2c in-place → forward then apply `iperm` manually (a memory
  pass), or restructure to a roundtrip where the scramble cancels.
