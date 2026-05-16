# 62. DIF + LOG3 + spill recipe — use-before-decl bug & workaround

**Date:** 2026-05-16
**Status:** Workaround landed in [lib/dft.ml:899](../lib/dft.ml#L899). Root
cause not fixed. Surface area: 8 codelets (R=12 and R=25 DIF log3
variants). Production impact: zero (DIF codelets never invoked at runtime
in current wisdom).

**Files:**
- Workaround — [lib/dft.ml](../lib/dft.ml) — one new pattern-match arm in `dft_expand_twiddled_spill`.
- Underlying machinery (where the real fix would live) — [lib/emit_c.ml:1427](../lib/emit_c.ml#L1427) (`spill_pass1`), [lib/regalloc.ml](../lib/regalloc.ml), [lib/algsimp.ml](../lib/algsimp.ml).

## How it surfaced

The new prototype `registry.h` ([emit_registry_h.ml](../bin/emit_registry_h.ml))
declares externs for every emitted codelet. The first mass-link attempt
(against all 832 avx2 codelets) failed at link time with `t398 undeclared`
and `t399 undeclared` errors inside `r25_t1s_dif_fwd_log3.c` — the
emitted C source references variables before they're declared.

Per-codelet scan revealed exactly **8 broken codelets**:

```
codelets/avx2/composites/r12_t1_dif_{fwd,bwd}_log3.c
codelets/avx2/composites/r12_t1s_dif_{fwd,bwd}_log3.c
codelets/avx2/composites/r25_t1_dif_{fwd,bwd}_log3.c
codelets/avx2/composites/r25_t1s_dif_{fwd,bwd}_log3.c
```

824 / 832 codelets compiled cleanly. The 8 failures shared a clear
pattern.

## Root cause analysis

A diagnostic ablation across flag combinations isolated the trigger:

| Test | Errors |
| --- | --- |
| R=25 t1s DIF fwd LOG3, M-active on | 15 |
| R=25 t1s DIF fwd LOG3, M-active off | 15 |
| R=25 t1 DIF fwd LOG3 (no t1s) | 15 |
| R=25 t1s DIT fwd LOG3 | 0 |
| R=25 t1s DIF fwd (no log3) | 0 |
| R=25 t1s DIF fwd LOG3 `--no-recipe` | 0 |
| R=12 t1s DIF fwd LOG3 | 9 |
| R=12 t1s DIT fwd LOG3 | 0 |

**Conclusion: the bug requires the simultaneous presence of**

1. **`DIF` orientation** (DIT is fine)
2. **`TP_Log3` twiddle policy** (TP_Flat is fine)
3. **Spill recipe enabled** (default; `--no-recipe` avoids the bug)

It does NOT require:
- M-active register allocation (broken with or without)
- `t1s` rendering choice (broken with both `t1` and `t1s`)
- Any specific radix family — affects every composite where the bug
  manifests, but only R=12 and R=25 currently in the prototype's
  composite list

R=6, R=10, R=20 compile cleanly with DIF + LOG3 + spill recipe. The
common feature of R=12 and R=25 is that their internal CT factorization
involves a non-pow-2 sub-radix (R=12 = 4×3; R=25 = 5×5) — others
(R=6 = 3×2, R=10 = 5×2, R=20 = 5×4) have a pow-2 sub-radix that
produces a different pass-1 topology.

## Mechanism

In [lib/dft.ml](../lib/dft.ml)'s `dft_expand_twiddled_spill`, the
`Cooley_Tukey (n1, n2)` arm:

1. Builds the input array (in DIF mode: raw inputs; in DIT: pre-twiddled
   via `cmul_pattern`).
2. Runs pass-1: N1 inner DFTs of size N2 over the input.
3. Captures each pass-1 output as a spill marker.
4. Pass 2 combines them; output stores happen at the end.

For `TP_Log3` in DIF orientation, the **output-side** twiddles are
Cmul-derived from a sparse base set:
- R=4 LOG3: derive W^3 from W^1·W^2 → one extra Cmul intermediate
- R=8 LOG3: derive W^3=W^1·W^2, W^5=W^4·W^1, W^6=W^4·W^2, W^7=W^4·W^3
- R=25 LOG3: similar pattern across all non-power-of-2 indices

These derived twiddles are **high-fanout** values: each is consumed by
multiple output stores. The spill recipe's `install_alloc` pass picks
them as spill candidates. But the spill recipe's reload-insertion logic
mishandles the cross-pass references — the original symbol name is left
in some use sites, while the declaration is moved to (or kept at) the
"spill point" which lands later in the function body. Result:
`t398 used at line 389, declared at line 576`.

DIT works because in DIT the Cmul derivation happens at the INPUT side,
before pass-1 even starts. The Cmuls' declarations are in the prologue,
naturally preceding all uses. DIF puts them at the OUTPUT side, after
pass-2 — that's where the ordering fails.

The exact failure mode in `regalloc.ml` / `emit_c.ml`'s spill recipe
hasn't been chased — investigating it would be a 1-3 hour OCaml debugging
session in code that's already had several rounds of doc-56 fma-lift
and selective-pinning work.

## The workaround

[lib/dft.ml:899](../lib/dft.ml#L899) adds one pattern-match arm:

```ocaml
| Cooley_Tukey (_n1, _n2) when policy = TP_Log3 && direction = DIF ->
    (* See docs/62 — DIF + TP_Log3 + spill recipe produces
     * use-before-decl in emitted C. Fall back to plain expansion
     * for this combo only. *)
    (dft_expand_twiddled ~policy ~direction ~sign n, [], None)
```

For the (DIF, TP_Log3) combination only, fall back to non-spill expansion
(same as the existing `Direct` and `Split_radix` fallback paths in this
function). Other combinations (DIT × any, DIF × TP_Flat) keep the spill
recipe.

## Validation

Post-fix isolated tests (9 of them) all pass:

| Test | Before | After |
| --- | --- | --- |
| R=25 t1s DIF fwd LOG3 | 15 errors | **0** |
| R=25 t1 DIF fwd LOG3 | 15 errors | **0** |
| R=12 t1s DIF fwd LOG3 | 9 errors | **0** |
| R=25 t1s DIT fwd LOG3 | 0 | 0 (regression-clean) |
| R=32 t1 DIT fwd LOG3 | 0 | 0 (regression-clean) |
| R=64 t1s DIT fwd LOG3 | 0 | 0 (regression-clean) |
| R=512 t1 DIT fwd LOG3 | 0 | 0 (regression-clean) |
| R=25 t1s DIT fwd (no log3) | 0 | 0 (regression-clean) |
| R=25 t1s DIF fwd (no log3) | 0 | 0 (regression-clean) |

Full primes + composites regen confirms **832 / 832 codelets compile
cleanly** across the avx2 tree after the fix.

## Cost of the workaround

The 8 affected codelets lose the spill recipe optimization — they fall
back to plain expansion which doesn't track high-fanout values for stack
spilling. At R=25 the codelet body grows from a register-pressure-aware
schedule to a straightforward emission, picking up some redundant moves
that GCC normally absorbs in register allocation. Empirical impact on
the regenerated tree:

| Family | Lines before fix | Lines after fix | Delta |
| --- | --- | --- | --- |
| avx2/composites | 33,320 | 35,098 | +5.3% |
| avx512/composites | 30,442 | 30,754 | +1.0% |

The growth is concentrated in the 8 affected codelets. Approximate per-
codelet bloat: 100-400 lines (DIF log3 versions slightly larger than
their previous broken state).

**Runtime impact: zero**, because:
- Current production wisdom uses `use_dif_forward = 0` on every entry
- DIF codelets are never invoked at runtime
- Even if a future calibration picks DIF forward, the 8 affected codelets
  are at composite radixes R=12 and R=25 specifically with LOG3 — a
  narrow corner of the wisdom search space

## When to revisit

The workaround should be replaced with a real fix when:

1. **A future wisdom search picks DIF forward** for a cell that uses
   R=12 or R=25 with LOG3 — at that point the perf delta from missing
   spill recipe matters and the underlying bug needs solving.
2. **Spill recipe gets refactored** for any other reason. The reload-
   insertion logic in `install_alloc` would naturally pick up this
   case during a rewrite.

If you investigate the root cause, the lines to start with:
[lib/emit_c.ml:1427-1500](../lib/emit_c.ml#L1427) (`spill_pass1` and
`install_alloc` for that pass), and trace what happens to a high-fanout
Cmul output that's both a pass-1 spill marker AND has a use site
emitted before its declaration. The reload site might be missing for
that specific cluster topology.

## Related

- [docs/56_strided_batch_2d_design_c.md](56_strided_batch_2d_design_c.md)
  — the doc-56 work that brought single-use inlining + selective pinning
  into the codegen pipeline. The DIF + LOG3 bug pattern may or may not
  have existed before doc-56 — wasn't tested until the registry mass-
  link this session, which is the first thing that compiles every codelet
  at once.
- [docs/61_plan_shaped_executor_spike.md](61_plan_shaped_executor_spike.md)
  — the plan-shaped executor spike. The registry was needed because
  the spike's plan-executor architecture wants a single source of truth
  for "what codelets exist." The bug surfaced as a side effect of that
  work; the spike itself doesn't depend on the broken codelets.
