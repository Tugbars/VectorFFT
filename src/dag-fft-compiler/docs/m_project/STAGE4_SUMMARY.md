# Stage 4: Wire Primes Through Canonical Prep

**Status:** COMPLETE — 14/14 prime codelets generate correct output.

## What Stage 4 delivers

The M-project (SSA-based precoloring register allocator with Belady spilling) now covers prime codelets R=3..R=19 in addition to the composite codelets it already handled. The wiring goes through `Regalloc.prepare_for_simple_codelet[_from_oref]` (Stage 3's canonical prep), so the implicit contract `Regalloc.allocate` requires is satisfied structurally — not by happenstance as in the original M7 attempt.

## Results

### Correctness (gcc-11 -O3, bit-exact diff vs default codelet)

| Codelet     | regalloc status   | output diff | use-after-clobber bugs |
|-------------|-------------------|-------------|------------------------|
| R=3 AVX2    | 28 tags bound     | 0           | 0/0 |
| R=3 AVX-512 | 28 tags bound     | 0           | 0/0 |
| R=5 AVX2    | 56 tags bound     | 0           | 0/0 |
| R=5 AVX-512 | 56 tags bound     | 0           | 0/0 |
| R=7 AVX2    | 82 tags bound     | 0           | 0/0 |
| R=7 AVX-512 | 82 tags bound     | 0           | 0/0 |
| R=11 AVX2   | 134 tags bound    | 0           | 0/0 |
| R=11 AVX-512| 134 tags bound    | 0           | 0/0 |
| R=13 AVX2   | 163 tags bound    | 0           | 0/0 |
| R=13 AVX-512| 163 tags bound    | 0           | 0/0 |
| R=17 AVX2   | OVERFLOW→fallback | 0           | 0/0 |
| R=17 AVX-512| 213 tags bound    | 0           | 0/0 |
| R=19 AVX2   | OVERFLOW→fallback | 0           | 0/0 |
| R=19 AVX-512| 241 tags bound    | 0           | 0/0 |

Composite regression (9/9 cases R=8..R=256 with and without `--log3`): **still passes**.

### Code size (object bytes, gcc-11 -O3)

| Codelet     | default | Stage 4 | Δ        |
|-------------|---------|---------|----------|
| R=11 AVX2   | 6,928   | 5,568   | **-19.6%** |
| R=11 AVX-512| 6,304   | 5,200   | **-17.5%** |
| R=13 AVX2   | 9,248   | 7,656   | **-17.2%** |
| R=13 AVX-512| 7,616   | 6,664   | **-12.5%** |
| R=17 AVX-512| 11,416  | 9,016   | **-21.0%** |
| R=19 AVX-512| 14,504  | 10,776  | **-25.7%** |

### Runtime (gcc-11 -O3, RTX-host SPR, ns/call, 100k iters)

| Codelet      | default       | Stage 4       | speedup |
|--------------|---------------|---------------|---------|
| R=11 AVX2    | 96.43         | 81.26         | **1.19×** |
| R=11 AVX-512 | 91.38         | 85.11         | 1.07×   |
| R=13 AVX2    | 138.06        | 128.21        | 1.08×   |
| R=13 AVX-512 | 134.38        | 123.82        | 1.09×   |
| R=17 AVX-512 | 226.73        | 199.92        | **1.13×** |
| R=19 AVX-512 | 283.61        | 246.83        | **1.15×** |

R=17/R=19 AVX2 fall back via Overflow (identical to default; same speed).

## Two bugs found and fixed during Stage 4

Both bugs lived in `pick_belady_victim` in `lib/regalloc.ml`. They were silently miscompiling at the cluster-spill level too, but never tripped the 9-composite regression set because the cluster-spill's uniform end-of-schedule force_last_use happened to align all relevant tag uses at the same position. The prime-path sentinel coverage uses non-trivial position sets and exposed both bugs.

### Bug 1: `next_use_after tag p` semantic (Part 1 fix)

`pick_belady_victim p` computed each candidate's next-use distance via `next_use_after tag p`, which returns positions STRICTLY > p. A sentinel covering `[134, 136, 138]` at p=138 would return None — treated as `max_int` (ideal victim). Belady would pick it as the farthest-future-use candidate, freeing its color for a fresh reload. The fresh reload's value then overwrites the register the position-138 consumer reads from.

**Fix:** use `next_use_after tag (p - 1)` to catch the current position as a use. If tag is used at p, nu = p (smallest possible value), and Belady (which picks LARGEST nu) effectively protects it. Caught R=11 AVX2 (was 16 diffs → 0 diffs).

### Bug 2: tie-breaking at extreme pressure (Part 2 fix)

Part 1 alone wasn't enough for R=17 AVX2. At peak_live 70 vs budget 14, every allocated tag/sentinel can have `nu = p` simultaneously (every register holds a value being read at p). With everyone tied at the smallest priority, `Hashtbl.iter`'s unstable ordering causes the picker to arbitrarily pick someone — silent miscompile.

**Fix:** filter candidates to `nu > p`. If no candidate survives, return None. The caller surfaces this as `Overflow`, triggering graceful fallback to default emission rather than silent corruption. Two old `failwith` sites (the Step 2 allocation failure and the `do_reload` "must-succeed" path) were rewired to use the existing Exit+overflow_at pattern.

## Architecture

The Stage 4 wiring in `lib/emit_c.ml` adds three branches under `| None -> match scheduler with` (Topological / Bisection / SU). Each follows the same pattern:

1. Run scheduler → `raw_scheduled`
2. `Regalloc.prepare_for_simple_codelet[_from_oref] ~raw_scheduled ~assigns ...` → `input`
   - Dedupes (satisfies I1)
   - Builds force_last_use mapping output tags to end-of-schedule (satisfies I5)
3. `install_alloc_canonical "name" input` → registers allocation (may Overflow → falls back silently)
4. `emit_regalloc_spill_decl buf` → emits `__m256d regalloc_spill[N];`
5. `List.iteri` walking `input.scheduled` emits per-position: spill_sites, reload_sites, node def
6. **End-of-schedule** spill_sites + reload_sites at position n (CRITICAL — was missed in initial wiring, caused 30 spill slots allocated but only 16 written for R=7 AVX2)
7. `List.iter assigns` emits stores
8. `clear_alloc ()`

Three factored helpers (`install_alloc_canonical`, `emit_node_spill_sites`, `emit_node_reload_sites`) sit next to the existing `install_alloc` to keep the cluster-spill recipe working unchanged.

The Annotated_bisection and Annotated_SU branches use `Annotate.emit_scope` (nested-block emission discipline) and were left untouched — primes don't reach them, so non-blocking.

## Tooling

`find_bad_reuse_v3.py` (the existing register-pinning conflict detector) only handles `__m512d` — it silently passed AVX2 cases that had real bugs. Stage 4 surfaced this gap; the companion `find_bad_avx2.py` at `/tmp/find_bad_avx2.py` extends the detector to AVX2. Both should be used in tandem going forward.

## Files changed

- `lib/regalloc.ml`: Stages 2 + 3 + Belady fixes (Part 1 + Part 2) + graceful Overflow signaling
- `lib/emit_c.ml`: Stage 4 wiring (Topological, Bisection, SU branches)

No changes to `lib/isa.ml`, `lib/algsimp.ml`, `lib/schedule.ml`, or any radix-specific generator.

## Open follow-ups (non-blocking)

1. **R=17/R=19 AVX2 still fall back.** Peak_live 70 and 79 vs budget 14 (5×-5.6× overflow) is genuinely extreme. The fallback emits the existing FFTW-style codelet, which is correct but doesn't benefit from M-project pinning. A future "rematerialize twiddle loads" pass could relieve pressure (twiddle Loads are cheap to recompute from memory).

2. **Annotated_bisection / Annotated_SU branches** still use the old emission discipline. They're used for nested-block scopes (the cluster-spill recipe), not for simple codelets, so primes don't reach them. Wiring them through canonical prep is a cleanup, not a correctness issue.

3. **R=19 AVX2 falls back even with the Belady fix** — confirms the failure is fundamentally about budget, not allocator logic.
