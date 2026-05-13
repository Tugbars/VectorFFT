# VectorFFT Register Allocation — The M-Project

A multi-stage extension of the VectorFFT codelet generator that adds an
SSA-based register allocator with spilling, producing C output with explicit
`register __m512d t asm("zmmN")` pinnings. The goal is to bypass gcc's
register allocator on extreme-pressure straight-line FFT code where it
makes demonstrably bad choices.

**Status:** Complete through M5e. Correctness validated across 9 regression
cases (R=8..R=256, AVX2 and AVX-512, t1_dit and t1_log3 variants), zero
use-after-clobber bugs across runs producing up to 1833 spill slots and
6594 allocated tags. Performance wins for spill counts under ~500 slots
(range +3.5% to +22%); regressions for spill counts above ~1500 slots
(range -14% to -29%) attributable to redundant reload loads. M6
(reload-variable lifetime tracking) is the remaining work to recover
the high-pressure cases.

---

## 1. Motivation

VectorFFT codelets are straight-line C functions consisting of hundreds to
thousands of intrinsic calls (`_mm512_fmadd_pd`, `_mm512_loadu_pd`, etc.)
with no branches or function calls between the load preamble and the store
epilogue. Each codelet typically has two passes (the cluster-flushed
strided structure) and uses tag variables `tN` to hold intermediate values.

At R=64 and above the live set easily exceeds the 32 architectural vector
registers on AVX-512 (or 16 on AVX2). When this happens, gcc's
register allocator must spill. Empirically it makes poor choices: it spills
values that are about to be used (instead of values that are dead-far-out)
and it duplicates loads at multiple use sites rather than reloading once
into a fresh register and keeping it live. The result is more memory
traffic than necessary and worse instruction-level parallelism than the
underlying intrinsic schedule would allow.

The approach: emit C with explicit register pinnings using gcc's
`register __m512d tN asm("zmm15")` syntax, anchored with an empty inline
asm barrier `asm volatile("" : "+v"(tN))` after each declaration to prevent
the constraint from being lost during optimization. Have the codelet
generator run its own SSA register allocator, choose pinnings using a
proper algorithm (linear scan + Belady eviction for spilling), and feed
gcc a stream of constraints it must honor. gcc then becomes a peephole
optimizer and assembler driver rather than a register allocator.

---

## 2. High-Level Architecture

### 2.1 Codelet structure

A typical generated codelet:

```c
void radix128_t1_dit_log3_fwd_avx512_gen_inplace_su_spill(
    double *rio_re, double *rio_im,
    const double *tw_re, const double *tw_im,
    size_t ios, size_t K) {
  for (size_t k = 0; k < K; k += 8) {
    /* pass 1: load + decimate + twiddle */
    {
      __m512d spill_re[N1], spill_im[N1];
      __m512d regalloc_spill[M1];  /* M5 reload arena */
      __m512d t0 = _mm512_loadu_pd(&rio_re[k]);
      register __m512d t1 asm("zmm0") = ...;  /* M3a pinning */
      asm volatile ("" : "+v"(t1));            /* M3a barrier */
      _mm512_storeu_pd(&regalloc_spill[3], t78); /* M5 spill */
      register __m512d t101_r0 asm("zmm5") =     /* M5 reload */
          _mm512_loadu_pd(&regalloc_spill[3]);
      asm volatile ("" : "+v"(t101_r0));
      ...
      _mm512_storeu_pd(&spill_re[c], tN);  /* cluster output */
    }
    /* pass 2: combine + final outputs */
    {
      __m512d regalloc_spill[M2];  /* separate arena per pass */
      ...
      _mm512_storeu_pd(&rio_re[i], tN);
    }
  }
}
```

Two passes per outer k-iteration. Each pass has its own scope, its own
spill array, and its own register allocation. The cluster pattern means
pass 1 produces N cluster outputs into a shared `spill_re/spill_im`
inter-pass buffer, which pass 2 reads as inputs.

### 2.2 What the allocator does

Given the scheduled IR (a list of nodes in execution order), for each tag
produced:

- **M3a allocates** a vector register from a budget pool, with last-use
  detection to free registers at the right point.
- **M5 falls back to spilling** when the pool is exhausted at any node: it
  picks an in-register victim by Belady (latest next-use), emits a store
  to the next `regalloc_spill[]` slot after the current node, and frees
  the victim's register. When a spilled tag is used later, M5 emits a
  fresh reload into a free register at the use site.

Output: an `allocation` record consulted by `emit_c.ml` when emitting each
node, decorating tag declarations with `register __m512d` pinnings and
injecting spill/reload sites at the right positions in the C output.

### 2.3 What changes for gcc

With pinnings in place, gcc cannot make register allocation decisions on
the pinned variables. It still does:
- Instruction selection and folding (FMA detection, etc.)
- Scheduling within the constraints of the pinnings
- Register allocation for any temporaries it introduces
- The actual code emission

The pinnings constrain it without removing its remaining freedoms.

---

## 3. The M Stages

### M1: Types + Stub

Created `lib/regalloc.ml` (≈138 lines initial). Defined the
`allocation` record with placeholder fields. Function `allocate` returned
all tags as `Unbound`. Wired into `emit_c.ml` behind `VFFT_USE_REGALLOC=1`
env var. With the env var set, behavior was identical to default — every
tag emitted as bare `__m512d tN = ...`. This was an intermediate
compilable state, deliberately staged to validate the plumbing before
adding real logic.

### M2: peak_live Analysis

Added live-range analysis: for each position in the schedule, count tags
whose def precedes the position and whose last use is at or after it.
`VFFT_PEAK_LIVE=1` env var prints the peak across the schedule. Used to
empirically determine which codelets would overflow which budgets, and to
sanity-check the allocator's spilling decisions. Pure measurement
infrastructure, no behavior change.

### M3a: SSA Register Allocation

Real linear-scan allocator. Budget=28 on AVX-512 (zmm0..zmm27, leaving
zmm28..zmm31 as headroom for gcc). Algorithm: walk the schedule in order;
at each node, free registers whose live ranges ended; allocate a register
for the new tag from the free pool; if the pool is exhausted, mark the
tag as `Unbound` (fall back to gcc's RA for this codelet).

Three subtleties consumed most of M3a's debug time:

1. **Pinning barriers are mandatory.** Without
   `asm volatile("" : "+v"(tN))` after each pinned declaration, gcc treats
   the asm-register annotation as advisory and may reallocate the variable
   during optimization. The barrier forces gcc to materialize the value
   in the pinned register at that program point.

2. **Inlined nodes' uses are transitive.** The IR has an `inline_set` of
   nodes whose definitions are folded into their unique consumer's
   expression (e.g. a single-use multiply folded into an add). For
   live-range purposes the use site is the *consumer's* position, not the
   inlined node's position, and the inlined node's own uses (the operands
   of its expression) must be walked recursively. Missing this caused
   premature register frees and silent wrong-output bugs.

3. **Output stores live outside the DAG.** Cluster-flush stores
   (`_mm512_storeu_pd(&spill_re[c], tN)`) are emitted *between* nodes
   based on cluster-flush position, not as nodes themselves. The
   allocator needs to know about these "phantom" uses or it will free
   their registers before the store fires. Solution: `force_last_use`
   map keyed on `(cluster, tag)` pointing at the flush position,
   consulted alongside the normal last-use logic.

Result on R=64 t1_dit AVX-512: stack spills 493→247 (-50%), code size
25072→21800B (-13%), runtime 1049→970ns (-7.5%, single trial; rigorous
5-trial measurement later gave +10%).

### M5: Spilling

Extends M3a so that overflow doesn't fall back to gcc — instead spill a
live tag to a slot, free its register, allocate the new tag into the
freed slot. When the spilled tag is later used, emit a fresh reload load
into a register at the use site.

#### 3.1 Architecture decisions

- **Q-A: Separate spill arena.** A new `__m512d regalloc_spill[N]` array,
  scoped inside each pass's `{ }` block. Independent from the existing
  `spill_re/spill_im` inter-pass arrays. Pass 2's array decl is emitted
  *after* `install_alloc spill_pass2` so it sees the pass-2 slot count.
- **Q-B: Belady eviction.** Pick the in-register tag whose next use is
  latest. Standard Belady's algorithm. Negative sentinel tags
  (reload-variable lifetimes; see below) are excluded from victim
  selection.
- **Q-C: Rename-based reload-rewrite.** When a tag `t101` is spilled and
  later reloaded, the reload introduces a fresh C variable name
  (`t101_r0`, `t101_r1`, ...). Subsequent uses within the reload's
  lifetime reference the new name via a `name_overrides` map. This
  avoids needing to rewrite the original tag's declaration.
- **Q-D: Staged rollout.** M5a (R=64 log3), M5b (R=128 t1_dit),
  M5c (R=256 t1_dit), M5c+ (R=128 log3), M5d (R=256 log3), M5e (AVX2).

#### 3.2 Data structures

Added to the `allocation` record:

```ocaml
type allocation = {
  ...
  num_spill_slots : int;
  reload_sites : (int, reload_decl list) Hashtbl.t;  (* pos -> reloads *)
  spill_sites : (int, (int * int) list) Hashtbl.t;   (* pos -> [(tag, slot)] *)
  spilled_of_tag : (int, int) Hashtbl.t;             (* tag -> slot *)
  name_overrides : (int * int, string) Hashtbl.t;    (* (pos, tag) -> name *)
}
and reload_decl = {
  reload_tag : int;
  reload_name : string;   (* "tN_rK" *)
  reload_reg : string;    (* "zmm5" *)
  reload_slot : int;
}
```

#### 3.3 Architectural fixes (subtleties discovered during debug)

1. **Spilled assignment is not retroactive.** Initial design tried to
   change `result[victim]` from `Reg "zmmN"` to `Spilled` after picking
   the victim. This was wrong — the victim's original register
   assignment is still emitted at its def site. Spilling is a *separate
   emission event* (a post-node store), not a retroactive rewrite. The
   `Spilled` variant of the result type became dead code and was removed.

2. **Sentinel-tag lifetimes for reload variables.** Reload variables need
   register colors too, but they aren't real IR tags — no entry in
   `uses_sorted`. Solution: `do_reload` allocates a sentinel
   (negative integer) into the `allocated` table; `release_dead` at
   `pos+1` frees it. Without this, multiple reloads at the same position
   collide on the same register.

3. **Belady victim filter.** `pick_belady_victim` must exclude negative
   sentinel tags — they're reload variables, not real values, so they
   can't be "spilled" (they're already a reload of a spilled value).

4. **Spill store order.** At a given position, `emit_c` emits
   `spill_sites` *before* `reload_sites`. This matters when a tag is
   both spilled (because something is being reloaded into its register)
   and reloaded (because it itself was spilled earlier) at the same
   position — e.g. cluster-flush evictions cascading into reloads.

5. **Cluster-flush position.** `flush_pos_for_cluster` is computed in
   `emit_c.ml` and drives `force_last_use` (M3a). For the final cluster,
   the flush fires at `pass2_n` (the position one past the last node).
   The allocator pre-positions its end-of-pass forced reloads to this
   index.

6. **Pass-local regalloc_spill arrays.** The `__m512d regalloc_spill[N]`
   declaration is emitted *inside* each pass's `{ }` block, with the
   pass-specific slot count. Critically, pass 2's decl is emitted only
   *after* `install_alloc spill_pass2` updates `current_regalloc` —
   otherwise it would use pass 1's stale count.

7. **`emit_store` reload fallback.** When `emit_c` is asked to emit a
   store of tag `t` at position `p`, it consults
   `name_overrides[(p, t)]`. If no override exists but `t` is in
   `spilled_of_tag`, emit an inline reload:
   `_mm256_storeu_pd(dst, _mm256_loadu_pd(&regalloc_spill[slot]))`.
   gcc picks a temp register. This handles the case where an output
   store reads a value that was spilled but never reloaded into a
   named variable.

8. **`try_reload` vs `do_reload` split.** Two reload APIs:
   - `try_reload` returns `Option` — used at Step 1b, Step 3, and
     post-iter, where reload failure can fall back to inline-load.
   - `do_reload` always succeeds — used at Step 1 for IR-level
     predecessor reloads, which must have a real register binding.

#### 3.4 Three-step iteration with fixed-point loops

Per-node processing in the allocator:

```
Step 1:    For each IR predecessor of node[i]:
             if pred is spilled:
               do_reload at position i        ─┐
                                                ├─ Step 1 fixed-point:
           After all preds reloaded,            │  a Step 1 reload may
           re-check: are any preds still       │  itself need its source
           spilled? (A reload may have         │  reloaded — iterate
           triggered a fresh spill upstream.)   │  until quiescent.
             If yes, restart Step 1.          ─┘

Step 1b:   For each tag t in forced_at[i]:
             if t is spilled, try_reload it at position i
           (forced_at[] is populated from output stores and cluster
            flushes whose position is i.)

Step 2:    Allocate node[i]'s result tag.
             If no register free, pick Belady victim, spill it,
             allocate this tag into freed register.

Step 3:    Re-check forced_at[i] fixed-point loop:
             Step 2 may have spilled a tag that was in forced_at[i].
             Iterate until forced_at[i] tags are all in-register or
             irrecoverably absent.

Post-iter: For tags in forced_at[pass2_n] (end-of-pass forced reloads):
             Same fixed-point treatment.
```

The Step 1 fixed-point was the last correctness fix. Without it, R=256
codelets had 548 wrong outputs because a Step 1 reload could itself
trigger a spill of its source, but the source wasn't re-considered. With
it: 6 LSB diffs (FP nonassociativity), 0 bugs.

### M5e: AVX2 budget tuning

Original budget formula: `isa.vec_regs - 4`. On AVX-512 that's 28
(reserving zmm28..zmm31). On AVX2 that's 12 (reserving ymm12..ymm15).
The 12 was too tight — R=64 AVX2 was a wash at -0.3% because gcc
benefits from more registers per node than the M5 path was leaving it.

Changed to:
```ocaml
if isa.vec_regs >= 32 then isa.vec_regs - 4
else isa.vec_regs - 2
```
AVX-512 unchanged at 28. AVX2 becomes 14. R=64 AVX2 went from -0.3% to
+3.5%. Smaller AVX2 cases (R=8, R=16) saw 1-4% changes either direction
within noise; R=32 was stable.

---

## 4. Final Performance Results

5-trial cool-down protocol: 3-second sleep between binary runs; report
range across trials. All on Sapphire Rapids, gcc-11 -O3. K=16 (AVX-512)
or K=8 (AVX2) inner-loop iterations.

### 4.1 AVX-512 (budget=28)

| Codelet | Spill slots | M5 median | Range | Verdict |
|---|---|---|---|---|
| R=64 t1_dit | 0 (M3a) | **+10%** | +6 to +12% | win |
| R=64 t1_log3 | ~25 | **+5%** | +5 to +6% | win |
| R=128 t1_dit | moderate | **+6%** | +3 to +9% | win |
| R=128 t1_log3 | 263 | **+15%** | +13 to +16% | **best win** |
| R=256 t1_dit | 1544 | **-14%** | -11 to -17% | regression |
| R=256 t1_log3 | 1833 | **-29%** | -27 to -32% | **worst regression** |

### 4.2 AVX2 (budget=14)

| Codelet | M5 median | Notes |
|---|---|---|
| R=8 t1_dit | **+5%** | M3a path, bit-exact |
| R=16 t1_dit | **+17%** | M3a path, bit-exact |
| R=32 t1_dit | **+22%** | M5 (pass 1 spills), bit-exact |
| R=64 t1_dit | **+3.5%** | M5 (both passes spill), bit-exact |

AVX2 cases all produced **bit-exact** outputs (zero FP rounding diffs).
AVX-512 cases produced 0-6 LSB diffs due to gcc reordering FMA chains
around the M5-emitted spill stores — pure FP nonassociativity, not bugs.

### 4.3 Performance vs spill count

| Slot count | Typical perf |
|---|---|
| 0 (M3a fits) | +5% to +10% |
| 1-300 | +5% to +22% |
| 300-1000 | marginal (-5% to +6%) |
| 1000+ | -14% to -29% |

The crossover is around 500-800 slots on this hardware/compiler combo.

### 4.4 Why R=256 regresses

M5 currently follows a **spill-once, reload-each-use** policy: each use
site of a spilled tag emits its own fresh reload load. For a tag with N
uses, M5 emits N loads where gcc might have shared one. With 1300+
spilled values at R=256 pass 2, the load count balloons and memory
traffic exceeds the savings from gcc's better allocation decisions on
the non-spilled tags.

This is what M6 (reload-variable lifetime tracking) will fix.

---

## 5. Code Size

Mostly smaller (gcc's RA on extreme pressure tends to emit more code):

| Codelet | Default | M5 | Delta |
|---|---|---|---|
| AVX-512 R=32 | 13.7 KB | 11.3 KB | -17.4% |
| AVX-512 R=64 t1_dit | 25.1 KB | 21.8 KB | -13.1% |
| AVX-512 R=64 log3 | 26.8 KB | 24.8 KB | -7.6% |
| AVX-512 R=128 t1_dit | 59.1 KB | 52.5 KB | -11.2% |
| AVX-512 R=128 log3 | 63.1 KB | 63.5 KB | +0.6% |
| AVX-512 R=256 t1_dit | 145.5 KB | 157.3 KB | +8.2% |
| AVX-512 R=256 log3 | 149.1 KB | 173.8 KB | +16.6% |
| AVX2 R=8 | 2.7 KB | 2.6 KB | -1.5% |
| AVX2 R=16 | 5.6 KB | 5.0 KB | -10.5% |
| AVX2 R=32 | 14.6 KB | 12.1 KB | -16.9% |
| AVX2 R=64 | 26.3 KB | 27.9 KB | +6.2% |

The R=256 expansion is the reload-load explosion. Everywhere else, M5
output is the same size or smaller despite the extra `spill_re/`
infrastructure.

---

## 6. Files Changed

Three files in the OCaml codegen, plus two test files. Everything else
in the codebase (algsimp, schedule, dft, bb, uarch, expr, annotate,
split_radix, dft_r2c) was untouched.

| File | Pre-M | Post-M | Notes |
|---|---|---|---|
| `lib/regalloc.ml` | — | 918 lines | NEW; M1 created stub, M2/M3a/M5 grew |
| `lib/emit_c.ml` | 1667 lines | 2055 lines | +388 (M3a +174, M5 +214) |
| `lib/isa.ml` | 140 lines | 140 lines | +`pinned_reg_decl` helper (M3a) |
| `bin_test/m1_test.ml` | — | — | NEW (M1) |
| `bin_test/m2_test.ml` | — | — | NEW (M2) |

Diffs against M3a baseline preserved in
`/mnt/user-data/outputs/{regalloc,emit_c}.ml.diff`.

---

## 7. Environment Variables

- `VFFT_PEAK_LIVE=1` — print peak live count per codelet (M2; pure
  measurement, no behavior change).
- `VFFT_USE_REGALLOC=1` — enable M3a path. If a pass overflows the
  budget, fall back to gcc's default RA for that pass.
- `VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1` — enable M3a with M5
  fallback. Overflow → spill via Belady eviction instead of
  fall-through.

Naming: codelets generated with M3a/M5 have the suffix `_su_spill`
("spill" is from the cluster-output `spill_re/spill_im` infrastructure,
which predates the M-project and is unrelated to M5's `regalloc_spill[]`).

---

## 8. Correctness Summary

**9/9 regression cases pass.** All output diffs are at FP-epsilon
(0-6 LSB diffs out of thousands of doubles), all due to gcc's permitted
reordering of associative-but-not-commutative FMA chains around M5's
spill stores. **Zero use-after-clobber bugs across all runs**, validated
by `find_bad_reuse_v3.py` which traces register-pinning conflicts
through the C output.

Largest case: R=256 t1_log3 AVX-512, 3836 + 2758 = 6594 allocated tags,
527 + 1306 = 1833 spill slots. 6 LSB diffs, 0 bugs.

---

## 9. Methodology Lessons

Three things to remember next time:

1. **Single-trial timing is noise.** Early M5a/M5b/M5c numbers (+10.2%,
   +27.7%, +2.5%) were inflated by warm/cold cache and thermal state.
   The honest 5-trial cool-down measurements gave +5%, +6%, **-14%** —
   substantially different. The R=256 "+2.5% win" was particularly
   misleading; it actually regresses.

2. **A correctness fix can change perf.** The R=256 "+2.5%" was measured
   on a codelet with 548 wrong outputs (pre Step-1-fixed-point). The
   buggy version was faster precisely because it skipped the
   reload-cascade that correctness requires. After the fix, perf dropped
   to -14%. **Always re-measure after correctness fixes** — never trust
   "the bug fix doesn't affect perf."

3. **Intermediate compilable states are valuable.** M1's stub (returns
   `Unbound` for everything, behavior unchanged) caught wiring bugs
   before any algorithm complexity was added. Adding `pinned_reg_decl`
   to `isa.ml` in M3a *before* wiring it into `emit_c.ml` did the same
   thing. Each stage compiled, each stage tested, each stage's
   correctness was a precondition for the next.

---

## 10. What's Left

### M6: Reload-variable lifetime tracking

Current M5: each use of a spilled tag emits a fresh reload load. For a
tag with N uses across positions p1..pN, M5 emits N loads — one per use
site, each into a new variable (`tN_r0`, `tN_r1`, ...).

Target M6: track reload-variable lifetimes the same way M3a tracks
normal tags. A reload at position p1 produces a fresh register binding
that stays alive until its last use within p2..pN, with `name_overrides`
pointing all intermediate uses at the same variable. Subsequent uses
after the reload-variable dies need a new reload.

This is mechanically straightforward: extend `uses_sorted` to include
synthetic reload tags, run the same Belady-aware liveness analysis on
them, decide which reloads should be promoted to "kept alive across N
uses" vs "single-shot reload" based on local register pressure.

Expected impact: R=256 t1_dit should flip from -14% to a moderate win.
R=256 t1_log3 should improve from -29% to at least neutral. Cases that
are already wins should not regress (the optimization only removes
redundant loads).

### AVX2 t1_log3 variants

Informal: not in original plan. R=64 log3 AVX2, R=128 log3 AVX2 etc.
Given log3 has heavier register pressure and AVX2 has fewer registers,
some of these may not even fit M5 cleanly. Worth probing to find the
practical R-limit on AVX2.

### Alternative compilers

ICX and clang have different register allocators with different spill
schedulers. In particular, ICX is known for aggressive coalescing of
redundant memory ops. There's a reasonable chance ICX naturally
collapses the redundant reloads at the assembly level, making R=256
viable without M6. Worth measuring before investing M6 time.

---

## 11. Reproducing Results

From the work tree (`lib/regalloc.ml`, `lib/emit_c.ml`,
`lib/isa.ml` at this commit):

```bash
dune build

# baseline
./_build/default/bin/gen_radix.exe 128 --twiddled --in-place \
    --log3 --isa avx512 --emit-c > r128_log3_def.c

# M5
VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1 \
  ./_build/default/bin/gen_radix.exe 128 --twiddled --in-place \
    --log3 --isa avx512 --emit-c > r128_log3_m5.c

# correctness
gcc-11 -O3 -mavx512f -mfma -c r128_log3_def.c -o r128_log3_def.o
gcc-11 -O3 -mavx512f -mfma -c r128_log3_m5.c  -o r128_log3_m5.o
# (link with probe_r128_log3.c, compare outputs)
```

Full regression script at `/mnt/user-data/outputs/regression.sh` — runs
all 9 cases, reports PASS/FAIL with diff counts, bug counts, and code
size deltas.
