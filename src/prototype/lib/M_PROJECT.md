# VectorFFT Register Allocation — The M-Project

A multi-stage extension of the VectorFFT codelet generator that adds an
SSA-based register allocator with spilling, producing C output with explicit
`register __m512d t asm("zmmN")` pinnings. The goal is to bypass gcc's
register allocator on extreme-pressure straight-line FFT code where it
makes demonstrably bad choices.

**Status:** Complete through M6. Correctness validated across 9 regression
cases (R=8..R=256, AVX2 and AVX-512, t1_dit and t1_log3 variants), zero
use-after-clobber bugs across runs producing up to 1833 spill slots and
6594 allocated tags. M6 (reload-variable lifetime tracking) flipped
R=256 t1_dit from -14% regression to +5% win, partially fixed R=256
t1_log3 (-29% → -26%), and preserved all other wins within noise.
Performance now ranges from +4% to +23% across the test matrix, with a
single residual regression at R=256 t1_log3 attributable to the
codelet's inherent dependency-chain depth rather than allocator
behavior.

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

## 1.1 Why gcc degrades at scale

"gcc makes poor choices" is the empirical observation that drove the
M-project. The deeper question is *why* — and the answer ties together
two compiler-internal limits with one structural property of our IR
that we can exploit.

### 1.1.1 The two heuristic limits gcc hits

**Register allocation is NP-hard, and gcc uses heuristic approximations.**
Optimal register allocation is equivalent to graph coloring with spill
costs, which is NP-hard for general interference graphs. gcc's RA
(integrated register allocator, IRA) walks the basic block in priority
order and makes locally-optimal spill decisions using cost estimates.
At small scale (few dozen live values, sparse interference graph) the
heuristic is near-optimal — the search space is small enough that
locally-reasonable choices don't accumulate into globally-bad outcomes.
At our codelet sizes the search space explodes: R=256 t1_log3 has 6594
SSA values and an interference graph with thousands of edges. The
heuristic's locally-reasonable spill decisions compound into
globally-suboptimal choices. The result is more spill stores than
necessary, redundant reload loads, and registers held by values that
should have been evicted earlier.

**Instruction scheduling has a finite lookahead window.** gcc's
scheduler considers a sliding window of instructions when deciding
issue order. At codelet sizes beyond a few hundred instructions, the
scheduler's view of the program is effectively local: it can't reorder
operations across the whole codelet to expose ILP or to relieve
register pressure. The schedule order becomes essentially the
emission order — which is the order our scheduler (algsimp +
topological/SU sort) picked, not gcc's.

On R=64 t1_dit AVX-512, the default gcc emit contained 493 stack-spill
stores (`vmovapd %zmmN, off(%rsp)`); the M3a-precolored emit reduced
this to 247 — a 50% reduction with no algorithmic change to the FFT
itself, just a different register assignment. The 246 saved spills
represent locally-suboptimal choices the IRA made on a codelet that
fits in 28 registers if you pick the right ones.

The cross-compiler measurement on R=256 codelets makes the same point
more sharply. gcc-11 -O3 emits noticeably fewer stack spills than
gcc-15 -O3 on the same input C; we did not bisect which heuristic
change between gcc-11 and gcc-15 is responsible, but the regression
is reproducible on our entire test suite. Specifically, on the R=256
codelet body the gcc-11 default path runs at ~565 ns/FFT while gcc-15
default is meaningfully worse — the M-project ended up tracking
gcc-11 as its baseline for that reason.

What we did NOT do: systematic exploration of
`-fira-algorithm=priority` vs `-fira-algorithm=CB`,
`-fsched-stalled-insns`, `-flive-range-shrinkage` (we use the last one
as a compile flag but did not study whether its absence changes the
picture), or `-fno-schedule-insns2`. The intervention we built
(precoloring + barrier) bypasses these knobs entirely, so we didn't
need to characterize them. A reader who wants to understand the
gcc-side failure modes specifically would want to run that flag matrix
on a fixed codelet and read the resulting objdump. Worth doing if the
goal is to file a gcc bug rather than to ship a workaround.

### 1.1.2 How the two limits compound

Either problem alone would be survivable — gcc could compensate for
one through the other. The damage comes from the interaction:

1. Heuristic RA emits a suboptimal spill → an extra `vmovapd` to
   stack at position p.
2. The spill store creates a dependency edge the scheduler can't
   reorder around (it's a memory write; subsequent reads must wait).
3. The scheduler's window is now filled with memory-dependent
   instructions; it can't find independent arithmetic to hide the
   spill latency.
4. While the spill is round-tripping through L1, *other* live values
   stay live longer than necessary — extending their live ranges and
   forcing more spills later.

Once both limits are saturated, every additional instruction makes
things worse, not better. This is the regime our R=256 codelets live
in under default gcc handling. It's also why the size where M-project
wins flip from "small" to "huge" is so sharp: it's the size at which
gcc's two limits both saturate.

We have indirect evidence of the cascade through M5 vs M6 measurements.
On R=128 t1_log3 AVX-512, M5 emitted 946 references to
`regalloc_spill[]`; M6 reduced this to 446 — a 53% reduction at the
source level. The runtime change between M5 (+15%) and M6 (+14%) is
within noise: the half-of-loads reduction barely shows up in the wall
clock. This is consistent with the cascade story: at moderate pressure
each load is independent and gcc's scheduler can hide latency by
reordering surrounding FMAs; removing 500 loads doesn't help much
because each was already partially hidden.

R=256 t1_dit makes the cascade visible. M5 emitted 3239 references;
M6 reduced to 2811 — only a 13% source-level reduction. But the
runtime swing was +19 percentage points (M5 -14% → M6 +5%). The
runtime gain is dramatically larger than the load-count reduction
would predict. The explanation we landed on: at R=256 pressure, M5's
loads were no longer independent — they had filled gcc's scheduler
window and were stalling on each other. Removing 428 loads from a
saturated window opens up scheduling slack that benefits the
remaining ~2800 loads too. The 19% swing is the cascade unwinding,
not just the 13% direct effect.

R=256 t1_log3 is the case where the cascade unwinding hit a different
ceiling. M6 reduced reload refs by 30% (4551 → 3195) — a larger
fractional reduction than R=256 t1_dit. But the runtime improvement
was only +3% (-29% → -26%). The cascade IS unwinding at the load
level, but the dependency-chain depth of the split-radix structure
imposes a separate ceiling that loads have nothing to do with. This
is §4.4's diagnosis: when the scheduler limit is exhausted by chain
depth rather than by spill traffic, removing spill traffic recovers
nothing.

What we did NOT do: trace a specific gcc-side cascade through objdump
diffs ("removed 1 spill, then N other spills disappeared"). The
evidence we have is at the aggregate level — load counts and wall
times — not at the instruction level. A reader skeptical of the
cascade story could reasonably ask for a single-spill manipulation
experiment; we haven't run one.

### 1.1.3 Why our approach can do better

Two properties of our pipeline make the optimization tractable where
gcc's is heuristic:

**SSA programs have chordal interference graphs (Hack & Goos 2006).**
Because our IR is hash-consed, every value has exactly one definition
site — it's structurally SSA. The live-range interference graph of an
SSA program is *chordal*: every cycle of length ≥ 4 has a chord. This
isn't an empirical observation, it's a theorem about SSA programs.
Chordal graphs admit *optimal* greedy coloring in linear time via a
perfect elimination ordering. Our schedule order IS a perfect
elimination ordering (this falls out of how we hash-cons + topo-sort).
**So we compute the provable optimum in linear time where gcc's
heuristic guesses with a polynomial-time approximation.**

**Codegen time is unconstrained.** gcc's RA must finish in a small
fraction of total compile time. The whole compiler budget is on the
order of seconds. Our codelet generator can take *minutes* per codelet
if it needs to — there's no production hot path being blocked. We can
afford algorithms that are correct and slow rather than fast and
approximate.

The combination is what makes the M-project work: we solve the
problem optimally (per chordal coloring) and slowly (because we can),
where gcc was solving it approximately and quickly (because it had to).

### 1.1.4 How we transfer the solution through gcc

Computing the optimal coloring is half the work. The other half is
making gcc honor it. Two GNU C constructs do this:

- **`register T x asm("zmm5")`** pins a C variable to a specific
  physical register. Without further coercion, gcc treats this as a
  *hint* — strongly preferred but not binding. During subsequent
  optimization passes (instruction scheduling, peephole, dead-store
  elimination), gcc may rewrite the variable into a different register
  if the constraint becomes inconvenient. This is by design — gcc was
  built assuming the user wants suggestions, not mandates.
- **`asm volatile ("" : "+v"(x))`** is an empty inline-asm block that
  declares `x` as both an input and an output via the `+v` constraint
  (read-write, vector register class). This is the load-bearing piece:
  it forces gcc to *materialize* `x` in a vector register at exactly
  that program point. The empty body emits no instructions, but the
  constraint discipline is real — gcc must comply.

Used together, the pin tells gcc *which* register, and the barrier
tells gcc *that* the value must be there *now*. The combination
converts the suggestion into a mandate without preventing gcc from
making other optimization decisions on either side of the barrier.

The discovery happened during M3a, in the "first non-trivial codelet
breaks correctness" stage. With `register T x asm("zmmN")` declarations
in place but no barrier, the emitted assembly showed individual
variables getting moved out of their pinned registers during gcc's
optimization passes — the C-level `register asm()` decoration was
being respected at declaration time but not maintained through the
rest of the function. The resulting wrong-output bugs were the kind
that look mysterious until you read the `.s` output: the algorithm is
right, the register choice is right at the moment of definition, and
then 30 instructions later the value gets clobbered because gcc
decided zmm5 was free.

The fix we landed on — `asm volatile ("" : "+v"(x))` immediately after
each pinned declaration — converts the suggestion into a constraint
at that program point. The empty asm body emits no instructions; the
`"+v"(x)` constraint says "x is both read and written here, in a
vector register, specifically the one I named on the LHS." gcc
respects this in the same way it respects any other volatile asm
constraint: it materializes the value in the requested register at
exactly that program point, and any subsequent rewrite has to
re-materialize it.

We did NOT explore alternative constraint letters systematically. The
ones we know exist:

- `"v"` — any vector register
- `"x"` — XMM (legacy SSE class, also accepted for AVX/AVX-512)
- `"Yz"` — restricts to xmm0 specifically (for legacy ABI cases)

Of these, `"v"` was the obvious choice for `__m512d` and it worked,
so we stopped there. Whether `"x"` would also work and whether
there's any codegen difference between the two — we don't know. We
also did not verify whether the barrier-after-each-decl is the
minimum necessary discipline: a single barrier at end-of-block, or
barriers only at points where the optimizer would otherwise move
values, might suffice. What we know is that barrier-after-each-decl
works reliably across the combinations we've tested (gcc-11, gcc-15,
ICX). A more permissive discipline might also work and could in
principle produce smaller code or marginally better schedules; we
haven't tested.

The minimal failing repro we'd want for documentation purposes (a
tiny codelet that produces wrong output without the barrier and
correct output with it) we don't have saved. It's recoverable —
generate any M3a codelet, strip the barriers, recompile, observe
the `.s` diff — but it's a few hours of work to make a clean
reproduction. Filed under "would be nice to have for a paper, not
blocking anything."

### 1.1.5 The elevator pitch

We don't compete with gcc on its home turf — locally-optimal RA
decisions at small scale, fast compile times, general-purpose code.
We move the global RA decision out of gcc's hot path entirely, solve
it optimally at codegen time using the SSA chordal-coloring guarantee
plus all the time we want, and use the `register asm()` + barrier
pattern to transfer the solution intact through gcc's optimization
pipeline. gcc retains everything it's good at (instruction selection,
peephole optimization, scheduling within our constraints, the emit
itself); it just doesn't make the global coloring decisions on our
huge codelets anymore.

The +10% to +23% wins, the elimination of M5's R=256 regressions via
M6, the cleanliness of the ICX compat — all of these follow from this
single architectural choice.

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

### M6: Reload-variable lifetime tracking

M5's reload policy was **spill-once, reload-each-use**: every use site
of a spilled tag emitted a fresh `_mm512_loadu_pd(&regalloc_spill[N])`
into a new C variable. For a tag with N uses across positions
p₁..pₙ, M5 emitted N loads — one per use site, each into a new
variable (`tN_r0`, `tN_r1`, ...). For R≤128 this was acceptable; for
R=256 it produced enough redundant memory traffic to overwhelm gcc's
RA savings, causing -14% to -29% regressions.

M6 extends each reload's lifetime so one load can service multiple
uses, with Belady-driven eviction handling register pressure naturally.

#### 3.5.1 Architecture

When `try_reload(tag, pos)` is called:

1. **Fast path:** if `live_reload_of[tag]` exists AND its sentinel is
   still in `allocated` AND it covers the current position, return the
   existing sentinel's name. `name_overrides` was pre-populated at the
   original reload site, so emit_c will use the existing reload
   variable without an additional load.

2. **Fresh reload path:** compute the covered positions as
   `[pos]` ∪ `{p ∈ uses_sorted[tag] | p > pos}` — i.e., the current
   position plus all remaining future uses. Allocate a register,
   register a sentinel with that covered set, and pre-populate
   `name_overrides[(p, tag)] = name` for every covered p.

3. The sentinel becomes a first-class participant in liveness analysis.
   `next_use_after sentinel p` consults `covered_of_sentinel`; Belady
   considers it a valid victim with a real next-use value.

#### 3.5.2 New data structures

Five auxiliary tables, all keyed by sentinel ID (negative integer):

```ocaml
live_reload_of      : (tag, sentinel) Hashtbl.t
tag_of_sentinel     : (sentinel, tag) Hashtbl.t
name_of_sentinel    : (sentinel, string) Hashtbl.t
covered_of_sentinel : (sentinel, int list) Hashtbl.t
creation_of_sentinel: (sentinel, int) Hashtbl.t
```

Plus a forward-ref hack: `next_use_after` is defined before
`covered_of_sentinel` exists (because `pick_belady_victim` needs it
and is also defined early), so a `ref None` is bound at definition
time and assigned `Some covered_of_sentinel` after the table is
created. Five lines of OCaml awkwardness, no semantic impact.

#### 3.5.3 Sentinel eviction handling

When `pick_belady_victim` picks a sentinel as victim:

- **No spill store needed.** The underlying tag's value is already
  in `regalloc_spill[slot]` from the original spill. The sentinel was
  just a register reservation; freeing it costs nothing beyond the
  loss of in-register access.
- **`name_overrides` invalidation.** For every covered position
  `p_i > eviction_pos`, remove `name_overrides[(p_i, underlying_tag)]`.
  Subsequent uses of the tag will now fail the `name_overrides` lookup
  and re-trigger Step 1's reload check, which creates a fresh sentinel
  via `try_reload`.
- **Aux-table cleanup.** Remove the sentinel from all four aux tables.

#### 3.5.4 Self-eviction prevention

A sentinel just created at position p must not be evicted at p — its
register holds the value being used by the current node's RHS.
`pick_belady_victim` filters out sentinels where
`creation_of_sentinel[s] = p`, leaving them protected only for the
position they're needed at. From p+1 onward, the sentinel competes
normally with regular tags.

#### 3.5.5 Worst-case bound

If Belady evicts the sentinel at every use position (extreme
pressure), M6 degrades exactly to M5's per-use reload behavior. **M6
is strictly ≥ M5 in performance.** This is the structural guarantee
that made aggressive lifetime extension a safe default.

In practice on the test matrix:
- **R≤128 cases**: spill counts are too low for lifetime extension to
  bite (each spilled tag has few uses). M6 ≈ M5 within noise.
- **R=256 t1_dit**: moderate eviction pressure on sentinels. About 30%
  of would-be M5 loads collapse into shared reloads. Result: +19%
  improvement, flipping a regression to a win.
- **R=256 t1_log3**: extreme dependency depth + 1759 spill slots.
  Sentinels get evicted frequently. About 30% of loads still collapse,
  but the residual register pressure on regular tags (caused by the
  longer-lived sentinels) limits net improvement to ~3%.

#### 3.5.6 Empirical reload reduction

| Case | M5 regalloc_spill refs | M6 refs | Reduction |
|---|---|---|---|
| R=128 t1_log3 AVX-512 | 946 | 446 | -53% |
| R=256 t1_dit AVX-512 | 3239 | 2811 | -13% |
| R=256 t1_log3 AVX-512 | 4551 | 3195 | -30% |

R=128 log3 sees the biggest fractional reduction (low pressure, every
sentinel survives its full lifetime). R=256 t1_dit sees a smaller
fraction but biggest absolute impact (the extra loads were on the
critical path). R=256 log3 has a large reduction that doesn't fully
translate to runtime because the dependency chains were the dominant
bottleneck.

---

## 4. Final Performance Results

5-trial cool-down protocol: 3-second sleep between binary runs; report
range across trials. All on Sapphire Rapids, gcc-11 -O3. K=16 (AVX-512)
or K=8 (AVX2) inner-loop iterations.

### 4.1 AVX-512 (budget=28)

| Codelet | Spill slots | M5 median | M6 median | Δ (M6-M5) |
|---|---|---|---|---|
| R=64 t1_dit | 0 (M3a) | +10% | +10% | ≈ noise |
| R=64 t1_log3 | ~25 | +5% | +6% | ≈ noise |
| R=128 t1_dit | moderate | +6% | +6% | ≈ noise (verified rigorously) |
| R=128 t1_log3 | ~260 | +15% | +14% | ≈ noise |
| **R=256 t1_dit** | **~1500** | **-14%** | **+5%** | **+19% (fix)** |
| R=256 t1_log3 | ~1750 | -29% | -26% | +3% (partial) |

The single major change is R=256 t1_dit, which M6 lifts from a -14%
regression to a +5% win — a swing of about 19 percentage points,
consistent across 5 trials with ranges entirely non-overlapping with
M5's distribution.

### 4.2 AVX2 (budget=14)

| Codelet | M5 median | M6 median | Notes |
|---|---|---|---|
| R=8 t1_dit | +2% | +10% | M3a, bit-exact; M6 slightly improves |
| R=16 t1_dit | +19% | +19% | M3a, bit-exact |
| R=32 t1_dit | +23% | +23% | M5 (pass 1 spills), bit-exact |
| R=64 t1_dit | +3% | +4% | M5 (both passes spill), bit-exact |

AVX2 cases all produced **bit-exact** outputs (zero FP rounding diffs).
AVX-512 cases produced 0-6 LSB diffs due to gcc reordering FMA chains
around the spill stores — pure FP nonassociativity, not bugs.

### 4.3 Performance vs spill count (post-M6)

| Slot count | Typical perf |
|---|---|
| 0 (M3a fits) | +5% to +10% |
| 1-300 | +5% to +23% |
| 300-1000 | +5% to +14% |
| 1000-1600 | +5% to +6% (post-M6; was -14% under M5) |
| 1700+ | -26% residual (only R=256 t1_log3) |

The pre-M6 "crossover around 500-800 slots" pattern is gone. M6's
worst-case bound (degrades to M5 under extreme pressure) means M5's
regression cliff is replaced by a gentle slope toward neutral, with
only the most extreme single case showing a residual regression.

### 4.4 Why R=256 t1_log3 still regresses

M6 reduced R=256 t1_log3's regalloc_spill references by 30% (4551 →
3195), but runtime only improved from -29% to -26%. The remaining cost
isn't load-driven — it's the codelet's inherent **dependency-chain
depth** in the log3 variant. Even with optimal allocation and optimal
reload reuse, the FMA chains in t1_log3 limit ILP enough that any
spill-induced register pressure compounds with the chain latencies.
Fixing this further requires upstream work on the scheduler or codelet
decomposition (smaller building blocks, deeper unrolling), not the
register allocator.

For comparison, R=256 t1_dit has the same spill volume but shallower
dependency chains; M6 fully recovers it because the loads were the
bottleneck and removing them exposes parallelism gcc can exploit.

---

## 5. Code Size

Mostly smaller (gcc's RA on extreme pressure tends to emit more code).
M6 added a column where it differs meaningfully from M5; for cases where
M5 and M6 produce essentially identical output, the column is collapsed.

| Codelet | Default | M5 | M6 | Notes |
|---|---|---|---|---|
| AVX-512 R=32 | 13.7 KB | 11.3 KB | 11.3 KB | -17% (no change M5→M6) |
| AVX-512 R=64 t1_dit | 25.1 KB | 21.8 KB | 21.8 KB | -13% |
| AVX-512 R=64 log3 | 26.8 KB | 24.8 KB | 24.5 KB | -9% |
| AVX-512 R=128 t1_dit | 59.1 KB | 52.5 KB | 51.8 KB | -12% |
| AVX-512 R=128 log3 | 63.1 KB | 63.5 KB | 54.1 KB | **-14% (M6)** vs +0.6% (M5) |
| **AVX-512 R=256 t1_dit** | 145.5 KB | 157.3 KB | 144.8 KB | **-0.4% (M6)** vs +8.2% (M5) |
| **AVX-512 R=256 log3** | 149.1 KB | 173.8 KB | 140.2 KB | **-6% (M6)** vs +16.6% (M5) |
| AVX2 R=8 | 2.7 KB | 2.6 KB | 2.6 KB | -1.5% |
| AVX2 R=16 | 5.6 KB | 5.0 KB | 5.0 KB | -10.5% |
| AVX2 R=32 | 14.6 KB | 12.1 KB | 11.7 KB | -20% |
| AVX2 R=64 | 26.3 KB | 27.9 KB | 27.5 KB | +4.7% |

Every M5 code-size regression (R=128 log3 +0.6%, R=256 t1_dit +8.2%,
R=256 log3 +16.6%) is **eliminated** by M6. R=256 log3 in particular
flips from 17% larger than default to 6% smaller — the redundant
reload loads that M5 emitted were physically a large fraction of the
codelet's text size.

---

## 6. Files Changed

Three files in the OCaml codegen, plus two test files. Everything else
in the codebase (algsimp, schedule, dft, bb, uarch, expr, annotate,
split_radix, dft_r2c) was untouched.

| File | Pre-M | Post-M6 | Notes |
|---|---|---|---|
| `lib/regalloc.ml` | — | 1019 lines | NEW; M1 stub, grew through M5 (~875) then M6 (+~144) |
| `lib/emit_c.ml` | 1621 lines | 2005 lines | +384 (M3a +174, M5 +210); unchanged in M6 |
| `lib/isa.ml` | 106 lines | 119 lines | +`pinned_reg_decl` helper + barrier (M3a, +13 lines) |
| `bin_test/m1_test.ml` | — | — | NEW (M1) |
| `bin_test/m2_test.ml` | — | — | NEW (M2) |

Diffs against M3a baseline preserved in `src/prototype/diffs/`
(`emit_c.ml.diff`, `dft.ml.diff`, `generate_codelets.sh.diff`).

---

## 7. Environment Variables

- `VFFT_PEAK_LIVE=1` — print peak live count per codelet (M2; pure
  measurement, no behavior change).
- `VFFT_USE_REGALLOC=1` — enable M3a path. If a pass overflows the
  budget, fall back to gcc's default RA for that pass.
- `VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1` — enable M3a with M5/M6
  fallback. Overflow → spill via Belady eviction instead of
  fall-through. M6's reload-variable lifetime tracking is part of this
  path (no separate env var; it's the default behavior when M5 is on).

Naming: codelets generated with M3a/M5/M6 have the suffix `_su_spill`
("spill" is from the cluster-output `spill_re/spill_im` infrastructure,
which predates the M-project and is unrelated to M5/M6's
`regalloc_spill[]`).

---

## 8. Correctness Summary

**9/9 regression cases pass.** All output diffs are at FP-epsilon
(0-6 LSB diffs out of thousands of doubles), all due to gcc's permitted
reordering of associative-but-not-commutative FMA chains around M5/M6's
spill stores. **Zero use-after-clobber bugs across all runs**, validated
by `find_bad_reuse_v3.py` which traces register-pinning conflicts
through the C output.

Largest case: R=256 t1_log3 AVX-512, 3836 + 2758 = 6594 allocated tags,
527 + 1306 = 1833 spill slots. 6 LSB diffs, 0 bugs.

**Additional validation (2026-05-14):** Full codelet tree regenerated
with M3a + M5 + M6 active. 536 codelets total across both ISAs and
8 families (primes/small_pow2/mid_pow2/large_pow2/xl_pow2/composites/
trig/strided); 240 of them go through the spill-recipe path and
therefore exercise M5/M6 (the remaining 296 are primes/trig/strided
which don't use the `_su_spill` recipe). All 536 compile cleanly under
gcc-15 -O3 with `-flive-range-shrinkage -Wno-incompatible-pointer-types`.

- **Structural M6 verification:** R=256 t1_log3 AVX-512 codelet shows
  4.02 references per reload variable (1395 unique reload vars across
  5615 references). M5 would give exactly 2.0; the observed 4× reuse
  confirms M6's reload-variable lifetime tracking is functionally
  active.
- **AVX2 regression bench:** `bench/regression/regression_bench_avx2.c`
  (M6 OCaml output vs hand-coded Python reference) at R=16/25/32/64
  fwd_t1_dit. **17/20 cells OCaml WINS, 3 TIE, 0 regressions.** All
  correctness checks PASSED. Best wins: R=16 K=64 ratio 0.789, R=25
  K=64 ratio 0.794, R=64 K=1024 ratio 0.831.
- **ICX compile compatibility:** ICX accepts the `register asm()` +
  `asm volatile ("" : "+v"(x))` pattern cleanly. R=64 t1_dit_fwd
  AVX-512: ICX 22838 B vs gcc-15 22792 B (0.2% diff). R=256
  t1_dit_fwd_log3 AVX-512: ICX 127563 B vs gcc-15 134632 B
  (**ICX 5.2% smaller**). Both compilers use all 32 zmm registers at
  R=64; load/store instruction counts at R=256 log3 are essentially
  equivalent (5582 ICX vs 5535 gcc). M6 already removed the
  redundant-load explosion at the source level, so ICX can't make
  much additional progress by collapsing memory ops.

---

## 9. Methodology Lessons

Four things to remember next time:

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

4. **Worst-case bounds enable aggressive defaults.** M6's aggressive
   lifetime extension (cover all remaining uses) sounds risky at face
   value — what if pressure forces eviction at every use? But the
   answer is that Belady eviction handles it: the worst case degrades
   exactly to M5's per-use reload behavior. **M6 is provably ≥ M5 in
   performance.** That structural guarantee removed any need for
   heuristic gating or per-case tuning. The lesson generalizes:
   designing the worst-case bound first lets the common-case
   optimization be as aggressive as it needs to be.

---

## 10. What's Left

### Alternative compilers — partial result available

**Initial measurement (2026-05-14):** ICX (Intel oneAPI clang-LLVM-based,
2025.3) accepts M6-emitted codelets without modification. Compile-only
comparison:

- R=64 t1_dit_fwd AVX-512: ICX .o = 22838 B, gcc-15 .o = 22792 B (0.2%
  larger; essentially tied)
- R=256 t1_dit_fwd_log3 AVX-512: ICX .o = 127563 B, gcc-15 .o = 134632 B
  (**ICX 5.2% smaller**)
- Load/store instruction count at R=256 log3: ICX 5582, gcc 5535
  (essentially equivalent — within 1%)
- zmm register usage at R=64: 32 distinct registers in both compilers
  (the M3a coloring is honored by both)

The R=256 log3 size difference is real but doesn't come from collapsing
memory ops — load/store counts are essentially identical. ICX is making
different instruction-selection / scheduling choices on the surrounding
code. This validates the M_PROJECT thesis: M6 already removed the
redundant-load explosion at the source level, so compiler-level
optimization can't collapse loads further. **What's preserved is the
M3a coloring (both compilers honor the pinning) and the M5/M6
spill/reload structure.** What differs is the surrounding optimization,
and the differences are modest (single-digit percent).

**Still pending:** runtime perf measurement of M6 codelets under ICX
vs gcc. The compile-time check rules out the original "ICX collapses
M5's reload explosion" hypothesis (moot after M6) but doesn't tell us
whether ICX's instruction scheduling extracts additional speedup at
runtime. The R=256 t1_log3 residual regression is the most informative
test case: if ICX recovers it, that's evidence the dependency-chain-depth
diagnosis (§4.4) is gcc-specific; if it doesn't, the diagnosis is
confirmed and the next lever is upstream scheduling, not RA.

### Upstream scheduler work (only for R=256 t1_log3)

The 1750-slot residual case isn't allocator-bound; it's chain-depth
bound. Possible levers:

- **Schedule reordering** to break long FMA chains into shorter
  parallel sub-chains where possible.
- **Codelet decomposition** — split R=256 t1_log3 into two smaller
  inner codelets composed via the strided pattern.
- **Different radix choice** at planner level — if R=256 t1_log3 is
  worse than R=128 × R=2 combination at the relevant K, the planner
  should pick the combination.

These are weeks of work and only matter if R=256 t1_log3 actually
gets selected by the planner. If R=128 + something beats it at every
K of interest, this is unnecessary.

### AVX2 t1_log3 variants

Informal: not in original plan. R=64 log3 AVX2, R=128 log3 AVX2.
Given log3 has heavier register pressure and AVX2 has fewer registers,
some of these may stress M6's eviction logic in ways the AVX-512
suite doesn't. Worth probing to find the practical R-limit on AVX2.

### Heuristic: dynamic lifetime sizing

M6's aggressive policy (cover all remaining uses) is optimal when
pressure is moderate. Under extreme pressure it relies entirely on
Belady eviction to handle the consequences, which costs a re-reload
per evicted sentinel. A potential M7 would size the initial lifetime
based on local pressure — cover only N uses ahead instead of all
remaining ones — to reduce eviction churn. Unclear whether the gain
would be measurable; the worst-case bound (degrades to M5) is already
tight enough that there isn't much room above it.

---

## 11. Reproducing Results

From the work tree (`lib/regalloc.ml`, `lib/emit_c.ml`,
`lib/isa.ml` at this commit):

```bash
dune build

# baseline (no regalloc, gcc handles allocation)
./_build/default/bin/gen_radix.exe 128 --twiddled --in-place \
    --log3 --isa avx512 --emit-c > r128_log3_def.c

# M3a + M5 + M6 (current default when the env vars are set)
VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1 \
  ./_build/default/bin/gen_radix.exe 128 --twiddled --in-place \
    --log3 --isa avx512 --emit-c > r128_log3_m6.c

# correctness
gcc-11 -O3 -mavx512f -mfma -c r128_log3_def.c -o r128_log3_def.o
gcc-11 -O3 -mavx512f -mfma -c r128_log3_m6.c  -o r128_log3_m6.o
# (link with probe_r128_log3.c, compare outputs)
```

For a full-tree regeneration (all radixes, both ISAs, every family),
the canonical workflow is:

```bash
dune build
VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1 \
  ISA=both ./scripts/generate_codelets.sh
CC=gcc-11 EXTRA_CFLAGS='-flive-range-shrinkage -Wno-incompatible-pointer-types' \
  ./scripts/compile_codelets.sh
```

This produces 536 `.c` files across `codelets/{avx2,avx512}/{primes,
small_pow2,mid_pow2,large_pow2,xl_pow2,composites,trig,strided}/` and
their `.o` counterparts. The `-Wno-incompatible-pointer-types` flag
is needed for gcc-12+ which treats `_mm256_loadu_pd(&regalloc_spill[N])`
(where `regalloc_spill` is declared `__m256d[]`) as an error rather
than a warning. gcc-11 accepts it as a warning, hence the flag wasn't
needed during the original M-project development.

For AVX2 correctness regression: `bench/regression/regression_bench_avx2.c`
compares M6 OCaml output against hand-coded Python references for
R=16/25/32/64 at K ∈ {64,128,256,512,1024}. Build via
`build_and_run.sh` (sets PYGEN to the Python generators in
`src/vectorfft_tune/radixes/r{R}/`).

Full regression script at `/mnt/user-data/outputs/regression.sh` — runs
all 9 cases, reports PASS/FAIL with diff counts, bug counts, and code
size deltas.