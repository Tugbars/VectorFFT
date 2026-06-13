# M-Project — Influences and References

The M-project's algorithms didn't come from a clean-slate design; they're
applications and adaptations of classical compiler-construction work. This
document maps each major design choice back to the canonical paper that
introduced the idea, notes what we used directly, and notes where we
deviated.

The M-project itself was developed iteratively rather than from a literature
review, so the mapping here is partly reverse-engineered: "this is the
result we got — these are the papers that established the same result
earlier." For anyone extending the system, these are the right starting
points in the literature.

---

## 1. Belady's MIN algorithm (1966) — used for M5 victim selection

> Belady, L.A. (1966). **"A Study of Replacement Algorithms for a
> Virtual-Storage Computer."** *IBM Systems Journal*, 5(2), 78–101.

### Key idea

When a page must be evicted from a fixed-size cache to make room for a new
one, the optimal choice (in the offline case, where future accesses are
known) is to evict the page whose **next** use is **furthest in the
future**. Belady's paper proved this for virtual-memory paging; the same
result applies to any fixed-size-cache replacement problem, including
register allocation.

### Application in M5

When a node's result needs a register and none are free, M5 picks a
victim from the currently-live in-register tags. The victim-selection
function in `regalloc.ml` walks the live set and chooses the tag whose
`next_use[t]` is largest:

```ocaml
let pick_belady_victim ~allocated ~uses_sorted ~current_pos =
  (* of all in-register tags, pick the one whose next use after
     current_pos is the latest *)
```

This is Belady's MIN, applied to vector registers instead of memory pages.

### Why MIN works here despite being unrealizable in general

Belady's algorithm is normally infeasible because online schedulers don't
know the future. In a codelet generator, **we know the entire schedule
before any allocation runs.** The `uses_sorted` table gives every tag's
use positions exactly. So the optimal-but-impractical offline algorithm
becomes a perfectly practical online algorithm: M5's victim selection is
provably optimal *per-step* (though local greedy optimality doesn't imply
global allocation optimality — see Bouchez et al. on spill NP-hardness
below).

### What we omit

Belady's paper also studies LRU, FIFO, and Random replacement and
characterizes when each is closest to MIN. None of that matters for
our setting since we have the full schedule. The "Belady anomaly" (more
cache slots producing more misses for FIFO/LRU) is also irrelevant —
MIN doesn't suffer from it.

---

## 2. Poletto & Sarkar — Linear Scan (1999) — used for M3a

> Poletto, M. and Sarkar, V. (1999). **"Linear Scan Register Allocation."**
> *ACM Transactions on Programming Languages and Systems*, 21(5), 895–913.

### Key idea

Earlier register allocators (since Chaitin 1982) used graph coloring on
the interference graph — expensive (NP-hard in general) and complex to
implement. Poletto & Sarkar introduced a much simpler approach: compute
live intervals (one contiguous range per variable, ignoring sub-range
holes), sort them by start position, walk in order, and at each interval
allocate a free register or spill the interval whose end-point is
latest. Linear time, dramatically simpler to implement, and within a few
percent of graph-coloring quality on most benchmarks.

### Application in M3a

The M3a allocator is a direct application of linear scan to a single
straight-line block (a codelet pass) where the IR is already in
scheduled order:

- **Live intervals**: derived from `uses_sorted[t]` — the def position is
  the interval start; the last use (max of uses_sorted, plus any forced
  use from `force_last_use`) is the interval end.
- **Walk in order**: iterate over the scheduled IR positions; at each
  position, free intervals that have ended, allocate a register for the
  new interval from the free pool.
- **Spill**: when the free pool is empty, M3a marks the new tag
  `Unbound` (falls back to gcc). M5 replaces this fallback with Belady
  spilling (next section).

### Where we deviate from the paper

- **No interval splitting.** Poletto-Sarkar splits long-lived intervals
  to allow sub-range allocation. We don't — each tag gets exactly one
  register binding from def to last use, or it's fully spilled. The
  cost is some lost reuse opportunity; the benefit is much simpler
  emit-time bookkeeping (no per-position name remapping for non-spilled
  variables).
- **Single basic block.** Codelets are straight-line code with one
  outer loop wrapping two inner passes, each a single basic block.
  Linear scan was designed for whole-procedure allocation across
  multiple blocks. The single-block restriction removes most of the
  complexity (no live-in/live-out reconciliation, no φ-functions, no
  block boundary fixup).
- **Pre-computed schedule.** Standard linear scan computes liveness
  from the IR before allocation. Our IR is pre-scheduled by an earlier
  pipeline pass; we just read off live ranges from `uses_sorted`. Closer
  in spirit to local allocation than to global linear scan.

### Why we picked linear scan over graph coloring

Codelets are large straight-line blocks (hundreds to thousands of nodes,
peak live counts up to ~50 on AVX-512 R=256), but they're regular. The
interference graph is dense but has obvious structure. Graph coloring
would require building and reducing an interference graph that
in our largest case has thousands of nodes. Linear scan handles all of
this in a single pass over the scheduled IR with no graph construction.
The simplicity vs slight-quality-loss tradeoff is heavily in favor of
linear scan when the regularity of the input means linear scan finds the
"right" answer most of the time anyway.

---

## 3. Hack, Grund & Goos — SSA Register Allocation (2006) — structural inspiration

> Hack, S., Grund, D. and Goos, G. (2006). **"Register Allocation for
> Programs in SSA-Form."** *Compiler Construction (CC '06)*, LNCS 3923,
> 247–262.
>
> Hack, S. and Goos, G. (2006). **"Optimal register allocation for
> SSA-form programs in polynomial time."** *Information Processing
> Letters*, 98(4), 150–155.

### Key idea

The interference graph of a program in **static single assignment** form
is chordal (every cycle of length ≥4 has a chord). Chordal graphs can
be optimally colored in polynomial time, in contrast to the
NP-completeness of general graph coloring. More importantly for
practical compilers: in SSA, **coloring, spilling, and coalescing can be
fully decoupled.** Spilling decisions are made before coloring, and
coloring then finds an optimal assignment if one exists.

### Application in M-project

We don't use chordal coloring directly (we use linear scan; see §2). But
the structural insight — **decouple spilling from allocation** — is
the load-bearing idea behind M5's architecture:

- **M3a (allocation)**: linear-scan coloring of an already-spill-free
  problem. If allocation fails (overflow), report it.
- **M5 (spilling)**: when allocation overflows, pick a victim by
  Belady, spill it, free its register, allocate the new tag. The spill
  is a separate emission event (a store + reload pair), *not* a rewrite
  of the original tag's binding.

The "spilling is a separate emission event, not a retroactive rewrite"
fix (M5 architectural fix #1 in `M_PROJECT.md`) is exactly the
decoupling Hack et al. advocate. Our pre-fix design had spilling
mutating allocation decisions; the post-fix design has spilling adding
new events that allocation never touches.

### What's SSA-like and what isn't

The VectorFFT IR is naturally SSA-ish: each tag `tN` is defined once,
used multiple times, never reassigned. We don't have φ-functions
because we don't have control flow inside codelets. So the SSA
preconditions for Hack-Goos chordality apply, even though we never
explicitly call our IR "SSA" or build an interference graph.

### What we don't take from them

The chordal-coloring algorithm itself (perfect elimination ordering on
the dominator tree). Linear scan is simpler and the loss of optimality
doesn't matter at our problem size — gcc was leaving so much on the
table that any reasonable allocator wins.

---

## 4. Frigo — A Fast Fourier Transform Compiler (1999) — architectural ancestor

> Frigo, M. (1999). **"A Fast Fourier Transform Compiler."**
> *Proceedings of the 1999 ACM SIGPLAN Conference on Programming
> Language Design and Implementation (PLDI '99)*, Atlanta, GA.

### Key idea

Generate FFT codelets from a small set of mathematical rules using a
dedicated code generator (genfft) written in OCaml. The generator
produces straight-line C code containing the unrolled FFT for a fixed
size; this C code is then compiled by gcc/icc/clang and called from
FFTW's plan executor. The generator is small (~5000 lines of OCaml) but
the codelets it produces are large (often thousands of lines of C). The
key insight: small high-level FFT transformations are easy to express
symbolically and let the optimizer handle algebraic simplification,
scheduling, and code emission separately.

### Application in VectorFFT

VectorFFT is the direct architectural descendant of genfft:

- **OCaml-based code generator producing C codelets**: same approach.
- **Codelet naming conventions** (n1/n2/t1/t2): same convention. Our
  `radix64_t1_dit_log3_fwd_avx512_gen_inplace_su_spill` follows the
  FFTW naming scheme.
- **Pipeline stages**: split-radix → schedule → algebraic
  simplification → emit C. Same conceptual pipeline.
- **AST/DAG representation in OCaml**: same data structure approach.

### Where VectorFFT extends genfft

- **SIMD-native code emission**: genfft emits scalar C; VectorFFT emits
  AVX-512/AVX2 intrinsics with vector-width-aware scheduling.
- **Two-pass cluster-flushed structure**: genfft codelets are
  single-pass; VectorFFT's strided 2D structure has pass 1 producing
  cluster outputs to a `spill_re/spill_im` buffer that pass 2 consumes.
- **Higher radixes**: genfft tops out at R≈64 in practice; VectorFFT
  has working codelets to R=256.
- **The M-project itself**: genfft has no register allocator and relies
  entirely on gcc/icc's RA. M3a/M5 are the new layer that VectorFFT
  adds on top.

### Where Frigo's paper is silent

Frigo treats register allocation as the C compiler's job. At his
problem size (R≤64 scalar, peak live count under 16), gcc-3.x's
allocator was good enough. At our problem size (R≥64 vector, peak live
counts ~50), it isn't. Hence the M-project.

---

## 5. Related work we considered and didn't adopt

These appeared during research but weren't used directly.

### Chaitin et al. (1981) — Graph-coloring RA

> Chaitin, G.J. et al. (1981). **"Register Allocation via Coloring."**
> *Computer Languages*, 6(1), 47–57.

The foundational paper on register allocation as graph coloring.
Influential but solving a harder problem (general control flow,
arbitrary spilling cost) than ours. Linear scan is the strictly simpler
choice for straight-line code.

### Briggs, Cooper & Torczon (1994) — Improvements to graph-coloring RA

> Briggs, P., Cooper, K.D. and Torczon, L. (1994). **"Improvements to
> Graph Coloring Register Allocation."** *ACM Transactions on
> Programming Languages and Systems*, 16(3), 428–455.

Refined Chaitin with optimistic coloring and better spill cost
heuristics. Their spill cost formula (use frequency × loop depth /
live-range length) doesn't generalize cleanly to our case where all
uses are at depth 0 inside a single straight-line block. We don't use
it.

### Wimmer & Mössenböck (2005) — Linear scan with interval splitting

> Wimmer, C. and Mössenböck, H. (2005). **"Optimized Interval Splitting
> in a Linear Scan Register Allocator."** *Proceedings of the 1st
> ACM/USENIX International Conference on Virtual Execution Environments
> (VEE '05)*, 132–141.

Extends linear scan to split long intervals at use-set boundaries, so
a variable can occupy different registers (or be spilled) in different
sub-ranges of its lifetime. Conceptually similar to M5's "rename on
reload" approach: a reloaded value gets a fresh variable name
(`tN_r0`), much like a Wimmer split-interval gets a fresh interval ID.
The key difference is that our reload-rename is driven by an explicit
spill event rather than by interval-graph analysis. **M6 (reload
lifetime tracking)** would bring our system closer to Wimmer-style
split-interval allocation.

### Wimmer & Franz (2010) — Linear scan on SSA form

> Wimmer, C. and Franz, M. (2010). **"Linear Scan Register Allocation
> on SSA Form."** *Proceedings of the 8th IEEE/ACM International
> Symposium on Code Generation and Optimization (CGO '10)*, 170–179.

Combines linear scan (§2) with SSA decoupling (§3) — exactly what the
M-project does, although we arrived at the combination empirically.
This paper is the closest single-source reference to what M3a + M5
together implement.

### Bouchez, Darte, Rastello (2007) — Spilling NP-completeness

> Bouchez, F., Darte, A. and Rastello, F. (2007). **"On the Complexity
> of Spill Everywhere under SSA Form."** *Proceedings of the 2007 ACM
> SIGPLAN/SIGBED Conference on Languages, Compilers, and Tools for
> Embedded Systems (LCTES '07)*, 103–112.

Proves that even when coloring is polynomial (chordal SSA graphs),
finding the optimal spill set is NP-complete. Justifies using a
heuristic (Belady) rather than searching for optimality.

---

## 6. Tools we exploit but aren't papers

### GCC's `register __m512d t asm("zmm15")` extension

> GCC Manual, **"Variables in Specified Registers."**
> https://gcc.gnu.org/onlinedocs/gcc/Explicit-Register-Variables.html

The mechanism that makes the M-project work at all. Without per-variable
register pinning, generating C with explicit register assignments would
be impossible and we'd have to emit assembly directly.

The barrier idiom `asm volatile("" : "+v"(t))` after each pinned
declaration comes from collective gcc folklore — it's the recommended
way to prevent the optimizer from violating the constraint. Not a
paper, but worth knowing.

### Sebastian Hack's PhD thesis (2007)

> Hack, S. (2007). **"Register Allocation for Programs in SSA Form."**
> PhD dissertation, Universität Karlsruhe.
> https://publikationen.bibliothek.kit.edu/1000007166/6532

Book-length treatment of SSA-based RA from the same author as the CC'06
paper. The most thorough single reference for anyone implementing
SSA-aware register allocation from scratch. Section 3.1.5 in particular
covers "register targeting in Chaitin-style allocators" — the technique
of forcing certain variables into specific registers, which is exactly
what the M-project's `pinned_reg_decl` does.

---

## 7. The synthesis

The M-project is, in effect:

> **Linear-scan register allocation (Poletto-Sarkar 1999) on
> straight-line SSA-ish IR (structural argument from Hack-Goos 2006),
> with Belady's optimal eviction (Belady 1966) as the spill heuristic,
> emitting register-pinned C using gcc's specified-register extension,
> wrapped around a code generator architecturally derived from FFTW's
> genfft (Frigo 1999).**

Every piece is well-established work. The contribution of the
M-project (such as it is) is in the application and in the engineering:
making linear scan work cleanly on a generated SIMD codelet, making
Belady eviction interact correctly with cluster-flushed output stores,
discovering that gcc's `register asm()` mechanism is reliable enough at
this scale, and noticing that the spill-once-reload-each-use policy
tips into a regression past ~500 spill slots (and that M6 is the
classical-literature answer to that — Wimmer-style interval splitting).

For someone reading the code without context, the four papers in §§1–4
are sufficient background. §5 is for anyone implementing a
similar system or extending M6.
