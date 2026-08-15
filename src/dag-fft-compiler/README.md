# dag-fft-compiler — the OCaml codelet compiler

This tree is a **compiler**, not a library. It takes a description of one FFT
kernel — a radix, a direction, a memory layout, a twiddle policy — and emits a
single straight-line C function with the SIMD intrinsics already scheduled and
register-allocated. The 1,432 `.c` files under `codelets/` are its output, and
the C runtime in `src/core/` is the thing that links and dispatches them.

Everything downstream depends on one property: **the emitter is deterministic
and its output is byte-reproducible.** 1,403 of the 1,432 shipped files
regenerate byte-for-byte from their recorded recipes (97.97%; 99.93% counting
code bodies only), and the gate in `generator/gates/` exists to keep it that
way. That is what makes it safe to refactor a compiler whose output nobody
reads line by line.

```text
codelets/     the product — 1,432 emitted .c files (see codelets/README.md)
generator/    the compiler itself (dune project) — this document is mostly about this
jit/          runtime codegen: emit_*.py + prelude/runtime headers for JIT plans
tools/        research harnesses (schedulers, ablations, probes) — not in the build
archive/      old-lib/, the pre-restructure monolith. Reference only; deleted at v1.0
build.sh      convenience build
```

> 🔴 **Never run a bare `dune build` in `generator/`.** The `@default` alias
> promotes tracked headers into `generated/` **even when the build fails**.
> Always name a target: `dune build bin/gen_radix.exe`. The generator is also
> **WSL/opam only** (`~/.opam/5.2.0`, dune 3.23, `DUNE_CACHE=disabled`).

---

## Part I — Architecture

### Three libraries, and the law between them

`generator/lib/` is three dune libraries with enforced dependencies. This is not
organisational taste; it is a compile-time law.

| library | modules | may depend on | the rule it enforces |
|---|---:|---|---|
| `vfft_kernel` | 23 | unix | **cannot name a family.** No `Codelet`, no `Real`, no `C2c_il` |
| `vfft_cx` | 7 | kernel | the packed-complex (IL) sub-engine |
| `vfft_gen` | 8 | kernel, cx, unix | families, the descriptor, the driver |

All three are `(wrapped false)`, so module names stay global and references
never carry a library prefix; `(implicit_transitive_deps false)` in
`dune-project` makes an accidental edge a build error rather than a habit.

The point of the kernel boundary: **the engine is feature-blind.** It knows
about expressions, scheduling, registers, ABIs and C text. It does not know
that r2c exists. When a family needs to inject family-specific text into the
middle of the engine's output, it does so through a declared seam
(`family_hooks`, below) — never by the engine testing a family flag it defined
itself.

### The data that flows

Four representations, in order:

```text
Expr.expr        pure symbolic arithmetic — what the FFT algebra produces
   │             (Expr / Cnum: complex numbers as symbolic pairs)
   ▼  Ir.of_assignments + hash-consing
Ir.t             the shared DAG — CSE'd, hash-consed, the thing passes rewrite
   │             (Algsimp, Fma_passes, Simplify operate here)
   ▼  Schedule / Regalloc
scheduled IR     an ordered list with register/spill decisions attached
   │
   ▼  Emit_render + Emit_body + Isa
C text           intrinsics, one straight-line function
```

`Ir` owns the only surviving mutable state in the kernel — four memoisation
tables. Everything else that used to be a global was eliminated during the
restructure: 66 globals went to 4, and `emit_state.ml` (a buffer for *backward*
dependency edges between passes) was deleted outright. This matters because
`gen_set` emits the whole corpus **in one warm process** — any state that
survives between emissions is a correctness bug waiting for a specific ordering.

### The keystone: `Codelet.t`

[`lib/gen/codelet.ml`](generator/lib/gen/codelet.ml) is the word for the thing
being compiled:

```ocaml
type t =
  { radix  : int
  ; isa    : string option
  ; uarch  : string option
  ; kind   : kind        (* 17 constructors — one per family shape *)
  ; mods   : modifiers   (* the FIVE global modifiers: dir dif table t1s su *)
  ; emit_c : bool }
```

The shape was **measured, not designed**: every shipped codelet carries exactly
one kind selector; exactly five modifiers are global; everything else is
family-scoped and lives in the kind's payload record. So `Trig of trig8` is one
constructor covering nine transforms, and `C2c_oop of { load; store; tw; fuse;
… }` carries the oop family's nine knobs. The rule that keeps this from
sprawling: **a family earns a module only when it has its own emission
strategy; a variant is a record field.**

`of_argv` / `to_argv` round-trip the recorded command lines **verbatim, flag
order included** — checked against 1,410 recorded argv lines. That identity is
what lets provenance headers, the corpus, and the regeneration recipes all be
the same fact rather than three that can drift.

⚠ Flag order is discovered from the corpus, not chosen. `--isa` sits
mid-sequence and its position differs per family; `--log3` hugs its twiddle
token; `r2c-term-ls` records `--isa` *after* `--emit-c`.

### The signature side: `Layout` and `Abi`

A codelet's parameter list is derived, not printed by hand.

[`Layout`](generator/lib/kernel/layout.mli) models where complex data lives:

| plane | pointers |
|---|---|
| `Split` | `p_re`, `p_im` — two arrays |
| `Inter` | `p_z` — one array of (re,im) pairs |
| `Inter_sw` | one array of (im,re) pairs — the backward-swap enabler |
| `Real` | one bare real pointer — the r2c/c2r strided family |

The **anti-hybrid law** lives here and is expressed as a type: `param` is
private, and pointers are *total on one plane*. A codelet that reads split and
writes interleaved is therefore not a bug to be caught — it is unrepresentable.

[`Abi`](generator/lib/kernel/abi.mli) turns a kind into a signature:
`Abi.shape` is a 13-arm total variant, `Abi.signature` renders it. Before the
old hand-written 13-arm ladder was deleted, both were emitted in-process for
every one of the 1,432 codelets and asserted equal — 1,432 independent proofs.
That cross-check survives as a permanent debug env, `VFFT_ABI_XCHECK=1`.

### The pass cascade: `Pipeline`

[`Pipeline.prepare_codelet`](generator/lib/kernel/pipeline.mli) is the **sole**
cascade — a ~555-line inline copy in the driver was deleted once it was proven
byte-equivalent across a 7-cell environment matrix.

```ocaml
type recipe = { butterfly_share : bool; dup : dup_ctx option }
val prepare_codelet : recipe:recipe -> … -> prepared
```

`~recipe` is **required**, not optional: every route must *declare* which arms
it runs. The main driver enables both; the split and cascade families pass
`default_recipe`. Inside: `Ir.reset`, `of_assignments`, dedup, the aggressive
prime passes, then the FMA cascade with frozen-tag threading and the spill
marker remap chain.

One knob is worth knowing about because it is a trap: `VFFT_FORCE_REASSOC`
looks like a free win — it cuts 17–23% of vector ops on power-of-two Cooley-Tukey
shapes. It was raced and **refuted 10/10**: reassociation flattens butterfly FMA
chains and costs ILP. Static op-count has now failed to predict runtime four
times in a row here. It stays wired as a diagnostic only.

### The `cx` sub-engine (packed complex / IL)

The interleaved-complex family has enough of its own algebra to justify a
parallel stack in [`lib/cx/`](generator/lib/cx/):

| module | role |
|---|---|
| `cx_ir` | the packed-complex IR + `ctx`, the per-emission context |
| `cx_math` | math-layer DAG builders |
| `cx_sched` | SR (Starve-Retire) scheduler |
| `cx_cpl` | CPL (Critical-Path List) scheduler — the ILP-objective alternative |
| `cx_spill` | Belady spill planner for mono emission |
| `cx_render` | C-text rendering of scheduled cx nodes |
| `cx_pipeline` | the one optimizer entry point every cil emission runs through |

`Cx_ir.ctx` is a record threaded per emission (twiddle policy, store mode,
spill slots, and the `VFFT_CX_*` knobs captured at creation). It replaced nine
module-level refs that the driver set and never reset — harmless in one-shot
`gen_radix`, a leak the moment cil entered the warm `gen_set` process.

### The families

[`lib/gen/`](generator/lib/gen/) — each owns an emission strategy, and each
installs `Emit_body.family_hooks`:

| module | covers |
|---|---|
| `real.ml` | r2cf, r2cb, r2c_term*, hc2c*, hc_ranged, r2r, strided_r2c |
| `c2c_split.ml` | in-place, oop, twidsq, strided, k1_mono — and hosts `emit_engine` |
| `cascade_z.ml` | the N≥2048 boundary-split cascade |
| `c2c_il.ml` | pure IL, driving the `cx` stack |
| `dft_r2c.ml` | the real-input math layer |
| `corpus.ml` | the typed corpus (below) |
| `gen_main.ml` | the driver |

`family_hooks` is 8 positional `(Buffer.t -> unit) option` slots plus
`no_hooks`. The engine keeps its own dispatch tests and **`failwith`s loudly**
if a family-owned arm is reached with no hook installed — a missing hook is a
crash, never silently-wrong C.

---

## Part II — Life of a codelet

`gen_radix.exe 16 --cil-t2 --isa avx2 --uarch raptor_lake_avx2 --emit-c`:

```text
argv
 └─ Codelet.of_argv ────────────────► Codelet.t {radix; isa; uarch; kind; mods}
     │                                 raises on conflicting layout flags
     └─ gen_main projects it into Emit_render.Cfg   (config flows FORWARD only)
         │
         ├─ MATH   Dft.* / Dft_recurse / Dft_select   (c2c)
         │         Dft_r2c.*                          (real-input)
         │         Cx_math.*                          (packed complex)
         │                    ──────────────────────► (elem_ref * Ir.t) list
         │
         └─ Pipeline.prepare_codelet ~recipe ───────► simplified · scheduled ·
             │                                        register-allocated IR
             └─ ROUTE BY FAMILY        (gen_main.ml:1532)
                  Real.emit_codelet        real family
                  C2c_split.emit_engine    everything else
                  C2c_split.emit_codelet   oop proper
                  Cascade_z.emit_codelet   boundary-split cascade
                  C2c_il.emit / emit_k1    pure IL (via lib/cx)
                  C2c_split.emit_k1_mono   k1 mono
                    each installs family_hooks, then calls…
                 └─ Emit_body.emit_codelet ── THE ENGINE
                      Abi.signature ← Layout        the function signature
                      Emit_render.body_preamble     spill decls, hoisted consts
                      Simd.load/store_transpose_*   feature-blind lattices
                      the 8 hooks at their seams
                                          ──────────► C text on stdout
```

`gen_radix` emits one codelet to stdout. `gen_set` emits the whole tree from
`Corpus` in a single warm process — which is exactly why per-emission state has
to be threaded rather than global.

---

## Part III — How a codelet becomes reachable from C

Emitting a `.c` file is only half of shipping one. The other half is dispatch.

```text
Corpus.cells / Corpus.files          18 typed quadrants over Codelet.t
   │                                 two LAWS enforced lazily at first use:
   │                                   1. round-trip verbatim
   │                                   2. uniqueness (per-dir file, global argv)
   ├──► bin/gen_set.exe              emits the codelet tree
   └──► bin/emit_*_registry.ml       7 registry emitters + the executor table
            │                        (oop, rfft, trig, strided, c2r, il, c2c)
            ▼
        generated/dune (mode promote)
            │
            ▼
        tracked headers in generator/generated/    ← the C build's input
            │
            ▼
        src/core/ includes them (e.g. oop/il2p.h)
```

The two corpus laws turn "the codelet exists", "the coverage matrix lists it"
and "the recipe rebuilds it" into a single checked fact. They fire in `gen_set`
and the registry emitters — the writers of tracked files refuse a lawless
corpus — but never in `gen_radix`, so one-off emission pays nothing.

🔴 **Which quadrant you add to decides your blast radius.** The 2026-08-15
coverage raise added 75 cells with *zero* registry churn, because the four new
quadrants are named by **no registry emitter** — those kernels dispatch through
wisdom and `il2p.h` instead. Put the same cells in an existing quadrant and they
land in the oop/strided registries, changing ABI slots and dragging the C
dispatcher along. The IL family is the counter-example worth studying:
`bin/emit_il_registry.ml` derives 253 extern declarations *and* the radix
X-macro lists that `il2p.h`'s resolvers expand, so "the codelet exists" and "a
resolver can reach it" cannot drift apart.

> ⚠ **filename ≠ symbol.** `radixR_z_KIND_avx2.c` defines
> `radixR_z_KIND_**fwd**_avx2` — forward is *implicit* in the filename and
> *explicit* in the symbol. `_bwd` files keep `_bwd` in both. And the family a
> file belongs to is decided by its **provenance header**, not its name.

---

## Part IV — Adding a new codelet type

First decide which of two things you are doing.

### A. A new *variant* of an existing kind — the common case

A new store mode, a new twiddle placement, a new interior form. Three edits:

1. add a field to that kind's payload record in `codelet.ml`
2. handle the flag in `of_argv` **and** `to_argv` (order matters)
3. branch on it inside the owning family module

No new module, no ABI change, no registry change. Gate with
`bin_test/argv_roundtrip.exe` and the corpus gate.

### B. A genuinely new kind — the full ladder

Walk it in dependency order. Each row has a gate that catches you if you skip it.

| # | where | what you add | what catches a mistake |
|---|---|---|---|
| 1 | `kernel/dft.ml`, `dft_recurse.ml`, `dft_select.ml`, or `gen/dft_r2c.ml` | the expansion producing assignments | `bin/dbg_eval.exe` (per-pass numeric prober), `bin/dump_ir.exe` |
| 2 | `kernel/layout.ml` | plane + pointer set, if the data layout is new | `gates/layout_smoke.sh` |
| 3 | `kernel/abi.ml` | an `Abi.shape` arm + its params | `VFFT_ABI_XCHECK=1`, `layout_smoke.sh` |
| 4 | `gen/codelet.ml` (+`.mli`) | `kind` constructor, `of_argv` flags, selector-set match, `to_argv`, `validate` | `bin_test/argv_roundtrip.exe` — **verbatim** |
| 5 | `Emit_render.Cfg` + `gen_main` | a config field, projected from the descriptor | build error if the kernel names `Codelet` |
| 6 | a family in `gen/` | a branch, or a new module installing `family_hooks` + a route | the engine `failwith`s if an arm is unhooked |
| 7 | `gen/corpus.ml` | cells, in a chosen quadrant | the two corpus laws |
| 8 | `bin/emit_*_registry.ml` + `generated/dune` | dispatch header, if C must reach it | link failure; registry byte-diff |
| 9 | `gates/` | `manifest` → `record` → `verify`, **own commit** | the corpus gate |

Steps 2–3 deserve emphasis: **the corpus contains zero representatives of the
five IL-layout arms**, so byte-identity proves nothing about them.
`layout_smoke.sh` is the only net over that space, and it also tests the
*negative* space — illegal flag combinations must be refused loudly at
emission.

---

## Part V — The gates

`generator/gates/` (see its own README for detail):

| gate | what it proves | cost |
|---|---|---|
| `full_corpus_gate.sh` | every one of 1,432 files still regenerates to its **recorded verdict class** | ~60 s |
| `layout_smoke.sh` | 13 layout shapes emit + compile under `gcc -Werror` with every declared pointer referenced; 4 illegal combinations refused loudly | seconds |
| `cil_matrix.sh` | the 183-case cx emission matrix over off-default `VFFT_CX_*` knobs | ~10 s |
| `bin_test/cx_pipeline_test` | the cx stack's unit gate (built **and run** by the corpus gate) | instant |
| `bin_test/argv_roundtrip` | descriptor ↔ argv fidelity | instant |

The corpus gate does **not** demand every file be identical. 29 files
legitimately do not reproduce (dead-era orphans, sunset copies, drifted
bodies); demanding perfection would mean a permanently red gate, which trains
everyone to ignore it. Instead each file is pinned to its recorded verdict
class and the gate fails if any file moves **in either direction** — a
non-reproducer that suddenly starts matching is as much a signal as the reverse.

**Run both `full_corpus_gate.sh` and `layout_smoke.sh`.** A compensating gate
covers, by construction, exactly what the main gate cannot see — so the main
gate can never tell you the other one went red. That is not hypothetical: the
smoke sat unrun from M3 to 2026-08-15 and was red for most of it, while the
corpus gate was green after every single step.

---

## Part VI — Working rules

- 🔴 **No bare `dune build`** — `@default` promotes tracked headers even on
  failure. Name your target.
- 🔴 **WSL/opam only** for the generator. A Windows-only session cannot land a
  generator change.
- 🔴 **Comparisons use LF-canonical bytes** (`git cat-file`), never worktree
  bytes — `core.autocrlf` leaves ~99 corpus files CRLF, and comparing raw bytes
  measures your checkout, not the emitter. Never let a `sed` rename script near
  `codelets/`.
- 🔴 **C gates can bank.** Several gate binaries re-race and *rewrite* the
  wisdom directory they are pointed at. Always pass a **scratch copy**, and run
  `git status generated/` afterwards.
- **A gate failure is a question, not a verdict.** Read the transition matrix
  before concluding: one announced regen moved 329 files, and every one was a
  comment header.
- **Static op-count does not predict time** — 4-for-4 against, here. Race it.
- Emission is deterministic *by design*: `Date`, randomness and ambient state
  are all absent from the emitters, which is what makes byte-identity a usable
  acceptance criterion at all.

---

## Historical note

An earlier version of this README documented an **N=1024, AVX-512,
out-of-place natural-order reference engine** and its benchmark suite
(`engine/`, `benchmarks/`, `docs/OOP_DESIGN.md`). Those directories no longer
exist in this tree; the engine they described was superseded by the planner and
executor now in `src/core/`. The measured conclusions are worth preserving:

- FFTW at N=1024 is **neither Stockham nor 6-step Bailey** — it is recursive
  cache-oblivious Cooley-Tukey, confirmed from its own plan dump; the tree *is*
  the recursion and the decomposition is planner-selected per size.
- Blocking the batch into one-vector-wide blocks was worth ~3×, the single
  largest lever.
- Six candidate explanations for the residual gap vs FFTW were each tested and
  **refuted**: leaf spills, transpose scatter, call fragmentation, leaf codelet
  quality (ours were *faster*), twiddle overhead, and block width. The residual
  was compositional, not a pointable defect — and only in the L3-resident regime.
- A padded-transpose fix for L1 set aliasing **lost**: the extra serialized pass
  cost more than the aliasing it removed. The isolated 1.8× was a reused-buffer
  micro-benchmark artifact.

Those numbers were taken on a virtualized AVX-512 Xeon. The deployment target
is an AVX2-only 14900KF, where none of those codelets run.
