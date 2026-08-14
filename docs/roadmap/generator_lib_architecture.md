# `generator/lib` — architecture of record

*docs/roadmap/generator_lib_architecture.md — 2026-08-14. Written against `main` @ `4b19f9ca`,
tree clean. Target: `src/dag-fft-compiler/generator/lib` — the OCaml DAG-FFT codelet compiler.
Working artifacts (gitignored): `docs/research/generator_arch/` — eight recon reports, two
proposals, three judgments, three adversarial verifications, and the corpus gate.*

> 🟢 **READ §21 FIRST if you are auditing this document.** An independent completeness pass on 2026-08-14
> re-executed the gate (`GATE PASS`, 1403/1432, 61.99 s), re-measured every headline number from the tree,
> and recorded twelve gaps (G1–G12) plus its own verdict on the owner's two halves. Corrections it applied
> are marked ⚠ at their site. **G1 is that this file is untracked — `git add` it.**
>
> 🔴 **NOTHING IN §2–§7 EXISTS IN THE TREE.** No part of this architecture has been built. The type
> sketches were transcribed into scratch dune projects and compiled during adversarial review; the
> ones that **failed to compile are marked in place and fixed here** (§3.1, §4.2, §5.2). The library
> has never been built in this shape, and every line-count and edit-count figure for the *after*
> state is an ESTIMATE unless it says otherwise. Numbers for the *today* state are measured, with
> the command or the `file:line` recorded.

---

## 0 · The verdict in one page

`generator/lib` is a working, high-quality FFT codelet compiler whose **content** is good and whose
**structure has no vocabulary**. Two sentences carry the whole diagnosis:

> **① The compiler has no word for the thing it compiles.** No value, type, or module answers *"what
> codelet am I emitting?"*. That fact — kind × direction × input layout × output layout × placement
> × iteration shape × twiddle source — is reconstructed ad hoc at ~40 sites from 66 untyped mutable
> cells, in five mutually inconsistent subsets, disambiguated by the **source order of `if/else if`
> chains**.
>
> **② The mutable cells are not configuration. They are backward dependency edges in disguise.** A
> family module (`Codelet_oop`, topological level 10) must tell a renderer (`Emit_render`, level 7)
> how to behave. That dependency is illegal — the renderer is *upstream* — so the value travels
> through a mutable cell in a module low enough for both to see. `emit_state.ml:43` says so in prose;
> `cx_ir.ml:5-8` says it again, in a module carved out fresh three weeks ago.

Everything the owner named follows. `emit_c.ml` is 4,167 lines because a function that must re-derive
its own identity 13 times cannot be split. There are 66 process-global mutable cells because the
consumer is a 4,066-line function too big to add a parameter to — a rationale the source states
**verbatim, three times, in three different files**. There are zero `.mli` because three `include`
chains dissolved the module boundaries an interface would have described. And the repo's standing law
— *never build hybrid IL-boundary/split-interior codelets* — is enforced by runtime `failwith`s in a
CLI parser, because **layout is not in any type**.

**Three facts that were not known when this campaign started and that change how it must be run:**

1. 🥇 **The ≥95 % bit-verbatim acceptance bar is already met, today, unmodified: 1403 / 1432 =
   97.97 %.** The job is **"do not regress"**, not "close a gap". A gate covering all 1,432 files
   exists and runs in ~59 s (`docs/research/generator_arch/gates/full_corpus_gate.sh`, executed and
   PASS during review). `EMIT_FAILED = 0` — every one of the 1,432 has a runnable recipe.
2. 🥇 **The family taxonomy is no longer unknown — it was measured out of the shipped corpus.** 83
   distinct argv shapes over 1,199 provenance-bearing codelets; every one carries exactly one of **40
   mutually-exclusive kind selectors — 1,199/1,199, zero exceptions** — *alongside a separate
   modifier matrix*. The compiler encodes that as ~40 booleans plus 22 derived emitter booleans.
3. 🔴 **"More modules" has already been tried here, and it regrew the disease.** The `cx_*`
   decomposition (`02f7c633`, 2026-08-09) produced seven clean modules under a byte-identity gate —
   and **re-invented the back-edge global buffer inside `cx_ir.ml` within weeks**, admitted in its own
   header. That is the owner's pre-rejected failure mode, already observed in-tree, in the
   best-executed refactor this codebase has done. It cost **+196 lines (+28 %)** for identical
   behaviour.

**The answer this document gives to the owner's central tension** — modular enough that each feature
has a home, without module explosion or a sprawl of shared values — is: **five feature modules over a
feature-blind kernel, and everything else is DATA.** A family earns a module when it needs its own
*emission strategy*; the other ~35 kinds are variant arms and record literals. That is not invented:
`codelet_zsplit.ml:63-167` already produces **21–22 distinct codelet kinds by record update from one
base value**, owns **zero** module refs, and `Fun.protect`s the one global it borrows.

---

# PART I — THE DIAGNOSIS

## 1 · Quantified baseline

All re-executed 2026-08-14 from `src/dag-fft-compiler/generator/`.

### 1.1 Size and shape

| metric | value | how measured |
|---|---:|---|
| `.ml` files in `lib/` | **35** | `ls lib/*.ml \| wc -l` |
| total lines in `lib/` | **30,776** | `cat lib/*.ml \| wc -l` |
| **`.mli` files, whole generator tree** | **0** | `find . -name '*.mli' -not -path './_build/*'` |
| modules in the hand-maintained `(modules …)` stanza | **34** | `lib/dune:26` |
| `.ml` on disk but **not** in the stanza | **1** — `number.ml` (133 lines) ⇒ compiled by nothing | set difference |
| stanza entries that are placeholders | **1** — `simd_ir` (1 comment line, 63 B) | `wc -l` |
| top-level definitions in `emit_c.ml` | **4** (`:60`, `:4126`, `:4145`, `:4167`) | `grep '^let \|^and \|^type \|^module '` |
| top-level definitions in `gen_main.ml` | **1** | same |
| dead `.ml` outside `lib/` that answers greps | **3 files / 5,305 lines** (`old-lib/`) | `wc -l old-lib/*.ml` |

**Largest files:** `emit_c` 4,167 · `codelet_oop` 2,540 · `fma_passes` 2,075 · `gen_main` 1,996 ·
`schedule` 1,781 · `codelet_zsplit` 1,741 · `dft_r2c` 1,625 · `simplify` 1,522 · `emit_render` 1,447 ·
`regalloc` 1,428 · `codelet_cil` 1,342.

**🥇 The numbers that matter more than file size — size is a FUNCTION-level problem:**

| function | span | lines | inputs |
|---|---|---:|---|
| `Emit_c.emit_codelet` | `emit_c.ml:60` → `:4125` | **4,066** | 12 optional + 2 required params **+ ~26 globals read at 96 sites** |
| `Gen_main.run` | `gen_main.ml:52` → `:1996` | **1,945** | argv + **101 local `ref`s** |
| `Codelet_zsplit.emit_codelet` | `:410` → `:1741` | **1,332** | incl. a 607-line *anonymous* nested block |
| `Codelet_cil.emit` | `:162` → `:1085` | **924** | 11 labelled args + cx globals |
| **total** | | **8,267** | **26.9 % of the library, in four functions** |

**No `.mli`, module map, or dune restructuring touches any of these four.**

### 1.2 Configuration surface — four planes, no owner

| plane | size | how measured |
|---|---:|---|
| CLI flag strings | **133 distinct** | `grep -ohE '"--[a-z0-9-]+"' gen_main.ml \| sort -u` |
| driver-local `ref`s inside `Gen_main.run` | **101** | grep |
| **process-global mutable cells** | **66** = 59 `ref` + 7 containers | multi-line-tolerant perl scan |
| distinct environment variables read | **55** at **91** sites in **20 of 34** modules | `grep -ohE 'getenv(_opt)? *"[A-Z_0-9]+"'` |
| `emit_codelet` labelled parameters | 12 optional + 2 required | `emit_c.ml:60-76` |
| `mutable` record fields in the whole library | **0** | comments only |

Global-cell census, per module: `emit_state` **34** (32 `ref` + `il_seen` Hashtbl `:107` + `il_pending`
Buffer `:108`) · `codelet_oop` **10** (`:230,232,238,249,264,265,274,275,282,1139`) · `cx_ir` 7 ·
`emit_render` 4 · `cx_math` 3 (`:47,154,205`, multi-line defs) · `ir` 4 · `schedule` 2 · `dft_select` 1
· `cx_render` 1.

⚠ **Any `:=` grep must be multi-line tolerant.** ocamlformat splits long assignments; a same-line grep
missed `Emit_c.r2r_signature := …` at `gen_main.ml:780` and wrongly reported the flag dead.

### 1.3 Coupling

| metric | value |
|---|---|
| include chains | **3** (+1 legitimate functor include at `schedule.ml:1179`) |
| chain declarations | `simplify.ml:32 include Ir` · `fma_passes.ml:166 include Simplify` · `algsimp.ml:26 include Fma_passes` ; `emit_render.ml:33 include Emit_state` · `emit_c.ml:58 include Emit_render` ; `dft_recurse.ml:27 include Dft_select` · `dft.ml:34 include Dft_recurse` |
| ⚠ first link of chain 1 is `open`, not `include` | `ir.ml:47 open Expr` — **`Expr` is NOT re-exported** |
| `Algsimp.` refs resolving to an `Algsimp` definition | **16 of 359 (4 %)** — 298 belong to `Ir` |
| `Emit_c.` refs resolving to an `emit_c.ml` definition | **2 of 77 (3 %)** — 48 `Emit_state`, 27 `Emit_render` |
| `Dft.` refs resolving to a `dft.ml` definition | **90 of 139 (65 %)** — **not the same disease** |
| **`Emit_state.` / `Emit_render.` references in code** | **0** — all 7 occurrences are in comments |
| `render_node_def` call sites in code | **15** (emit_c 8, codelet_oop 5, codelet_zsplit 2) |
| `failwith` in `lib/` | **123** |
| `gen_main` participation in 91 feature commits | **47 (52 %)**, solo-rate **0.06**; P(`gen_main` \| `emit_state` changed) = **1.00** |

### 1.4 Duplication

| item | size |
|---|---:|
| AVX2 ↔ AVX-512 lattice clones inside `emit_c.ml` | **1,533 lines spanned, 88 % normalized-identical** |
| signature-arm boilerplate in `emit_c.ml` | 12 copies of the 9-line spill-decl block; 11 of `render_hoisted_consts` |
| **total mechanical duplication inside `emit_c.ml`** | **~800–1,050 lines (19–25 %)** |
| hard-coded SIMD intrinsics vs `Isa.` calls in `emit_c.ml` | **181 : 24 ≈ 7.5 : 1** ⚠ corrected from 189 — `grep -oE '_mm[0-9]*_[a-z0-9_]+' emit_c.ml \| wc -l` = **181**; `grep -oE 'Isa\.[A-Za-z_]+'` = **24**. (A looser `_mm[a-z0-9_]+` gives 194; the doc previously quoted an intermediate figure with no stated regex.) |
| interleave-at-the-boundary implementations | **5** (1 legitimate + 4 hand-written copies) |
| the frozen 11-arg z ABI | **printed twice, byte-identically, on two different compiler stacks** |
| the pass cascade | **three live copies, already diverged** |
| symbol-name construction schemes | **4** |

### 1.5 🥇 The safety net — it bounds every claim in this document

⚠ **Superseded numbers warning:** earlier recons reported reproducibility at 75–82 %. Those counted
*`Coverage`-driven regeneration only*. **recon 08 is authoritative** — it ran all five recipe arms and
diffed bytes.

| fact | value |
|---|---|
| committed emitted `.c` under `codelets/` | **1,432** across 16 directories |
| **byte-identical with today's unmodified emitter** | 🥇 **1403 / 1432 = 97.97 %** |
| emitted-**body** identical (hand-written prologues excluded) | **1415 / 1432 = 98.81 %** |
| files that failed to emit at all | **0** |
| **live emitter regressions** | **0** |
| directories at 100 % | **12 of 16**; a 13th misses by one floating-point constant |
| the `gen_set` arm alone | **1074 / 1074 = 100 %** |

**The 29 non-identical files, fully enumerated — there is no unexamined remainder:**

| class | n | what |
|---|---:|---|
| stale shipped file | **1** | `rfft/avx2/radix256_r2c_term_ls_r8_avx2.c:36` — disk `0.70710678118655002`, emitter `0.70710678118654757`. √½ = `0.70710678118654752…` ⇒ **the shipped file is ~11 ULP LESS accurate**; its AVX-512 twin is already correct. **A live accuracy defect the audit found.** |
| stale **and** orphaned | **16** | all of `rfft/avx512_regen/` — predate the odd-count-tail feature, referenced by **nothing** repo-wide, in no CMake or `build.py` list, one commit ever (`4f16a2e3`). **55 % of the entire gap.** Textbook POOL SUNSET. |
| hand-augmented comment prologue, **body byte-identical** | **12** | 6 `tangent/` (documented at `emit_ship.sh:6-9`: *"re-add that header by hand if you regenerate"*) + 6 `pure_il` promotion/race notes. |
| **live regressions** | **0** | — |

**🔴 Two measurement traps. A naive comparison reports ~89 %, below the bar, for reasons that have
nothing to do with the generator:**

1. **CRLF is a checkout artifact.** `core.autocrlf=true` + `.gitattributes:2 * text=auto` ⇒ 99 worktree
   files are CRLF; **all 1,432 committed blobs are LF**. ⚠ **`git archive` applies the eol filter and
   extracts 100 % CRLF — unusable**; use `git cat-file`. Ignoring this = 99 spurious failures.
2. **`argv[0]` leaks into shipped bytes.** `emit_render.ml:1395-1414` stamps the whole argv including
   argv[0]. **Five spellings are baked into the corpus; 123 files hard-code a developer's path, 38 an
   absolute one carrying the username.** Replay needs `exec -a`. Ignoring this = 123 more failures.

**The counter-fact that makes the gate viable at all:** emission is otherwise **fully deterministic** —
no date, no version stamp, no hash-order or build-path dependence; four full runs byte-identical.
🔴 **Protect this property.**

---

## 2 · Root causes vs symptoms

A 4,167-line file is a **symptom**. Here is what produced it and keeps producing it.

**RC1 — There is no codelet descriptor.** A codelet's identity exists only as a scattered tuple of
booleans; every consumer re-derives an approximation locally. *Structural property that keeps it true:*
adding a feature is cheapest as one more boolean, and **every existing site keeps compiling when you
do**. Nothing forces a new family to be handled everywhere, because there is no exhaustive match to
break. Contrast `Codelet_oop.edge_pattern`, where dune's dev profile makes a missed arm a **fatal**
non-exhaustive-match error (noted at `codelet_cil.ml:34`) — the enforcement this codebase already
relies on elsewhere and threw away in `emit_c`. And it is *provably* a variant: 1,199/1,199 shipped
codelets carry exactly one of 40 mutually-exclusive kind selectors.

**RC2 — The drivers are orchestration functions, not pipelines.** Four functions hold 8,267 lines.
Nothing inside them has a name, so nothing inside can be tested, reused, given an interface, or
**passed a parameter**. The source states the consequence verbatim, three times:

> `emit_state.ml:9-14` — *"Why refs and not parameters: the renderers have many call sites scattered
> through emit_codelet's scheduler and spill variants; threading ~20 mode parameters through each would
> dwarf the feature code."*
> `emit_state.ml:184` — *"Why a top-level ref instead of a parameter: render_node_def has ~10 call
> sites…"*
> `dft_select.ml:100-103` — *"pick_algorithm has 7 call sites including recursive descent… threading
> would require visible churn across multiple modules."*

**This is a feedback loop: size makes globals cheap; globals make the function un-splittable.** *(And
the stated justification is measurably overstated: `render_node_def` has **15** call sites in code, not
"many". A `~ctx` argument is a 15-edit change.)*

**RC3 — The globals are backward dependency edges, and `include` is the mechanism that hides them.**
🥇 The sharpest finding of the recon, and it merges the two suspected diseases into one. Five measured
back-edges:

| writer (level) | global | reader (level) |
|---|---|---|
| `Codelet_oop` (L10) | `Emit_state.current_tw_perpos`, `current_tw_linear` | `Emit_render` (L7) |
| `Codelet_zsplit` (L10) | `Emit_state.current_tw_zsplit` | `Emit_render` (L7) |
| `Codelet_cil` (L8) | `Cx_ir.tw_log3` | `Cx_render` (L1) |
| `Emit_c` (L8) | `Schedule.order_source`, `injection_log` | `Schedule` (L5) |
| `Gen_main` (L11) | `Dft_select.target_vec_regs` | `Dft_select` (L0) |

Both modules **admit it in prose**. `emit_state.ml:43`: *"Set by emit_codelet from
`Codelet_oop.current_oop_tw_linear`"* — but `Emit_c` **cannot** reference `Codelet_oop`, because
`Codelet_oop` already depends on `Emit_c`. `cx_ir.ml:5-8`: the render-state refs *"are render/store form
state, not IR state, but they live beside the IR so every `cx_*` module sees one copy."*

**`emit_state.ml` is therefore not a configuration module. It is a back-edge buffer, and its own MODULE
CARD proves it** (`:22`: *"PUBLIC SURFACE (measured): zero direct `Emit_state.X` references"*). **Two
modules exist on disk with zero addressable surface.** Two consequences bind everything below: (a)
cutting the include chains **without** giving families a forward-flowing config value only renames the
problem; (b) **the disease reproduces under decomposition alone** — `cx_*`, carved fresh with no
include chains, regrew the buffer in weeks.

**RC4 — There is no abstraction for SIMD *lattice* operations.** `isa.ml` (386 lines) is a good
abstraction over *value* operations with all width branches internal. It covers roughly 15 % of what the
emitters need. Every *lattice* operation — transpose, deinterleave, interleave, mirror, mask-pack — is
hand-written per width. Ratio in `emit_c.ml`: **181 raw `_mm*` literals : 24 `Isa.` calls** (⚠ 181, not
189 — see §1.4 for the regex).

### Root cause → problem

| | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 | P16 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **RC1** no descriptor | ● | ◐ | ◐ | | | ● | ● | | ◐ | ● | | | | ● | | |
| **RC2** monolith drivers | ◐ | ● | ● | ◐ | | | | | ◐ | ● | | | | | | |
| **RC3** globals-as-back-edges | | ● | ◐ | | ● | ◐ | ● | | | ◐ | | ◐ | ◐ | | | |
| **RC4** no lattice abstraction | | | ◐ | | | | | ● | ● | | | | | | | |
| **independent** | | | | ● | | | | | | | ● | ● | ● | | ● | ● |

● primary · ◐ contributing

**Problems that will NOT be fixed by RC1–RC4 and need their own remedy:** P4 (triplicated cascade),
P11 (no owning corpus description), P12 (backward layering into the math layer), P13 (the two-stack
fork), P15/P16 (build hazards, dead weight).

### 🔴 The trap to avoid

**"More modules" fixes RC2 and RC3's symptoms, does nothing for RC1 — and RC3 regrows.** The repo has
run this experiment. Any proposal that answers the owner with modules alone will reproduce it at larger
scale.

---

## 3 · The ranked problem list

Ranked by **daily cost = breadth of forced edits × frequency × silence of failure**.
🔴 critical · 🟠 high · 🟡 medium · ⚪ low.

**P1 — No codelet family taxonomy; the ABI is re-derived by ad-hoc disjunction 🔴.** Five boolean
disjunctions exist over the same 8-flag space **and they disagree**:

```ocaml
(* emit_render.ml:316 *) let stride = if in_place then "ios" else if twidsq then "is"
    else if !r2cf_signature || !r2cb_signature || !hc_strided || !n1_oop_strided then "is" else "K" in
(* emit_render.ml:325 *) let loop_var = if twidsq || !r2cf_signature || !r2cb_signature
    || !hc_strided || !n1_oop_strided || !r2c_term_signature || !r2c_term_laststage then "v" else "k" in
(* emit_c.ml:415 *)     let real_fft_sig = !r2cf_signature || !r2cb_signature || !hc_strided in
```

`real_fft_sig` omits `n1_oop_strided`; `stride` omits the `r2c_term_*` pair; `loop_var` includes both.
`emit_c.ml:1883-1908` computes the same facts a **fourth and fifth** time from three different subsets
in three different arm orders. The dispatch is one 13-arm ordered `if/else if` chain spanning
`emit_c.ml:522 → 1864`: `:522 strided` → `:1446 in_place` → `:1513 twidsq` → `:1544 r2cb` → `:1572 r2cf`
→ `:1601 r2c_term_laststage` → `:1639 r2c_term_signature` → `:1672 hc2c_natural` → `:1708
hc2c_natural_bwd` → `:1747 hc_strided` → `:1782 n1_oop_strided` → `:1810 r2r` → `:1841 else`.
**The arm order IS the priority specification and it is written nowhere else** — `hc2c_nat` sets *both*
`hc2c_natural` (`gen_main:772`) and `hc_strided` (`:759`), and works only because `:1672` precedes
`:1747`. Four exact invariants fall out of the corpus that appear nowhere in code: in-place `--su` ⊎
`--twiddled` (72 + 576 = 648, exact); `hc2c-nat`: `--bwd` ⟺ `--dif`; `--ranged` only with hc2hc/hc2c-nat;
`--t1s` = ALL for real-cascade kinds. And the normaliser already exists, written imperatively:

```ocaml
(* gen_main.ml:759-779 *)
Emit_c.hc_strided       := !hc2hc || !hc2c || !hc2c_nat;       (* derived predicate *)
Emit_c.hc2c_natural     := !hc2c_nat && not !bwd;              (* kind x direction, arm 1 *)
Emit_c.hc2c_natural_bwd := !hc2c_nat && !bwd;                  (* kind x direction, arm 2 *)
if !hc2c_nat then (Emit_c.hc2c_nat_r := n;
                   Emit_c.hc2c_nat_sstar := if n mod 2 = 0 then (n/2)-1 else (n-1)/2);
```

**This is a smart constructor with validation and derivation, written as statements against globals.**
`:772-773` is one concept crossed with direction encoded as two booleans whose *both-true* state and
whose *neither-true-while-`hc2c_nat`-is-set* state are both representable.

**P2 — Configuration has no owner: four untyped planes, 66 cells, five backward edges 🔴.** 133 CLI
flags → 101 driver-local refs → a lossy projection into 66 process globals at `gen_main.ml:753-790` and
`:1950-1962` → plus 55 env vars read at the point of use, bypassing everything. **The only typed channel
is `emit_codelet`'s 12 labelled args, and it is the smallest.** The owner's phrase, settled: *shared
config is not missing — it exists in the wrong form.* `codelet_oop.ml:152-164` defines a proper `config`
record with three purpose-built variants and a `validate` at `:175`; **sixty-six lines below it**,
`:230-282` declares **ten more knobs for the same family** as bare globals, each commented "Set from
gen_main --X", set at `gen_main.ml:1950-1959` **one statement before** `cfg` is handed to `emit_codelet`
at `:1960`. **Seven config fields travel as a value; ten travel as globals, for one family, five lines
apart.** Validation splits along exactly that seam. `provenance_env_overrides`
(`emit_render.ml:1364-1394`) watches **11 of 55** env keys while `:1358` claims *"the header can never
drift from behavior"* — **false for 47**.

**P3 — Four orchestration functions hold 27 % of the library 🔴.** The file defends this
(`emit_c.ml:29-32`): *"emit_codelet is deliberately one large function: it is the single place where
scheduling, spill structure, regalloc and rendering decisions meet, and every codelet family shares that
one meeting point rather than duplicating it."* **The "rather than duplicating it" half is falsified
inside the same file** — the 9-line spill-decl block appears 12 times
(`:1492,1537,1564,1593,1631,1664,1699,1738,1773,1802,1821,1855`) and the AVX2/AVX-512 lattices are 88 %
clones across 1,533 lines. 🔴 **Attacking P3 first, without a descriptor, just distributes the same flag
reads across more files.**

**P4 — The pass cascade exists three times and the copies have already diverged 🔴.**
`Pipeline.prepare_codelet` (`pipeline.ml:145-309`, 165 lines) serves 179 files; a hand-inlined copy
(`gen_main.ml:921-1430`, ~510 lines) serves 1,020; `cx_pipeline.ml` (14 lines) serves 233 *with
legitimately different algebra*; plus a partial-prefix ghost at `bin/dbg_eval.ml:121-134`.
`pipeline.ml:84-87` predicted the failure in its own MODULE CARD (*"a pass added or reordered in one and
not the other **silently diverges the oop family from the in-place families**"*). **The predicted
divergence has occurred:** `Algsimp.butterfly_share_mul` (`gen_main.ml:1202`) and `duplicate_uncse`
(`:1245`) are **absent from `pipeline.ml`**; `VFFT_FORCE_REASSOC` (`:916`) and `policy_n` (`:914`, the
dct1/dst1 correctness fix) are honoured **only** in `gen_main` and are **silent no-ops on the oop and
zsplit paths today**. They also differ in *form*, so no textual diff can police them; and
`pipeline.ml:277` justifies a correctness exclusion by citing *"gen_radix.ml line 588"* — `bin/gen_radix.ml`
is now **4 lines long**. Dormant only because both passes are env-gated off. **The highest correctness
risk in the tree, and invisible to a gate that runs at default env.**

**P5 — `include`-chaining has deleted three module interfaces; zero `.mli` is the symptom 🟠.** The
motive is documented, not inferred (`emit_c.ml:51-56`): *"`include` re-exports both layers so every
external `Emit_c.X` reference compiles unchanged. The `open` of `Algsimp` precedes the include so names
defined in the render chain (e.g. its `topo_sort_reachable`) keep shadowing `Algsimp`'s."* **The order of
two source lines is the only thing keeping 4,066 lines of emission on the intended
`topo_sort_reachable`.** Three daily costs: (1) no `.mli` is writable as a contract — a truthful
`emit_c.mli` today is ~76 lines of *someone else's* surface including ~36 `val current_* : bool ref`;
after the cut it is 2 `val`s; (2) **grep is unreliable** — writers say `Emit_c.X :=`, readers say bare
`!X`, and `Emit_state` never appears outside comments, so **no single grep answers "who touches this
flag?"**; (3) two modules exist with zero addressable surface. **The blocking myth, killed:**
`algsimp.ml:6-8` justifies the chain by claiming it keeps hash-cons state *"physically single"* — that is
not what `include` does; it **aliases** `Ir.hcons_table`. The state is already shared. **Cost to cut: 467
mechanical, 100 %-compiler-checked renames across ~20 files.**

**P6 — Layout has no type, so the anti-hybrid law cannot be a type error 🟠.** `Expr.elem_ref` is
layout-agnostic; layout is resolved at *render* time by reading globals (`emit_render.ml:388`);
`Codelet_oop.config.buffer` is only `InPlace | OutOfPlace` while IL-ness lives in **four separate
globals** (`:264-275`) consulted via `il_in_active ()` / `il_out_active ()`; and **a signature is a run of
`Buffer.add_string` literals** — 131 `__restrict__` lines emitted from four modules (`emit_c` 94,
`codelet_oop` 22, `codelet_zsplit` 9, `codelet_cil` 6). Here is the actual printer,
`codelet_oop.ml:296-318`:

```ocaml
| OutOfPlace ->
  if il_in_active ()
  then (Buffer.add_string buf "    const double * __restrict__ in_z, …";
        Buffer.add_string buf "    const double * __restrict__ in_unused,\n")
  else (Buffer.add_string buf "    const double * __restrict__ in_re,\n";
        Buffer.add_string buf "    const double * __restrict__ in_im,\n");
```

**Two independent `if`s over two independent global pairs. The law is respected by convention, in a
string printer.** Six representable illegal states are enumerated; `--ip-il-in --ip-il-out` together is
accepted with **no mutual-exclusion check anywhere** and resolved by `if/else if` order in **two places
that must agree by hand**; `strided_il_in := true` with `strided_r2c := true` emits `const double *
__restrict__ rio_re = rio;` referencing an **undeclared `rio`** — **OCaml compiles happily; the failure
lands in `gcc`**. **Nuance the design must preserve:** `il_in` *without* `il_out` **is legal and
shipped** (the cascade boundary-conversion codelet). The encoding is 4 booleans for what is one 3-valued
choice per edge. **The contrast is in-tree:** `codelet_zsplit`'s `zs_edge` (`:63-77`) is the only place in
the library where a memory boundary is **data carrying its C stride name** rather than a boolean.

**P7 — An unenforced temporal protocol across 1,074 codelets in one warm process 🟠.**
`bin/gen_set.ml:26-38` calls `Gen_main.run` in-process in a loop — **all 66 globals persist across
codelet boundaries** — while `Ir.reset` (`ir.ml:255-259`) sets `next_tag := 0` per codelet, so **any
tag-keyed table that survives a codelet boundary aliases different nodes**. Two lifetimes, one protocol:
`gen_set` runs everything in one process (resets mandatory); `emit_ship.sh`/`emit_r32.sh` fork per file
(resets irrelevant). **The flags cannot tell which world they are in — every defect lives on that seam.**
D-1 is **LIVE under `VFFT_DUP=1`**: `dup_barrier_tags` has a conditional setter (`gen_main.ml:1252`) and
**no reset path anywhere**. D-2 is one line from live: `current_tw_perpos`/`current_tw_linear` are set at
the top of every OOP emission (`codelet_oop.ml:1965-1966`) and cleared by nothing, leaking into six later
quadrants — harmless **only** because the readers sit behind `if strided` and the strided quadrant is
n1-only today. **Add one twiddled strided codelet to `Coverage` and the tree silently regenerates with
broadcast twiddles instead of vector loads.** Three reset disciplines coexist. The smoking gun,
`codelet_oop.ml:2248`, verified verbatim: `(* neutralize every mode ref that could leak from a prior
emission *)` — a hand-maintained list covering **9 of 66 cells**, inside `emit_k1_mono` only.

**P8 — `Isa` abstracts values but not lattices; adding an ISA is a shotgun edit 🟠.** Nine
`isa.vec_width = N` branch sites in `emit_c.ml` each open a hand-written region, including a 307-line
AVX2 transpose preamble (`:662`), a 477-line AVX-512 preamble (`:969`), and 313/496-line inverse
postambles (`:3190`, `:3509`). Adding an ISA is **~1,700 new lines inside `emit_codelet`** and a traced
blast radius of **21 files**. **The library already contains the cure and stopped applying it:**
`codelet_zsplit` 87 `Isa.` : 0 literals, `codelet_cil` 57 : 1, `cx_render` 71 : 2 — they **extended
`isa.ml`** instead of inlining. **The literals are historical, not necessary.**

**P9 — ~800–1,050 duplicated lines in `emit_c.ml`, plus five hand copies of one lattice 🟡.** The file
admits it (`:3607`: *"§6a45: avx512 r2c split postamble. Same formulas as the avx2 …"*). 🔴 The frozen
11-arg z ABI is printed **twice, byte-identically, on two different compiler stacks** —
`codelet_zsplit.ml:1539-1546` and `codelet_cil.ml:866-873` — six parameter lines character-for-character
identical, each with its own `(void)` silencer list (zsplit **derives** it at `:1510-1532`, cil
**hardcodes** it at `:874`). **One frozen ABI contract, two sources of truth, derived-vs-hardcoded
silencers that can already disagree. Highest value-to-risk extraction in the library.**

**P10 — `gen_main.ml` is a god driver 🟡.** `Gen_main.run` is simultaneously the CLI parser (133 flags),
the config normaliser (`:753-790`), the math dispatcher (`:791-905`), a second copy of the pass cascade
(`:921-1430`), the symbol constructor (`:1492-1642` + a substring post-process at `:1896-1937`), the
cross-family validator (`:1651-1688`, `:1866-1895`), and the six-way emitter dispatcher (`:1741-1964`).
History confirms it: 47 of 91 feature commits, solo-rate 0.06.

**P11 — The corpus has no owning description 🟡 (🔴 for this campaign).** (a) **Provenance is a
per-emitter *choice*, not a pipeline seam** — `Emit_render.provenance_block` has exactly three callers
(`emit_c.ml:474`, `codelet_oop.ml:2123`, `codelet_zsplit.ml:773`); **`codelet_cil.ml` never calls it, and
neither does `emit_k1_mono`** ⇒ **229 files (16.0 %) ship with no recorded invocation**, their recipes
reverse-engineered from a filename grammar plus a `--cil-split` table that exists **only in
`docs/research/il_tail_handling/artifacts/regen_cil.sh:14-16`**, a shell script in a gitignored folder.
(b) Emission is not hermetic in `argv[0]`. (c) `coverage.ml:1` calls itself *"THE single source of truth
for codelet generation"* and covers **75 %** (1,074/1,432); the claim is **false in seven places** —
`--post-tw`, `--oop-load UL`, `--oop-store UL`, `--oop-tw-linear`, `--oop-spec-named`, `--strided-r2c`,
`--r2c-term-ls` all ship codelets with no `Coverage` entry. (d) **The two build systems disagree about
what the corpus IS** — `CMakeLists.txt:180` globs a **nonexistent `codelets/il/`** and never globs `zil/`;
CMake compiles **598** files, `build_tuned/build.py` compiles **863**. (e) There is **no per-codelet
documentation channel**, so measured facts get stapled on by hand and fall straight out of the
reproducible set. (f) **82 of 265 zil codelets are consumer-orphaned** in a coherent shape.

**P12 — Back-end concerns leak upstream into the math layer 🟡.** V1: `dft_select.ml:103 target_vec_regs`
— pure mathematics (which CT factorization for N=64) decided by an ISA property in a process-global whose
default is 32 (AVX-512); its own MODULE CARD calls it *"set once at program startup"*, **falsified by
`gen_set`'s warm process**. V2: `dft.ml:408-413` — the math layer emits regalloc hints and **forked its
own CT recursion to do so** (*"we do this by manually expanding the outermost CT step instead of recursing
through `dft`"*), so `dft_expand_twiddled` and `dft_expand_twiddled_spill` are hand-synced parallel
implementations of the same mathematics. 🟢 **The reverse direction is clean** — no trig or DFT math is
re-derived in `emit_c`, `emit_render`, `regalloc`, `schedule` or `isa`. Record that as a strength.

**P13 — Two complete parallel compiler stacks, undeclared and asymmetric 🟡.** `codelet_cil` has **0**
code references to `Emit_c`, `Dft`, `Pipeline`, `Regalloc`, `Ir`. **Shared between the two stacks: `Isa`,
`Uarch`, `Schedule.Make`, `Expr.elem_ref`. That is all.** The fork is justified by a measured refutation
(`codelet_cil.ml:28-42`: merging packed-complex into `Ir.node_kind` = **~150–180 exhaustive-match arms
across 7 modules**) and must be preserved. **The problem is that the fork is undeclared and the halves are
asymmetric.** Corpus attribution: `Emit_c.emit_codelet` **1,020 (71.2 %)** · `Codelet_cil.emit` 233 ·
`Codelet_oop.emit_codelet` 141 · `Codelet_zsplit.emit_codelet` 32 · `Codelet_oop.emit_k1_mono` 6. **Six
emission entry points exist, not three.** 🔴 **Any plan that leaves `emit_c.ml`/`gen_main.ml` alone touches
under a third of the acceptance number.**

**P14 — Symbol naming is a cross-cutting concern with four owners 🟡.** `gen_main.ml:1492-1642` (150-line
chain, 24 arms) + `:1896-1937` (substring replace `"t1_oop"` → `"t1_dif_oop"` + six suffixes) ·
`codelet_oop.ml:2166-2222` · `codelet_zsplit.ml:566-577` · `codelet_cil.ml:866,877-886` — and
`coverage.ml` must **independently predict** the result. Worse: `emit_c.ml:188-201` decides codelet family
by **string-searching the symbol for `"_n1_"`** — the precise pattern the repo's standing law forbids —
and returns `false` for a `_n1t_` codelet.

**P15 — Build and gate hazards that will corrupt a migration 🟡 (🔴 during execution).**
🔴 `scripts/regen_codelets.sh:13` is `dune build bin/gen_set.exe 2>/dev/null || dune build` — **the
fallback IS the bare build, its stderr is discarded, and it fires precisely when the targeted build fails,
i.e. exactly when someone is mid-refactor** (`scripts/bootstrap.sh:113` has one too). 🔴 **A bare `dune
build` rewrites tracked source** — `generated/dune` declares **13 `(mode promote)` rules**, and
`generated/plan_executors.h` (1,035,972 B, dated Jul 27) is older than its declared dep. **Reproduced
during review: `dune build @default` FAILED (rc=1) and still promoted that file to 1,078,074 B.** The law
is therefore stronger than "never bare `dune build`": **`@default` promotes even on the failure path.**
🔴 The `(modules …)` stanza silently drops files. 🔴 **The only unit gate for the cx stack does not
compile** — `bin_test/cx_pipeline_test.ml` fails on `warning 8, CRotAdd (_, _)`; the "ALL PASS" `.exe` on
disk is a stale Aug-9 binary, and `bin_test/dune` uses `(executables)` not `(tests)`, so nothing ever built
it. **`.mli` cannot be argued on build time** — measured cold full build of all 30,776 lines: **1.0–1.4 s
wall**. Argue `.mli` on encapsulation and layering only.

**P16 — Dead weight and grep poison ⚪ (🔴 in a campaign whose hazard is "filename ≠ family").**
`generator/old-lib/` — **3 `.ml`, 5,305 lines**, tracked, un-citable by standing law, added by the cx
commit as a "reference snapshot": **the refactor's own safety net became a permanent liability**, and it
still contains `codelet_zil.ml` (1,828 lines), so the name still answers greps. `lib/gen_main.ml.orig`
(39,080 B, header says *`gen_radix.ml`*). `lib/number.ml` (133 lines, tracked, not in the stanza, `open
Num` with no `num` in `(libraries)` ⇒ never compiled, greps as live; licence check first — FFTW-derived).
`lib/simd_ir.ml` (1 comment line, **is** in the stanza). `emit_c.ml:4126-4167` — three stubs, two
unconditional `failwith`. `emit_c.ml:129-365` — the pin/fence policy is a **fossil**: ~235 lines of
commentary for 2 lines of live logic (`:235-236`), both defaults OFF. **~5,600 lines of dead/misleading
OCaml that answers greps.**

🔴 **And the hazard extends to directory names.** `zil/` is a fossil DIRECTORY name: all 233 IL files carry
`codelet_cil.ml` provenance, `codelet_zil.ml` **no longer exists in `lib/`**, and all 35 `*_z_n1_*.c` —
which MEMORY.md calls legacy-zil under *"`_n1_` = zil, `_n1t_` = cil"* — reproduce byte-identically from
`--cil-n1`. **That MEMORY.md rule is STALE.** The law *"the header decides, never the filename"* now
extends one level up.

---

## 4 · The convergence map

### 4.1 Cross-cutting concerns and who owns them

**🥇 The deepest finding: the concerns with the widest footprint have no owning module at all.**

| concern | modules touched | owner | how many representations |
|---|---:|---|---|
| **Codelet family / ABI** | ~8 | ❌ **none** | 5 inconsistent disjunctions + a 13-arm ordered chain + a 24-arm naming chain + a `Coverage` argv list |
| **Layout (split vs interleaved)** | 12 | ❌ **none — no type carries it** | 5 implementations; 8+ booleans across 2 files |
| **The codelet SIGNATURE itself** | 4 | ❌ **none** | 131 `__restrict__` lines from 4 modules; no signature datatype exists |
| **Twiddle policy** | 14 | ⚠️ partial — `Dft.twiddle_policy` (`dft.ml:64-67`), which nobody is obliged to use | **9** |
| **Direction** | 14 | ❌ **none** | **5** encodings — and the word means **two different things** (fwd/bwd vs DIT/DIF). `codelet_zsplit.ml:153-161` documents that `sign=Fwd` alone computes a *wrong kernel*, found by a numerical gate, not review |
| **SIMD lattice ops** | 11 | ❌ **none** | per-width hand-written; **181:24** in `emit_c` |
| **Symbol naming** | 12 | ❌ **none** | 4 schemes + a substring post-process + `Coverage` predicting the result |
| **The codelet-file shell** | 4 | ❌ **none** | 3 hand-printers; one frozen ABI printed twice |
| **The schedule body walk** | 4 | ❌ **none** | 15 call sites, 3 mode-bit conventions |
| **Admission gates** ("is (radix, isa, kind) legal?") | 3 | ❌ **none** | 3 hand-rolled `failwith` blocks (`codelet_oop:175-200`, `codelet_zsplit:421-536`, `codelet_cil:175-196`) |
| **Provenance / reproducibility** | 5+ | ❌ **none** | 3 formats (one empty), 2 emitters that never stamp, `Coverage` at 75 %, 2 disagreeing build systems |
| **Per-codelet documentation** | — | ❌ **none** | stapled on by hand ⇒ 12 files leave the reproducible set |
| **Reset / lifetime discipline** | 4 | ❌ **none** | 3 disciplines; a 9-of-66 hand-maintained list |
| **ISA value ops** | 16 | ✅ **`Isa`** — 11 users, 35-name surface, all width branches internal | **the model** |
| **Scheduling algorithm** | 10 | ✅ **`Schedule.Make`** — 3 instantiations, both stacks | **the other model** |
| **µarch latency** | — | ✅ **`Uarch`** | — |

### 4.2 The tangle, compactly

```
                       ┌─────────────────────── 133 CLI flags ─────────────────────────┐
                       │            Gen_main.run   (1,945 lines, ONE function)          │
                       │   parse → 101 local refs → NORMALISE → dispatch → name → guard │
                       └───┬──────────┬───────────┬──────────────┬──────────┬───────────┘
     writes 21 flags ──────┘          │           │              │          └── INLINE COPY of
     + 10 oop flags                   │           │              │              the pass cascade
     + target_vec_regs ───────────────┼───────────┼──────────────┼──────────────  (DIVERGED — P4)
                                      │           │              │
   ┌──────────────────────────────────▼───────────▼──────────────▼───────┐
   │  emit_state.ml   34 cells, ZERO addressable surface                  │  ◀── BACK-EDGE BUFFER
   │       │  include                                                     │      (RC3): downstream
   │  emit_render.ml  ── 49 flag reads, input-address chain               │      families write
   │       │  include                                                     │      here so an UPSTREAM
   │  emit_c.ml       ── 96 flag reads, 4,066-line emit_codelet,          │      renderer can read it
   │                     13-arm signature ladder, output-address chain    │
   └──────┬────────────────────────────────────────────┬─────────────────┘
          │ render_node_def ×15, 3 conventions          │  Emit_c.* (97% not its own)
   ┌──────▼─────────┐  ┌──────────────────┐      ┌──────▼────────┐
   │ codelet_oop    │  │ codelet_zsplit   │      │  Pipeline     │  ← the OTHER cascade
   │ config(7 flds) │  │ zs_kind+zs_edge  │      └───────────────┘     (serves 179 files)
   │ + 10 globals   │  │ (the good design)│
   └───────┬────────┘  └────────┬─────────┘
           └──── writes tw flags ┘  ────────────────────────────▶ read by emit_render (BACKWARD)
   ═══════════════════ STACK A (real/split) — 1,193 shipped files ═══════════════════
   ═══════════════════ STACK B (packed-complex) — 233 shipped files ═════════════════
   codelet_cil ─→ cx_ir(7 cells, ITS OWN back-edge buffer) cx_math cx_render
                  cx_sched cx_cpl cx_spill cx_pipeline
        shared between stacks:  Isa · Uarch · Schedule.Make · Expr.elem_ref   ← that is ALL
```

---

## 5 · `emit_c.ml`, at length

The owner named this file. The honest answer is more nuanced than "it is too big".

### 5.1 What it is

**`emit_c.ml` is not a module containing functions. It is one function containing a module's worth of
concerns.** Four top-level definitions, three of which are stubs. Lines 60–4125 are a single
`let emit_codelet`: **4,066 lines, ~440 decision points, 96 reads of 26 distinct process-global flags,
478 `Buffer.add_string` calls, 241 `Printf.sprintf`, ~359 lines of raw C held in OCaml string literals,
181 hard-coded SIMD intrinsic occurrences against 24 uses of `Isa`, and 6 blank lines in the whole
file.** It is also **well documented** — 726 comment-only lines (17.4 %) — but the documentation is
*narrative provenance*, not *contract*, and with no `.mli` none of it is enforceable.

| concern | lines | % | where |
|---|---:|---:|---|
| **B. ISA lattice + transpose emission** | **1,750** | **42.0 %** | 601–1445, 1466–1485, ~1997–2076, 3186–3994 |
| **A. Scheduling / spill / regalloc orchestration** | **1,380** | **33.1 %** | 77–365, 2099–3183 |
| **C. Signature / ABI generation** | **501** | **12.0 %** | 522–600, 1446–1512, 1513–1864 |
| F. Generic C rendering (provenance, includes, metadata) | 184 | 4.4 % | 451–521, 4063–4115 |
| E. Tail / remainder control flow | 130 | 3.1 % | 389–450, 3995–4062 |
| D. Addressing policy + store dispatch | 95 | 2.3 % | 1865–1959 |
| G. Store dispatch | 76 | 1.8 % | part of 1960–2098 |
| H. Header / dead stubs | 51 | 1.2 % | 4116–4167 |

Cross-cutting slices that overlap the above: **split-vs-IL layout ~468 lines (11.2 %)**; **real-FFT
(r2c/c2r/hc2c) feature code ~874 lines (21.0 %)**.

### 5.2 Why it grew this way

1. **A correct insight, over-applied.** `emit_c.ml:29-32` is right that scheduling, spill, regalloc and
   rendering meet in one place. What does not hold is the extension to *"therefore everything lives
   here"*: the ABI table, the transpose lattices and the addressing policy **never touch `assigns`,
   `spill`, `scheduler`, or `current_regalloc`.**
2. **RC1.** With no family descriptor, the only way to serve 13 ABI families is 13 ordered branches in one
   function — and once the branches are there, everything that varies per branch must be there too.
3. **RC2's feedback loop.** Every new feature needed a parameter; the function was too big to take one; it
   got a global; the globals made the function bigger.

**There is direct evidence of an attempted decomposition that was abandoned**, `emit_c.ml:4117-4125`
verbatim: *"Factoring that out parameterized over buffer/stride names is the unfinished M2 phase-2
extraction."* **A previous author identified the cleanest cut in the file, stubbed it, and stopped.**
⚠ **Correction to a widely-repeated claim:** those stubs have **zero callers repo-wide**, and **22
UnitLeg codelets ship and reproduce at 100 %** (`oop/avx2`, 91/91). They are dead code, not a blocked
feature — the extraction they name is a **de-duplication**, not an unblocking.

### 5.3 Which parts are irreducible — ~1,100–1,300 lines (~28 %)

- **The blocked two-pass spill emitter (`:2099-3007`, 909 lines).** A real compiler back-end phase: DAG
  pass classification, block-sequential PASS-1 ordering to hold peak-live at O(N₂), C scope construction,
  a transitive reload walk, cluster-grouped store flushing, store-on-compute. **Despite being 22 % of the
  file it reads only 2 distinct global flags across ~152 branch points**, and is 100 % unchanged since the
  tree move. 🔴 **Do not promise that the spill path shrinks — this is the one place the file's defence of
  itself is correct.** After any decomposition it is still **700–900 lines**.
- The scheduler dispatch (`:3008-3183`, 176 lines) — four genuinely different strategies.
- The arbitrary-K tail driver (`:3995-4062`, 68 lines).
- ~250–400 lines of irreducible lattice construction, **down from 1,750**.

### 5.4 Which parts are accidental co-location — ~2,800 lines (~68 %)

Signature/ABI generation (501 lines) **lives here because it writes into the same `Buffer`** ·
transpose/interleave lattices (1,750) never reference `assigns` · addressing policy (95) is a pure
function of codelet kind, re-derived four times · pin/fence policy (237) is a fossil · provenance (184)
already lives in `Emit_render` · **~200 lines of FFT MATH live in the emitter, outside the DAG** — the
r2c fused conjugate split (`:3269-3312`, duplicated at `:3605-3650`) and the c2r Hermitian merge
(`:845-876`, `:1181-1216`). They get **no CSE, no algsimp, no FMA lift, no scheduling, no regalloc.**
🔴 **Moving them is the only seam that should be *expected* to change emitted bytes.**

### 5.5 🥇 The density finding — the mess is not where the mass is

| region | lines | decision points | distinct globals read |
|---|---:|---:|---:|
| spill / blocked two-pass path (`:2099-3007`) | **909** | ~152 | **2** |
| `emit_store` (`:1960-2098`) | 139 | ~28 | 5 |
| `render_output_addr` (`:1920-1959`) | **40** | ~15 | **5** |
| addressing policy (`:1865-1917`) | **53** | ~19 | **7** |

**The worst code in the file, per line, is 53 lines long. The biggest region is big but coherent. Any
proposal that targets file size rather than flag density will move the wrong 909 lines.**

### 5.6 Three excerpts that are genuinely hard to follow

**(a) Four re-derivations of the same two facts** — `emit_c.ml:1882-1908` vs `emit_render.ml:316-336`, in
a *different* arm order.
**(b) One flag set, three chains, three different orders** (`emit_c.ml:537-600`): chain 1 (parameters)
tests `strided_il_in` first; chains 2 and 3 do not mention it. The postamble chain at `:3194` tests
`strided_il_out` *first*, opposite to the signature chain.
**(c) Five invisible state dimensions for one `match` arm** (`emit_c.ml:1997-2076`). To read one store you
must simultaneously hold: which of 13 signature branches fired; whether `ip_il_out` was set by a driver
3,000 lines away; the **cross-call mutable handshake** through `il_stash` (whose correctness depends on
the *scheduler* emitting re/im in adjacent order, enforced only by a runtime `failwith` at `:2004`); which
of 5 ISA widths is active; and which of 2 `ls_mode`s the tail installed. **Four of the five are invisible
at the call site.**

### 5.7 Which ranked problems it exhibits

**P1 · P2 · P3 · P5 · P6 · P8 · P9 · P14 · P16.** It does **not** exhibit P4, P11, P12, P13 or P15.
🔴 **It is the display case, not the root.**

---

## 6 · The shotgun-surgery table

### 6.1 Traced cost of five realistic changes

| change request | files | why |
|---|---:|---|
| **Add an ISA** | **13–14 lib + 7–8 bin = 21** | `isa.ml`(record + 26 `vec_width` sites) · `uarch.ml` · **`dft_select.ml` — `target_vec_regs` changes the *factorization*** · `gen_main.ml`(`:618`,`:636`) · `emit_c.ml`(**29 width sites, ~1,700 new lines**) · `codelet_oop.ml`(**42**) · `cx_render.ml`(11) · `emit_render.ml`(7) · `codelet_cil.ml`(4) · `regalloc.ml` · `codelet_zsplit.ml` · `annotate.ml` · `coverage.ml`(quadrants + 2 radix lists + 31 ISA literals) · 7 `bin/emit_*_registry.ml` + `bin/gen_set.ml` |
| **Add a layout variant** | **6–8** | `emit_state.ml`(new ref) · `gen_main.ml`(**3 edits**) · `emit_render.ml`(**audit 5 independent disjunctions**) · `emit_c.ml`(signature block + ~9 flag sites) · `coverage.ml` · `bin/emit_strided_registry.ml` · C-side executor headers |
| **Change a codelet signature** | **4–6** | `emit_c.ml`(`~:415-560`) · `emit_render.ml`(4 sites) · `coverage.ml`(**name prediction**) · `bin/emit_*_registry.ml` · `generated/plan_executors.h` |
| **Add a twiddle policy** | **5–9** | `dft.ml` · `dft_recurse.ml` · `emit_render.ml`(`:231-286`) · `emit_state.ml`(a fourth ref matching three existing) · `gen_main.ml` · **plus whichever of the 8 existing twiddle refs it must not contradict** |
| **Add a radix** | **2–4** | `dft_select.ml` · `coverage.ml`(**9 hand-maintained radix lists**). **The cheapest change in the tree** — the only tax is 9 unrelated lists with no type relating them. |

### 6.2 What real feature commits actually touched (91 commits)

| module | commits | solo | solo-rate |
|---|---:|---:|---:|
| **`gen_main`** | **47 (52 %)** | 3 | **0.06** |
| `codelet_cil` (stack B) | 26 | 7 | **0.27** |
| `codelet_zsplit` | 15 | 2 | 0.13 |
| `codelet_oop` | 11 | 3 | 0.27 |
| `emit_c` | 10 | 3 | 0.30 |
| `cx_math`/`cx_render`/`cx_ir`/`isa`/`emit_render`/`emit_state` | 7/7/5/5/5/4 | 0 | **0.00** |

**🥇 Width is not the metric — *width of unowned edits* is.** Stack A's typical change (`26b9f6fb`) puts
**890 lines into one file** and still drags three others. Stack B's **widest** change (`f253bbc7`, adding a
`CRotAdd` IR node) touches 7 files but adds **only ~+130 lines outside `cx_math`** — each file gets the one
clause it owns: constructor, smart constructor, render case, latency, scheduler case. **That is healthy
ripple, and it is exactly the difference this restructure is trying to buy.**

### 6.3 The contrast — the two families that got it right

- **`codelet_zsplit`: adding a kind is ONE record literal + one match arm.** `stf2` (`:224-234`) is
  `{ mid with base = "stf2"; uj2 = true; policy = TP_PowW1; tw_off = "2*(size_t)k";
  in_edge = E_sect_tap "OLs"; out_edge = E_z "OLs" }` — six fields, one place. **The signature, the
  `(void)` silencers, the addressing and the filename all *derive* from the record.** 21 named kinds by
  record update from one base. **That is why 22 kinds fit in 1,741 lines while 13 `emit_c` families need
  4,066.**
- **`codelet_oop`: adding an edge pattern is one constructor + one arm each in `emit_load_edge` (`:676`)
  and `emit_store_edge` (`:1040`) — and the compiler tells you if you miss one**, because dune's dev
  profile makes non-exhaustive matches **fatal**. **Exhaustive matching over a variant is the enforcement
  mechanism this codebase already relies on elsewhere and has thrown away in `emit_c`.**

---

## 7 · What is GOOD and must be preserved

A restructure that destroys working machinery is a regression. These are load-bearing.

**7.1 The `cx_*` cluster — the target shape, already built here, graded honestly.** Seven modules
(`cx_ir` 183 · `cx_math` 638 · `cx_render` 300 · `cx_sched` 88 · `cx_cpl` 232 · `cx_spill` 160 ·
`cx_pipeline` 198 = **1,799 lines**), average 257 lines, **zero include chains**, each with a single
`open Cx_ir` and otherwise qualified access, each with a MODULE CARD, carved out under a byte-identity
gate. **It answers both halves of the owner's tension: every stage has a home, and there are seven
modules, not thirty.** The seam was load-bearing within two days — `cx_cpl` wired by a **3-line
dispatcher** (`codelet_cil.ml:107-112`), `cx_spill` consumed at `:625`, and `bin_test/cx_pipeline_test.ml`
exists *only* because a pass became a named module. **The most transferable micro-lesson: the cx family
chose `open` where the old code chose `include`.** `Cx_math` opening `Cx_ir` does **not** make `Cx_ir`'s
names reachable through `Cx_math`. **That single choice is why seven small modules stayed tractable.**
🔴 **But grade it honestly, because proposals will cite it:** it cost **+196 lines (+28 %)** for identical
behaviour; **it did not fix configuration** (`cx_ir.ml` became the cx family's `emit_state.ml`;
`Cx_ir.mono_spill_slots` at `:147` is written and read **only inside one function of `codelet_cil`** — a
local promoted to a global purely by decomposition pressure); **it did not shrink the emitter**
(1,744 → 1,051 → **1,342 today**, and the 924-line `emit` was never decomposed); it added **zero `.mli`**;
and it left `old-lib/` behind. **cx is precedent for LAYER extraction and none at all for DRIVER
decomposition.**

**7.2 `Schedule.Make` — the one functor, and the answer to the functor question.** `schedule.ml:334
module type SCHED_NODE` / `:414 module Make`. Small 6-method signature (`preds`, `latency`, `is_load`,
`is_store`, `is_const`, `kind_char`), **three instantiations** (`Ir_node` `:1179`, `ZNode`
`codelet_zsplit.ml:347-406`, `Cx_sched.Node` `cx_sched.ml:88`), 50–90 lines each, **zero reported
inference or build pain**, and it delivered measured wins (−39 % reg-reg moves at r32, −49 % at r64).
**Direct in-repo evidence FOR functors where the shared thing is an *algorithm over an abstract node with
a small signature* — and the existence proof against the "functor nightmare".**

**7.3 `Isa` — the model shared module.** L0, 11 users, 35-name surface, one record type, **all width
branches internal**. **The shape every shared module should have.** Its limit (value ops only) is P8 — an
extension, not a repair.

**7.4 `Expr` — proof that plain qualified access scales here.** 13 names, **19 users**, referenced
directly with **no facade**. The widest fan-in in the tree and the least problematic. **Breadth is not the
disease; re-export is.**

**7.5 The data-modelled feature axes that already exist.** `zs_edge` (`codelet_zsplit.ml:63-77`) — the only
place a memory boundary is data carrying its C stride name · `zs_kind` (`:94-167`) — a 14-field record,
21–22 kinds by record update, with `dif : bool` as a *kind field* whose comment records the bug history ·
`Codelet_oop.config` (`:152-164`) — the in-repo precedent for record-of-config **and the cautionary tale of
what happens when later features bypass it** · `Dft.twiddle_policy`, `Isa.ls_mode`, `Codelet_cil.kind` —
real types for real axes, all under-used.

**7.6 The DAG / CSE core and the clean layering direction.** The hash-consed IR, the rewrite cascade, the
DFT construction layer, `Regalloc`, `Bb`, `Annotate` — 20 years of FFTW-lineage technique, **none of it
implicated except through its facade (P5) and two backward edges (P12)**. In particular: **IR-variant
leakage into drivers is a NON-PROBLEM** — 790 of 805 `NK_*` uses are inside `ir`/passes/`emit`; drivers use
~3 each, nearly all in comments. **Do not waste effort abstracting it.**

**7.7 The byte-identity discipline and its machinery.** The gate exists, covers the whole corpus, is
validated, and **never writes the real corpus**. The house rule that makes it work: **every new capability
ships default-OFF** — visible verbatim across the source (*"Default false keeps every existing kernel
byte-identical"*, `cx_ir.ml:99,116,139`). ⚠ **And the irony this diagnosis must record: that same rule is a
documented cause of P2.** `cx_ir.ml:116` correctly diagnoses a coupling accident (*"position and direction
are INDEPENDENT properties"*) and then encodes the fix as one more boolean anyway, justified by the
byte-identity sentence. **The bit-identity requirement itself selects for one-more-global.** A restructure
must supply a landing spot *cheaper than a new global* or the loop continues.

**7.8 The blast radius is a gift.** Total external surface of the whole library: **39 values/types + 25
constructors across 10 modules**; **24 of 34 modules have ZERO external consumers** and can be given
restrictive `.mli` with **zero `bin/` churn.** The interface problem is almost entirely intra-library.

**7.9 The MODULE CARD convention — a prose `.mli` that already exists.** **31 of 35 files** carry a
structured header: `ROLE / PIPELINE / PUBLIC SURFACE (measured) / DEPS / STATE / ENV / GOTCHA`. **The
codebase has already invented an interface language and measured its own surfaces.** **And they have
measurably drifted** — `algsimp.ml:16-19` claims `schedule(68)`, actual 55; `emit_c.ml:40` claims
`codelet_oop(46)`, actual 49. **🥇 `.mli` files are the compiler-checked version of a document this team
already writes by hand and cannot keep true. That is the strongest argument for them available in this
repo** — and the *build-time* argument is worth exactly zero here.

**7.10 `Fun.protect` — the construct the tree needs, already written twice** (`codelet_zsplit.ml:1281,1449`;
`codelet_oop.ml:1947`). **It was never named or shared.**

**7.11 The hand-written-first workflow.** Supported by evidence and **bounded**:
`codelet_zsplit.ml:1552-1649` is *"a VERBATIM transcription of the proven prototype
(`src/core/oop/zturn_proto.h`), memcmp-EXACT at all four cells… **Source order is LOAD-BEARING**"*;
`Cx_math.dft_cx16_wing_t2` reproduces a hand kernel's listing order. **Mechanism:** an absorbed kernel
arrives with its own knobs; the knob's natural owner is the family module, which sits *below* the renderer
that must obey it (RC3); with no forward config channel, the cheapest legal landing spot is one more global
plus one more branch. **Architectural consequence — one requirement, not a whole design: the architecture
must keep a place to host a hand kernel verbatim, with its own listing order, without that becoming a
global flag.** 🔴 **Do not architect around the hypothesis beyond that.**

---

# PART II — THE ARCHITECTURE

## 8 · The organizing principle

> **A piece of code belongs to a FEATURE module if it must know *which family* it is serving; it belongs
> to the KERNEL if it varies only by things a feature hands it — an ISA, a radix, an edge, a layout, an
> ABI value.**
>
> **A family earns a MODULE when it needs its own *emission strategy* — its own body/edge emitter — not
> when it needs its own ABI, its own knob, or its own name. Everything else is DATA.**

The first sentence produces a *location*, which is what the owner asked for: *"where is the r2c
terminator"* has the answer `gen/real.ml`, all 318 files of it, end to end. The second sentence is the
answer to the central tension, and it is not invented — `codelet_zsplit` already builds 21–22 kinds from
one record.

**This is not a convention.** `Codelet.t` is defined **above** the kernel, in a **different dune
library**, so a kernel module that tries to branch on a family **does not compile**. 🟢 That claim was
tested during review: a kernel module referencing `Vfft_gen.Codelet` gives `Unbound module "Vfft_gen"`.
**The design's single most important structural claim is true.**

### 8.1 The three tests that stop the kernel becoming the new dumping ground

This is the half of the brief a feature-first split classically fails.

**T1 — the naming test (compiler-enforced).** A kernel module may not mention a family name. Enforced
twice: `vfft_kernel` does not depend on `vfft_gen`, so `Codelet.t` is not in scope; and
`grep -l 'Codelet\.' lib/kernel/*.mli` must print nothing, forever. Today `emit_c.ml` reads 26 family flags
at 96 sites. **Under T1 that file cannot exist.**

**T2 — the deletion test (gate-enforced).** Kernel code is code that emits *the same bytes* if every family
but one is deleted. Mechanically checkable: restrict `Corpus` to one family, regenerate, diff with the full
corpus gate.

**T3 — the NO-ANONYMOUS-PARAMETER rule (human, and load-bearing).**

> 🔴 **Extraction into the kernel is FORBIDDEN until the shared parameter has a name that mentions no
> family. If the only honest name for the argument you would have to add is `~is_r2c_term` or
> `~hc2c_nat_sstar`, the abstraction is wrong and the code STAYS IN THE FEATURE until someone finds the
> real vocabulary.**

Worked on `emit_render.ml:316-336`: naively "shared", those three lines become a kernel function taking
**seven booleans** — the disease, relocated one level down. That is *literally* what happened to
`cx_ir.ml`. T3 forbids it and forces the real vocabulary: `Abi.loop = Batched of {var;bound;stride} |
Ranged of {var;count;stride} | Columns of {var;stride}`, a value carrying its own C identifiers —
generalising `zs_edge`'s `E_planes of string`, which is already in the tree. **Five disagreeing
disjunctions collapse to three total projections, and the arm ORDER that is currently the specification
disappears.**

🔴 **T3 has a known unresolved collision and it must be settled before M4** — see §11 OQ-3
(`Abi.prologue` and the strided-r2c C prologue).

---

## 9 · The target module tree — 38 modules, three libraries

Layers are strict *within a library*; the library boundary is the enforcement. ⚠ Honest correction to an
earlier draft: **"layers are strict" is decorative inside a library** — the design's own types need
`Layout`→`Abi`, `Render`→`Abi`, `Emit_body`→`Render`, `Render`→`Simd`, all intra-L3. The *library*
boundary is real; the intra-library layer numbers are documentation.

### Library `vfft_kernel` — 23 modules, structurally incapable of naming a family

| # | module | L | one line | today |
|---:|---|:-:|---|---|
| 1 | `Isa` | 0 | ISA record + *value* intrinsics | `isa.ml`, vocabulary grown |
| 2 | `Uarch` | 0 | µarch latency/port tables | unchanged |
| 3 | `Expr` | 0 | `elem_ref` + symbolic expression — the one type both stacks share | `expr.ml` |
| 4 | `Cnum` | 0 | exact complex constants | unchanged |
| 5 | **`Layout`** ★ | 0 | `plane` / `buffers` / `param` / `pointers`. **The anti-hybrid law lives here** | new, <150 lines |
| 6 | **`Knobs`** ★ | 0 | the 43 byte-affecting env vars, declared once, read once, into per-consumer recipes. **Contains `module Trace`** — the 12 pure-diagnostic keys, ambient, never threaded (§11.5) | from 90 scattered `Sys.getenv` sites |
| 7 | `Ir` | 1 | hash-consed DAG + smart constructors | `ir.ml`, chain cut |
| 8 | `Simplify` | 1 | algebraic rewrites | chain cut |
| 9 | `Fma_passes` | 1 | the FMA rewrite family | chain cut |
| 10 | `Algsimp` | 1 | spill lifting, `butterfly_share_mul`, DAG stats | chain cut → 16 own names |
| 11 | `Bb` | 1 | basic-block / budget analysis | unchanged |
| 12 | `Annotate` | 1 | DAG annotation | unchanged |
| 13 | `Schedule` | 1 | `SCHED_NODE` + `Make` + `Ir_node` — **the library's one functor** | unchanged |
| 14 | `Regalloc` | 1 | linear-scan allocation, peak-live | unchanged |
| 15 | `Pipeline` | 1 | **THE** pass cascade, parameterised by `Knobs.pipeline_recipe` | sole owner |
| 16 | `Dft_select` | 2 | c2c algorithm choice (`target_vec_regs` becomes a parameter) | chain cut |
| 17 | `Dft_recurse` | 2 | c2c recursive constructions | chain cut |
| 18 | `Dft` | 2 | c2c twiddle policy + expansion wrappers | chain cut (65 % its own — **not** the disease) |
| 19 | `Split_radix` | 2 | existing SR construction. **Placed, frozen, never developed** (banned topic) | unchanged |
| 20 | **`Simd`** ★ | 3 | width-parametric **lattice** vocabulary: transpose, deinterleave, interleave, mask-pack, stream | `emit_c`'s feature-blind lattice lines + 4 hand copies |
| 21 | **`Abi`** ★ | 3 | a codelet's C **signature** as data: buffers × twiddles × loop × scalars, one **total** `make` | `emit_c` 501+95 lines + 4 printers |
| 22 | `Render` | 3 | per-node C rendering, `type ctx`, `body_preamble` | `emit_render.ml` minus flags |
| 23 | **`Emit_body`** ★ | 3 | schedule / spill / regalloc / render meeting point — **the irreducible core** | `emit_c:2099-3183` |

⚠ **`Trace` is a submodule of `Knobs`, not a compilation unit** — deliberately, because the count
discipline applies to the design's own additions. Making it a 39th module would be the exact failure the
owner pre-rejected, arriving as a *fix*.

### Library `vfft_cx` — 7 modules (**#24–#30**), `C2c_il`'s private compiler

**24** `Cx_ir` · **25** `Cx_math` · **26** `Cx_render` · **27** `Cx_sched` · **28** `Cx_cpl` · **29**
`Cx_spill` · **30** `Cx_pipeline` — content unchanged; state
fixed at M12. **A separate library, not `(private_modules)`** — see §9.2.

### Library `vfft_gen` — 8 modules

⚠ **Numbering corrected.** An earlier draft numbered this table 25–32, colliding with the seven `cx_*`
modules and leaving #24 unassigned; the tree is 1–23 kernel, 24–30 cx, **31–38 gen**.

| # | module | L | one line | owns |
|---:|---|:-:|---|---|
| 31 | **`Codelet`** ★ | 4 | **the word for the thing we compile**: kind × modifiers, `sense`, validation, `symbol`, `to_argv`/`of_argv` | replaces 40 CLI booleans + 22 emitter booleans + 4 naming schemes |
| 32 | `C2c_split` | 5 | split-complex c2c: in-place 648 · OOP edges 147 · strided/batched 54 | **849 files (59.3 %)**. 🔴 **NO SIZE ESTIMATE IS GIVEN ANYWHERE IN THIS DOCUMENT** — it absorbs all of `codelet_oop.ml` (2,540) plus `emit_c`'s in-place/strided share, so it is plausibly **2,000–3,500 lines and the largest module in the tree after the restructure**. See §21 G5. |
| 33 | `Real` | 5 | r2c · c2r · hc2hc/hc2c · the r2r trig zoo | **318 files** |
| 34 | `Real_math` | 5 | real/Hermitian DAG constructions | `dft_r2c.ml`; split **declined, owner-ratified** at `:22-32` |
| 35 | `Cascade_z` | 5 | the zil block-split cascade kinds | **32 files** — already the best-behaved emitter |
| 36 | `C2c_il` | 5 | packed-complex full-IL family | **233 files** |
| 37 | `Corpus` | 6 | the corpus = **union of per-feature `recipes`**; global uniqueness; quadrants | `coverage.ml`, inverted from hand-list to derivation |
| 38 | `Driver` | 6 | argv → `Codelet.of_argv` → dispatch to ONE feature → stdout | what remains of `gen_main.run` |

**Deleted outright:** `emit_state.ml` (34 cells → fields of `Abi.t` / `Render.ctx`) · `emit_c.ml`
(redistributed) · `gen_main.ml` (**renamed** to `Driver` after redistribution — it is a survivor in the
38-count, not a deletion; only `emit_state`, `emit_c` and `simd_ir` leave the stanza) · `simd_ir.ml` · `number.ml` (licence check first) ·
`gen_main.ml.orig`. 🟢 **`old-lib/` — OWNER DECISION REVISED (2026-08-14, later the same day): MOVED to
`src/dag-fft-compiler/archive/old-lib/`** ("you can put the old lib to another folder, so I can
follow the new files easier") — out of the generator working tree, still tracked, banners + `.ignore`
travel with it, deletion still scheduled at v1.0. The earlier keep-in-place text below is retained
for the record:
An earlier draft of this document recommended moving it out of the tree (3 files, 5,305 lines,
un-citable by standing law). That recommendation is **withdrawn**: the directory is retained
deliberately for historical tracking and **will be deleted when v1.0 releases — not before.**

The underlying problem it was solving is real but smaller than the remedy: old-lib **answers greps**
in a codebase whose central hazard is *"filename ≠ family"*. Address that without moving anything —
add an `.ignore` (ripgrep honours it, and `rg --no-ignore` still reaches the files when you actually
want them) plus a one-line STATUS BANNER at the top of each file, matching the repo's existing
banner-over-commit-message convention. Zero risk, keeps the history, kills the false hits.

### 9.1 🥇 The central tension, quantified — CORRECTED

**This is the table the owner will judge the design on. Four rows in an earlier draft were refuted during
adversarial review; they are corrected here, with the refutation named.**

| metric | today | after | Δ | note |
|---|---:|---:|---|---|
| **compiled modules** | 34 | **38** | **+4** | 23 kernel + 7 cx + 8 gen. Survivors: 31 of today's 34 are kept or renamed; `emit_state`, `emit_c`, `simd_ir` are deleted; 7 are new (`Layout` `Knobs` `Simd` `Abi` `Emit_body` `Codelet` `Real`) |
| **modules with an external consumer today** | **10** | **~10** | ≈0 | ⚠ **CORRECTED.** An earlier "34 → 31" row was refuted: only 10 modules have any external consumer, and `(private_modules)` does not restrict resolution on dune 3.23 (§9.2). **This row is not a win; it is reported so the count is honest.** |
| the `cx_*` subsystem as names a reader carries | 7 | **1** (`C2c_il`) | −6 | via a separate library + `(implicit_transitive_deps false)` |
| `.ml` files on disk | 35 | 38 | +3 | |
| `.mli` files | **0** | **~28** | +28 | §10 |
| **total files in `lib/`** | 35 | **~66**, in **two directories** (`kernel/` ~48, `gen/` ~18) | +31 | ⚠ the two-library split requires two directories anyway, so this is not 66 files in one flat listing |
| names visible through `Algsimp.` | 359 refs, 16 own (4 %) | **16** | −96 % | from M1 alone |
| names visible through `Emit_c.` | 77 refs, 2 own (3 %) | **2** | −97 % | from M1 alone |
| **process-global mutable cells** | **66** | **4** (`Ir` hash-cons memos) | **−94 %** | ledger in §11.4 |
| **total shared *named* surface** (cells + env keys + CLI flags) | **254** | **~200** | **−21 %** | ⚠ **CORRECTED.** 55 env vars become 55 `Knobs`/`Trace` fields one-for-one; 133 CLI flags survive verbatim. Only the mutable-cell plane collapses. |
| `Sys.getenv` sites / modules reading env | 90 / 20 | **2 / 1** | — | `Knobs` + its `Trace` submodule |
| `include` used as inheritance | 7 sites | **0** | — | ⚠ 7, not 5 (`dft.ml:34`, `dft_recurse.ml:27` included) |
| **largest function** | 4,066 lines | **1,332** | −67 % | ⚠ **CORRECTED from "~900".** `Codelet_zsplit.emit_codelet` (`:410`–`:1741`, end of file) is **1,332** lines and M8 only *renames* it to `Cascade_z`. |
| **lines in the four god-functions** | 8,267 (27 %) | **~3,155** | **−62 %** | ⚠ **CORRECTED from "−89 %".** Only `emit_c` (4,066) and `gen_main.run` (1,945) are decomposed; `Cascade_z` (1,331) and `C2c_il` (924) are renames. **A 3.5× understatement in the earlier draft.** ⚠ the two rows above use 1,332 (inclusive of both endpoints); an earlier draft wrote 1,331 in one place and 1,332 in another. |
| illegal ABI states representable | ≥ 6 enumerated | **0**, *if* OQ-1 lands | — | §11 OQ-1: the `extra : param list` hole was demonstrated exploitable |
| corpus files with no `Coverage` entry | 358 (25 %) | **0** (derived) | — | M10 |
| corpus files with no recorded recipe | 229 (16 %) | **0** | — | M12b, byte-changing |
| places `__restrict__` is printed | 4 modules, 131 lines | **1** (`Abi`) | — | subject to OQ-3 |
| places `_mm*` literals live outside `isa.ml` | **278, in 6 modules** (⚠ corrected from "266, in 4+") | **1** (`Simd`) | — | partial: see M7 scoping |
| pass cascades | **3** (already diverged) | **1** | — | M11 |

**+4 modules. −62 shared mutable cells. −21 % of the total named shared surface. The god functions lose
62 % of their mass, not 89 %. And the file count nearly doubles across two directories.**

### 9.2 Why +4 is not module explosion — four defences, one of them withdrawn

**(a) ⚠ WITHDRAWN.** An earlier draft claimed `(private_modules cx_*)` makes the seven `cx_*` invisible,
taking the public count 34 → 31. **Refuted by experiment:** on dune 3.23, `Vfft_gen.Cx_pipeline.p`
compiles from an external executable; the negative control gives `Unbound value`, not `Unbound module`,
and `_build/…/vfft_gen.ml-gen` shows dune aliasing the private modules anyway (tried at lang 3.0 and
3.16, with and without `(modules …)` and `(public_name)`). **The field is validated but does not restrict
resolution.** The replacement mechanism is a **third library `vfft_cx`, not listed in `bin/dune`'s
`(libraries)`, plus `(implicit_transitive_deps false)`** — which is *also* required because transitive
deps are currently reachable without being declared. That works, and it is the only reason the "cx becomes
one name" row survives.

**(b) The +4 buys four *nouns the library currently lacks*, not four more places to look.** `Layout`,
`Simd`, `Abi`, `Codelet`. Each replaces a concept with **zero owner and 4–5 implementations** today (§4.1:
SIMD lattice — owner ❌, 181:24 · the codelet *signature* — owner ❌, 131 `__restrict__` lines from 4
modules · codelet family/ABI — owner ❌, 5 mutually inconsistent disjunctions). **Net change in *places you
must look* is strongly negative.** Each is a **type**, not a behaviour: `layout.ml` <150 lines, `abi.ml`
~300, `codelet.ml` **450–1,000 (ESTIMATE, widened — see §11 cost 7)**.

**(c) A new feature adds ZERO modules. This is a checkable commitment.**

| adding | modules added | edits |
|---|---:|---|
| a new flag / knob | **0** | one field on a `Knobs` recipe or one `Codelet` modifier |
| a new kind inside an existing family (a 23rd `zs_kind`) | **0** | one variant arm + one record literal |
| a new ABI shape | **0** | one `Abi.t` value |
| a new ISA | **0** | one `Isa.t` record + the arms `Simd` demands |
| a genuinely new **emission strategy** | **1** | the only case that earns a module |

**(d) Trig and strided prove the rule was applied against the design's own thesis.** The owner named five
features. This gives **three modules** for those five, deliberately:

- **Trig** (dct1–4, dst1–4, dht, rdft — 72 files) differs from r2c in exactly two things: its math
  constructions (already in `Real_math`) and one ABI — `(const double *in, double *out, size_t K)`
  (`emit_state.ml:59-65`). **An ABI is now a value.** A `Trig` module would be ~8 lines and one more
  dependency edge. *A feature that differs only in ABI does not earn a module.*
- **Strided/batched** (54 files) is a **kind**, not a **module**. It selects a loop shape and an edge
  pattern, both of which are `Abi`/`Codelet` values; `emit_c:522-1445` is ~90 % AOS↔SOA lattice (→ `Simd`,
  shared with `codelet_oop`'s UnitLeg) + an ABI + r2c math (→ `Real`). **Strided lives in `C2c_split` with
  `Edge = Unit_leg` and `Loop = Batched`.** ⚠ An earlier draft contradicted itself by *also* giving `c2c` a
  `Strided` constructor and calling that "not a family". **Resolved here: `Strided` IS a kind constructor
  (the corpus selects it) and is NOT a module.** Kind ≠ module is the whole discipline.

---

## 10 · The `.mli` policy

**The rule.** A module gets an `.mli` **iff** it has more than one consumer **or** it sits on a layer
boundary. A module under active experimentation with a single consumer does not get one. That yields
**~28 of 38**. The ten exemptions are the `cx_*` seven (single consumer, `open`-coupled — 53 of `Cx_ir`'s
57 names are reachable only via `open`, so their interfaces need a design decision first), `Emit_body`
(one consumer), `Split_radix` (frozen), and `Real_math` (one consumer).

**What they buy here, specifically.** Not build time — measured cold full build of all 30,776 lines is
**1.0–1.4 s wall / ~2.5 CPU-s**, so that argument is worth exactly zero. They buy: (1) the
compiler-checked version of the MODULE CARD, a document this team already writes by hand for 31 of 35
files **and cannot keep true** (`algsimp.ml:16-19` claims `schedule(68)`, actual 55); (2) enforced
layering — `layout.mli` not exporting a second printer is what makes §12's law hold; (3) `Codelet_zsplit`
defines 39 names and has **1** public consumer — 38 names of encapsulation for free.

**What they cost.** Type definitions duplicate. Adding a variant to `Ir.node_kind` becomes two edits.
That is real and permanent for 28 modules.

**Timing.** Leaves early (M2), chain tails and emitters late (M9). A truthful `emit_c.mli` *today* is ~76
lines of *someone else's* surface including ~36 `val current_* : bool ref`; after the chain cut it is 2
`val`s. `.mli` on stable leaves costs nothing, cannot change bytes, and locks the layering **as you go**.

**Generation recipe** — ⚠ **the commonly-quoted form does not run.** `-open Vfft_v2__` gives
`Error: Unbound module`. The working form, taken from dune's own `--verbose` line and executed during
review (exit 0, 81 lines for `ir.ml`):

```sh
cd _build/default && ocamlc -i -short-paths -no-alias-deps \
  -I lib/.vfft_v2.objs/byte -open Vfft_v2 -impl lib/ir.ml > /tmp/ir.mli
```

Then **delete lines**. It is an editing task, not an authoring task, and the starting point type-checks.

### 10.1 Sketch — `kernel/layout.mli`

```ocaml
(* kernel/layout.ml — LAYER 0.  THE ANTI-HYBRID LAW LIVES HERE AND NOWHERE ELSE.
   It also owns the C parameter type, deliberately: the law is about WHICH POINTERS
   appear in a signature, so the pointer-parameter vocabulary belongs with the law.
   (An earlier draft put `param` in `Abi` and declared Layout "no deps"; that is a
   MODULE CYCLE and it was reproduced as a build error during review.) *)

type plane =
  | Split                      (** two planes: <p>_re, <p>_im *)
  | Inter                      (** one plane: <p>_z, pairs (re, im) *)
  | Inter_sw                   (** one plane: <p>_z, pairs (im, re) — the bwd-swap enabler *)

type buffers =
  | Rio    of plane            (** true in-place: ONE buffer.  `ip_il_in && ip_il_out`
                                   is NOT EXPRESSIBLE — there is one plane, not two. *)
  | From_z                     (** in_z -> rio_re, rio_im : the ip_il_in ABI,
                                   VERIFIED against emit_c.ml:1448-1452 *)
  | To_z                       (** rio_re, rio_im -> out_z : the ip_il_out ABI *)
  | Oop    of { load : plane; store : plane }
                               (** 3 x 3 = 9, ALL legal — including {load=Inter;
                                   store=Split}, the SHIPPED boundary-conversion codelet *)

type role = In | Out | Rio_role

type param = private
  { ctype : string; name : string; restrict_ : bool; silence : bool }

val pointers : plane -> role -> prefix:string -> param list
(** THE data-plane printer.  Total, and takes ONE plane.  Its existence IS the
    enforcement: there is no overload taking two, no `plane list`, no `plane option`,
    no pair of bools, and no other function in this library returns a data-plane
    `param`.  A second such printer would have to be added HERE and exported HERE. *)

val scalar : ctype:string -> name:string -> param
(** The ONLY other way to make a `param`, and it cannot produce a pointer:
    `ctype` is validated against a closed list of scalar C types.  See §12. *)
```

### 10.2 Sketch — `gen/codelet.mli` (surface only)

```ocaml
(* gen/codelet.ml — LAYER 4.  THE WORD FOR THE THING WE COMPILE.
   Derived from the corpus, not invented: 1,199/1,199 shipped codelets carry exactly
   ONE of 40 mutually-exclusive KIND selectors — alongside a SEPARATE modifier matrix.
   Both are encoded.  An earlier draft encoded only the kinds and could not name the
   32 shipped `*_strided_r2c.c` files. *)

type direction = Fwd | Bwd
type placement = Dit | Dif   (** twiddle PLACEMENT — NOT direction.  Confusing the two
                                 produced a silently WRONG kernel that review missed and
                                 a conj-identity gate caught (codelet_zsplit.ml:153-161) *)
type sense = private { dir : direction; placement : placement; table_conj : bool }
val sense : dir:direction -> kind_default:placement -> ?override:placement -> unit -> sense

type kind      (* the measured 40-way sum; arms in codelet.ml *)
type modifiers (* the measured modifier matrix as a record of typed options,
                  e.g. { real_edge : real_edge option; tail : tail option;
                         spill : spill option; store : store_mode; ... } *)

type t = private
  { kind : kind; mods : modifiers; radix : int; isa : Isa.t
  ; sense : sense; layout : Layout.buffers; recipe : Knobs.render_recipe }

val make    : … -> t          (** validating smart constructor; enforces the four measured
                                  invariants (in-place `--su` (+) `--twiddled`; hc2c-nat
                                  `--bwd` <=> `--dif`; `--ranged` only with hc2hc/hc2c-nat;
                                  `--t1s` = ALL for real-cascade kinds) *)
val symbol  : t -> string     (** ONE naming scheme.  Today there are FOUR plus a substring
                                  post-process at gen_main.ml:1896-1937 plus `Coverage`
                                  independently PREDICTING the result. *)
val to_argv : t -> string list  (** provenance == coverage == regen recipe, one fact *)
val of_argv : string list -> t
val abi     : t -> Abi.t        (** total; see §11 cost 7 for the size risk this carries *)
```

---

## 11 · The typed replacement for the global flags

### 11.1 The three problems hiding in "66 globals"

Two independent censuses split them the same way, and **that split is a measurement, not a design
choice**:

- **~22 are configuration** — the codelet-family flags. **Exactly one writer each**, all in the single
  contiguous block `gen_main.ml:753-790`, and **pure readers** downstream: 96 reads in `emit_c`, 49 in
  `emit_render` (70 writes / 171 reads overall). ⇒ fields of a value that flows **forward**.
- **~10 are per-emission scratch** — `current_ls_mode` (19r/10w), `current_regalloc` (16r/5w),
  `current_emit_position`, `current_fence_only`, `il_seen`/`il_pending`/`il_stash`, `dup_barrier_tags`.
  Genuinely mutable, genuinely short-lived. ⇒ **`mutable` fields of a record created per emission**.
- The rest are **back-edge buffers** that stop existing once configuration flows forward, plus memo tables
  that belong to their own module.

> 🔴 **`emit_state.ml:9-14` defends the globals with *"threading ~20 mode parameters through each would
> dwarf the feature code."* That defence is correct about 20 parameters and wrong about one record.** And
> it is empirically wrong on its own terms: `render_node_def` (`emit_render.ml:1421-1445`) **already takes
> 8 mode parameters + 1 expression**, at **15** call sites. Under `~ctx` its arity goes **9 → 3**.

### 11.2 The types

```ocaml
(* kernel/abi.ml — LAYER 3.  Depends on Isa + Layout.  A codelet's C SIGNATURE, as data. *)

type twiddles =                          (* replaces NINE representations *)
  | No_tw                                (** signature keeps tw_re/tw_im, body (void)s them *)
  | Per_group   of { stride : string }
  | Broadcast | Per_position
  | Linear      of { nlegs : int }       (** the MKL streaming-cursor layout *)
  | Records     of { off : string; width : int }   (** zsplit's [c x VW][s x VW] *)

type loop =                              (* the T3 neutral parameter *)
  | Batched of { var : string; bound : string; stride : string }
  | Ranged  of { var : string; count : string; stride : string }
  | Columns of { var : string; stride : string }

type t = private
  { symbol : string; target : Isa.t; buffers : Layout.buffers
  ; twiddles : twiddles; loop : loop; extra : Layout.param list }
  (** INVARIANT, enforced by `make`: every element of `extra` came from
      `Layout.scalar`.  A data-plane pointer is not constructible here — see §12. *)

val make : symbol:string -> target:Isa.t -> buffers:Layout.buffers ->
           twiddles:twiddles -> loop:loop -> ?extra:Layout.param list -> unit -> t

val params    : t -> string      val silencers  : t -> string   (* DERIVED, not hardcoded *)
val signature : t -> string      (* the parameter list, target attr, and open brace.
                                    THE ONLY place __restrict__ is printed. *)
val loop_var  : t -> string
val in_stride : t -> string      val out_stride : t -> string
val in_addr   : t -> leg:int -> re:bool -> string
val out_addr  : t -> slot:int -> re:bool -> string
val z11       : symbol:string -> Isa.t -> t
(** The frozen 11-arg z ABI as ONE value.  Today printed TWICE, six parameter lines
    character-for-character identical, on two different compiler stacks
    (codelet_zsplit.ml:1539-1546, codelet_cil.ml:866-873), with zsplit DERIVING its
    (void) silencers (:1510-1532) and cil HARDCODING them (:874).
    Highest value-to-risk extraction in the library. *)
```

⚠ **`Abi.prologue : t -> string` does NOT exist, and an earlier draft's promise that
`Buffer.add_string ctx.buf (Abi.prologue ctx.cfg)` replaces the whole 13-arm ladder was refuted.** All 11
arms of `emit_c.ml:1446-1911` end with a spill-decl block plus
`render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns))` — functions of the **regalloc
result** and the **scheduled DAG**, neither of which is a field of `Abi.t`, and `render_hoisted_consts`
lives at `emit_render.ml:522`, i.e. in `Render`, which already depends on `Abi`. **That is a second module
cycle and it is not a typo.** The fix adopted here: **split the concept.**

```ocaml
(* kernel/render.ml — LAYER 3 *)
val body_preamble : ctx -> spill:Regalloc.spill option -> assigns:(… * Ir.t) list -> string
(** The spill declarations + hoisted constants.  This is NOT the ABI; it is the body's
    first statements.  Today it is the SAME 9-line block copy-pasted 12 times
    (emit_c.ml:1492,1537,1564,1593,1631,1664,1699,1738,1773,1802,1821,1855) plus 11
    copies of the render_hoisted_consts call.  It becomes ONE function called once per
    feature — so the 149 lines of boilerplate collapse without inverting the layering. *)
```

Each feature therefore writes two lines, not one:

```ocaml
Buffer.add_string ctx.buf (Abi.signature ctx.cfg);
Buffer.add_string ctx.buf (Render.body_preamble ctx ~spill ~assigns);
```

```ocaml
(* kernel/render.ml — the context that flows DOWN.  Two lifetimes, visible in the type. *)
module Scratch : sig
  type t =                                 (* NOT `private` — see below *)
    { mutable ls_mode : Isa.ls_mode; mutable regalloc : Regalloc.allocation option
    ; mutable position : int; mutable fence_only : bool
    ; mutable store_on_compute : bool
    ; il_seen : (int, unit) Hashtbl.t; il_pending : Buffer.t
    ; mutable il_stash : (int * string) option
    ; dup_barrier_tags : (int, unit) Hashtbl.t }
  val create : Abi.t -> t
end

type ctx =
  { cfg    : Abi.t                         (* NO mutable fields: "this is a decision"  *)
  ; isa    : Isa.t
  ; recipe : Knobs.render_recipe
  ; buf    : Buffer.t
  ; sc     : Scratch.t }                   (* ONLY mutable fields: "this is bookkeeping" *)

val ctx      : cfg:Abi.t -> isa:Isa.t -> recipe:Knobs.render_recipe -> ctx
val node_def : ctx -> Ir.t -> string       (* was 8 mode params + 1 expr, 15 sites *)
```

⚠ **`Scratch.t` must NOT be `private`.** Reproduced during review: `Error: Cannot assign field
"Render.Scratch.ls_mode" of the private type`. Reads compile; **assignments do not** — and `current_ls_mode`
has 10 writes, `current_regalloc` 5, and `codelet_oop` performs 12 writes across 3 top-level defs. The
alternatives are ~8 setters (accessor sprawl the brief pre-rejects) or dropping `private`. **Drop it.** The
guarantee that matters — *a fresh one per emission, so nothing leaks* — comes from `Scratch.create` being
called once per codelet, not from `private`. That win is fully preserved.

> 🥇 **The temporal protocol dissolves.** `bin/gen_set.ml:26-38` runs 1,074 codelets in **one warm
> process**, so today every flag must be reset by hand, and **three different disciplines coexist** —
> `Fun.protect` (`codelet_zsplit.ml:1281,1449`), set/clear 285 lines apart (`codelet_oop.ml:1965`/`:2250`),
> and unconditional re-set — with a hand-maintained list at `codelet_oop.ml:2248` covering **9 of 66
> cells**. **A fresh `ctx` per codelet makes all eight recorded defects (D-1…D-8) unrepresentable,
> including D-1 which is LIVE under `VFFT_DUP=1`.** No `Fun.protect`, no reset list, no discipline to
> remember. **This is the largest correctness win in the design and it is a consequence of the data model,
> not an added feature.** *(The library today has **zero** `mutable` record fields. These are the first,
> and this is the right use: short-lived scratch inside a value with a bounded lifetime.)*

### 11.3 A real call site, before and after

**WRITER — `gen_main.ml:759-779`, verbatim:**

```ocaml
Emit_c.hc_strided       := !hc2hc || !hc2c || !hc2c_nat;
Emit_c.strided_r2c_bwd  := !strided_r2c && !bwd;
Emit_c.hc2c_natural     := !hc2c_nat && not !bwd;
Emit_c.hc2c_natural_bwd := !hc2c_nat && !bwd;
Emit_c.hc_ranged        := !ranged && (!hc2hc || !hc2c_nat);
if !ranged then Emit_c.hc_ranged_r := n;
if !hc2c_nat then (Emit_c.hc2c_nat_r := n;
                   Emit_c.hc2c_nat_sstar := if n mod 2 = 0 then (n/2)-1 else (n-1)/2);
```

`:772-773` is **one concept crossed with direction, written as two booleans.** Both-true is representable;
neither-true-while-`hc2c_nat`-is-set is representable. `hc_ranged` + `hc_ranged_r` is a bool paired with a
**conditionally written, never reset** int.

**AFTER — `gen/codelet.ml`:**

```ocaml
let hc2c_nat ~dir ~ranged ~radix:n =
  Hc2c { natural = true; ranged; r = n;
         sstar = (if n mod 2 = 0 then n/2 - 1 else (n-1)/2) }
```

One constructor. `dir` is a field of `sense`. `ranged` becomes `int option` inside the arm that owns it.
Both illegal states have **no representation**; the derivation lives with the thing it derives from.
**21 lines of statements → 3 lines of value.**

**READER — `emit_render.ml:316-336`**, the five disagreeing disjunctions:

```ocaml
let stride    = Abi.in_stride  ctx.cfg in
let loop_var  = Abi.loop_var   ctx.cfg in
let tw_stride = Abi.tw_stride  ctx.cfg in
```

**Threading cost, measured.** ⚠ **Cheaper than earlier drafts claimed, on the emit chain**: reads
attributed to enclosing top-level defs give `emit_c` 96 reads / **1** def, `emit_render` 46 reads / **4**
defs (`render_load` 39, `render_node_def_core` 5, `il_in_name` 1, `compute_inline_set` 1),
`codelet_oop`/`codelet_zsplit` **0 reads** (pure writers). **M6 is ~5 functions gaining `~ctx`, not "171
read-site edits."** The 20 writes at `gen_main.ml:753-790` and `:1950-1962` become one constructor call.
**All of it is compiler-checked: either `ctx` is in scope or the file does not build.**

### 11.4 The shared-state ledger — every one of the 66 cells accounted for

| module | cells | fate | after |
|---|---:|---|---:|
| `emit_state` | 34 | **22** → fields of `Abi.t` (immutable); **9** → `Render.Scratch` (per-emission); **3** back-edge buffers (`current_tw_perpos`, `current_tw_linear`, `current_tw_zsplit`) cease to exist. ⚠ **corrected from “22 / ~10 / 2”**: the scratch set is exactly `current_ls_mode`, `current_regalloc`, `current_emit_position`, `current_fence_only`, `current_store_on_compute`, `il_seen`, `il_pending`, `il_stash`, `dup_barrier_tags` = 9, and the tw back-edges are 3, not 2 (§2 RC3 lists all three). 22+9+3 = 34. ⚠ See §21 G3–G4: two of those nine are misclassified and at least one of the 22 has no `Abi.t` field. | **0** |
| `codelet_oop` | 10 | fields of `Codelet.t` / `Render.ctx`; **the hand-written reset block at `:2248-2258` is deleted, not fixed** | **0** |
| `emit_render` | 4 | `Render.Scratch` | **0** |
| `schedule` | 2 | parameters (`order_source` becomes an argument, not an ambient) | **0** |
| `dft_select` | 1 | parameter — `target_vec_regs` (`:103`), whose own comment *"set once at program startup"* is **falsified** by `gen_set`'s warm process | **0** |
| `ir` | 4 | **STAY.** A hash-cons table *is* a memo, not configuration. `ir.mli` **names the reset contract** instead of leaving it to a comment at `pipeline.ml:136` | **4** |
| `cx_ir` | 7 | **M12a** — a per-emission `Cx_ctx`. Includes `mono_spill_slots` (`:147`), a local promoted to a global purely by decomposition pressure | **0** |
| `cx_math` | 3 | **M12a** — includes `tangent` (`:47/:154/:205`), a one-way latch set at `gen_main:524` with no `:= false` anywhere | **0** |
| `cx_render` | 1 | **M12a** | **0** |
| **total** | **66** | | **4** |

**66 → 15 after M0–M11 → 66 → 4 after M12a.** The four survivors are the only cells in the library that
are legitimately process state, and they are the only ones whose contract is written down. **Nothing is
left as an unexplained remainder.**

### 11.5 `Knobs` and `Trace` — the second config protocol

**55 distinct `VFFT_*` env vars are read at ~90 sites in 20 of 34 modules, and they change emitted
bytes.** The provenance stamp watches **11** while `emit_render.ml:1358` claims *"the header can never
drift from behavior"* — false for 47.

⚠ **Two corrections that an earlier draft got wrong, both from measurement:**

1. **Do not build one 55-field record.** Three per-consumer recipes, so no module sees a knob it does not
   use: `pipeline_recipe` → `Pipeline`; `sched_recipe` → `Schedule`; `render_recipe` → `Render`.
2. 🔴 **Split byte-affecting knobs from trace knobs.** **12 of the 55 are pure diagnostics** —
   `BSM_TRACE FACTOR_TRACE FLATTEN_FMA_MUL_TRACE FLATTEN_FMA_MUL_TRACE_VERBOSE FMA_ADDEND_TRACE
   MULFMA_TRACE MULIFT_TRACE SPILL_MARKER_TRACE VFFT_CX_STATS VFFT_DEEP_COLLECT_TRACE VFFT_DUP_TRACE
   VFFT_SCHED_DUMP` — none of which can change emitted bytes. Threading them as recipe fields buys **zero
   safety at the cost of a parameter in the hottest functions in the library**. They go in `Trace`, read
   once at startup, ambient, never threaded. A design that takes its own T3 rule seriously makes this
   split, and the earlier draft did not.

```ocaml
type pipeline_recipe = { force_fma_lift : bool; aggressive : bool; reassoc : bool
                       ; no_subdedup : bool; policy_n : int option; … }
type sched_recipe    = { order : order_source; su_tiebreak : … ; load_pace : int; … }
type render_recipe   = { dup_barriers : bool; fence : fence_policy; … }
val snapshot : unit -> pipeline_recipe * sched_recipe * render_recipe
val stamp    : Buffer.t -> unit    (* provenance: ALL of them, not 11 *)
```

⚠ **And an honest correction to a cost line.** An earlier draft promised `Fma_passes` (2,075 lines),
`Simplify` (1,522), `Schedule` (1,781) and `Regalloc` (1,428) *"stay approximately as they are"*. Measured
`Sys.getenv` sites by module: `fma_passes` **14** (e.g. `:1316`, inside a nested lambda inside a pass),
`gen_main` 13, `schedule` **10** (e.g. `:1223`, inside a scheduler tiebreak, one of ten in that file),
`codelet_cil` 8, `emit_c` 7, `algsimp` 6, `codelet_oop` 6, `simplify` **5**. **Those three modules DO gain
a threaded parameter, and the promise cannot both hold.** The mitigation is (a) `Trace` removes the
diagnostic subset from the problem entirely, and (b) the remaining byte-affecting knobs are bound **once at
each module's top-level entry point** and read from a local, not threaded to every leaf. **OQ-6 measures
whether that is enough before M6 commits.**

`Knobs` also fixes a **live divergence**: the recipe is re-derived in `gen_main`,
`codelet_oop:1211-1224` and `codelet_zsplit:579-586`, and `VFFT_FORCE_REASSOC` is honoured **only** in
`gen_main` — a silent no-op on the oop and zsplit paths today.

**And it is subject to T3 like everything else** — a knob whose only honest name mentions a family is not a
knob; it is a `Codelet` modifier. Expect the consolidation to surface two or three knobs that do nothing:
`VFFT_USE_REGALLOC` is documented at `emit_c.ml:131` and **never read**.

### 11.6 Why one explicit `ctx` parameter, and not the alternatives

| candidate | verdict |
|---|---|
| **one explicit `ctx` parameter** | ✅ **chosen.** Zero inference cost, greppable, one word per site, and the mutable scratch gets a lifetime for free |
| reader/state monad | ❌ every emission site becomes `let*`; inference errors get worse; OCaml has no `do`-notation to pay for it |
| functor over a `CONFIG` module | ❌ forces functor application **per codelet**, poisons every type error with `Make(X).t` — the exact "OO nightmare" the owner pre-rejected |
| first-class module | ❌ same cost plus pack/unpack noise, for a 5-field record |
| a `Config` singleton with a mutable "current" | ❌ *the same disease with a type annotation.* Preserves every temporal hazard and makes the leaks **harder** to see |
| keep some globals "just for scratch" | ❌ that is how `cx_ir.ml` regrew the buffer three weeks after a clean decomposition |

---

## 12 · Laws as types

### 12.1 The anti-hybrid law

**The law:** never build hybrid IL-boundary/split-interior codelets; a signature carrying both
`in_re`+`in_im` alongside interleaved pointers is HYBRID; banned at codelet **and** route level.

**Why it cannot be a type error today:** §3 P6. Layout lives in four booleans and a string printer.

**The fix, in three parts.**

1. **`Layout.plane` is a single value per side, and `Layout.pointers` is total on ONE plane.** There is no
   value that carries both `re`/`im` and `z` for one side.
2. **`Abi.params` folds over `buffers`.** Every data-plane pointer in the emitted signature comes from
   exactly one call to `Layout.pointers`.
3. 🔴 **`extra` cannot name a data plane.** This third part is **new here and it is load-bearing.** An
   earlier draft declared the law enforced with `extra : param list` where `param = { ctype : string;
   name : string; restrict_ : bool; silence : bool }` — four unconstrained strings and bools. **That was
   demonstrated exploitable during review**: using only the public API, with `pointers` total and taking one
   plane, a "feature module" emitted

   ```
       const double * __restrict__ in_z,      const double * __restrict__ in_unused,
       double       * __restrict__ out_re,    double       * __restrict__ out_im,
       const double * __restrict__ in_re,     const double * __restrict__ in_im
   ```

   — `in_re`+`in_im` beside an interleaved pointer, the law's exact stated test, with no new printer, no
   cast, and no edit to `layout.ml`. **The fix is the T3-shaped one: `Layout.param` is `private`, and the
   only two constructors are `Layout.pointers` (data planes, total on one plane) and `Layout.scalar`
   (whose `ctype` is validated against a closed list of scalar C types).** A pointer parameter is then
   **not constructible outside `layout.ml`**, and the hybrid signature above does not compile.

   ⚠ **This is the design's most important unverified assumption.** It requires that no shipped family
   needs an *unmodelled pointer* parameter — i.e. that every pointer in all 23 distinct C ABIs is a data
   plane or a twiddle table. Strides, counts and `K` are scalars, so this is plausible; it has **not been
   checked against the ABI census.** See §14 OQ-1.

**The state-space reduction, counted:** in-place goes from `ip_il_in × ip_il_out` = 4 states (1 illegal,
**unguarded**) to 3 constructors (0 illegal). OOP goes from `il_in × il_in_sw × il_out × il_out_sw` = 16
states (4 illegal, 2 guarded by hand at `gen_main.ml:1869-1872`) to 9 (0 illegal). **20 → 12 states;
illegal 5 → 0.**

**The nuance is preserved and it is not optional:** `il_in` *without* `il_out` is **legal and shipped** —
`Oop { load = Inter; store = Split }`. A whole-codelet layout enum would wrongly ban a file that is in the
corpus today.

### 12.2 The other laws

| law | today | after |
|---|---|---|
| **one interior per slot** (symbol uniqueness) | pairwise hand-written `failwith`s at `gen_main.ml:1651-1675` | `Corpus.check_unique` runs **once over all 1,432** `Codelet.symbol` values. Global, not pairwise — catches a collision between families nobody thought to pair. *(Honest: a runtime raise at program start, not a type error.)* |
| **forward-only families** | a comment: *"fwd only: the bwd 2-quad was REFUTED (+29..36 %)"* (`codelet_zsplit.ml:198`) | the kind carries no direction: `Sterm2_fwd_only`, `Stf2_fwd_only` have no `dir` payload. **`stfb2` is not a value.** |
| **direction ≠ placement** | four booleans with an undocumented consistency law; getting it wrong produced a grossly wrong kernel found only by a conj-identity gate | `sense` is `private` with one smart constructor; `table_conj` is **derived** |
| **`--ranged` only with hc2hc/hc2c-nat** | a bool + a conditionally-written, never-reset int | `ranged : int option` is a field of the `Hc2hc`/`Hc2c` arms only |
| **in-place: `--su` ⊎ `--twiddled`** (648 files, exact) | two independent CLI bools | disjoint `c2c` constructors |
| **hc2c-nat: `--bwd` ⟺ `--dif`** | two flags that must agree by hand | `placement` derived from `dir` inside `sense` |
| **radix × ISA admissibility** | three independent hand-rolled `failwith` blocks answering the same question (`codelet_oop:175-200`, `codelet_zsplit:421-536`, `codelet_cil:175-196`) | one `Codelet.make` validation + one per-feature `admissible` predicate that `recipes` is **derived from** — an inadmissible codelet is never constructed |
| **an intrinsic that does not exist at this width** | `Isa.intr isa "permute4x64_pd"` (`codelet_zsplit.ml:823`) composes a name at any width regardless of whether the instruction exists (**there is no AVX-512 form**) | `Simd` exposes **operations, not names**: `Simd.deinterleave_pair : Isa.t -> …`, width dispatch internal — the way `Isa`'s value constructors already work |
| **every family has a corpus recipe** | `coverage.ml:1` claims "THE single source of truth" and covers **75 %** | **both mechanisms composed:** `recipes : kind -> Codelet.t list` is a **total match** (a missing arm is a compile error; "deliberately none" must be written `\| X -> []`) **and** `Corpus.all = concat_map recipes kinds` is what **drives emission**, so there is no second list to fall out of sync with |
| **never split-radix** | `split_radix.ml` exists, `VFFT_NEWSPLIT` off, cx port REFUTED+PARKED | `Split_radix` is placed in the kernel and **referenced by no feature's `recipes`**. It cannot ship a codelet without someone adding it to a corpus list — a visible, reviewable act |
| **hybrid ban at ROUTE level** (K≥2 dein→split→inter bridge) | runtime C in `vfft.c` | 🔴 **NOT REACHED.** That law lives outside this library and no OCaml type touches it. Stated, not papered over. |

**What is deliberately NOT a type error:** anything whose legality is a *measurement* — which chain the
planner picks, which t2q placement wins, whether `sterm` or `sterm2` is faster on a cell. Those are race
results and belong in wisdom files. **The line is: structure is typed, performance is measured.**

### 12.3 Two structural laws considered and their verdicts

**`type t = private { tag : int; node : node_kind }` on `Ir` — ⚠ REJECTED for this campaign.** It was
proposed as *"one keyword, verified safe, costs nothing"*. **Refuted by building it.** Three breakages:
(1) `fma_passes.ml:1246` constructs `{ tag; node = NK_Const 0.0 }` **punned**, invisible to the grep that
"verified" it; (2) 🥇 `algsimp.ml:369-376` defines `let fresh nk = … { tag; node = nk }` that
**deliberately bypasses `hcons_table`**, inside the spill-lifting pass that `Pipeline` runs — **the
invariant `private` is sold as enforcing is already exempted by the library itself on the production
path**, so `Ir.Unsafe` is required on day one and the invariant becomes documentation again; (3)
`schedule.ml:337-340` declares `SCHED_NODE.t` as a **public concrete record**, so `Ir_node`'s re-export
fails (*"A private record constructor would be revealed"*), and making `SCHED_NODE.t` private breaks
`codelet_zsplit.ml:1354`. Full census: **13 `{ … node = … }` literals, 7 of which are construction sites
that break.** *(Re-exporting a private record is legal OCaml — that was proved separately; the blocker is
specifically the functor signature.)* **Verdict: not one keyword. Document the exception in `ir.mli` prose
instead, and revisit only if `Algsimp.fresh` is ever removed.**

**Fatal warnings — CONFIRMED and load-bearing.** Dune's dev profile band is
`-w @1..3@5..28@30..39@43@46..47@49..57@61..62-40`. **Warning 8 (non-exhaustive match) ∈ 5..28** — proven
empirically: `bin_test/cx_pipeline_test.ml` currently *fails to build* on `warning 8, CRotAdd (_, _)`.
That is what turns "total match" from a convention into enforcement, and it is why `recipes` being total is
a real law. **Warning 33 (unused `open`) ∈ 30..39 is also an error**, so any converted `open` that turns
out unused after requalification fails the build loudly — a feature; expect it. ⚠ **But warning 40 is
DISABLED**, so constructor disambiguation is silent (which makes plain variants ergonomically cheap and
strengthens the GADT/phantom rejection) — **and warnings 44 and 45 are omitted from the band entirely**,
so an `open`-shadowing flip is **completely silent**. That is a live hazard for M1; see §13.

---

## 13 · Where a new feature goes

⚠ **This walkthrough uses `--strided-r2c`, a feature that was actually built.** An earlier draft used a
*hypothetical* ("add an interleaved-boundary r2c terminator", 18 edits / 6 files / 4 silent failures →
claimed 5 edits / 2 files), and adversarial review correctly objected that a campaign's headline expansion
number should not be an estimate on an unbuilt feature. **The hypothetical is still worth reading — it is
§6-adjacent and appears in DIAGNOSIS §7.3 — but the numbers below are the ones to quote.**

**TODAY — measured** (`grep -rn strided_r2c lib/*.ml` → 19 sites, **3** files — `emit_state.ml`, `gen_main.ml`, `emit_c.ml`, exactly the three listed below; recon/03 counts 24 edit sites
across 9 files/regions because it includes regions the flag does not name):

```
emit_state.ml:88,89                       2 refs
gen_main.ml:108,321,762,763,1626-1627     local ref, parse arm, 2 projections, name concat
emit_c.ml:547,552,562,566,574,588,636,747,1084,3269,3605    11 read sites
```

**~350 lines written, of which ~290 are a second copy of the other 60 at a different vector width**, and
the r2c butterfly is emitted as raw C strings **twice** (`emit_c.ml:3269-3407` AVX2, `:3605-3759` AVX-512,
the latter commented *"Same formulas as the avx2 edition"*). **Zero entry in `Coverage.ml`.**

**AFTER — traced edit by edit:**

| # | file | edit |
|---:|---|---|
| 1 | `gen/codelet.ml` | the `strided` kind gains an `r2c` modifier (see §10.2 `modifiers`) |
| 2 | `gen/codelet.mli` | mirror it |
| 3 | `gen/codelet.ml` `of_argv` | parse `--strided-r2c`. ⚠ **NOT “forced by totality”** — an earlier draft said so; OCaml never exhaustiveness-checks a `string` match, so `of_argv` is the ONE direction the compiler cannot police. Only `to_argv`/`symbol`/`abi`/`recipes`, which match on the `kind` **variant**, get warning-8 enforcement. A round-trip property test over the 1,199 recorded argv lines is the compensating control, and it is an M5 deliverable. |
| 4 | `gen/codelet.ml` `to_argv` | emit it — **provenance and gate input are now the same fact** |
| 5 | `gen/codelet.ml` `symbol` | the `_strided_r2c` suffix (today `gen_main.ml:1626-1627`) |
| 6 | `kernel/abi.ml` | signature arms + stride params — 🔴 **and the C prologue that aliases `rio_re`/`rio_im`, which is where T1 fires. See OQ-3.** |
| 7–9 | `gen/real.ml` | AVX2 merge prologue, AVX-512 merge prologue, per-iteration lane declarations |
| 10–11 | `gen/real.ml` | AVX2 + AVX-512 conjugate-split postambles, ~290 lines, **moved verbatim** under the `emit_*_verbatim` rule |
| 12 | `gen/real.ml` `recipes` | the 32-file row — **a NEW edit the design ADDS**, because these 32 files have **zero** `Coverage` entry today. Good change; still +1 edit. |

| | today (measured) | after (traced) |
|---|---:|---:|
| edit sites | 24 | **~12** |
| files | 9 | **4** |
| lines written | ~350 | **~350** |
| duplicated ISA-twin lines | ~290 | **~290** |
| silent-failure edits | ≥4 | **0–1** |

**~2× better on ceremony and files. 0× better on the metric that costs an engineer a day — lines of
hand-written duplicated intrinsics.** The reason is stated plainly: the AVX2/AVX-512 postambles are raw
`_mm256_*`/`_mm512_*` string emission of FFT math that never enters the DAG, §15 puts that seam
**explicitly out of scope**, and the `emit_*_verbatim` hosting rule *blesses* hand-shaped regions inside
feature modules. **The design sanctions the duplication it measured.** That is a defensible engineering
choice — it protects the byte gate — but the honest claim is that the improvement is in *ceremony and
silent-failure elimination*, not in *work*.

**What genuinely gets better, then:** four of the 24 edits today fail **silently** (a split codelet shipping
under an IL name; a symbol colliding with its split twin so the linker picks one at random; split addresses
written through an IL signature; a missing `hoist_consts_enabled` disjunction term). After, those are 0–1,
because `to_argv`/`symbol`/`recipes` are the *same* value and totality forces the arms. **And there is now a
single place that answers "where is strided r2c" — `gen/real.ml`.**

---

## 14 · The migration plan

**The bar is already met: 1403/1432 = 97.97 % byte-identical, `EMIT_FAILED = 0`, zero live emitter
regressions. The job is DO NOT REGRESS.** The 29 known misses are enumerated in §1.5 and **must not be
charged to this restructure**.

### 14.1 The gate — verdict verbatim, and what it can actually cover today

**It covers all 1,432 files in all 16 directories.** `gates/full_corpus_gate.sh verify` was executed
during review: **PASS, exit 0, 61.9 s, 1432 files, 1403 BYTE-IDENTICAL (97.97 %), corpus drift 0**,
per-directory output reproducing recon/08 line for line. **No fallback to the 183-case `cil_matrix.sh`
subset** (which is 12.8 % of the corpus and is **not** sufficient for this bar; keep it as a 10-second
inner loop). The gate is shaped as a **verdict diff**, so the 29 pre-existing misses are recorded with
their current verdict and it is **green today** — it will not become the permanently-red gate that gets
disabled. It also sha256-fingerprints the corpus so **a corpus edit is distinguishable from an emitter
change**, and it **never writes the real corpus**.

Recipe arms: `genset 1074 · derive 221 · replay 125 · ship 6 · k1 6`.

🔴 **Five defects in the gate machinery that must be fixed as step M0, before any code moves:**

| # | defect | fix |
|---:|---|---|
| G1 | **The entire gate is gitignored** — `.gitignore:96 /docs/research` matches `full_corpus_gate.sh`, `recipes.tsv`, `baseline_manifest.tsv`, `baseline_verdicts.tsv`, and `regen_cil.sh` (the only home of the `--cil-split` table). **The acceptance criterion for a multi-week restructure is one `git clean -xfd` from gone.** | move to a tracked path, e.g. `src/dag-fft-compiler/generator/gates/`. ~1 h. |
| G2 | **`record` overwrites the baseline untracked**, so a re-record leaves no reviewable trace — and the plan *requires* re-records (argv[0], the 29 misses, M12b). **The 97.97 % number can drift without anyone noticing.** | subsumed by G1: once tracked, every `record` is a reviewable diff. |
| G3 | **The gate builds 2 of 19 executables.** `dune build bin/gen_radix.exe bin/gen_set.exe` compiles only those two; `bin_test/`, `facdrv/`, `emit_tool/` are never created. M1 must rewrite **91 chain-tail refs across 7 consumer files** (`dbg_eval` 29, `dump_ir` 15, `test_mk_plus` 14, `m2_test` 11, `stage3_test` 7, `dbg_zil_math` 5, `facdrv/main` 4) — **it can break all seven and the gate reports PASS.** | an **18-target scoped build**, proven rc=0 and tree-clean during review. Exclude `bin_test/cx_pipeline_test.exe` until M0 fixes it. |
| G4 | **`CORPUS_DRIFT` is advisory** — computed at `:80`, only `printf`ed at `:171`; the exit decision is a verdict diff over `recipes.tsv`'s fixed 1,432 rows, so an **ADDED** corpus file passes with exit 0. Additions are exactly what M8, M10 and M12 produce. | one line: `[ "${CORPUS_DRIFT:-0}" -eq 0 ] \|\| { echo "GATE FAIL — corpus changed"; exit 1; }` |
| G5 | 🔴 **The build law is stronger than "never bare `dune build`".** Reproduced during review: **`dune build @default` FAILED (rc=1) and still promoted** `generated/plan_executors.h`, 1,035,972 → 1,078,074 B. So `scripts/regen_codelets.sh:13`'s `\|\| dune build` corrupts a tracked 1 MB header **precisely on the failure path**. | fix both fallbacks (`regen_codelets.sh:13`, `bootstrap.sh:113`); `scripts/README.md` prescribes bare `dune build` at `:72,101,124,465` and must be corrected too. |

**What is NOT covered, stated plainly:**

- **46–47 of 55 env vars change emitted bytes and are unstamped.** Everything in M3–M11 is gated at
  **default env only**. M11 lives **entirely** inside that blind spot.
- **M3's layout arms have zero corpus representatives** — `ip_il_in`, `ip_il_out`, `strided_il_in`,
  `strided_il_out`, `strided_ilo_nt`: `__restrict__ in_z` matches **0 files repo-wide**, and their CLI
  spellings appear in no script and no `Coverage` entry. **M3 refactors code the gate cannot see.**
- **229 files' recipes are RECOVERED, not RECORDED** — reverse-engineered from a filename grammar plus a
  `--cil-split` table whose only home is a shell script in a gitignored research folder. They reproduce
  byte-identically today, **but the gate's input is not durable**, and a new radix whose split is not in
  that table drops out of the gate silently. **M12b fixes it at the root.**
- **12 tracked `generated/*.h` registry headers are outside the gate entirely.** See M10.

**Build discipline, every step:** WSL/opam toolchain (`~/.opam/5.2.0/bin`, dune 3.23.0, **not on PATH**);
`export DUNE_CACHE=disabled`; **scoped targets only**. 🔴 **A Windows-only session cannot land M3–M11.**

### 14.2 The steps

| # | step | byte-identity | risk / notes |
|---:|---|:-:|---|
| **M0** | **Gate hardening + hygiene.** G1–G5 above. Then delete `simd_ir.ml` (+ its stanza line, together, or hard error), `number.ml` (licence check first), `gen_main.ml.orig`; ⚠ old-lib/ is **NOT moved** — the §9 owner decision (KEEP IN PLACE until v1.0) supersedes this row's earlier text; treatment = `old-lib/.ignore` + STATUS BANNERS. Fix the cx test so it actually builds (G3 builds + RUNS it). 🟢 **EXECUTED 2026-08-14** — gate moved to `generator/gates/`, G3/G4/G5 landed, hygiene done, `cx_pipeline_test` fixed (missing `CRotAdd` eval arm) + ALL PASS freshly built, gate re-run from the tracked home: **GATE PASS 1403/1432, drift 0**. `number.ml` licence check: header says "Ported from FFTW" (copied GPL source, never compiled) — deletion also resolves the licence question. | ✅ **by construction** | **The gate must be tracked and complete BEFORE anything else moves.** Also removes ~5,600 lines of grep poison in a campaign whose central hazard is *"filename ≠ family"*. |
| **M1** | **Cut the chains: `include` → `open`, SEVEN one-word edits** (`simplify.ml:32`, `fma_passes.ml:166`, `algsimp.ml:26`, `emit_render.ml:33`, `emit_c.ml:58`, `dft.ml:34`, `dft_recurse.ml:27`), then requalify **~395 refs in `lib/`, ~467 including `bin/`**. | ✅ **but see the hazard** | 🥇 `include` and `open` bring *identical* names into scope with *identical* shadowing — only `include` re-exports. Proved during review that `include` aliases the *same* `ref` cell (`physically_equal = true`) and that there is **0 real shadowing** across all 7 chain links. **⚠ But "compile error, never different bytes" is REFUTED in one configuration:** `emit_render.ml:42` defines its own `topo_sort_reachable` shadowing `Ir`'s, used at **14 sites** inside `emit_codelet`; cutting the chains *forces* new `open`s whose placement decides resolution — and **warnings 44/45 are omitted from dune's warning band, so an open-shadow flip is completely silent.** 🔴 **Mitigation, mandatory: run M1 under `-w +44+45`, and pre-qualify `topo_sort_reachable` at all 14 sites in its own separate commit first.** ⚠ any `:=` grep must be multi-line tolerant (`perl -0777`); a naive grep finds 11 of `gen_main`'s 26 writers. ⚠ never let a rename script near `codelets/` — Git-Bash `sed` strips CRLF and 99 corpus files are CRLF in the worktree. 🟢 **EXECUTED 2026-08-14**: topo_sort_reachable pre-qualified ×14 (own gated step, PASS); **522 requalifications** landed via ownership-map rewrite (407 value/type + 115 constructor/field); 7 chains cut; compensating `open Ir` added where the chain had been the only supplier — and warning 33 then PROVED `annotate`, `schedule`, `emit_c` and `test_mk_plus` never used Algsimp bare at all (their `open Algsimp` became unused and was removed — the pass-layer dependency was an include-chain artifact, and the topo_sort_reachable shadow hazard is structurally dead, not just mitigated). All 19 executables build; cx test PASS; **full corpus gate PASS (1403/1432, drift 0)**; a `-w +44+45` rebuild shows ZERO shadow warnings; `generated/` tree-clean throughout. |
| **M2** | **`.mli` for L0–L2 leaves** (`Isa` `Uarch` `Cnum` `Expr` `Regalloc` `Schedule` `Split_radix` `Pipeline` `Bb`…). | ✅ **by construction** | Generation recipe in §10 (the `-open Vfft_v2__` form does **not** run). **`Ir.t = private` is NOT part of this step** — rejected, §12.3. 🟢 **EXECUTED 2026-08-14**: G8 landed FIRST (`spill_info` + `make_spill_info` moved emit_render→algsimp beside `spill_tag_marker`; consumers requalified in pipeline/codelet_oop/emit_c/gen_main/regalloc; render-side ALGORITHMS over the data stayed put) — the Pipeline(L1)→Render(L3) inversion is gone. Then **11 `.mli`s installed** (isa uarch cnum expr bb annotate schedule regalloc split_radix pipeline ir; 0→11, the tree's first), generated via the §10 `ocamlc -i` recipe + MODULE-CARD headers; `ir.mli` NAMES the reset contract incl. the Algsimp.fresh hcons exemption. 19/19 build, cx test PASS, **full corpus gate PASS (1403/1432, drift 0)**. ⚠ Incident during acceptance: WSL 9p flake produced one phantom-drift scan and one TRUNCATED verdict file (~169/1432 rows reported as "verdicts moved") — both bracketed by clean runs; fix = `wsl --shutdown` + rerun, and the gate now self-checks NV==NBASE and exits 2 (harness error) on an incomplete run. |
| **M3** | **`Layout`.** Replace the 8 layout bools with `plane`/`buffers`/`param`. | ⚠️ **partly UNGATEABLE** | 🔴 Zero corpus representatives (§14.1). Compensating control, and it is a **deliverable of this step, not a nice-to-have**: `gates/layout_smoke.sh` invokes all 12 `buffers`/`plane` combinations, compiles with `gcc -c -mavx2 -Werror`, and asserts declared pointers == referenced pointers. *(Effectively prototyped during review: gcc rejects both illegal states with `'out_z' undeclared` and `'rio_re' redeclared`.)* **Weaker than byte-identity — stated, not oversold** — but strictly stronger than today, where `--ip-il-in --ip-il-out` silently produces uncompilable C. **Write `layout.mli` + `abi.mli` and build them scoped FIRST** — the cycle in §10.1 was a real build error and the module count depends on the resolution. 🟢 **EXECUTED 2026-08-14**: `lib/layout.ml`+`.mli` landed (12th .mli) — with a plane the sketch missed and the corpus taught: **`Real`** (bare `rio`/`out`, the strided-r2c family), partially answering OQ-1's census. Three printer sites rewired through `Layout.pointers` (codelet_oop signature, emit_c in-place arm, emit_c strided chain) — **byte-identical, full gate PASS**. The law now FIRES: `--ip-il-in --ip-il-out` (previously silent, order-resolved) raises; the oop sw-pair illegal states (previously unguarded) raise; and the demonstrated `strided-il + strided-r2c` broken-C fallthrough is a loud failwith. `gates/layout_smoke.sh` delivered: **17/17** — 13 positive arms (all five zero-representative IL arms emit + gcc-compile, data planes all referenced) + 4 negative arms refused loudly. Harness calibrations recorded in the script: `-Wno-error=incompatible-pointer-types` (pre-existing spill-store idiom, 648 shipped files) and tw_re/tw_im exempt from the referenced-check (declared-unused uniformity convention on no-twiddle kinds). |
| **M4** | **`Abi`.** The 13-arm ladder (`emit_c.ml:1446-1911`) + the second derivation (`:1885-1936`) + the third (`:428`) become one total `make` + projections; `Render.body_preamble` absorbs the 12 duplicated spill/const blocks; `Abi.z11` replaces the twice-printed z ABI. | ✅ **+ self-differential** | 🥇 **`VFFT_ABI_XCHECK` is a PREREQUISITE, not a mitigation.** Land `Abi` alongside the legacy ladder; emit **both** signatures in-process for every codelet; `assert` string equality. **1,432 independent equality proofs between ladder and match — so divergence localises to a SIGNATURE, not a file.** It is also the diagnostic: if it cannot go clean, the ABI is not a value and M4 re-scopes (§16 R2). ⚠ **Disclosed:** the xcheck inherits M3's blind spot — 0 of 1,432 recipes exercise the five layout arms, so the xcheck proves nothing about them. Delete the legacy arm in the following commit. 🟢 **PHASE 1 EXECUTED 2026-08-14 (Abi beside the ladder, xcheck CLEAN)**: `kernel/abi.ml`+`.mli` landed — `Abi.shape` (13 arms, total match) → `params_of_shape` builds every parameter list from Layout (pointers/scalar/tw_pair; two more Real-plane users surfaced: Hc2c_nat's Rp/Ip/Rm/Im); `Abi.signature` renders byte-exact. `VFFT_ABI_XCHECK=1` hook at emit_codelet's tail extracts the ladder's emitted signature (first `__attribute__((target` → `)
{
`) and asserts equality. **Full corpus gate PASS with the xcheck armed — zero mismatches across every emit_c emission, first run** — and a sabotage positive-control (me→meX) tripped rc=2 with a precise diff, proving the assert is live, then reverted. 🟢 **PHASE 2 CORE EXECUTED 2026-08-14 (same day)**: all 13 arms' signature emissions DELETED (203 lines, one breadcrumb each); the hoisted `abi_shape` derivation + `Abi.signature` is now THE one signature printer for the emit_c family; the M3 legality guards moved up with it; the xcheck hook retained as a permanent debug env (post-ladder it self-checks against any future non-Abi writer). **19/19 build + full corpus gate PASS with VFFT_ABI_XCHECK=1 armed.** Surgery incidents, recorded for the next mechanical step: a span anchor matched a BODY emission (`"    size_t k = 0;` carries the param-line prefix) and swallowed r2r's tail loop + the `else (` marker — caught by paren-balance build failure, repaired from the transcript; the owner committed mid-surgery (twice), so a `git checkout --` landed on unexpected states AND flipped the worktree file to CRLF via autocrlf — all subsequent splices must be CR-tolerant; and an env-based smoke passes VACUOUSLY when the hook is absent — always verify the hook exists before trusting its pass. 🟢 **PHASE 3 EXECUTED 2026-08-14 — M4 COMPLETE**: `Render.body_preamble` = one definition (the 12 spill-decl copies + 11 hoisted-consts calls collapsed to one call per arm; census matched 12/11, twidsq consts-free); `Abi.z11_signature` = THE frozen z ABI (plain-literal concat — string-continuation escapes are CRLF-fragile), byte-verified against a shipped cil file by the standing probe `bin_test/z11probe.ml`, wired into all THREE historical prints (codelet_cil + zsplit's emit_signature + zsplit's driver wrapper — the census said two, the tree had three); silencer policies stayed caller-side. **Full corpus gate PASS with VFFT_ABI_XCHECK=1.** OQ-3's prologue split holds (signature ends at the brace; in-body aliases stay feature-side). |
| **M5** | **`Codelet` + `Knobs`, in the `Driver` only.** Features still read globals; the globals are now *set from the descriptor* in one place. | ✅ | Pure addition. The descriptor is exercised over the whole corpus before anything depends on it, and `to_argv` round-trip fidelity is checkable against the 1,199 recorded argv lines. 🔴 **`to_argv` must reproduce the recorded string, flag order included, or M5 becomes a second baseline regeneration** (up to 1,203 `Generated by:` lines). 🟢 **CODELET HALF EXECUTED 2026-08-14/15**: `gen/codelet.ml`+`.mli` landed — 15-kind sum (X2's 14 + `K1_mono`, which the GATE surfaced: the k1 arm's 6 files are provenance-silent and were missing from the round-trip input — the gate is the completeness net), 5 global modifiers, family payloads; constructors concrete per G9. **Round-trip 1,410/1,410 VERBATIM** (1,183 provenance + 221 derive + 6 k1; the 16 orphaned avx512_regen files excluded with cause — dead-era flag order, 0/16 in the gate). Canonical to_argv orders read off the corpus, including the surprises: --isa sits MID-sequence per family, --log3 hugs its twiddle token, r2c-term-ls records --isa AFTER --emit-c, and the avx512_regen era alone used isa-last. **gen_main's two projection blocks now derive the config globals FROM the descriptor** (argv parsed once, `of_argv ~strict:false`; the --dct2-trigII debug variant OR'd from its local until the driver rewrite). **Full corpus gate PASS with VFFT_ABI_XCHECK=1.** REMAINING for M5: the `Knobs` module (recipes + Trace + snapshot) — landing as M6's opening move alongside its consumers, per X4's thread-everything verdict. |
| **M6** | 🔴 **Thread `Render.ctx`. Delete `emit_state.ml`.** (6a) the 22 config flags; (6b) the ~10 scratch cells. | ✅ at default env **AND** `VFFT_DUP=1` | Cheaper than feared — ~5 functions gain `~ctx` (§11.3). 6b is the semantically load-bearing half: `il_stash`'s correctness depends on the *scheduler* emitting re/im in adjacent order, enforced only by a runtime `failwith` at `emit_c.ml:2004`. **Standing rule if a row moves: "the gate decides, not the argument" — record it as a latent bug the corpus encodes and re-bank with a numeric gate; do not preserve the leak.** (D-2: `current_tw_perpos`/`current_tw_linear`, `codelet_oop.ml:1965-1966`, never cleared, leaking into six later quadrants.) Apply the xcheck pattern to the flag reads. 🟢 **M6.0 EXECUTED 2026-08-15 (Knobs, the M5 row's second half, landed beside its consumers)**: `kernel/knobs.ml`+`.mli` — the env REGISTRY: 8 byte-affecting keys + 8 Trace keys (the §11.5 split), each read ONCE lazily; all **29 scattered `Sys.getenv` sites in schedule/fma_passes/simplify swapped** (zero getenv left in the three modules), consumers keep their historical parses on the raw `string option` so bytes cannot move (typed recipes arrive with ctx at M6.1). Trace liveness verified (FACTOR_TRACE through Knobs produces the trace), **full corpus gate PASS**. 🟢 **M6.1 EXECUTED 2026-08-15**: `Emit_render.Scratch` landed — the 10 per-emission cells (ls_mode, regalloc, emit_position, fence_only, il_seen, il_pending, il_stash, dup_barrier_tags, unpin_candidates, hoisted_const_tags) as ONE record, created fresh by each driver (gen_main + codelet_oop's emit_codelet/emit_k1_mono + codelet_zsplit). §11.3's measurement confirmed to the letter: the doc's 4 emit_render defs + il_in_name = 5 gained `~sc` (plus wrappers/dispatchers the compiler enumerated: render_node_def, body_preamble, filter_inline_set_cross_pass, emit_load_edge, oop's 8-helper chain). THREE reset disciplines RETIRED: il_reset became a Scratch function, emit_k1_mono's Fun.protect fence save/restore deleted outright (fresh-per-emission makes it meaningless), gen_main's conditional dup_barrier_tags replacement became a fill of the driver's own record — D-1 is now unrepresentable. 21/21 build, cx PASS, **full corpus gate PASS**, and the row's dual-env requirement met: **VFFT_DUP=1 byte-identical** vs pre-change references (modulo the pre-existing random /tmp/vfft_dup_order* name stamped in comments — nondeterministic by design, recorded not hidden). STILL REMAINING: M6.1 Render.ctx + Scratch threading (~5 functions per §11.3), M6.2 config reads from the descriptor + DELETE emit_state — the X6 per-cell ledger is the map; gate at default env AND VFFT_DUP=1. |
| **M7** | **Extract `Simd`** from `emit_c`'s **feature-blind** lattice ranges + the 4 hand copies + the ≥5 transposes. Text-preserving. | ✅ | ⚠ **Scope corrected twice.** (a) The stated justification — *"`codelet_oop`'s UnitLeg path `failwith`s to this day"* — is **wrong**: those stubs have **zero callers repo-wide** and 22 UnitLeg codelets ship and reproduce 100 %. M7 is de-duplication, not unblocking. (b) 🔴 **M7 cannot be both text-preserving and feature-blind as originally scoped**: the lattice ranges read `strided_r2c`/`strided_r2c_bwd` at **11 lines / 14 occurrences** (⚠ **corrected from “24 sites”**, which was inherited from a review note: `grep -o 'strided_r2c[a-z_]*' emit_c.ml | wc -l` = 14, on lines 547,552,562,566,574,588,636,747,1084,3269,3605 — 9 lines in the 522–1445 load lattice, 2 in the 3186–3994 store lattice. The substantive point stands; the magnitude does not), and the r2c/c2r math §15 declares out-of-scope is *physically inside them* (`emit_c.ml:3269` sits inside the 3190–3994 store lattice; `:843-860` inside the 522–1445 load lattice). **Therefore M7 extracts ONLY the feature-blind sub-ranges; the r2c-conditional regions stay put until M8 moves them into `Real`.** The file names this seam itself at `:4117-4125`. |
| **M8** | 🔴 **Feature modules — PER FAMILY, SMALLEST FIRST, each independently gated + xchecked.** `Cascade_z` (32) → `Real` (318) → `C2c_il` (233) → `C2c_split` (849). What remains of `emit_c` becomes `Emit_body`. | ✅ each | **THE RISKIEST STEP — see §16 R1. GATED ON A GO/NO-GO PROBE.** Four small red-or-green events instead of one large one. ⚠ **`Cascade_z` first is for an early visible payoff, NOT for risk information** — `codelet_zsplit` already has zero own refs and `Fun.protect`s its one borrowed global, so that sub-step **cuts nothing and carries no risk signal**. The first *informative* sub-step is `Real`. 🔴 **Do NOT promise the spill path shrinks:** it reads **2** distinct globals across ~152 branch points, is 100 % unchanged since the tree move, and stays 700–900 lines. **It moves whole; it does not decompose.** |
| **M9** | **Three dune libraries** (`vfft_kernel` / `vfft_cx` / `vfft_gen`, with `(implicit_transitive_deps false)`) **+ the remaining `.mli`** (chain tails, emitters). | ✅ | Interfaces and packaging only. 24 of 34 modules have zero external consumers ⇒ near-zero `bin/` churn (but 9 files `open Vfft_v2`, including all 4 debug tools). **This is where T1 stops being a rule and becomes a build error.** ⚠ Attempting the split while back-edges are live produces a dune cycle reported as an unreadable error that gets blamed on the refactor — which is why it lands after M6. ⚠ `(private_modules)` is **not** the mechanism (§9.2). 🟢 **PHYSICAL PRECURSOR EXECUTED 2026-08-14 (owner request: "put the old lib to another folder… the 'current' lib too — so I can follow the new files easier")**: `lib/` now spans `kernel/` (21 modules incl. layout + the emit chain pending decomposition), `cx/` (7), `gen/` (6: codelet_oop/zsplit/cil, dft_r2c, coverage, gen_main) via `(include_subdirs unqualified)` — ONE library still, every module name and reference unchanged, 19/19 build + full gate PASS. New-architecture modules now land in their destined folder on arrival; M9 proper converts the folders into the three enforced libraries. (Also: `generator/old-lib/` moved to `src/dag-fft-compiler/archive/old-lib/` the same day — §9's revised owner decision.) |
| **M10** | 🔴 **`Corpus` from `recipes`.** | ⚠️ **CROSS-TREE — the point of no return** | ⚠ **Re-priced.** `Coverage.files` is not internal: it feeds `bin/gen_set.ml:62` **and six registry emitters**, which are the `(deps …)` of **12 `(mode promote)` rules writing TRACKED headers** — `registry_avx2.h` (93 KB), `registry_avx512.h` (95 KB), and the rfft/oop/trig/strided/c2r pairs — consumed by the C build, whose two systems already **disagree about what the corpus is** (CMake 598 files and globs a nonexistent `codelets/il/`; `build.py` 863). `coverage.ml:29-31` says so itself: *"regen + registry emit must ship together."* Raising `Coverage` 75 % → ~98.5 % lands **71 additions in existing quadrants**, so registries change, so new ABI-typed slots appear, so the runtime dispatcher changes. **Everything up to M9 is a `git revert` inside `generator/lib`; M10 is the first step whose output is consumed by C code outside the generator.** Requires its own C-side gate. |
| **M11** | **`Pipeline` becomes the sole cascade.** Delete the inline copy at `gen_main.ml:906-1430`; rewire `bin/dbg_eval.ml:121-134`. | ⚠️ **default env only** | 🔴 **The one step the gate cannot fully see.** **Split it:** (a) the *mechanical* unification, with the two divergent passes preserved as explicit `pipeline_recipe` fields so default emission is byte-identical — gateable, ship it; (b) the *behavioural* question *"should `VFFT_FORCE_REASSOC` now fire on oop and zsplit?"* — an **owner decision needing a race, not a diff**. Run the gate for (a) under a **6-cell env matrix** (`VFFT_FORCE_REASSOC`, `VFFT_DUP`, `VFFT_COLLECT_M`, `VFFT_DEEP_COLLECT`, `VFFT_SCHED_ORDER`, `VFFT_NO_SUBDEDUP`), recording a baseline per cell. |
| **M12a** | **The IL state fix.** `Cx_ir`'s 7 cells + `Cx_math`'s 3 + `Cx_render`'s 1 become a `Cx_ctx` created per emission; the 265 zil files enter `Corpus` and therefore the gate. | ✅ | **The campaign's shared blind spot.** 233 shipped files (16 %), an emitter absent from `Coverage` entirely, and the subsystem that **regrew a globals buffer within weeks of its own cleanup**. cil is safe today **only because it is not in `Coverage`**; adding it to the warm `gen_set` process activates its missing resets (`codelet_cil.ml:168-180` launders 4 labelled args into `Cx_ir` globals and never resets them) — **so M12a's state fix must land before or with the `Corpus` entry, never after.** |
| **M12b** | 🔴 **`codelet_cil` gains a `provenance_block` call.** | ❌ **BYTE-CHANGING — announced baseline regeneration** | ⚠ **Re-classified.** `provenance_block` emits **before `#include`**; adding the call to `codelet_cil` moves **233 files from `IDENTICAL` to `PROLOGUE_ONLY`** ⇒ **GATE FAIL on 16 % of the corpus**. This is arithmetic, not risk. It gets exactly the treatment the argv[0] fix gets: a **separately announced, separately reviewed baseline regeneration**, never folded into a structural step. It is the prerequisite for raising machine-readable reproducibility past 84 % and for making the 221 recovered recipes *recorded*. |

### 14.3 Explicitly out of scope

- **The ~200 lines of FFT math in the emitter** (`emit_c.ml:3269-3312` / `:3605-3650` / `:845-876` /
  `:1181-1216`). The **one** seam expected to change bytes — this math gets no CSE, no algsimp, no FMA
  lift, no scheduling, no regalloc. It belongs to a **numerical** campaign with numeric gates. *(If ever
  done: budget explicitly against the 3.0 points of headroom, enumerate every changed file, re-gate
  numerically.)*
- **The `argv[0]` hermeticity fix** (`emit_render.ml:1395-1414`). One line; changes 123 provenance lines;
  bodies identical. A **separately-announced baseline regeneration**.
- **Regenerating the 29 known misses**, including the live accuracy defect at
  `rfft/avx2/radix256_r2c_term_ls_r8_avx2.c:36` and the 16 orphaned `rfft/avx512_regen` files (pure sunset
  candidates = 55 % of the gap). **Corpus decisions, pre-existing, not chargeable here.**

### 14.4 Pre-work — six experiments that must run before any code is written

Each is hours, not days, and each settles a defect found by adversarial review or by the §21 completeness pass.

| # | experiment | settles | cost |
|---:|---|---|---|
| **X1** | Write `layout.mli` + `abi.mli` for real and build them scoped. Report whether the module count is 37, 38 or 39. | §10.1's cycle — reproduced as a real build error | ~30 min |
| **X2** | Type all **83 corpus argv shapes** with `Codelet.t`, including the 32 `*_strided_r2c.c` files and the full modifier matrix. Report whether `kind` stays a sum and how many `modifiers` fields there are. | §10.2 — the earlier pure-sum sketch **could not name 32 shipped files** | ~4 h |
| **X3** | The **M8 liveness probe**, with its measurement procedure defined **first** (see §16 R1). | the GO/NO-GO for the riskiest step | ~1 day |
| **X4** | Count, for `fma_passes`, `schedule` and `simplify`, how many function signatures gain a `recipe` parameter under `Knobs`. If > 30 for a module, that module keeps ambient reads and is documented as an exception. | §11.5's contradiction with cost 10 | ~2 h |
| **X5** | Build the **three-library** layout (`vfft_kernel` / `vfft_cx` / `vfft_gen`) with `(implicit_transitive_deps false)` in a scratch project and confirm an external executable linking only `vfft_gen` cannot resolve `Cx_pipeline`. | §9.2(b), which asserts *"That works"* without an experiment — §21 G11. The `(private_modules)` refutation WAS reproduced; its replacement was not. | ~30 min, beside X1 |
| **X6** | Per-cell ledger: assign each of the 66 globals to a named field of `Abi.t` / `Codelet.modifiers` / `Render.Scratch` / `Simd`, one row each. | §21 G2–G3 — `strided_ilo_nt` has no `Abi.t` field, and `current_store_on_compute` / `hoist_consts_enabled` are bucketed as scratch despite being single-writer driver config | ~2 h |

---

## 15 · Honest costs — what gets worse

1. **The file count nearly doubles: 35 → ~66.** Navigating the *directory* gets noisier even as navigating
   the *code* gets easier. ⚠ Mitigated more than an earlier draft admitted: **two (now three) dune libraries
   require separate directories anyway**, which works at `(lang dune 3.0)` with no extra stanza, so the end
   state is ~48 files in `kernel/` and ~18 in `gen/`, not 66 flat. `(include_subdirs qualified)` for finer
   grouping needs lang ≥ 3.7 and is **not proposed inside this campaign**.
2. **`.mli` files duplicate type definitions and will annoy someone within a week.** Adding a variant to
   `Ir.node_kind` becomes two edits. That is why 10 modules deliberately get none — but the 28 that do will
   pay it forever.
3. 🔴 **Adding a feature gets SLOWER, and that is the point and the cost.** Today: one CLI bool, one global,
   one `else if`. After: one `Codelet` arm, one `Abi` arm, one `recipes` row. **This is a real tax on the
   repo's actual working rhythm** — kernels are hand-written first and absorbed afterwards, and the
   absorption step's cheapest landing spot has historically been one more boolean. The design deliberately
   removes the cheap landing spot. **If the owner values absorption velocity above location, this is the cost
   line to argue with** — and §16 R3's falsifier is written for exactly that.
4. **Expect the line count to go UP first.** The one directly comparable data point in this repo: `02f7c633`
   moved 693 lines out and added 889 — **net +196 (+28 %) for identical behaviour**. **ESTIMATE: 30,776 →
   32,000–33,000 at peak (M4–M8), returning to ~29,000–30,500 once M0's dead lines and `emit_c`'s
   ~800–1,050 duplicated lines are actually deleted.** Anyone promising a smaller library on day one is
   quoting the destination and hiding the journey.
5. **~550 mechanical edits, and every in-flight branch conflicts.** ~395–467 renames (M1) + the M6 threading.
   All compiler-checked, none silent (**except the warning-44/45 hazard — see M1**), but the merge pain lands
   on whoever has an open feature branch. **Schedule M1 and M6 when the tree is quiet**, and add a
   `.git-blame-ignore-revs`.
6. **Cross-feature changes get worse. This is the honest structural trade.** Today, changing the arbitrary-K
   tail policy touches 3 copies (`emit_c.ml:3995-4062`, `codelet_oop.ml:2032-2090`, `codelet_cil.ml:721-800`).
   After, the *policy* is one kernel value but each feature still drives its own body emitter — so the change
   touches kernel + N features. **Net wash at best, and worse if an unanticipated cross-cutting concern
   appears.** Feature-first optimises for "add a feature", which the commit history says is 91 of the last 91
   changes. **It is a bet, and this is what it costs if the bet is wrong.**
7. 🔴 **`Codelet` is a new hub, and hubs are how this happened last time.** ⚠ **ESTIMATE widened from
   450–650 to 450–1,000** after review pointed out that `Codelet.abi : t -> Abi.t`, declared *total*, was
   never counted: five total functions (`symbol`, `to_argv`, `of_argv`, `make`/validate, `abi`) over ~40
   kinds × modifiers is plausibly 600–1,600 lines on its own. **The guard is a rule, not a hope: `Codelet`
   contains types and total functions over those types — never emission, never a `Buffer`, never an
   `Isa`-dependent decision.** A `Buffer.add_string` in `codelet.ml` is a review defect. **If it passes ~800
   lines it IS the new `gen_main`, and the correct response is to move `abi` into the feature modules and
   accept two homes for that fact.**
8. **Field duplication in the new records.** Across the four core records: the ISA is stored **3×**
   (`Codelet.t.isa`, `Abi.t.target`, `Render.ctx.isa`), the layout **2×**, the recipe **2×**, the symbol
   **2×** — **7 redundant copies of 4 facts across 26 fields**, with no stated equality invariant. That is
   structurally the same class of defect as the five disagreeing disjunctions, reduced from five to two but
   **silent instead of greppable**. **Mitigation adopted: `Abi.make` is sealed inside `Codelet.abi` — feature
   modules obtain an `Abi.t` only from a `Codelet.t`**, so the two cannot disagree. That costs `Abi.make`'s
   public availability, which is accepted.
9. **Three dune libraries remove an escape hatch.** A cross-library cycle becomes a hard error — the point —
   but the day someone genuinely needs the kernel to know something about a feature there is no quick fix,
   only a redesign of the vocabulary. **That will hurt at least once**, and the correct response (find the
   neutral parameter, per T3) is slower than the incorrect one (add a boolean).
10. **The cx library move breaks the only cx unit test.** `bin_test/cx_pipeline_test.ml` tests `Cx_pipeline`
    directly; it must become an inline test. Partly notional — **it does not currently compile at all**
    (warning 8, `CRotAdd`), and the "ALL PASS" `.exe` on disk is a stale Aug-9 binary.
11. **~50 % of the library's mass is deliberately untouched.** `Dft_r2c`→`Real_math` (1,625, split declined
    and owner-ratified at `:22-32`), `Fma_passes` (2,075), `Simplify` (1,522), `Schedule` (1,781),
    `Regalloc` (1,428), and the ~900-line spill engine stay approximately as they are — ⚠ **except for the
    `Knobs` parameter they may gain (§11.5, X4).** That is *correct* — none of it is implicated — but a
    reader expecting "the library got restructured" should know that half of it, by line count, is
    intentionally left alone.
12. **`Real` merges four things the owner called separate features** (r2c, c2r, hc2c, trig — 318 files,
    ESTIMATE ~900 lines). §9.2(d) argues for it and this document stands by it, but **it is the one place
    this design answers "each feature has its own module" with "no."** Splitting `Real` into `Real` + `Trig`
    costs one module and ~20 lines of dispatch — a deliberately reversible decision.
13. **The ~59-second gate is on the critical path for every commit in M3–M11, and it requires WSL.**
14. **The hand-written-first workflow needs a hosting rule, and that rule is a small ugliness.** Feature
    modules may contain verbatim transcriptions of proven hand kernels — precedent:
    `codelet_zsplit.ml:1552-1649` (*"Source order is LOAD-BEARING"*), `Cx_math.dft_cx16_wing_t2`. **They are
    allowed, by name convention (`emit_*_verbatim`), taking an `Abi.t` for their signature and nothing
    else.** This is the deliberate answer to §7.7 — the architecture must supply a landing spot for a new
    knob **cheaper than one more global**, and here the cheapest is *one variant arm + one verbatim function,
    both inside the one feature module, with zero reach outside it.* But it means the codebase permanently
    contains hand-shaped regions, and pretending otherwise would be dishonest. **It is also why §13's
    walkthrough shows 0× improvement on lines written.**
15. **Exploratory flag-flipping gets slower for exactly the experiments the standing law forbids.** A
    researcher wanting a genuinely hybrid signature must now edit a type, not flip a CLI flag. That is the
    requested outcome and it is still a cost.
16. **The CLI-flag search path gets strictly worse, and that is how tickets arrive.** Today
    `grep -rn strided_r2c lib/*.ml` returns 19 sites in **3** files with 100 % recall, because CLI flag ≡ global
    name ≡ filename fragment. After, that token survives only in `Codelet.of_argv`. **Mitigation adopted as a
    standing rule: constructor and modifier-field names MUST match their CLI spelling** (`--strided-r2c` ⇒
    `strided_r2c`). It is free discipline and the earlier draft declined to mandate it.

---

## 16 · Open questions and the experiments that settle them

Everything the adversarial reviews refuted or left underdetermined, each with its falsifier. **A claim
listed here is not a recommendation.**

| # | open question | experiment |
|---:|---|---|
| **OQ-1** | 🔴 **Does the `Layout.param = private` + `scalar`-only-`extra` fix actually close the hybrid hole?** The hole was *demonstrated* exploitable in the earlier `extra : param list` form. The fix is designed here but **not built**, and it assumes no shipped family needs an unmodelled pointer parameter. | Census all 23 distinct C ABIs from the corpus; classify every parameter as data-plane / twiddle-table / scalar. If any is none of those, the fix is incomplete and `extra` needs a fourth typed component. **~3 h.** Then rebuild the exploit against the fixed types — it must not compile. |
| **OQ-2** | 🔴 **Is `Codelet.family` a sum or a product?** The pure 40-arm sum could not name the 32 shipped `*_strided_r2c.c` files, because the corpus carries **kinds AND a separate modifier matrix**. This document proposes `kind × modifiers`, which is untested. | **X2** (§14.4). Type all 83 argv shapes; count the arms and the modifier fields. **If `modifiers` exceeds ~12 fields, `Codelet` is a config record with a different name and the design must say so.** |
| **OQ-3** | 🔴 **T1 vs `Abi`: who prints the strided-r2c C prologue?** `emit_c.ml:562-600` aliases `rio_re`/`rio_im`. Either `Abi` grows an arm that knows about r2c (**T1 violated by the design's own type**) or the prologue moves into `Real` (**and `Abi` is no longer "the only place a declaration is printed"**). The design must pick, and does not. | Decide before M4. **Recommendation to be tested: `Abi.signature` prints only the parameter list, target attribute and open brace; every in-body pointer alias is a feature-module statement.** Then `Abi` stays family-blind and §9.1's `__restrict__` row holds, at the cost of "one place a declaration is printed" becoming "one place a *parameter* is printed". Verify by attempting the split on the strided-r2c prologue alone. **~2 h.** |
| **OQ-3b** | 🔴 **Who prints the provenance block after the library split?** `emit_render.ml:1362 provenance_argv` is set by `gen_main.ml:53` and read at `:1406`; §12.2 makes provenance `Codelet.to_argv`, but `Codelet` is `vfft_gen` and `Render` is `vfft_kernel` — a kernel module reaching for it is the exact `Unbound module "Vfft_gen"` T1 exists to produce. Same class as OQ-3. | Settle with OQ-3's experiment: `Render.provenance_block` takes `~argv:string list` as a parameter (names no family ⇒ T3-clean) and the feature module supplies `Codelet.to_argv c`. Confirm no kernel module names `Codelet`. **~1 h.** |
| **OQ-4** | 🔴 **Is `emit_c.ml`'s lexical cut set small enough for M8?** Nobody has measured it. See R1 below. | **X3**, with the procedure defined first. |
| **OQ-5** | **Is the ABI expressible as a value at all?** Adjudicated *product* on the evidence (conjunctive arms at `emit_c.ml:547/566/588`, two further derivations at `:1885-1936` and `:428`), but the deeper risk is that the arms overlap **semantically** and no compositional algebra reproduces the ladder without a residual "legacy order" tiebreak. | **`VFFT_ABI_XCHECK=1` must go clean across all 1,432 at default env** (M4). If it does not, `Abi` becomes a *printer over a `param list`* consulting an explicit, **named and documented** precedence order rather than a total constructor. §12.1's law survives that (it rests on `Layout.pointers`, not on `Abi` being a sum); §12's *"illegal states → 0"* row does not. |
| **OQ-6** | **Does `Knobs` threading make `fma_passes`/`schedule`/`simplify` worse than the globals it replaces?** 14/10/5 getenv sites, several inside nested lambdas in hot passes. | **X4** (§14.4). Threshold: > 30 signatures gaining a parameter for a single module ⇒ that module keeps ambient reads, documented as an exception. |
| **OQ-7** | **Is the modifier legality matrix COMPLETE or merely CURRENT?** Absent combinations may be *illegal* or merely *unbuilt*. This decides whether `Codelet.make` **forbids** them or merely **doesn't construct** them. ⚠ An earlier draft claimed a total `Abi.make` **dissolves** this question; that dissolution depended on the `extra` hole being closed and is therefore contingent on OQ-1. | Ask the owner per row; or diff `gen_main`'s 133 flags against the 65 observed in the corpus and classify the 68-flag gap. **~2 h**, same experiment as OQ-8. |
| **OQ-8** | **Are the seven `Coverage`-missing feature axes exhaustive?** Found by set-difference, so a flag with **zero** shipped codelets would not appear. | Diff `gen_main`'s 133 flags against `Coverage`'s used set. ~1 h. |
| **OQ-9** | **Do the never-reset int refs and the `current_tw_perpos` pair actually leak in a real `gen_set` run?** (P7 D-2.) | Direct experiment: regen `strided-avx2` alone vs. as part of `all`, `cmp` the outputs. **~20 min, and it settles D-2 empirically.** |
| **OQ-10** | **Are the 221 recovered `pure_il` recipes TOTAL**, or a lucky fit for today's radices? Two concrete blockers: `--cil-split` is encoded inconsistently in names (`radix32_z_t2b_avx2.c` vs `..._t2b48_...`), and the 6 tangent files need 5 `VFFT_CX_*` knobs no filename carries. | **M12b fixes it at the root.** Until then, a new radix whose split is not in `regen_cil.sh:14-16` drops out of the gate silently. |
| **OQ-11** | **Cross-host determinism of emission.** Measured on one host/toolchain only (WSL, opam 5.2.0, dune 3.23.0). | Run the gate on a second host. |
| **OQ-12** | **`expr.ml:289-369` (81 lines) — dead?** Its names (`dft_expand`, `dft_expand_twiddled`) **shadow the live `Dft.*` ones** in the six files that `open Expr`. ⚠ **UNVERIFIED** — a qualified grep cannot settle it. | **Verify by deletion-and-build, not by grep.** |
| **OQ-13** | **Licensing status of FFTW-derived `number.ml`** before deletion. | Read the header + check the repo's licence posture. |
| **OQ-14** | ~~Committed `.o` files in `codelets/strided/avx2/`~~ 🟢 **SETTLED 2026-08-14 by the completeness pass.** 26 `.o` files exist **on disk** and **`git ls-files … | grep '\.o$'` returns nothing** — they are build residue, **not tracked**. recon 08 was right, recon 06 wrong. No action; do not “clean them from git”. | done |
| **OQ-15** | **`emit_tool/emit_executor_h.ml` vs `bin/emit_executor_h.ml`** — 153 lines of divergence, unread. `generated/dune:3` invokes the `bin/` one. | Read the diff before deleting; it may hold the newer logic. |
| **OQ-16** | **`.mli` friction at 30 kLOC in *iteration* terms** (not build time — settled at 1.0–1.4 s). | Only experience settles it. **24 of 34 modules have zero external consumers, so the experiment is cheap and reversible.** |

### The three biggest risks

**R1 — M8's lexical cut set. The biggest risk, and it has a pre-committed GO/NO-GO probe.**
`emit_c.ml:60-4115` is one `let emit_codelet`. That makes M6 cheap (`ctx` is lexically in scope at all 96
reads) and makes M8 dangerous — **the moment the function is cut, every `let` binding live across the cut
must become a parameter or a `ctx` field.** Aggravating: the identity derivation exists in **three** places
(`:1446-1841`, `:1885-1936`, `:428`) which M8 must reconcile *simultaneously* with the cut; and the gate's
failure granularity is a **file**, not a line.

**Failure mode, named:** the cut set is large → the natural fix is to widen `Render.ctx` with fields that
exist only to carry locals across the seam → **`ctx` becomes `emit_state.ml` with a record type** — which is
precisely how `cx_ir.ml` regrew its back-edge buffer.

🔴 **The threshold is definition-sensitive and plausible definitions straddle it.** A lexical probe run
during review gave, over the comment/string-stripped body:

| binding definition | bindings | max crossing a candidate cut | verdict under "< 15 ⇒ GO" |
|---|---:|---:|---|
| function-level only (indent ≤ 4) | 39 | **12** (at 365–2116) | **GO** |
| including one nesting level (indent ≤ 8) | 91 | **28** (at 1885) | **NO-GO** |

**The seam where the two answers diverge is 1885–2116 — precisely the boundary M8 must cut.** And **six of
the twelve function-level crossers are closures over local mutable state**, not values (`clear_alloc :159`,
`install_alloc_canonical :167`, `emit_regalloc_spill_decl :193`, `emit_node_spill_sites :204`,
`emit_node_reload_sites :222`, `record_peak_live :93`) — those do not become parameters; they become a
record of callbacks or they move *with* the spill engine, so counting each as "one name" understates the
work.

🔴 **MANDATORY, and the procedure is pre-committed here so the probe is obeyed rather than argued about:**

- Use a **scope-aware free-variable analysis** over the AST (`compiler-libs`/`ppxlib`), **not a grep** — a
  lexical probe conflates shadowed names like `n`, `h`, `g`, `j0` and only bounds the answer from above.
- Exclude `buf` (509 references, `:261`→`:3433`) — it becomes `ctx.buf` and is genuinely free.
- **Count a closure over local mutable state as 3**, because it becomes a callback record, not a parameter.
- **< 15 weighted names per boundary ⇒ GO.** M8 proceeds per-family, `Real` first for risk information
  (`Cascade_z` first for payoff, but it cuts nothing).
- **≥ 15 ⇒ NO-GO, and the fallback is pre-authorised: stop after M7.** That still lands `Layout`, `Abi`,
  `Codelet`, `Render.ctx`, `Simd`, `Knobs`, the laws, the `.mli` roster, and the libraries.

🔴 **And note what the corrected §9.1 implies for that fallback.** `Algsimp.` 359→16 and `Emit_c.` 77→2 come
from **M1**; 66 cells → ~15 comes from **M6**; the four new nouns come from M3–M5 and M7. **The corrected
god-function and largest-function rows are the ONLY rows attributable to M8, and both are rows this
document had to correct downward.** If the probe says NO-GO, the design loses very little of what it can
*demonstrably* deliver — which strengthens the fallback and weakens the case for taking the M8 risk. **The
owner should read that as: M0–M7 is the high-confidence core; M8 is the part that answers the location
complaint and it is optional.**

**R2 — the ABI may not be expressible as a value at all.** Falsifier and fallback: OQ-5. ⚠ **R2 has no
defined consequence for M8** — if `Abi` re-scopes to an ordered printer, whether the feature split still
makes sense is undetermined and must be re-asked at that point.

**R3 — the gate's blind spot, and the possibility that the answer is "location was not worth it".** The
three holes are named in §14.1. **And the falsifier for the whole recommendation:** the owner's brief
pre-rejects module explosion, but their *working rhythm* is hand-write-then-absorb, and cost 3 says
absorption gets slower. **If the owner, shown §9.1 and cost 3 together, says absorption velocity matters
more than a one-file answer to "where is the r2c terminator" — then the smaller footprint (M0–M7) is the
correct architecture and M8/M12 should be dropped.** That is a values question, not an evidence question,
and **it should be asked before M3, because M0–M2 are free and correct under either answer.**

> 🟢 **R3 ANSWERED BY THE OWNER (Tugbars, 2026-08-14):** *"there won't be much hand written codelets
> anymore, the development is about to be finished. the library's math and optimizations are 95%
> figured out."* — absorption velocity is no longer the operative value; the library is entering its
> finished/maintenance era, where cold-reader location, reproducibility and silent-failure elimination
> dominate. **VERDICT: the FULL plan is in scope — M8 proceeds (GO per §22 X3, conditional on the
> spill-cluster-moves-whole rule) and M12 stays.** The stop-after-M7 fallback remains pre-authorized
> only as a risk fallback, not as the target.

---

## 17 · Rejected alternatives

Three alternative architectures were argued and lost. ⚠ **Provenance the owner must see: two of the four
commissioned proposals were never written.** `A_layered_pipeline.md` and `D_typed_core.md` do not exist —
both agents returned no text. **Every "convergence" claim behind this document is n = 2, not n = 4.** The
consequence that matters: **nobody argued *for* GADTs or phantom types**, so their rejection below is
**unopposed rather than tested**. If the owner wants that stone turned over, the typed-core angle must be
re-run.

**A — "Interface-first": leave `emit_c.ml` intact; fix layering, `.mli` and shared state only.** *(Proposal
C, the real runner-up.)* Its principle — *"a module owns a stage; a value owns a feature"* — is the better
engineering sentence and it produced the better migration plan, most of which this document adopted
wholesale: `include`→`open` as one-word edits, the `VFFT_ABI_XCHECK` promoted to a prerequisite,
`Layout.buffers` as `Rio | From_z | To_z | Oop{load;store}`, the `.mli` timing split, the two-lifetime
`ctx`, *"the gate decides, not the argument"*, and the named gate blind spots. **It lost on one thing:** by
its own accounting it moves only 596 lines out of `emit_c.ml` (+1,750 in an *optional* step), so the r2c
signature arms and the two raw-string r2c butterflies stay inside the 4,167-line file the owner pointed at.
**It fixes *scattered* and declines *lumped*, and says so.** All three judges scored it identically on that
criterion. It also names three concession triggers, **two of which already fire on this repo's roadmap** —
"a third back end arrives" (AVX-512/AVX10, EPYC/GPU are in MEMORY.md) and "the IL family needs r2c/c2r"
(measured empty, and the owner named both axes in the brief). **By its own criteria it says a deeper
restructure is warranted.** 🟢 **It survives as the pre-authorised fallback if R1's probe says NO-GO** —
§16 R1, and that is a genuinely good outcome, not a consolation prize.

**B — Module-per-kind / module-per-flag: give every family and every axis its own module.** Rejected, and
**already refuted in-tree**: the `cx_*` decomposition produced seven clean modules and **regrew the
back-edge global buffer within weeks** (`cx_ir.ml:5-8`, admitted in its own header), at a cost of **+196
lines (+28 %) for identical behaviour**. `Cx_ir.mono_spill_slots` (`:147`) is written and read **only
inside one function of `codelet_cil`** — a local promoted to a global purely by decomposition pressure.
That is the owner's pre-rejected failure mode, observed here, in the best-executed refactor this codebase
has done. **Decomposition without a forward config value does not prevent regrowth; it relocates the
buffer one level down.** This is why `Codelet` is sequenced at M5, **before** any decomposition — the
single most important ordering decision in the plan. The same evidence rejects a module for `Trig` and one
for `Strided` (§9.2d): **data first, modules second.**

**C — Functors and type-level exotica: a `Make(Isa)` emitter, a `('layout,'dir) codelet` phantom, GADTs for
the family variant.** Rejected on measured grounds. **The ISA is a runtime CLI choice**, so `Emit.Make(Avx2)`
forces first-class modules at every dispatch — the exact inference pain the owner named as the nightmare —
and the real cost it addresses (181 in `emit_c`, **278** outside `isa.ml` across 6 modules) already has an in-tree cure:
`codelet_zsplit` is **87 `Isa.` calls : 0 literals**, `cx_render` 71:2, `codelet_cil` 57:1, because they
**extended `isa.ml`** instead of inlining. *Grow the record's vocabulary; do not turn the record into a
signature.* For layout, enforcement comes from `Layout.pointers` being **total on one plane**, not from a
type parameter; a `('layout,'dir) codelet` leaks type variables into every emit-chain signature for **zero**
additional safety and visibly damages error messages inside a 4,000-line function. **Keep exactly one
functor — `Schedule.Make`, three instantiations, a 6-method signature, zero reported inference pain, −39 %
reg-reg moves at r32 — and add none.** ⚠ Unopposed (§17 preamble). Also rejected on the same page: a
reader/state monad (one `~ctx` parameter is 15 edits and reads like OCaml), a `Config` singleton with a
mutable "current" (*the same disease with a type annotation*), **unifying `Ir` and `Cx_ir`** (~150–180
exhaustive-match arms across 7 modules — a **measurement**, not an opinion; re-litigating it is a trap), and
**splitting `emit_c.ml` by line count** (the 1,380-line spill region reads **2** globals; the 419-line
signature chain reads **13** — size and mess anti-correlate, so splitting by size extracts the healthy tissue
and leaves the disease).

---

## 18 · Where a reader looks — the one-place index

This table is the test of whether the design answered the brief.

| I want to understand… | open |
|---|---|
| what codelet kinds exist and what each one *is* | `gen/codelet.ml` |
| how in-place / OOP / strided c2c codelets are emitted | `gen/c2c_split.ml` (849 files) |
| how r2c, c2r, hc2c and the trig zoo are emitted | `gen/real.ml` (318) + `gen/real_math.ml` |
| how the full-IL family is emitted | `gen/c2c_il.ml` (233); its private compiler is library `vfft_cx` |
| how the z cascade is emitted | `gen/cascade_z.ml` (32) |
| **whether a signature is legal** | `kernel/layout.ml` — **the only** place the hybrid ban lives |
| what C signature a codelet has | `kernel/abi.ml` — **the only** place `__restrict__` is printed *(subject to OQ-3)* |
| how a transpose / deinterleave is emitted at width N | `kernel/simd.ml` — **the only** place `_mm*` lives outside `isa.ml` *(feature-blind ranges only until M8)* |
| which optimizer passes run, in what order | `kernel/pipeline.ml` — **the only** cascade |
| what knob does what | `kernel/knobs.ml` — 43 byte-affecting recipes + `Knobs.Trace`'s 12 diagnostic keys |
| what the corpus is and how a file is reproduced | `gen/corpus.ml` + `Codelet.to_argv` — **the same fact, once** |
| whether my change broke anything | `generator/gates/full_corpus_gate.sh verify` — 1,432 files, ~59 s |

---

## 19 · Scorecard against the seven sentences a proposal must satisfy

| # | requirement | this architecture |
|---:|---|---|
| 1 | the compiler has no word for what it compiles — **fix that FIRST** | `Codelet.t` at **M5**, before any decomposition (M6–M8). Measured out of the corpus: 40 kinds + a modifier matrix, 1,199/1,199. ⚠ the sum-vs-product shape is **OQ-2**, settled by X2 before M5. |
| 2 | the globals are **backward** edges; the fix is a config record flowing **FORWARD** | `Codelet.t` → `Abi.t` → `Render.ctx`, strictly downward. All measured back-edges are eliminated by construction — **at least eight, five of them from family modules; see §21 G10** —; the two remaining upward pulls (`Dft_select.target_vec_regs`, `Schedule.order_source`) become parameters. |
| 3 | size is a symptom, **flag density is the disease** | ranked by density: the 53-line addressing block (7 globals) is attacked at M4; the ~900-line spill path (2 globals) merely **relocates** at M8 and is explicitly **not** promised to shrink. |
| 4 | "more modules" already regrew the disease here and cost **+28 %** | **+4 modules**, cx becomes one name via a third library, the +28 % tax is quoted not hidden, and regrowth is prevented by **T1 (a library boundary)** rather than by discipline — with M12a finally applying the state fix inside cx. |
| 5 | ~467 compiler-checked renames stand between here and the first writable `.mli` | M1 pays them first and alone, as **seven one-word edits + compiler-enumerated fallout** — ⚠ under `-w +44+45`, with `topo_sort_reachable` pre-qualified, because the silent-shadowing hazard is real. |
| 6 | the bar is met at **97.97 %**; the job is **DO NOT REGRESS** | every step gated on all 1,432. The five that cannot be fully gated — M3 (zero-representative arms), M10 (12 tracked registry headers, cross-tree), M11 (non-default env), M12b (**byte-changing by arithmetic**), and the out-of-scope math seam — are **named with their reasons and their compensating controls**, and M12b is re-classified as a baseline regeneration rather than sold as ✅. |
| 7 | supply a landing spot **cheaper than one more global**, or the loop continues | **one variant arm + one record literal**, inside one feature module — the `zs_kind` pattern generalised. Plus the `emit_*_verbatim` hosting rule so an absorbed hand kernel needs **no flag at all**. This is the sentence the whole design is built to satisfy, because `cx_ir.ml:99,116,139` each justify a new global with the verbatim phrase *"Default false keeps every existing kernel byte-identical"* — **the bit-identity requirement itself selects for one-more-global.** |

---

## 20 · The one-line verdict

**Feature modules over a feature-blind kernel, with the descriptor built before the decomposition and every
step gated against all 1,432 corpus files: +4 modules, 66 shared mutable cells down to 4, the god functions
down 62 %, the anti-hybrid law a compile error — and a pre-authorised stop after M7 if the lexical cut set
says the last step is too big.**

---

## 21 · Completeness audit (independent pass, 2026-08-14)

*An independent reviewer re-measured this document against the source tree with fresh scripts and ran the
gate. This section records what survived, what was corrected in place above, and what is still missing.
Corrections applied above are marked ⚠ at their site.*

### 21.1 🟢 Re-verified independently — no change needed

**The headline was re-executed, not read.** `bash gates/full_corpus_gate.sh verify` in WSL:
**`GATE PASS`, exit 0, 61.99 s, corpus files 1432, corpus drift 0, BYTE-IDENTICAL 1403 (97.97 %)**,
tally `1403 IDENTICAL / 17 BODY_DIFFERS / 11 PROLOGUE_ONLY / 1 PROLOGUE+EOF_NL`, and the 16-line
per-directory block reproduced §1.5 exactly (`rfft/avx2 64/65`, `rfft/avx512_regen 0/16`,
`zil/…/pure_il 221/227`, `tangent 0/6`, everything else 100 %). **The acceptance criterion is real and the
bar is genuinely met.** ⚠ *Note for future agents: the campaign brief's assertion that "nothing in this
campaign was compiled" is FALSE — `docs/research/generator_arch/scratch/e1…e7` are real dune projects with
populated `_build/` trees, and the gate builds the real library.*

Also re-measured and **exact**: 35 `.ml` / 30,776 lines / **0** `.mli`; 34 names in `lib/dune:26` with
`number.ml` on disk-but-absent and `simd_ir.ml` listed-but-a-stub; 1,432 `.c` in 16 leaf directories with
the per-directory counts as printed; **66** process-global cells (59 `ref` + 7 containers) with the exact
per-module split 34/10/7/4/3/4/2/1/1 — including `ir.ml`'s fourth cell, `of_expr_memo` at `:242`, which a
naive `ref` grep misses; **32** refs in `emit_state.ml`, names as listed; **133** CLI flags; **55** env keys
at **91** sites in **20** of 34 modules, and all 12 named `Trace` keys are in that set; **7** `include`
chain links at the exact cited lines plus the functor include at `schedule.ml:1179`; **4** top-level defs in
`emit_c.ml` at `:60/:4126/:4145/:4167` and **1** in `gen_main.ml` at `:52`; **6** blank lines, **478**
`Buffer.add_string`, **241** `Printf.sprintf` in `emit_c.ml`; **123** `failwith` in `lib/`; **131**
`__restrict__` lines split 94/22/9/6 across exactly four modules; **15** `render_node_def` call sites
(8/5/2 — recon 07's "17" was the wrong one, this document has it right); `Algsimp.` 359 refs with
**14–16** own, `Emit_c.` 77 with **2** own and 48/27 from `Emit_state`/`Emit_render`, `Dft.` 139 with
~61–65 % own, and **0** `Emit_state.`/`Emit_render.` references in code — so the 467-rename figure
reconciles (343+75+49); **10** modules with an external consumer and **19** executables; **91** chain-tail
refs in consumer files; the 13-arm signature ladder at all thirteen cited lines, read one by one; the
18-term `hoist_consts_enabled` disjunction at `gen_main.ml:734-752`; every one of §13's eleven
`strided_r2c` read sites in `emit_c.ml`; **13** `(mode promote)` rules in `generated/dune` of which the
**12** registry headers are the ones fed by the six `Coverage`-reading emitters (both figures are correct
in their own context); `regen_codelets.sh:13`, `bootstrap.sh:113`, `.gitignore:96`, `CMakeLists.txt:180`
(`foreach(_fam inplace rfft c2r oop strided il trig)` — `il` does not exist, `zil` is never globbed),
`emit_ship.sh:6-9`, `regen_cil.sh:14-16` — all quoted **verbatim-correct**; and the stale constant at
`rfft/avx2/radix256_r2c_term_ls_r8_avx2.c:36` (`0.70710678118655002`) against its already-correct AVX-512
twin (`0.70710678118654757`). The corpus partition is exact: 849 + 318 + 233 + 32 = 1,432.

**This document does not have the fabricated-tight-number problem. Treat its measurements as sound.**

### 21.2 🔴 Gaps found — ranked

**G1 — This file is UNTRACKED, and that is the same defect it flags as M0/G1 for the gate.**
`git status --porcelain` returns exactly one line: `?? docs/roadmap/generator_lib_architecture.md`. It is
not gitignored (`git check-ignore` rc=1) — it has simply never been `git add`ed. **The document of record
for a multi-week restructure is one `git clean -xfd` from gone.** `git add` it before anything else.

**G2 — At least one of the 32 `emit_state` cells has no home in the design: `strided_ilo_nt`.** §11.4 says
all 22 config cells become **fields of `Abi.t`**. `Abi.t` as sketched in §11.2 is
`{ symbol; target; buffers; twiddles; loop; extra }` — six fields, none of which can carry
`strided_ilo_nt`, whose entire job is to pick a **store instruction**: `emit_c.ml:3198`
`let stfn = if !strided_ilo_nt then "_mm256_stream_pd" else "_mm256_storeu_pd"`, the same at `:3517` for
AVX-512, plus an `_mm_sfence()` at `:4086`. That is a `Simd` concern (or a `Codelet` modifier), **not a
signature fact** — and it is also in the symbol (`gen_main.ml:1628`), so it touches `Codelet.symbol` too.
**Action: the ledger needs a per-cell column, not a per-module count.** Until it has one, "nothing is left
as an unexplained remainder" is a claim about arithmetic, not about the 66 cells.

**G3 — Two of the "config" cells are in the wrong bucket, and the split is therefore NOT purely a
measurement.** §11.1 calls the 22/10 split *"a measurement, not a design choice."* Two counter-examples:
`current_store_on_compute` has **exactly one writer** (`gen_main.ml:1962`) and three pure reads in
`emit_c` — config by this document's own stated criterion — yet §11.2 puts it in `Render.Scratch` as a
`mutable` field. `emit_render.ml:520 hoist_consts_enabled` is written by the **18-term derived predicate**
at `gen_main.ml:734-752` and is likewise pure config, yet §11.4 sends all four `emit_render` cells to
`Scratch`. **Neither misplacement is fatal, but the sentence "measurement, not a design choice" must be
softened, and the 22/9/3 line above must be checked cell by cell before M6.**

**G4 — `provenance_argv` is unassigned, and under M9 it becomes a library-boundary violation.**
`emit_render.ml:1362 let provenance_argv : string array option ref` is set once by
`gen_main.ml:53 Emit_c.provenance_argv := Some argv` and read by `provenance_block` at `:1406`. §11.4
disposes of it as one of "`emit_render` 4 → `Render.Scratch`". But §12.2 promises *"provenance == coverage
== regen recipe, one fact"* via `Codelet.to_argv`, and **`Codelet` is in `vfft_gen` while `Render` is in
`vfft_kernel`** — so a kernel module printing the provenance block needs a gen value, which is exactly the
`Unbound module "Vfft_gen"` that T1 is designed to produce. **The fix is easy** (the feature module passes
`~argv:(Codelet.to_argv c)` into `Render.provenance_block`; "argv" names no family, so T3 is satisfied),
**but the document never says it, and it is the same class of defect as OQ-3's `Abi.prologue`.** Add it as
OQ-3b and settle it with the same 2-hour experiment.

**G5 — The largest module after the restructure is never estimated.** Line estimates are given for
`Layout` (<150), `Abi` (~300), `Codelet` (450–1,000, with an explicit ~800 red line) and `Real` (~900).
**`C2c_split` — 849 files, 59.3 % of the corpus, absorbing all 2,540 lines of `codelet_oop.ml` plus
`emit_c`'s in-place and strided share — gets none.** For a document whose thesis is that a 4,167-line file
is the display case, silence on a plausibly 2,000–3,500-line successor is the one number the owner will
ask for first. §9.1 reports "largest **function** after"; it needs a "largest **module** after" row.
**Add it to X3's deliverables** — the M8 probe has to bound this anyway.

**G6 — `Codelet_oop.emit_body_spill` (`codelet_oop.ml:1367`, 578 lines) is a second body/spill emitter
that the design never disposes of.** recon/03 measured **five** god-functions totalling 8,834 lines; §1.1
lists **four** totalling 8,267 and drops this one without comment. §9 sources `Emit_body` solely from
`emit_c:2099-3183`. So after M8, `C2c_split` still contains a private 578-line spill emitter duplicating
the kernel's `Emit_body`, and §15's honest-costs list does not mention it. **Either merge it into
`Emit_body` (and say so, with its own gate step) or record it in §15 as surviving duplication.**

**G7 — `Emit_body`'s `.mli` exemption contradicts the `.mli` rule.** §10 exempts it as *"one consumer"*.
But `emit_c:2099-3183` today serves the **1,020** `Emit_c.emit_codelet` files, which the design splits
between `C2c_split` (702) and `Real` (318) — **two consumers, on a library-internal layer boundary**, so
this document's own rule (*"more than one consumer **or** a layer boundary"*) says it gets one. The `.mli`
roster is therefore **~29 of 38**, not ~28, and the exemption list is nine, not ten.

**G8 — A layer inversion inside the kernel that §9 does not disclose.** `pipeline.ml:99` declares
`spill_info : Emit_c.spill_info option` and `:306` calls `Emit_c.make_spill_info`; both names are defined
at **`emit_render.ml:897`/`:906`**, i.e. in the future `Render`. So `Pipeline` (kernel, labelled **L1**)
depends on `Render` (kernel, **L3**). §9's honest note lists four intra-L3 exceptions and calls the rest
decorative; this one is a **cross-layer** inversion and it is not among them. It will surface the moment
`pipeline.mli` is written at **M2**. **Cheapest fix: move `spill_info` + `make_spill_info` down to a level
that both can see (they are data about spill markers, and `Algsimp.spill_tag_marker` is their input) —
a one-type move, and it belongs in M2, not M8.**

**G9 — `type kind` and `type modifiers` are declared ABSTRACT in `codelet.mli` (§10.2), which kills the
one law they are supposed to enforce.** §12.2 makes *"every family has a corpus recipe"* a compile error
because `recipes : kind -> Codelet.t list` is a **total match** policed by warning 8. A match is only
possible if the constructors are **visible**. With `type kind` abstract, `gen/real.ml` cannot pattern-match
at all, warning 8 never fires, and §13 edit #1 ("the `strided` kind gains an `r2c` modifier") has nothing
to mirror in the `.mli` (edit #2). **The `.mli` must expose `kind`'s arms and `modifiers`' fields
concretely** — keeping only `t` `private` and `sense` abstract-with-a-smart-constructor. That is a
one-line change to the sketch, but it is the difference between a law and a comment.

**G10 — the "five measured back-edges" is a floor, not a census.** A stripped `X.y := ` scan over the five
driver/family modules finds writes into upstream state that §2 RC3 does not list, including
`Emit_c.provenance_argv` and `Emit_c.hoist_consts_enabled` (both → `Emit_render`), `Cx_math.tangent`
(← `gen_main:524`, the one-way latch §11.4 names but does not count as an edge), and the six
`Emit_c.current_ls_mode` writes from `Codelet_oop`. §19 row 2's *"All five measured back-edges are
eliminated by construction"* should read *"all measured back-edges — at least eight, five of them from
family modules"*. The remedy is unchanged; only the completeness claim is.

**G11 — `(implicit_transitive_deps false)` + a third library is asserted, not tested.** §9.2(b) says of the
replacement for `(private_modules)`: *"That works, and it is the only reason the 'cx becomes one name' row
survives."* The refutation of `(private_modules)` was reproduced (`scratch/e3`); the **three-library**
configuration was not. It is the load-bearing mechanism for one of §9.1's surviving rows and for T1.
**Make it experiment X5 — it is ~30 minutes and sits beside X1.**

**G12 — incidental: `src/core/oop/zturn_proto.h` does not exist.** §7.11 quotes
`codelet_zsplit.ml:1558`'s claim that `emit_s0t_body` is *"a VERBATIM transcription of the proven
prototype (`src/core/oop/zturn_proto.h`)"*. The quote is faithful, but the file is **gone from the repo**
(referenced from `zturn.h:48,555`, `zsplit.h:61` and `src/core/README.md:107`, which calls it
*"permanent"*). The precedent for §15 cost 14's `emit_*_verbatim` hosting rule therefore has **no surviving
source of truth** — which strengthens the case for the rule and should be stated when it is cited.

### 21.3 Independent verdict on the owner's two halves

**Half 1 — "each feature has its own module": PASS, with G5 attached.** §18's index is a real answer, and
the corpus partition is exact and complete. The caveat is that the biggest answer, `gen/c2c_split.ml`, is
itself an 849-file, plausibly-3,000-line module. The owner's specific complaint — *"both split and
interleaved layout solutions, r2c/c2r solutions, all lumped in one file"* — **is materially answered**:
IL leaves for `C2c_il`, r2c/c2r/trig for `Real`, and the 4,167-line file the owner pointed at ceases to
exist.

**Half 2 — "no module explosion, no sprawl of shared values": PASS on modules, PARTIAL on values, and the
document says so itself.** +4 modules over 1,432 files, with `Trig` and `Strided` explicitly **refused**
modules, is a genuinely disciplined answer and the `zs_kind` precedent proves it works in-tree. On values:
66 mutable cells → 4 is the strongest result in the design and it dissolves eight recorded defects. But
§9.1's corrected row is the honest one — **total named shared surface 254 → ~200, −21 %**, because 133 CLI
flags and 55 env keys survive one-for-one, and four new records add 26 fields of which **7 are redundant
copies of 4 facts**. Read the owner's phrase as *"there is no shared config **type**"* and the design
answers it completely; read it as *"there are too many shared names"* and it answers a fifth of it.
**State that distinction to the owner explicitly — it is the one place a reader could feel over-sold.**

### 21.4 What this pass did NOT check

The M8 lexical cut set (R1/X3 — the biggest open risk, unmeasured by anyone); whether `Codelet.kind × modifiers`
can type all 83 argv shapes (X2); the 221 recovered `pure_il` recipes' totality (OQ-10); cross-host
determinism (OQ-11); `expr.ml:289-369` liveness (OQ-12, still needs deletion-and-build); the 46–47
unstamped env vars' effect on emitted bytes; and the C-side blast radius of M10.

---

## 22 · Pre-work results (X1–X6, executed 2026-08-14)

*All six §14.4 experiments ran the day M0 landed. Artifacts: `docs/research/generator_arch/scratch/`
(`x1_x5/` scratch dune project, `x2.ml` + `x2_shapes.tsv`, `x3.ml`, `x6_cell_ledger.tsv`).*

**X1 — `layout.mli` + `abi.mli` built for real: the module count is 38.** ✅ Layout owning `param`
resolves the §10.1 cycle; `Abi` builds against `Isa` + `Layout` with both `.mli`s enforced; the `z11`
value renders the frozen 11-arg ABI (verified against `codelet_cil.ml:866-873`) with derived
silencers. One finding the sketches omitted: under **wrapped libraries** a feature module writes
`open Vfft_kernel` (or qualifies) — one open, same discipline as the cx family.

**X5 — the three-library mechanism WORKS (G11 settled).** ✅ With `vfft_cx` a separate library and
`(implicit_transitive_deps false)`, an executable linking only `vfft_gen` gets `Unbound module
"Vfft_cx"` (rc=1) while `vfft_gen` uses cx internally. Stronger than needed: the app cannot resolve
`Vfft_kernel` either (reproduced accidentally by this probe's own harness). **And OQ-1's exploit is
dead both ways:** the hand-rolled hybrid `param` literal → `Cannot create values of the private type`
(compile error); `Layout.scalar ~ctype:"const double *"` → `Invalid_argument` with the positive path
intact. Remaining OQ-1 half: the 23-ABI parameter census (data-plane/twiddle/scalar) — not yet run.

**X2 — OQ-2 settled: `kind` stays a SUM; `Codelet` is not a config record.** ✅ 212/212 normalized
corpus argv shapes (superset of the 83: 1,199 provenance headers + the 221 derive-arm recipes,
values abstracted) parse into: **14 top-level kind constructors** carrying 3 small enums (trig 9-way,
cil-form 3-way, zs-kind 19-way ⇒ 41 leaf selectors ≈ the measured 40), **global `modifiers` = 5
fields** (`dir`, `dif`, `table`, `t1s`, `su`) — far under the ~12 red line — and **family-scoped
modifiers as kind-payload records** (oop 7 fields, cil 3, zsplit 2, hc 1): the `zs_kind` precedent
generalized. The earlier pure-sum failure ("could not name the 32 `*_strided_r2c.c`") dissolves —
`Strided_r2c` is its own selector. Prototype `of_argv`: `scratch/x2/x2.ml`.

**X3 — the M8 GO/NO-GO: NO-GO as literally pre-committed, GO under one condition the plan already
mandates.** AST probe (compiler-libs, spine-level scoping, nested shadowing ignored ⇒ overcounts ⇒
conservative toward NO-GO; `buf` excluded; the six named closures ×3): **weighted 23–25 at every
candidate boundary ⇒ NO-GO** — but the entire excess is ONE coherent cluster: the spill-engine
closure family (`record_peak_live:119` … `emit_node_reload_sites:340`, plus `max_pass_peak` and a
**seventh** mutable-state closure the review's list missed, `install_alloc:242`) and the tail
machinery (`anyk_tail`/`tail_bound`/`tail_var`/`emit_v_loop_header`, `:419-435`). Modeling the spill
engine + its closures moving **WHOLE** into `Emit_body` — which §5.3 and the M8 row already mandate
("it moves whole; it does not decompose") — every boundary drops to **weighted 3–5: GO with wide
margin**, the residue being the tail machinery plus one or two locals. **Verdict: M8 is GO, with the
spill-cluster-moves-whole treatment promoted from recommendation to LOAD-BEARING CONDITION** (and
`install_alloc` added to its manifest). Raw-name counts (11–13) independently confirm the review's
lexical probe.

**X4 — OQ-6 settled: thread the recipes; no module needs the ambient exception.** ✅ Enclosing-def
census: `fma_passes` 14 getenv sites in 4 defs, **11 of 14 are `*_TRACE`** ⇒ after the `Trace` split
exactly **one** byte-affecting knob (`VFFT_FMA_MULTIUSE`) in one def; `schedule` 10 sites / 3 defs
(byte-affecting: SCHED_ORDER + LOADS/PACE/TIEBREAK/GH_THRESHOLD in `su_schedule{,_subset}`);
`simplify` 5 / 3 (COLLECT_M, DEEP_COLLECT). **Total: ~6–8 top-level defs gain a recipe parameter
across all three modules** — nowhere near the >30/module threshold. The nested-lambda sites the doc
worried about are all Trace keys.

**X6 — the 66-cell ledger exists, per cell (G2/G3 closed).** ✅ `scratch/x6_cell_ledger.tsv`: every
cell → a named home. Survivors: exactly the **4 `Ir` memos**. Corrections encoded: `strided_ilo_nt`
→ a `Codelet` store-mode modifier consumed by `Simd` (+ symbol), not an `Abi` fact (G2);
`current_store_on_compute` and `hoist_consts_enabled` → `Knobs.render_recipe` config, not Scratch
(G3) — so the Scratch set is 8, not ~10, and §11.4's per-module arithmetic holds with the corrected
buckets. Bonus: `cx_math.tangent`'s one-way-latch defect dies structurally with the per-emission ctx.

**Net effect on the plan:** every §14.4 gate is green — M1 can start, M3/M4 are de-risked (X1),
M5's descriptor shape is settled (X2), M8 is GO conditional on the spill-cluster rule (X3), M6's
ledger is cell-exact (X6), and `Knobs` threading is cheap (X4). The remaining pre-M3 item is the
**owner's R3 values question** (absorption velocity vs one-place answers), which decides whether
M8/M12 stay in scope at all.
