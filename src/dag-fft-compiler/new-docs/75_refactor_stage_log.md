# 75. Module refactor — stage log and doctrine

Living document; one entry per completed stage. Motivation: 23,352
lines across 19 modules, zero .mli files, algsimp.ml as a
4,686-line god module (owned the IR type, hashcons, constructors,
AND every pass; depended on by 8 files, 77 references from
schedule.ml alone), gen_main.ml as 1,188 lines containing a single
`let`.

## Doctrine (binding for every stage)

1. MOTION ONLY per stage. A stage moves text; it never renames
   externally-used identifiers, never deduplicates, never deletes.
   Semantic deltas (e.g. the approved use-count dedup) are separate,
   separately-gated changes that land AFTER their destination module
   exists.
2. FACADE PATTERN: the old filename survives as `include <New>` plus
   re-exports, so every `Old.x` reference and `open Old` site
   compiles unchanged and module-level mutable state (hashcons,
   counters) stays physically single. External API is frozen.
3. COMMENTS TRAVEL WITH THEIR CODE — and so does CONTEXT: `open`
   statements and other preamble the extracted region depends on must
   be replicated into the new module (and removed again where a layer
   provably does not use them; warnings-as-errors enforces this). A comment block documenting a
   declaration moves iff the declaration moves (stage 1 caught and
   fixed an orphaned block; treat adjacency as part of the gate).
4. GATES PER STAGE (amended after stage 2): the gate battery MUST be
   chained behind build success (`dune build && { gates }`) — a failed
   build leaves the previous binary in _build and every gate runs
   green against stale code (this bit twice in one stage before the
   chaining was made mandatory). Battery: dune build clean; byte-identity 10/10 (all
   knobs off); reproduce.sh 11/11; at least two semantic
   spot-checks that exercise the moved code (stage 1 used dup R=13
   377/31 and dup+affinity R=17 693/108); CHANGES entry; patch/zip
   refresh.
5. DOCUMENTATION CONVENTION: docs cite FILES and SYMBOLS, never
   line numbers (line citations rot under motion; the 2026-07-02
   audit confirmed the existing docs/ tree already follows this).
   Historical ledger entries are never retro-edited — they were
   correct as of their date.
6. .mli interfaces come AFTER moves stabilize, starting with ir.mli.
10. HEADERS ARE FOR THE READER, NOT THE ARCHAEOLOGIST (owner
   directive): file headers and module cards describe what the
   module is and how it fits — never when it was refactored, what it
   was extracted from, or which stage moved it. That provenance
   lives here and in CHANGES. Dated MEASUREMENT verdicts and REMOVED
   tombstones are content, not narration — they stay. All headers
   were de-narrated accordingly.

## Stage 1 — ir.ml (DONE 2026-07-02)

Extracted algsimp.ml lines 1-785 verbatim into lib/ir.ml:
node_kind / t, the hashcons (the CSE mechanism itself), tag counter,
preds, and the mk_* smart-constructor recursion group through
of_assignments. algsimp.ml (3,925) = facade + passes + spill-marker
machinery + BSM + dup. dune modules += ir. The spill-marker essay
comment was reunited with `type spill_tag_marker` on the algsimp
side. All gates green (build / identity 10/10 / reproduce 11/11 /
both dup spot-checks exact).

## Stage D1 — module cards (DONE 2026-07-02)

Every lib/*.ml (20 files) now carries a standardized MODULE CARD
inside its leading comment: ROLE, PIPELINE position, PUBLIC SURFACE
(measured — which functions other files actually call, with caller
counts), DEPS/USED-BY (reference counts from the dependency matrix),
ENV vars read (grepped, matching REPRODUCE.md Part B), GOTCHAs
(session-verified only — e.g. schedule.ml's sink double-append),
and DOCS pointers. `grep "MODULE CARD" lib/*.ml` lists the set.
Existing header prose preserved; two demonstrably wrong lines
corrected (gen_main.ml header said "gen_radix.ml"; emit_c.ml's
"naive AVX-512 t1_dit emitter" line is flagged as historical by its
card). Surfaced for OWNER DECISION: number.ml opens with "Ported
from FFTW" (GPL provenance — belongs with the git-history scrub
item); simd_ir.ml is an empty placeholder (deletion candidate, dune
comment lists it aspirationally). Scope: lib/ complete; bin/ and
tools/ headers exist but are not carded (follow-up on request).
Gates: build clean, identity 10/10, reproduce.sh 11/11.

## Stage 2 — dft.ml layered split (DONE 2026-07-02)

Two-cut, three-layer split along the file's own section banners,
include-chained two deep (Dft_select ⊂ Dft_recurse ⊂ Dft facade) so
every `Dft.X` reference compiles unchanged:
- `dft_select.ml` (~290): algorithm type, enable predicates,
  factor_override, pick_algorithm, needs_reassoc, target_vec_regs.
- `dft_recurse.ml` (~960): const_cmul + the atomic mutual-recursion
  group dft / dft_direct / conjugate-pair / winograd 5,25,7 / dft_ct
  (one `and` chain — moves as a unit or not at all).
- `dft.ml` facade (~790): twiddle policy, assignment-list wrappers,
  spill planning (should_spill / should_block_n1 stay here — the
  blocked-recipe predicates gen_main queries).
Both new files carry MODULE CARDs. Gates green on a fresh binary
(build-chained): identity 10/10, dup R13 377/31, dup+affinity R17
693/108, reproduce.sh 11/11.

## Stage 3 — emit_c.ml: state + render extraction (DONE 2026-07-02, after one full revert)

emit_c.ml (3,799) → emit_state.ml (150: the two contiguous
content-verified ref blocks — the signature-flag wall and the
M3a→emit_position block) + emit_render.ml (1,132: preamble opens,
topo sort, renderers, selective pinning with its essay INTACT, const
hoisting, inlining set, spill configuration, metadata, provenance) +
emit_c.ml facade (2,575 ≈ 60 lines of chain + emit_codelet).
Include chain: Emit_state < Emit_render < Emit_c; every Emit_c.X and
`open Emit_c` unchanged. Three feature-local refs (unpin candidates,
hoisting gate, provenance_argv) deliberately stay beside their
features in emit_render — equally shared via the chain; the state
card says so. THE REMAINING DISEASE is named: emit_codelet is a
single 2,503-line function; its internal breakup is the designed
edit-heavy future stage.

HONEST RECORD: the first attempt at this stage FAILED and was fully
reverted. A heuristic comment back-scanner split two banner essays
across files, my own card prose contained a literal `*` `)` sequence
(an OCaml comment terminator), and one fix deleted a fragment without
rehoming it — text loss, a motion-only violation. The certified
state was restored from the last shipped zip (the revert path works
and was exercised), and the redo used content-asserted explicit
ranges with a per-line purity scan. Gates on the redo: identity
10/10, dup R13 377/31, dup+affinity R17 693/108, reproduce 11/11.

Doctrine amendments purchased by the failure:
7. NO heuristic comment scanning. Extractions use explicit,
   content-asserted line ranges with a purity scan (every line in a
   moved range must be provably comment / blank / target decl);
   anything fuzzier gets a human-verified boundary first.
8. Card and comment PROSE must be scanned for the OCaml comment
   terminator sequence before writing (asterisk-paren inside prose
   ends the comment).
9. The previous shipped zip is the rollback point; it is not
   overwritten until the current stage's full gate battery passes.

## Deliberately NOT split (OWNER-RATIFIED 2026-07-02)

- codelet_oop.ml (1,575): long but single-concern — one codelet
  family, types → emit primitives → prepare → bodies → driver.
- dft_r2c.ml (1,541): homogeneous math dump, one transform family
  per banner section; the banners already provide navigation.

## Planned stages (owner-approved direction; each gated separately)

| stage | extract | contents | unblocks |
|---|---|---|---|
| 3 | ir_walk.ml | shared use-count / topo / anchor walks | the approved ×6 dedup delta |
| 4 | simplify.ml, bsm.ml, dup.ml | algsimp pass decomposition | map_dag unification |
| 5 | cli.ml, recipe.ml | gen_main single-let breakup — EDIT-HEAVY (parameter threading, not pure motion; needs its own design) | navigable driver |
| 6 | emit_state.ml, then c_syntax / spill_recipe / sched_wisdom | emit_c decomposition — mode-state refs are SCATTERED through the renderers (lines 93-396), so a state-collection pass precedes any prefix cut | wisdom plumbing in one place |
| 7 | ir.mli first interfaces | freeze the node API | accretion prevention |
