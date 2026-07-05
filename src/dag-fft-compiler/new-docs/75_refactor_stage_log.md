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

---

# REPO-TREE APPLICATION (src/dag-fft-compiler, branch dev/arbitraryTail)

The stages above were executed on a different snapshot of the code
(the zip-shipped tree); none of the extracted files existed in the
repo's generator/lib. Entries below apply the same doctrine to the
repo tree, with two owner directives added:

11. PARALLEL TREE (owner directive 2026-07-05): the refactor lives in
    `generator/new-lib` (dune library `vfft_v2r`) with gate drivers in
    `generator/new-bin` (clones of bin/gen_radix + bin/gen_set against
    Vfft_v2r). Production `lib/` (`vfft_v2`), `bin/` and `codelets/`
    are NOT touched. Promotion is a separate, owner-gated step:
    replace lib module sources with the new-lib set + the dune modules
    line (library name stays `vfft_v2`), delete new-lib/new-bin,
    re-run gates through bin/ and the registry emitters.
12. HEADER STANDARD (owner-ratified 2026-07-05): regalloc.ml's header
    essay is the documentation example the other modules follow.

## Gate battery for this tree (replaces reproduce.sh, which does not
exist here)

Chained behind `dune build` of SCOPED targets only — never bare
`dune build`: generated/dune has `mode promote` rules that regenerate
generated/*.h into the source tree (one bare build emptied
plan_executors.h; restored from git).

1. `dune build new-bin/gen_set.exe new-bin/gen_radix.exe`
   (DUNE_CACHE=disabled — the stale-cached-exe trap is real).
2. Knob-ON spot-checks, 8 cells, byte-compared against
   production-built baselines: VFFT_DEEP_COLLECT R13 avx2,
   VFFT_COLLECT_M R25 avx512, VFFT_SPLIT_RADIX R32 avx512,
   VFFT_PIN_FORCE R32 log3 avx512, VFFT_FORCE_FENCE R7 avx2,
   VFFT_GH_THRESHOLD=8 R64 t1 avx2 raptor_lake, oop R16 UG/UG
   two-buffer t1 avx2, t1_dif R20 bwd avx2.
   Provenance note: emitted headers embed argv[0], so baseline and
   candidate drivers are both invoked via one copied neutral exe path.
3. Full-tree identity: new-bin gen_set regenerates all quadrants
   (1074 files) into a scratch root; `diff -r` against the
   production-generated baseline tree must be empty; stray-tree sweep
   (find avx2/avx512 dirs outside codelets/) must be empty.
   Repo checkout is CRLF (core.autocrlf=true) while the generator
   writes LF — which is why the gate diffs generator output against
   generator output, never against the working tree.

## Stage R1 — ir.ml (DONE 2026-07-05)

new-lib/algsimp.ml lines 1-766 (through of_assignments) verbatim into
new-lib/ir.ml: node_kind / t, hashcons, tag counter, preds,
topo_sort_reachable, the mk_* recursion group, of_expr/of_assignments,
reset. algsimp.ml = `include Ir` facade + spill lifting + passes.
Purity proof: chunk concatenation byte-equals the original (cmp).
Gates green (build / 8-cell spot / 1074-file identity).

## Stage R2 — dft.ml layered split (DONE 2026-07-05)

Same three-layer cut as stage 2 above, boundaries re-derived on this
tree's content: dft_select.ml (algorithm type, enable predicates,
factor_override, pick_algorithm, needs_reassoc, target_vec_regs),
dft_recurse.ml (const_cmul + the dft..dft_ct `and` chain, one unit),
dft.ml facade (twiddle policy, cmul_pattern, dft_expand wrappers,
spill markers, should_spill / should_block_n1). Include-chained
Dft_select < Dft_recurse < Dft. `open Expr` removed from the select
layer (provably unused — warnings-as-errors enforced it, doctrine
rule 3). Gates green.

## Stage R3 — emit_c.ml state + render extraction (DONE 2026-07-05)

emit_c.ml -> emit_state.ml (the two mode-ref blocks: the
signature-flag wall current_tw_perpos..current_ls_mode, and the M3a
block current_regalloc..current_emit_position) + emit_render.ml (topo
sort, render_load / render_node_def, selective pinning, const
hoisting, inline set, spill_info machinery, metadata, provenance) +
emit_c.ml facade (emit_codelet + strided helpers). Chain
Emit_state < Emit_render < Emit_c. Feature-local refs (unpin
candidates, hoisting gate, provenance argv) stay beside their
features in render, as ratified above. Resolution-order subtlety
handled: the facade's `open Algsimp` precedes `include Emit_render`
so the render chain's topo_sort_reachable keeps shadowing Algsimp's,
as in the single file. emit_codelet (~2.5k lines) remains THE
remaining disease — its breakup is still the designed edit-heavy
future stage. Gates green.

## Stage R4 — algsimp pass decomposition (DONE 2026-07-05)

algsimp.ml passes split along its own banners: simplify.ml
(dedup_sub_pairs, collect_m / deep_collect, lift_sub_neg_mul,
factor_common_muls, factor_by_atom, share_subsums, transpose) and
fma_passes.ml (fma_lift, factor_const_muls, multi_use_fma_lift,
fma_addend_factor, flatten_fma_mul_addend). algsimp.ml = facade
(`include Fma_passes`, chain Ir < Simplify < Fma_passes) + spill
lifting + butterfly_share_mul + stats/printing. This tree has no dup
pass (doc 65's selective duplication is not in this snapshot), so
stage-4's dup.ml does not apply. Gates green.

## Stage RD1 — headers + module cards (DONE 2026-07-05)

Every compiled new-lib module carries a MODULE CARD (bb.ml's format:
ROLE / PIPELINE / PUBLIC SURFACE measured by grep with counts /
DEPS / ENV / GOTCHA) and a reader-oriented header to the regalloc.ml
standard. Stale headers corrected: emit_c.ml's "naive AVX-512 t1_dit
emitter" replaced with the real emit_codelet contract; gen_main.ml's
header no longer claims to be gen_radix.ml and documents run's five
phases; schedule.ml's header now covers the SU + GH half;
dft_r2c.ml's TODO list replaced with what the file actually hosts.
Measured surfaces from a scripted dependency matrix (grep counts,
comment mentions included). Facade-chain modules (Ir, Simplify,
Fma_passes, Dft_select, Dft_recurse, Emit_state, Emit_render) have
ZERO direct qualified references — the frozen-API goal, now measured.
Gates green after the doc pass (comment-only changes still fully
gated; the stage-3 lesson).

## Owner-decision queue (surfaced by this tree's audit; no action taken)

- lib/gen_main.ml.orig: a git-TRACKED stale backup of gen_main.ml.
  Deletion candidate.
- lib/number.ml: NOT in the dune modules list (uncompiled orphan) and
  opens with "Ported from FFTW" — the GPL-provenance concern already
  flagged at stage D1; here it is dead code as well. Deletion
  candidate pending the git-history scrub decision.
- lib/simd_ir.ml: 1-line placeholder, IS in the modules list (both
  trees). Delete or fill.
- DUPLICATED PASS CASCADE: pipeline.ml (used by codelet_oop) and
  gen_main.run carry the same hash-cons -> dedup -> FMA-cascade ->
  marker-remap sequence as two live copies; pipeline.ml's own header
  names the drift hazard. Unifying gen_main onto
  Pipeline.prepare_codelet is a SEMANTIC change (doctrine rule 1) —
  proposed as its own gated stage.
- generated/dune promote rules: a bare `dune build` regenerates
  generated/*.h in-place (plan_executors.h was emptied by one during
  this session and restored). Consider (mode promote) -> explicit
  alias, or document the scoped-build rule in scripts.
