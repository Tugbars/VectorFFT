# prototype-core — parallel 1D C2C executor over the OCaml DAG-FFT machinery

**Status:** Phase 0 skeleton. Empty header placeholders only; no functional
code yet. Phases 1-5 progressively wire end-to-end execution.

## What this is

A parallel implementation of the FFT executor sitting one tier above
the prototype's auto-generated codelets / registry / plan executors.
It exists so we can develop and validate the prototype DAG-compiler
pipeline against a complete executor pipeline **without touching
production code** at `src/core/`.

Eventual goal: when prototype-core proves itself across the wisdom
portfolio, the `src/core/` directory gets deleted and prototype-core
moves into its place. Until then, the two trees coexist by file
separation — production keeps shipping while prototype-core matures.

## Scope (deliberately narrow)

**IN — 1D C2C, in-place, single-threaded:**
- Factorizable N (uses radixes {2..512} from the prototype codelet tree)
- Both DIT and DIF orientations (DIT is the default; DIF deferred)
- Wisdom-driven plan selection where available
- Estimate-mode (cost-model-driven) plan selection for cold cells
- The (B)+(A) plan-shaped executor fast path via `plan_executors.h`
- Generic per-stage loop fallback for cells not in plan_executors

**OUT (production keeps handling, prototype-core defers):**
- Bluestein (chirp-z for arbitrary / prime N) — non-staged plan path
- Rader (prime sizes via convolution) — non-staged plan path
- R2C / C2R (Hermitian-packed real FFT) — different signature, different planner
- 2D FFT (tiled + Bailey)
- DCT-II / DCT-III / DCT-IV / DST-II / DST-III / DHT
- K-split / group-parallel threading
- Blocked executor for N>512 (optimization on top of standard execution)

These are not architectural gaps — each is a separate workstream that
gets a prototype-core port after 1D C2C is solid.

## Boundary with `src/core/`

| Rule | Why |
| --- | --- |
| `src/core/` is **never modified** | Production keeps building. prototype-core is parallel; we never break the working production code path. |
| `src/prototype-core/` **does not include from `src/core/`** | No `#include "../core/..."` lines. Decouples the two trees so renames and edits on either side don't cascade. |
| Both directories define `stride_plan_t` / `stride_stage_t` independently | Types are conceptually the same but defined twice. When we eventually unify, prototype-core's types become the canonical ones. |
| Codelet inventory is shared via `src/prototype/codelets/{isa}/` | One codelet tree, two executor consumers. prototype-core uses prototype codelets exclusively (via `src/prototype/generated/registry.h`). |

## File map (Phase 0 — all placeholders)

```
src/prototype-core/
├── README.md              # this file
├── plan.h                 # stride_plan_t / stride_stage_t types          [Phase 1]
├── twiddle.h              # plan_compute_twiddles_c (per-stage tables)    [Phase 2]
├── planner.h              # stride_auto_plan / estimate / wise            [Phase 3]
├── executor.h             # lookup → specialized OR generic dispatch      [Phase 1]
├── executor_generic.h     # per-stage loop fallback for cold cells        [Phase 1]
├── wisdom_reader.h        # parse build_tuned/vfft_wisdom_tuned.txt        [Phase 3]
└── compat.h               # n1 6↔7-arg adapter (if needed)                [Phase 4]
```

## Dependencies

prototype-core's `#include` web (when complete):

```
executor.h            ──┬── plan.h
                        ├── executor_generic.h  ──┬── plan.h
                        │                          └── ../prototype/generated/registry.h
                        └── ../prototype/generated/plan_executors.h

planner.h             ──┬── plan.h
                        ├── twiddle.h
                        ├── wisdom_reader.h
                        └── ../prototype/cost_model/factorizer.h
                              (already exists — cost model)

twiddle.h             ── plan.h
```

External (built dependencies):
- `src/prototype/codelets/{isa}/` — codelet `.c` files
- `src/prototype/generated/registry_{avx2,avx512}.h` — codelet inventory
- `src/prototype/generated/plan_executors.h` — specialized fast paths
- `src/prototype/cost_model/generated/radix_cpe.h` — measured CPE
- `src/prototype/cost_model/generated/radix_profile.h` — DAG op counts
- `src/prototype/cost_model/factorizer.h` — cost model

## Phase plan

| Phase | What lands | Effort | Cumulative coverage |
| --- | --- | --- | --- |
| **0** | Skeleton + this README + empty headers | 30 min | Marker; nothing executable |
| **1** | Minimal `executor.h` + hand-crafted plan for N=1024 K=128 | 2 hours | End-to-end on 1 cell |
| **2** | Port `twiddle.h` from production (1D C2C subset only) | 3-4 hours | Arbitrary factorable plans |
| **3** | Port `planner.h` (no Bluestein/Rader) + `wisdom_reader.h` | 3 hours | Wisdom-driven plan selection |
| **4** | n1 signature alignment (or `compat.h` adapter) | 1-2 hours | Drop-in compatibility checked |
| **5** | A/B bench vs production over wisdom portfolio | 2 hours | Validates win/parity |

Estimated total to a usable prototype-core 1D C2C executor: **~12-14 hours**.

## Why a separate directory at all?

Three reasons:

1. **Boundary discipline.** Without a hard separation, well-intentioned
   edits to `src/core/executor.h` would creep in. A separate tree makes
   "did you touch production?" a `git diff --name-only src/core/` check.

2. **Coexistence.** Prototype-core can be partially complete while
   production keeps serving everything. R2C, 2D, DCT, threading — all
   stay in `src/core/` until their respective ports land.

3. **Eventual replacement.** When prototype-core reaches feature
   parity for the slice it claims (1D C2C first), the cutover is
   "delete `src/core/`, rename `prototype-core/` to `core/`." No
   in-place rewrite.

## Not in scope (now or ever)

- **Multi-host wisdom blending.** Each host has its own wisdom; we
  don't try to combine measurements across machines.
- **Run-time ISA dispatch (fat binary).** Per-ISA binaries, compile-
  time selection. Production does the same.
- **Float precision.** Double-only. The codelet tree is double-only.
