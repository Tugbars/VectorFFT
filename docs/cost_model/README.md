# VectorFFT cost model

How `VFFT_ESTIMATE` decides which factorization to use, without measuring.

The cost model is the brain behind `vfft_plan_*(N, K, VFFT_ESTIMATE)` — a
closed-form, sub-millisecond planner that picks a competitive plan
(typically within 1.0–1.3× of wisdom-tuned plans) for any (N, K) the
library can factor.

This folder explains the architecture, the data sources, the math, and
how to regenerate everything from scratch.

## Reading order

1. [01_architecture.md](01_architecture.md) — components and data flow
2. [02_static_profile.md](02_static_profile.md) — what `extract.py` produces (op counts, register pressure)
3. [03_dynamic_cpe.md](03_dynamic_cpe.md) — what `measure_cpe.c` produces (per-butterfly cycles)
4. [04_factorizer.md](04_factorizer.md) — the cost formula and how plans get scored
5. [05_variant_selection.md](05_variant_selection.md) — how the cost model mirrors plan-build's choice between t1, t1s, and log3 codelets
6. [06_validation.md](06_validation.md) — bench methodology and current results
7. [07_regeneration_workflow.md](07_regeneration_workflow.md) — exact commands to rebuild everything

## Audiences

- **Library maintainer adding a new radix**: read 02 + 03 + 07 — those
  cover the data inputs to the cost model. New radixes need profile
  rows in both auto-generated headers.
- **Library maintainer porting to a new ISA / CPU**: read 03 + 06 —
  the CPE table is host-specific; you'll need to regenerate it on the
  target hardware.
- **User comparing ESTIMATE / MEASURE / EXHAUSTIVE modes**: read 04 +
  06 — those cover what the model does and how accurate it is.
- **Reviewer auditing the methodology**: read 03 + 04 + 05 — those are
  where the technical decisions live.

## Source of truth

Code, not docs:

- [src/core/factorizer.h](../../src/core/factorizer.h) — cost model
- [src/core/wisdom_bridge.h](../../src/core/wisdom_bridge.h) — variant predicates
- [src/core/generated/radix_profile.h](../../src/core/generated/radix_profile.h) — auto-generated, op counts
- [src/core/generated/radix_cpe.h](../../src/core/generated/radix_cpe.h) — auto-generated, cycle costs
- [tools/radix_profile/extract.py](../../tools/radix_profile/extract.py) — generator: static analysis
- [tools/radix_profile/measure_cpe.c](../../tools/radix_profile/measure_cpe.c) — generator: dynamic measurement

If a doc disagrees with the code, the code wins. Open an issue.
