# engine/ — design-lineage experiments (superseded)

These five small .c files are the development progression of the OOP
engine, kept as documented lineage. Nothing builds against them; the
PRODUCTION composition layer is core/oop_plan.h (the fused four-step
BAILEY2 executor in vfft_oop_execute_fwd) selected by core/oop_auto.h.

Progression, oldest first:
1. engine_natural_oop.c            64x16 work-buffer reference (two OOP moves)
2. engine_natural_oop_4stage.c     explicit four-stage variant
3. engine_natural_oop_inplace_twiddle.c  FFTW-method in-place-twiddle form
                                   (~1.17-1.20x over #1; killed the second
                                   shuffle and the work buffer)
4. engine_natural_oop_onecall.c    single-call fused form — the direct
                                   ancestor of BAILEY2
5. engine_natural_oop_onecall_spec.c  + stride-specialized codelets

History and measurements: docs/OOP_DESIGN.md and the lab notebook.
Safe to delete if lineage-by-git-history is preferred.
