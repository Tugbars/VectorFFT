# Cost-model + registry port to estimate mode — changes

Ports the old prototype's cost_model + scripts + registry emitter forward
onto the current generator and wires them into VectorFFT estimate mode.
Drops the static op-count profile (the compiler rewrites op counts at
codegen, so a per-radix op table predicts something it cannot predict),
and trims the radix set to the production range.

## Radix set (trimmed, three encodings kept in sync)

Production set: primes {2,3,5,7,11,13,17,19}, small/mid pow2 {4,8,16,32,64},
composites {6,10,12,20,25}. Dropped: 128, 256, 512, 1024 (experimental
assembly-analysis sizes; a single large-radix codelet loses to multi-stage
composition, measured ~1.57x slower than MKL's composed 128). 128 stays
re-addable, off by default. The set is encoded in three places, all edited
to agree:
  - bin/emit_registry_h.ml   standard_radixes
  - scripts/generate_codelets.sh   FAMILIES_ALL (large_pow2/xl_pow2 out of
    default, opt-in only) + STRIDED_SIZES trimmed to 16/32/64
  - cost_model/factorizer.h   _radix_has_codelet

## Registry (regenerated)

bin/emit_registry_h.ml added to bin/dune and run for both ISAs:
  generated/registry_avx2.h, generated/registry_avx512.h
Externs match the current generator's in-place codelets exactly, verified
across the full 18-variant matrix for R=8 (n1 fwd/bwd + t1/t1s x dit/dif x
flat/log3 x fwd/bwd) with zero diff. ABI is the in-place stride-executor
ABI (rio_re, rio_im, tw_re, tw_im, ios, me).

## Profile removal

  - lib/pipeline.ml, bin/gen_radix.ml: removed the dead transpose
    fixed-point loop and its op-counter (count_ops, has_cmul). The loop was
    gated `aggressive && not is_direct`, always false, so output is
    unchanged. Verified: generator builds, codelets still generate and
    compile.
  - cost_model/generated/radix_profile.h: DELETED. Its only surviving
    symbol, the array bound, moved to generated/radix_dims.h (value kept at
    1025 so the CPE/memboundness tables stay in-bounds; the trimmed registry
    is what gates selection).
  - cost_model/extract.py: DELETED (regex op-counter, obsolete with the
    profile gone).
  - cost_model/factorizer.h: added _radix_coarse_cpe (5*R*log2(R)/width),
    the c2c fallback and the out-of-scope strided/trig scorers now use it
    instead of the profile tables. The coarse prior is fallback-only;
    measured CPE in radix_cpe.h overwrites it the moment it is populated.
    To sharpen relative ordering later without re-introducing codegen
    coupling, swap the _radix_coarse_cpe body for a small fixed table of
    theoretical per-radix op counts.
  - measure_cpe.c: predicted_cyc now reports the same coarse prior (so the
    measured-vs-predicted diagnostic column still means something).
  - include swaps radix_profile.h -> radix_dims.h in radix_cpe.h,
    radix_memboundness.h, measure_cpe.c, measure_memboundness.c,
    score_and_time_plans.c.

## Verified on this machine

  - dune build clean; sample codelets generate and compile.
  - cost model compiles; rankings sane: N=128 -> [64,2] (never [128]),
    N=1024 -> small-radix multi-stage (never the [1024] monolith), trim
    enforced by the registry-gated picker.
  - measure_cpe.c and measure_memboundness.c compile after the removal.
  - estimate_plan.h compiles end-to-end against the trimmed cost model +
    registry + core headers (touches nothing that was removed).

## Deferred

  - Real CPE / memboundness measurement: this VM's timing CV is too high to
    trust (single vCPU, descheduling spikes). radix_cpe.h is the day-one
    placeholder; bank real numbers on the 14900KF / EPYC. measure_cpe.c and
    measure_memboundness.c pull externs from the regenerated registry, so
    they retarget automatically once the trimmed codelets are built on the
    host.
  - emit_executor_h (specialized plan executors): not on the estimate path;
    the generic executor (executor_generic.h) runs plans without it.
  - An AVX-512 memboundness table for EPYC (current tables are _avx2,
    Raptor-Lake calibrated; per-host regen needed).
  - On-demand CPE at plan time (lazy per-radix slotting into the existing
    "measured else fallback" structure in _radix_butterfly_cost).
