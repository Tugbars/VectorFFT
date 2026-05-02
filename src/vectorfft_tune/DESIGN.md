# VectorFFT tuning — design notes

This file documents decisions that a reader of the codebase would otherwise
have to reverse-engineer from the layout. It is not a tutorial; see README
for that. It is the place to put "why is this not symmetric?" answers so
they live in one place instead of scattered across commit messages.

## R=8 has more Phase-A variants than other radixes

Every radix except R=8 has exactly four Phase-A variants, one per
dispatcher:

    radixes/r{16,32,64}/gen_radix{N}.py :
        ct_t1_dit        -> t1_dit
        ct_t1_dif        -> t1_dif
        ct_t1s_dit       -> t1s_dit
        ct_t1_dit_log3   -> t1_dit_log3

R=8 has **eight** Phase-A variants, mapping to the same four dispatchers:

    radixes/r8/gen_radix8.py :
        ct_t1_dit           -> t1_dit      (baseline)
        ct_t1_dit_prefetch  -> t1_dit      (SW prefetch on rio)
        ct_t1_dit_log1      -> t1_dit      (loads R-2 twiddles, derives W^(R-1))
        ct_t1_dit_u2        -> t1_dit      (U=2 software pipelining — AVX-512 only)
        ct_t1_dif           -> t1_dif      (baseline)
        ct_t1_dif_prefetch  -> t1_dif      (SW prefetch on rio)
        ct_t1_dit_log3      -> t1_dit_log3
        ct_t1s_dit          -> t1s_dit

### Why the extra variants at R=8

R=8 is a compute-bound radix on modern CPUs. The DFT-8 butterfly is small
enough (~52 arithmetic ops, critical path ~15 FMAs deep) that the
"obvious" memory optimizations — tiling, non-temporal stores, sparse
twiddle reads with derivation — often regress. Instead, the optimizations
that matter at R=8 are micro-level: software prefetch on the rio buffer,
squeezing one more derived twiddle out of R-2 loads, software pipelining
across two butterflies.

These are the "weird" optimizations: they compress the critical path
further when the baseline is already port-dense, and they tend to have
narrow win regions. Historically at R=8 some of them (`log1`, `u2`,
`prefetch`) have won small but real margins on specific microarchitectures
that the main Phase-A quartet missed.

On the two chips benchmarked in the initial report (Raptor Lake i9-14900KF
and a Sapphire Rapids cloud VM) the winners at R=8 were dominated by
`t1_dit` (flat baseline) on Raptor Lake and `t1_dit_log3` on SPR. The
other variants occupy narrower niches but did not dominate any region on
these specific chips.

They are kept in the tree because:

1. The bench infrastructure already emits and validates them for free.
   Removing them would forfeit a dimension of the sweep rather than
   simplifying anything important.
2. Our prior is that low-R variants matter more on narrower cores. The
   data we have so far says "not on these two chips"; new microarchitectures
   (Zen, Apple silicon, ARM Neoverse) may surface different winners.
3. At R=8 the generator is already ~2000 lines and the marginal cost of
   emitting four extra variants is negligible (< 100 lines of extra
   generator code paths).

### Why buf variants are skipped at R=8

Phase B (tiled/buffered codelets with tile × drain knobs) is present at
R=16, R=32, R=64 but not at R=8. This is deliberate. VTune analysis on
an earlier standalone version of the library showed that R=8 butterflies
are compute-bound, and the memory-centric optimizations that buf provides
(tile-outer iteration, drain-mode selection, non-temporal stores) regress
at that radix. Two measurements on two chips later showed the same thing
inferentially — flat DIT beats every memory-optimized variant on RL R=8
at 8 of 18 regions, exactly the regime where buf would theoretically
apply.

Adding buf at R=8 would cost ~2000 lines of generator code and gain
nothing demonstrable. It can be added later if contributor data suggests
otherwise.

## R=32 and R=64 have dead buf knobs

Both `gen_radix32.py` and `gen_radix64.py` have the function signature:

    emit_ct_file(isa, itw_set, ct_variant,
                 tile=None,
                 drain_mode='temporal',
                 drain_prefetch=False,          # <-- unused
                 twiddle_prefetch_distance=0,   # <-- unused
                 twiddle_prefetch_rows=0)       # <-- unused

The last three arguments are implemented in the emitter code paths but
are never exercised by the Phase-B bench. The candidates.py for R=32 and
R=64 enumerates only `(tile, drain_mode)` combinations, leaving the
prefetch knobs at their defaults of False/0.

These knobs exist because prior ad-hoc benches on the standalone
library hinted they might matter in specific regimes (drain_prefetch
helps when the store path straddles an L1D eviction boundary;
twiddle prefetch helps when log3 bases hit cold pages at large me).
Wiring them into the Phase-B sweep would 6× the candidate count at
each radix, which we judged not worthwhile before collecting baseline
Phase-A+B data across chips.

They are left in the emitter rather than removed because reviving them
is a candidates.py change only — no emitter work — and because the
if-branches they produce compile to nothing when the default values
propagate through. A reader seeing `if drain_prefetch:` should understand
this is a deferred feature, not orphaned code.

If the `drain_prefetch` or `twiddle_prefetch_*` knobs are benched in the
future, the candidates.py pattern to follow is:

    _BUF_PREFETCH = [False, True]
    for tile in _BUF_TILES:
        for drain in _BUF_DRAINS:
            for prefw in _BUF_PREFETCH:
                vid = f'ct_t1_buf_dit_tile{tile}_{drain}'
                if prefw: vid += '_prefw'
                ...

The function naming convention in the emitter already accounts for the
`_prefw` suffix, so only the candidates enumeration and the VARIANTS
dict need to be extended.

## Variant → dispatcher mapping is N:1

Each `ct_*` variant maps to exactly one dispatcher (the third tuple
element in VARIANTS). Multiple variants may map to the same dispatcher
— this is the core of the Phase-B design. At R=16, for example:

    ct_t1_dit                       -> t1_dit
    ct_t1_buf_dit_tile64_temporal   -> t1_buf_dit
    ct_t1_buf_dit_tile64_stream     -> t1_buf_dit
    ct_t1_buf_dit_tile128_temporal  -> t1_buf_dit
    ...

The dispatcher (emitted into `vfft_r16_t1_buf_dit_dispatch_avx2.h`) is
a static-inline wrapper that branches on (me, ios) and calls the
fastest variant from the bench. The consumer of the library sees one
function per dispatcher, not one function per variant.

Consequences:

- The planner never branches on codelet choice. All variants for a
  dispatcher compute the same mathematical function with the same
  twiddle-buffer layout. The planner calls `vfft_r16_t1_buf_dit_dispatch`
  once and does not care that six variants compete underneath.
- Adding a new variant is a generator + VARIANTS-dict change only. The
  planner, validator, and emit code do not need modifications.
- Bench results are tagged by variant (not by dispatcher). The emit phase
  consumes the variant-tagged measurements and decides which variant to
  call from the dispatcher at each (me, ios) region.

Exception: `ct_t1s_dit` has its own dispatcher (`t1s_dit`) even though it
could in principle share with `t1_dit`. They use different twiddle
layouts (t1s needs scalar broadcast table, flat variants need the
`(R-1)*me` table), so they cannot share a dispatcher that takes one
twiddle-table argument.

## Sparse log3 convention is unified across radixes

The log3 codelet family reads a small number of twiddle bases from the
same flat `(R-1)*me` buffer that the baseline DIT uses, and derives the
remaining twiddles via chained complex multiplies. The convention:

    R=4:  reads {0}*me                  (1 base; derives W^2, W^3)
    R=8:  reads {0,1,3}*me              (3 bases; derives 4 more)
    R=16: reads {0,1,3,7}*me            (4 bases; derives 11 more)
    R=32: reads {0,1,3,7,15}*me         (5 bases; derives 26 more)
    R=64: reads {0,7}*me                (2 bases; derives 61 more, FFTW-style)

The R=64 case is different because the R=64 FFTW kernel decomposes
differently — it uses a 2-level sub-FFT structure that needs fewer base
reads per butterfly at the cost of more cmul derivation. This
complication is absorbed into the radix-specific generator; the planner
and twiddle-table code see only "log3 uses the flat buffer, reads sparse
positions."

Unified buffer shape is important because it means the **planner never
branches on protocol choice either**. The twiddle table is flat (R-1)*me
doubles regardless of which variant the dispatcher picks.

Exception: `t1s` uses a different buffer layout (scalar broadcasts, one
set per K-block). The planner picks `t1s` vs `flat` at plan time based on
protocol choice. That is one top-level branch, not a per-region branch,
and it is reflected in the dispatcher split described above.

## Benchmarked-measurement storage and host fingerprint

Each `bench_out/r{N}/measurements.jsonl` is append-only. The bench is
restart-safe: if a run is interrupted, the already-measured entries are
recognized by their (variant, isa, protocol, radix, me, ios, direction)
key and skipped on restart.

A `host_fingerprint.json` sidecar file is written the first time the
run phase executes in a given `bench_out/r{N}/` directory. On subsequent
runs, the current host is compared to the stored fingerprint. If the
os + cpu_model + machine triple differs, a warning is printed but the
run continues — this protects contributors who copy a tree between
machines without realizing, without forcing them to start over if they
know what they are doing.

Hostname is deliberately excluded from the fingerprint comparison
because container restarts often assign different hostnames to the same
physical silicon.

## Per-variant validation, not just per-dispatcher

`common/validate.c` registers one validator case per variant (not only
per dispatcher). The reason: if the dispatcher happens to pick variant
A at the test `me` values but has a bug in variant B (picked at other
`me` values), a dispatcher-level test will pass and the bug will ship.

At R=16, R=32, R=64 Phase B the cost of this choice is 12 extra
validator cases per ISA (6 variants × 2 directions) plus the dispatcher
case itself. Total validation time overhead: about 5 seconds per radix.
It has caught no bugs so far but is a cheap insurance policy against
the specific class of errors it is designed for.

## Why the pipeline is phased (generate → compile → run → emit → validate)

Each phase writes artifacts to disk and reads only artifacts from
previous phases. There is no in-memory state carried between phases.
Consequences:

- Any phase can be re-run independently. The bench is re-entrant.
- If `compile` crashes (e.g. ICX can't find a header), the `generate`
  phase output is preserved and the user fixes the compile issue
  without regenerating.
- If `run` crashes partway through, the measurements already written
  to `measurements.jsonl` are read back in on restart and only the
  missing (variant, me, ios) combinations are re-benched.
- If the winner-selection logic in `emit` has a bug, we fix `emit`
  alone without touching the measurements.

The cost of this design is filesystem traffic: every measurement and
every generated file lives on disk. In practice this is a few MB per
radix per run, which is trivial relative to the benefit of having
each phase debuggable in isolation.
