# VectorFFT codelet bench + dispatch infrastructure

Radix-generic ATLAS-style autotuning for VectorFFT codelets. Per-chip
calibration emits dispatcher headers that the registry picks up.

Currently ported: **R=4, R=8**.

## What this does

For a given radix R, measures every codelet variant across a sweep of
`(me, ios)` points on the current machine, then emits per-dispatcher
header files plus a plan-wisdom header. Example R=8 output:

    vfft_r8_t1_dit_dispatch_avx2.h    # DIT dispatcher, picks among DIT variants
    vfft_r8_t1_dif_dispatch_avx2.h    # DIF dispatcher, picks among DIF variants
    vfft_r8_t1_dit_dispatch_avx512.h
    vfft_r8_t1_dif_dispatch_avx512.h
    vfft_r8_plan_wisdom.h             # protocol selection for planner

## Schema: protocol vs dispatcher

Each codelet variant has two orthogonal tags:

- **protocol**: the twiddle-table layout the variant expects
  (`flat` = `(R-1)*me` doubles; `log3` = `me` doubles; `t1s` = `(R-1)` scalars).
  The planner uses this to size and populate the twiddle buffer.

- **dispatcher**: the codelet slot the variant populates. Variants sharing
  a dispatcher key compute the **same mathematical function** and are
  mutually exchangeable based on measured ns/call. Variants in **different**
  dispatcher slots (e.g. DIT vs DIF at R=8) compute different functions
  from the same inputs, and the planner chooses between them at plan
  construction time, not at codelet-call time.

R=4 example: `ct_t1_dit`, `ct_t1_dit_u2`, `ct_t1_dit_log1` all have
dispatcher=`t1_dit` (all DIT, all flat). `ct_t1_dit_log3` has
dispatcher=`t1_dit_log3` (separate because it needs a different twiddle
buffer). `ct_t1s_dit` has dispatcher=`t1s_dit`.

R=8 example: `ct_t1_dit` and `ct_t1_dit_prefetch` both have
dispatcher=`t1_dit` (both DIT-family). `ct_t1_dif` has dispatcher=`t1_dif`
(DIF-family — computes a different function).

## Pipeline

```
python common/bench.py --radix-dir radixes/r4 --phase all   # all 5 phases
python common/bench.py --radix-dir radixes/r8 --phase all
```

Or run individual phases:

```
python common/bench.py --radix-dir radixes/r4 --phase generate
python common/bench.py --radix-dir radixes/r4 --phase compile
python common/bench.py --radix-dir radixes/r4 --phase run
python common/bench.py --radix-dir radixes/r4 --phase emit
python common/bench.py --radix-dir radixes/r4 --phase validate
```

Phases are idempotent; `--phase run` resumes from existing jsonl.

## Portability

### Linux (default)
```
python3 common/bench.py --radix-dir radixes/r4 --phase all
```

### Windows + ICX (production target)
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
python common\bench.py --radix-dir radixes\r4 --phase all
```

Sources oneAPI env (needed for libircmt.lib link). ICX on Windows uses
LLD linker by default. AVX-512 candidates auto-skip on hosts without
AVX-512 via CPUID at runtime.

### Compile flag isolation
Each translation unit is compiled with only its needed ISA flags.
`harness.c` and the aggregator get base (AVX2) flags. Per-ISA candidate
fragments get their ISA-specific flags. This prevents the compiler from
emitting AVX-512 instructions in outer harness code on hosts without
AVX-512 support (previous symptom: `STATUS_ILLEGAL_INSTRUCTION` at runtime).

On hosts without AVX-512, the AVX-512 fragment is replaced with an empty
stub so the aggregator's `extern` declarations still resolve.

## Adding a new radix

Create `radixes/rN/` with:

- `gen_radixN.py` — codelet generator. Must expose:
  - `VARIANTS` dict: `{variant_id: (function_base, protocol, dispatcher)}`
  - `function_name(variant_id, isa, direction) -> str`
  - `protocol(variant_id) -> str`
  - `dispatcher(variant_id) -> str`
  - CLI supporting `--isa avx2|avx512|scalar` to emit the ISA header.

- `candidates.py` — exposes `RADIX`, `GEN_SCRIPT`, `enumerate_all()`,
  `sweep_grid(variant)`, `function_name`, `protocol`, `dispatcher`.

The bench infrastructure in `common/` is radix-agnostic and requires no
changes.

After bench runs, extend:

- `common/validate.c` — add `#elif RADIX == N` block with `ADD_CASE`
  entries for each dispatcher you emit.
- `common/fft_radix_include.h` — add `#elif RADIX == N` block including
  the generator header + each dispatcher header + plan wisdom.

## Files in bench_out/r{R}/

- `staging/fft_radix{R}_{isa}.h` — generator output per ISA
- `build/vfft_harness_candidates_{isa}.c` — per-ISA candidate tables
- `build/vfft_harness_candidates.c` — aggregator
- `build/*.o | *.obj` — per-TU object files (multi-step compile)
- `build/harness[.exe]` — measurement binary
- `build/validate_{isa}[.exe]` — validator binaries
- `measurements.jsonl` — raw results, one JSON object per line

## Troubleshooting

**"No C compiler found"**: install gcc/clang/icx, or set `VFFT_CC`.

**AVX-512 candidates measured as "host_lacks_avx512"**: expected on Raptor
Lake (AVX-512 fused off). The harness's runtime CPUID probe skips them.

**Validator shows failures**: the dispatcher produces different output
than the reference codelet for that dispatcher's family. This usually
means a genuine codelet bug, OR a twiddle-table layout mismatch between
the validator's `populate_tw()` and what the codelet expects. Check
`common/validate.c`'s `populate_tw()` for the protocol in question.

**Multiple dispatchers emitted for same radix**: expected. DIT and DIF
families at R=8 get separate dispatchers because they compute different
functions.

