# The public header surface

What belongs in `include/vfft.h` and what moves to `include/vfft_diagnostics.h`.
Agreed with the owner on 2026-10-06. Nothing has been moved yet.

## The rule

- **`vfft.h` is the transform contract.** A caller needs it to compute an FFT:
  create, execute and destroy, the configuration, the allocator, and anything that
  changes what `vfft_create` does to the caller's process.
- **`vfft_diagnostics.h` holds what is shipped but not needed to compute an FFT.**
  This means facts for logs, the engagement counters, and the tools the gauntlet uses
  to time code under the races' protocol. It is installed, and nothing in `vfft.h`
  depends on it.
- **The library's internal use never depends on a public declaration.** The races
  enter the measurement scope through `race_scope.h` and `vfft_measure.h`, and the
  composite planners free their private child stores through the internal wisdom
  functions. Moving a declaration out of `vfft.h` changes nothing inside the library.

## Decided

| declaration | goes to | why | callers today |
| --- | --- | --- | --- |
| `vfft_isa()` | `vfft_diagnostics.h` | A build-time fact for logs. Buffer alignment comes from `vfft_alignment()`, and the ISA is already part of `vfft_wisdom_identity()`. It cannot guard a CPU without AVX-512: an AVX-512 build faults there regardless. | two Zen 4 benches print it in their banner |
| `vfft_measure_begin()` | `vfft_diagnostics.h` | Timing the caller's own code under the races' scope is a benchmarking tool. | `gauntlet/bench_scope.h` |
| `vfft_measure_end()` | `vfft_diagnostics.h` | Pairs with `vfft_measure_begin()`. | `gauntlet/bench_scope.h` |
| `vfft_measure_describe()` | `vfft_diagnostics.h` | One line for a bench log. | `gauntlet/bench_scope.h` |
| `vfft_measure_confine()` | `vfft_diagnostics.h` | It changes the affinity of the whole process, for every thread any library creates afterwards, and `vfft_measure_end()` does not undo it. That belongs in a tool, not in the main contract. | `gauntlet/bench_scope.h` |
| `vfft_version()` | stays in `vfft.h` | The one non-transform call libraries keep in their main header (zlib `zlibVersion`, FFTW `fftw_version`, MKL `mkl_get_version_string`): an application can log the exact library and detect a header/library mismatch. | two Zen 4 benches |

## Open

1. **`vfft_measure_configure()`, `vfft_measure_config_t` and its enums.** These have
   a user case. They are the only way for an application to stop a racing
   `vfft_create` from pinning its thread (the pool's pin of the caller to logical CPU 0
   stays), raising its priority, or waiting up to 60 s for another process's
   measurement. The owner's 10-04 design made the scope public for that reason (design
   step 3, `wisdom_system.md`). The question is whether it stays in `vfft.h` beside
   `vfft_create`, with one line in `vfft_create`'s documentation pointing to it, or
   moves to the optional header with the other four.
2. **`vfft_wisdom_load()`, `vfft_wisdom_save()`, `vfft_wisdom_free()`, the
   `vfft_wisdom` type and `config.wisdom`.** After the 10-05 wisdom system, a create
   finds this CPU's folder and saves by default, and `VFFT_WISDOM_DIR` names another
   store for the whole process. The three calls serve tooling today: the gauntlet
   bench, `recal_1d_probe`, the gates and `src/tools/baseline/api_sweep.c`.
   `vfft_wisdom_save()` has two callers, both tooling. The one thing a user would lose
   is choosing a store in code. The options are:
   - (a) move all of them out of `vfft.h`, so users rely on the automatic folder and
     `VFFT_WISDOM_DIR`;
   - (b) **recommended:** replace `config.wisdom` with a directory path,
     `const char *wisdom_dir` (NULL means automatic), so users choose a store in code
     with no handle and nothing to free, and move the three calls to a tooling header;
   - (c) keep `vfft_wisdom_load()` and `vfft_wisdom_free()` public and drop only
     `vfft_wisdom_save()`.
3. **`vfft_wisdom_folder()`, `vfft_wisdom_identity()`, `vfft_wisdom_build()` and
   `vfft_wisdom_report()`.** The same question as item 2. The gauntlet's tooling calls
   them through `gauntlet/wisdom_folder.py`. Their other callers have not been traced.

## Gap, reported and not filled

`vfft.h` defines no version macros. An application can log `vfft_version()`, but it
cannot compare the header it compiled against with the library it loaded. The check
would need three `#define`s beside the function, and adding them is the owner's call.
