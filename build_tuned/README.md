# build_tuned/ — the bench harness and its build system

Everything that compiles, runs and aggregates a measurement lives here. The
library itself is in `src/core/` (header-only bar `vfft.c`) and
`src/dag-fft-compiler/` (the codelet generator and its emitted corpus).

| | |
|---|---|
| `build.py` | the build system: codelet lib + driver TUs -> one exe |
| `benches/` | every bench, gate, probe and calibrator (one `.c` = one exe) |
| `run_bench.py` | paper-grade sweep, ONE FRESH PROCESS per (N,K) cell |
| `avg_runs.py` | averages two isolated runs, matched on (N, K, plan) |
| `results/`, `*.csv` | banked measurements (tracked on purpose) |
| `VtuneHarness/` | VTune capture + report |

## Building

```
cd build_tuned
python build.py --src benches/<file>.c [--mkl] [--fftw] [--vfft] [--jit] --compile
```

`--src` is required in practice — its argparse default points at a
`test/test_tuned_core.c` that no longer exists. `--vfft` adds `src/core/vfft.c`
(needed by anything using the public `vfft.h` front door); `--mkl` / `--fftw`
add the reference backend; omitting `--compile` runs the exe afterwards.

Most bench files carry their own build line in the header comment. Use it.

## The build model

Two independent halves, cached separately:

**Codelets** — the 899 avx2 `.c` files under `src/dag-fft-compiler/codelets/`
(`<family>/avx2/`, except zil which nests one level deeper as
`zil/avx2/{pure_il,pure_il/tangent,boundary_split}`) are OCaml-generated and change only
when the generator runs. Each compiles to its own object in
`src/dag-fft-compiler/.obj/<isa>/`, which is then archived into
`libdagcodelets.a`. That objdir path is fixed because `dag_write_jit_rsp()`
points the JIT runtime at the same objects.

**Driver TUs** — the bench itself plus (with `--vfft`) `vfft.c`. These compile
to `build_tuned/.obj/<flags-hash>/`. The hash covers compiler, flags and the
whole `-I` list, so a `--mkl` build, a `--fftw` build and an ASAN build can
never reuse each other's objects.

Both halves are per-object, compiled in parallel, and rebuilt only when stale.
Staleness is decided from the gcc `-MMD` depfile, so editing a core header
invalidates exactly the TUs that include it. Every uncertainty — missing
depfile, unparseable depfile, vanished dependency — answers "rebuild".

## Cost

Measured 2026-08-23, 32-thread host, mingw 15.2, `-O3 -mavx2 -mfma -march=native`.

| | before | now |
|---|---|---|
| nothing changed | 137s | **0.4s** |
| one codelet regenerated | ~317s | **1.2s** |
| one core header touched | 137s | 70s (only the dependent TU) |
| clean codelet corpus (899) | ~317s | ~45s |

The everyday cost is the driver TUs, not the codelets: `vfft.c` alone is 72s
and `bench_1d_vs_mkl.c` 65s, and both were previously handed to a single gcc
command that compiled them back to back on every build. Link is 0.25s. The
headers alone are 0.64s — the core is `static inline`, so the cost is
optimising *reachable* code, not expansion.

The split changes no codegen: the linked exe is byte-identical to the old
single-command path except two PE-header timestamp bytes.

## The CMake mirror

The root `CMakeLists.txt` builds the same corpus into the same exe and is the
second build system of record. It models codelets as a per-object target, so
it parallelises and does incremental correctly on its own:

```
cmake -S . -B build-ninja -G Ninja ^
  -DCMAKE_MAKE_PROGRAM=C:/mingw152/mingw64/bin/ninja.exe ^
  -DCMAKE_C_COMPILER=C:/mingw152/mingw64/bin/gcc.exe ^
  -DCMAKE_BUILD_TYPE=Release -DVFFT_ISA=avx2
cmake --build build-ninja
```

`cmake.exe` is not on PATH; it lives under `C:/Program Files/CMake/bin`.
Ninja ships with mingw as `C:/mingw152/mingw64/bin/ninja.exe`. Ninja
parallelises by default; MinGW Makefiles needs an explicit `-j`. Both build
systems emit an exe of the identical size.

| | build.py | cmake + ninja |
|---|---|---|
| nothing changed | 0.4s | 0.1s |
| one codelet | 1.2s | 1.6s |
| one core header | only the dependent TU | only the dependent TU |
| clean | ~120s | ~105s |

Two properties are structural to both and are NOT bugs to go fix:

- **Clean builds serialise codelets, then drivers.** The driver TUs do not
  depend on `libdagcodelets.a` to *compile*, only to link, so in principle they
  could start at t=0 and the clean build would be bounded by the slowest single
  TU (~75s). Ninja still finishes them last. Declaring the exe target before
  the library in `CMakeLists.txt` does NOT change this — measured, refuted,
  reverted: Ninja schedules from the target dependency graph, not declaration
  order.
- **`vfft.c` alone is the floor.** One 7751-line TU at 72-100s sets the cost of
  any core-header edit in either build system. The only lever left that does
  not touch `-O` is splitting that TU.

CMake `Release` adds `-DNDEBUG`, which `build.py` does not. Today that is inert
(the only assert in the core is a `_Static_assert` in `support/ref.h`, which
`NDEBUG` does not affect), but the two build systems are not flag-identical.

## Knobs

| | |
|---|---|
| `VFFT_BUILD_JOBS` | compile parallelism (default: CPU count; `1` = serial) |
| `VFFT_ISA` | `avx2` \| `avx512` — selects the codelet tree |
| `VFFT_ASAN` | AddressSanitizer; own flag-hash, so its objects stay separate |
| `CC` | compiler (default mingw gcc; MSVC-style front ends take the one-command path) |
| `MKLROOT` | MKL location if not at the oneAPI default |

## Invariants

**The optimisation level is fixed.** `-O2` on the driver TUs would be 2.6x
faster to compile (72s -> 28s). It is out of the question — this is a
high-performance library, and that includes not shipping a lower-`-O` "debug"
config that could leak into a measurement. Build time is bought with
parallelism and caching, which cost nothing in codegen.

**Depfile parsing has two Windows hazards**, both handled in `_deps_of`:
paths carry drive-letter colons (so `-MT` is given a colon-free target), and
the MKL include dir sits under `Program Files (x86)`, which gcc emits
space-escaped. A naive `split()` shreds that into non-existent fragments,
every object then looks stale, and the cache silently never hits.

**Builds run concurrently here.** The archive is written to a pid-temp and
`os.replace`d, so no window exists in which `libdagcodelets.a` is missing.
Per-object writes are not locked. A running `bench_*.exe` holds its own path,
so a relink fails with "Permission denied" — that is another session, not a
build defect.

**`dag_codelet_srcs()` and the root `CMakeLists.txt` share one corpus.** The
CMake mirror counts each codelet family separately and fails hard on an
undeclared empty family, because a single aggregate total once hid 265 absent
zil files behind a plausible-looking 598. A count mismatch between the two
build systems is a real defect, not a configuration difference.

**Codelet objects are named by source stem** across all families, which
requires basenames to stay unique tree-wide.
