# `core/support/` — platform foundation

The bottom layer everything else depends on: the runtime environment (denormals, memory,
pinning, search knobs), the thread pool, and the 2D strided-codelet registry type. No FFT
math here — this is the floor the engine stands on.

| file | role |
|------|------|
| `env.h` | CPU/runtime setup (FTZ-DAZ, aligned + huge-page alloc, ISA/CPU query, core pinning) **+** env-overridable search-tuning knobs |
| `threads.h` | the **from-scratch** spin-based thread pool (K-split parallelism; no OpenMP/TBB) |
| `strided_codelets.h` | ABI-typed registry struct for the 2D Design-C "strided" row-FFT codelets |

---

## `env.h` — runtime environment + search knobs

Two parts: PART 1 sets up the CPU/runtime for fast, low-variance FFT; PART 2 holds the
env-overridable knobs the exhaustive/joint search reads.

### PART 1 — CPU / runtime

**Denormal handling (FTZ/DAZ) — `stride_env_init()`.** Sets two MXCSR bits: **FTZ**
(*flush-to-zero*, bit 15 / `0x8000` — denormal *results* become zero) and **DAZ**
(*denormals-are-zero*, bit 6 / `0x0040` — denormal *inputs* treated as zero). **Why it
matters:** denormal arithmetic traps into microcode and runs **50–100× slower** on x86, and
FFTs generate denormals readily — near-zero inputs, twiddle-product underflow, inverse-scale
rounding. Both flags are safe (denormals sit below any real signal's noise floor; MKL/IPP/HPC
libraries enable them). **MXCSR is per-thread**, so `stride_env_init()` must be called from
*every* thread that does FFT work (it returns the old MXCSR; `stride_env_restore` puts it
back). Call it once at program start on the main thread, and inside each worker.

**Aligned + huge-page allocation.**
- `stride_alloc` / `stride_free` — 64-byte aligned (`STRIDE_ALIGNMENT`), one cache line, the
  SIMD load/store requirement. `_aligned_malloc` (Windows) / `posix_memalign` (POSIX).
- `stride_alloc_huge` / `stride_free_huge` — **2 MB huge pages** for the big `re[]`/`im[]`
  data buffers (above `STRIDE_HUGEPAGE_THRESHOLD = 64 KB`). **Why:** strided FFT access blows
  the DTLB with 4 KB pages — VTune measured **23% DTLB-Store overhead** at N=1000 K=256;
  2 MB pages cut the page count 512× and largely erase it. Windows: `VirtualAlloc` +
  `MEM_LARGE_PAGES` (needs the *"Lock pages in memory"* privilege). Linux: `mmap` +
  `MAP_HUGETLB` (needs `nr_hugepages > 0`) → **THP fallback** (`madvise(MADV_HUGEPAGE)`) →
  plain aligned alloc. `stride_free_huge` tells huge from fallback by 2 MB-alignment.

**Version / ISA / CPU query.** `STRIDE_ISA_NAME` resolves to `avx512`/`avx2`/`scalar` from
compile macros (per-binary ISA, no runtime fat-dispatch). `stride_set_verbose` +
`stride_print_info` dump version, ISA, the **CPU brand string** (`__cpuidex` on Windows,
`/proc/cpuinfo` on Linux), and whether FTZ+DAZ are actually live — a one-call sanity check.

**CPU affinity / core pinning.** `stride_pin_thread(core)` / `stride_unpin_thread` /
`stride_get_num_cores`. **Why:** unpinned threads migrate (L1/L2 invalidation, cross-CCX on
Zen, P↔E-core bounce on Intel hybrid, NUMA hops) — the single biggest source of run-to-run
variance on hybrid CPUs. For benchmarking, **pin to a P-core**. This is also the foundation
of the thread-pool contract: the pool pins worker *i* to core *i+1* and **assumes the caller
is on core 0** (see `threads.h` + the MT gotcha below).

### PART 2 — exhaustive / joint search knobs (env-overridable)

The thoroughness dials the exhaustive/joint planners read. Every one is `#ifndef`-guarded
*and* env-overridable, so tuning needs no recompile:

| knob | default | env var | meaning |
|------|:-------:|---------|---------|
| `EXH_MAX_DEPTH_POW2` | 5 | `VFFT_PROTO_EXH_MAX_DEPTH` | max stages for pow2 N (pow2 optima are shallow) |
| `EXH_MAX_DEPTH_NONPOW2` | 9 | `VFFT_PROTO_EXH_MAX_DEPTH` | max stages for non-pow2 (needs deeper plans) |
| `EXH_PRUNE_FACTOR` | 2.0 | `VFFT_PROTO_EXH_PRUNE` | skip a factorization's variant cartesian if its default bench > F× the running best (1e9 = disable) |
| `WISDOM_OVERWRITE` | 0 | `VFFT_PROTO_WISDOM_OVERWRITE` | 0 = fill-missing-only (safe incremental); 1 = re-calibrate + overwrite |

Accessors `vfft_proto_env_{max_depth,prune_factor,wisdom_overwrite}` read the default unless
the env var overrides; `max_depth` also clamps to a `hard_cap` (pass `STRIDE_MAX_STAGES`).

> **⚠ Validation scope — the defaults were tuned on N=1024 K=4 (pow2) ONLY** (2026-06-14,
> i9-14900KF). The absolutely-exhaustive run there (262,428 candidates: all 35 decompositions
> × all orderings × all variants, **no pruning**) showed latency is **monotonic in
> stage-count** (2-stage `64×16` = 3.2 µs optimal; 10-stage = 13.6 µs, 4.2×), and that
> `depth-5 + 2× prune` reach the **identical winner** at 32× fewer candidates / 56× faster.
> That is a **1024 result**, not validated elsewhere — larger pow2 may want 4–5 stages
> (depth-5 could clip), and the 2× prune may be too aggressive for many-small-prime
> non-pow2. Re-run the env-gated sweep (`VFFT_PROTO_EXH_MAX_DEPTH=16 VFFT_PROTO_EXH_PRUNE=1e9`)
> before trusting these for a new size class. (A third cap, `VFFT_PROTO_DP_MAX_PERMS=720`,
> lives in `planning/dp_planner.h` and can clip orderings for non-pow2 — also 1024-only.)

---

## `threads.h` — the from-scratch thread pool

**Built from scratch on raw OS primitives — Win32 `CreateThread`/`SetThreadAffinityMask` and
POSIX `pthreads`. No OpenMP, no TBB, no external dependency, no wrapper.** This is deliberate,
not a shortcut: the FFT MT pattern is **K-split** (each worker takes a contiguous slice of the
K batch, runs the full N-point FFT on it, stages independent across K — *no inter-thread sync
inside a transform*), dispatched **thousands of times per second**. A general framework's
~5 µs dispatch latency would dominate; this pool gets it to ~10 ns.

**Design:**
- **Persistent, pinned workers.** `stride_set_num_threads(n)` creates `n−1` workers (the
  caller is thread 0); worker *i* is pinned to **core `i+1`**. So the **caller must run on
  core 0** — that's the whole pinning contract (all P-cores 0–7 on the 14900KF).
- **Spin-based dispatch, not sleep.** Workers spin on a `volatile done` flag with `_mm_pause`
  / `__builtin_ia32_pause`. Posting work = setting `func/arg` then clearing `done` (the wake
  signal); waiting = spinning on `done`. **~10 ns wake latency vs ~5 µs for OS events.**
  Idle workers burn a core — acceptable because dispatch frequency is high and the pool is
  torn down (`stride_set_num_threads(1)`) when not in use. *(The file's top comment mentions
  "sleep on OS primitives" — aspirational; the implementation spins.)*
- **Thread 0 = caller, zero dispatch overhead.** The caller runs its own `1/T` slice inline,
  only the other `T−1` go through the pool.
- **Sense-reversing spin barrier** (`_stride_barrier_*`) for the group-parallel executor path:
  an atomic counter (`InterlockedIncrement` / `__sync_add_and_fetch`), last arrival flips the
  sense bit. **~100 ns vs ~1 µs for `pthread_barrier`.**

> **The MT caller-pin gotcha (load-bearing).** Because workers pin to cores `1..T−1` assuming
> the caller is on core 0, **the calling thread must be pinned to core 0** (`stride_pin_thread(0)`).
> Pin it anywhere in `1..T−1` and it collides with a worker → two threads spin-contend one
> core → *catastrophic* anti-scaling (T4/T8 collapse to 0.0–0.1×). Every MT bench in this repo
> pins the caller to core 0 for this reason.

---

## `strided_codelets.h` — 2D Design-C row-FFT codelets

The ABI-typed registry struct (`strided_codelets_t`) that the **auto-emitted**
`strided_registry_{isa}.h` populates (the OCaml pipeline emits it from coverage; this header
just declares the slot type). **One uniform 6-arg in-place ABI** —
`fn(rio_re, rio_im, tw_re, tw_im, row_stride, me)` — **no scratch buffer**: Design C does
matrix → register-transpose → butterfly DAG → inverse-transpose → matrix **all in registers**
(doc 56). **Single-stage `n1` only by design** (it's excluded from the auto-blocking recipe
and the AVX2 regalloc/pinning gate — both exclusions are load-bearing). Radix-indexed, two
directions (`n1_fwd`/`n1_bwd`); the radix set differs per ISA (avx2 `{4,8,12,16,20,32,64}`,
avx512 `{8,16,32,64}` — the avx2 16-register file spills past R=256, so coverage gates it and
the registry just mirrors NULL for ungenerated radices). Consumed by the 2D row-FFT path
(`transforms/fft2d/`).

---

## Gotchas

- **`stride_env_init()` is per-thread** (MXCSR is thread-local) — call it in every worker, not
  just main, or that worker eats the denormal penalty.
- **Caller must be on core 0** for the thread pool (see above) — the #1 MT footgun.
- **Huge pages need a privilege/sysctl** ("Lock pages in memory" / `nr_hugepages`); the alloc
  silently falls back to standard pages if unavailable, so a missing setup costs perf, not
  correctness.
- **Search knobs are 1024-tuned** — re-validate before trusting on a new size class.

See also: `core/engine/README.md` (the executor the pool parallelizes), `core/planning/README.md`
(what reads the PART-2 search knobs), `core/transforms/fft2d/` (the strided-codelet consumer).
