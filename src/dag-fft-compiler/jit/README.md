# JIT execution path (`src/dag-fft-compiler/jit/`)

Runtime code generation for 1D C2C: give a **plan discovered at runtime** the same
specialized machine code the build-time emitter bakes into `plan_executors.h`,
**without a rebuild**, and at **zero extra cost on the FFT hot path**.

---

## 1. Why this exists

The fast path in this library is the **static, plan-shaped executor**: for a known
`(N, K, factorization, variant-assignment)` tuple, `emit_executor_h.ml` bakes a
straight-line C function into `../generator/generated/plan_executors.h` (direct
codelet calls, variant branch collapsed). At runtime
`vfft_proto_lookup_fwd_avx2(plan)` matches a plan to that function.

The catch is a **phase mismatch**:

```
  CALIBRATOR / PLANNER  →  runs at RUNTIME, discovers factorizations
  STATIC EXECUTORS      →  baked at BUILD time, frozen set
```

A **live planner** (patient/measure/estimate) routinely produces a plan that was
**not in the wisdom file when the library was built**. That plan has no baked
executor, so it falls to the **generic executor** — which interprets the plan
stage-by-stage and pays a per-stage dispatch tax (≈5–10% at high stage counts,
worse the more stages). Nothing heals this between runs: the static set is frozen
until the next emit+rebuild.

**The JIT closes that gap.** When a plan has no baked executor, we generate one
*for that exact plan* at runtime, compile it, load it, and cache it. The plan then
runs at full static-executor speed.

> This was "option (c)" in the design discussion (a/b were: script the
> calibrate→emit→rebuild loop, and pre-bake a wide size grid). (c) is the only
> option that helps a plan the planner *discovers live*.

---

## 2. Where it sits — three-tier dispatch

```
        plan  ─►  vfft_proto_plan_jit_fwd(plan)        (call ONCE, planner phase)
                        │
                        ├── baked static in plan_executors.h?  ── yes ─► return it   (0 ms)
                        │
                        ├── already JIT'd this process?        ── yes ─► return it   (~µs, registry)
                        │
                        ├── cached .dll/.so on disk?           ── yes ─► dlopen      (~0.3 ms)
                        │
                        └── COLD: emit .c → gcc -shared → load → cache → register     (~0.7 s, ONCE EVER)

        hot loop:   fn(plan, re, im, slice_K, plan->K, 0)      ◄── direct call, ZERO JIT overhead
                    (fn == NULL  →  vfft_proto_execute_fwd_generic — always-correct fallback)
```

The **generic executor is the safety net**: correctness never depends on the JIT.
The JIT (like the baked statics) is purely a *speed cache* layered over it.

---

## 3. Components

| File | Role |
|------|------|
| `emit_jit.py` | **Cross-platform (Win+Linux) single-plan emitter.** Given one plan, prints a standalone `.c` with one exported executor `vfft_proto_jit_exec`. The body is a sequence of `VFFT_PROTO_STAGE_*` macro calls — **byte-identical** to what the OCaml build-time emitter produces (same macros → same machine code → same speed). Also emits self-contained codelet `extern`s so a *cold* plan (absent from `plan_executors.h`'s baked extern set) still compiles. |
| `jit_prelude.h` | Shared include surface for the emitted `.c`. Just re-exports `../core/plan.h`, which pulls in `plan_executors.h`'s `stride_plan_t` + the `VFFT_PROTO_STAGE_*` macros + SIMD stubs. |
| `jit_runtime.h` | **The resolver.** `vfft_proto_plan_jit_fwd(plan)` → `vfft_proto_exec_fn`. Hashes the plan, checks the registry + on-disk cache, shells out to `emit_jit.py` + `gcc` on a miss, `dlopen`s the result, registers + returns it. Header-only; platform `#ifdef`'d (Win `LoadLibrary`/`.dll`, POSIX `dlopen`/`.so`). |
| `jit_smoke.c` | Smoke + regression harness: drives the resolver on cold/baked cells, checks **bit-exact accuracy vs generic**, times it, and proves the persistent cache. |
| `../jit/generated/` | Persistent cache: `jit_<key>.c` + `jit_<key>.dll/.so`. Machine/toolchain-specific → **gitignored**. This is the "compiled wisdom" — a plan compiles **once, ever**. |

---

## 4. The ABI rule (critical)

The JIT'd executor receives a `stride_plan_t *` **from the core** and reads its
fields via the `STAGE_*` macros. It **must** compile against the *same* struct
layout the core uses, or it reads garbage.

The core's `stride_plan_t` is the **standalone** struct defined inside
`plan_executors.h` under `#ifndef VFFT_PROTO_USE_PRODUCTION_PLAN_T` — because
`../core/plan.h` includes `plan_executors.h` **without** setting that flag. So the
JIT translation unit gets ABI parity for free by including the same `core/plan.h`
(via `jit_prelude.h`). Do not introduce a second `stride_plan_t` definition.

---

## 5. The emitter mapping (`emit_jit.py`)

Recovered from / verified against `plan_executors.h`. Per-stage macro selection:

**DIT forward**
- stage `0` → `VFFT_PROTO_STAGE_OUTER` (the `n1`, twiddle-free first stage; its
  variant code is moot)
- stages `1..n-1` → `VFFT_PROTO_STAGE_{FLAT|LOG3|T1S}` for variant code `{0,1,2}`

**DIF forward**
- stages `0..n-2` → `VFFT_PROTO_STAGE_DIF_{FLAT|LOG3}`
- last stage → `VFFT_PROTO_STAGE_DIF_OUTER`

**Codelet symbol naming** (for the self-contained externs):
`OUTER → radix{R}_n1_{dir}_{isa}`, `FLAT → radix{R}_t1_dit_{dir}_{isa}`,
`LOG3 → radix{R}_t1_dit_log3_{dir}_{isa}`, `T1S → radix{R}_t1s_dit_{dir}_{isa}`.

The runtime recovers the variant code from the *wired* plan (no need to store it):
`use_log3 → LOG3`, else `t1s_fwd != NULL → T1S`, else `FLAT`
(`vfft_proto_jit_variant()` in `jit_runtime.h`).

```
# manual invocation examples
python emit_jit.py --N 131072 --K 4 --factors 4,4,4,32,64 --out out.c
python emit_jit.py --N 131072 --K 4 --factors 4,4,4,32,64 --variants 2,2,1,2,2 --body-only
```

---

## 6. Usage (caller pattern)

```c
#include ".../core/planner.h"
#include ".../core/executor.h"
#include ".../jit/jit_runtime.h"

stride_plan_t *plan = vfft_proto_plan_create_ex(N, K, factors, variants, nf, dif, &reg);

/* PLANNER PHASE — resolve once. May compile (cold) or just dlopen (cached). */
vfft_proto_exec_fn fwd = vfft_proto_plan_jit_fwd(plan);

/* HOT LOOP — direct call, zero JIT overhead. Graceful fallback if unresolved. */
for (...) {
    if (fwd) fwd(plan, re, im, K, plan->K, /*start_stage=*/0);
    else     vfft_proto_execute_fwd_generic(plan, re, im, K);
}
```

`vfft_proto_plan_jit_fwd` is the FFTW-style "planning returns an executor" handle.
**Hold the returned pointer**; do not re-resolve per call.

---

## 7. Key design decisions (and why)

- **Return the fn; don't store `plan->exec_fwd`.** `plan_executors.h` (where
  `stride_plan_t` lives) is **auto-generated, and the calibrator re-emits it**
  whenever `spike_wisdom.txt` changes — a hand-added struct field would be wiped.
  Returning the pointer is the regen-proof equivalent and keeps the same
  zero-overhead property (resolve once, direct call after).
- **`execute_fwd` is left untouched** → **zero regression** for every existing
  caller; JIT is strictly additive.
- **Compile cost locked to the planner phase.** All emit/compile/load/hash/registry
  work is inside `plan_jit`; the hot path never compiles or looks up.
- **Cross-platform Python emitter, not OCaml.** OCaml/dune isn't on the Windows
  box, and single-plan emit is trivial (a list of macro calls), so a ~30-line
  Python script serves *both* OSes with no OCaml runtime dependency — and it's the
  first step toward a future in-process codegen backend.
- **Persistent, hash-keyed cache.** Key = `(N, K, factors, variants, isa, version)`.
  Compile once, ever; bump `VFFT_PROTO_JIT_VERSION` to invalidate after a codegen
  or codelet change.
- **One executor per plan shape** (mirrors the one-best-per-cell wisdom rule): the
  planner's pick maps 1:1 to its executor.

---

## 8. Configuration (`jit_runtime.h`, `#if _WIN32`/`#else` defaults, all `#ifndef`-overridable)

| Macro | Windows / Linux default | Meaning |
|-------|---------|---------|
| `VFFT_PROTO_JIT_REPO` | `C:/…` / `/mnt/c/…` repo root | base path |
| `VFFT_PROTO_JIT_DIR`  | `…/jit/generated` | persistent `.c`/lib cache |
| `VFFT_PROTO_JIT_INC`  | `…/jit` | `emit_jit.py` + `jit_prelude.h` dir (also `-I`) |
| `VFFT_PROTO_JIT_GCC`  | `…/mingw152/…/gcc.exe` / `gcc` | compiler |
| `VFFT_PROTO_JIT_PYTHON` | `python` / `python3` | emitter interpreter |
| `VFFT_PROTO_JIT_CFLAGS` | `… -shared` / `… -shared -fPIC` | compile flags |
| `VFFT_PROTO_JIT_CODELETS` | `@…/codelets.rsp` / `@…/codelets_linux.rsp` | gcc `@response-file` of codelet `.o` |
| `VFFT_PROTO_JIT_VERSION` | `2` | bump to bust the on-disk cache |

---

## 9. Build & run the smoke test (gcc 15.2, both OSes)

**One-time:** build in-repo codelet objects (no `C:\tmp` dependency):
- Windows: `powershell -ExecutionPolicy Bypass -File jit\build_codelets.ps1` → `jit/generated/codelets.rsp`
- Linux:   `bash jit/build_codelets.sh` → `jit/generated/codelets_linux.rsp`

**Windows:**
```sh
GCC=C:/mingw152/mingw64/bin/gcc.exe
ROOT=C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler
$GCC -O3 -mavx2 -mfma -march=haswell -Wno-incompatible-pointer-types -Wno-unused-result \
     -c $ROOT/jit/jit_smoke.c -o jit_smoke.o
$GCC jit_smoke.o @$ROOT/jit/generated/codelets.rsp -o jit_smoke.exe -lm
./jit_smoke.exe "4,4,4,4,4,4,4,8"   # COLD 8-stage → emit+compile+load
./jit_smoke.exe "4,4,4,32,64"       # BAKED        → baked static
```

**Linux/WSL:** `bash jit/run_linux_smoke.sh` (links `codelets_linux.rsp`, `-lpthread -ldl`; the
source needs `_GNU_SOURCE` for env.h's pthread pin — already defined in `jit_smoke.c`).

Expected (both): `max_abs_diff vs generic = 0.000e+00` (bit-exact). Cold 1st-run
~0.7 s (Windows) / ~3.4 s (WSL — slow `/mnt/c` compile); fresh-process cache hit ~0.3 ms.

---

## 10. Extensibility — how to grow it

The runtime (hash → cache → emit → compile → load → register) is
**transform-agnostic and written once**. Each new feature adds only:

1. an **emit mode** in `emit_jit.py` for that family's body, and
2. its **prelude** (the macros + codelet externs it calls).

Roadmap: `c2c_fwd` (done) → `c2c_bwd`, `c2c_dif` → `r2c`/`c2r` → `2d`/strided →
`trig` (DCT/DST/DHT) → Bluestein/Rader. The OCaml emitter's `rfft`/`c2r`/`trig`/
`strided`/`oop` codegen is the reference spec for each. The plan hash should grow
a **family tag**, and the registry can be **signature-tagged** so families with
different executor ABIs coexist.

**Linux:** PROVEN under WSL (gcc 15.2) — one cross-platform source: `emit_jit.py`
via python3, `.so`/`dlopen`/`-fPIC`, ELF codelets from `build_codelets.sh`.
Bit-exact, and Windows re-verified with no regression.

**Future backend swap:** replace the `gcc -shared` step with an in-process codegen
backend (asmjit / copy-and-patch / LLVM) behind the same resolver interface — that
removes the gcc + Python runtime deps (the true MKL-style JIT) without touching
cache/registry/dispatch.

---

## 11. Known limitations / spike-isms

- **Codelet objects** are built in-repo by `build_codelets.{ps1,sh}` (gitignored
  `jit/generated/codelets*.rsp`). If absent, JIT compile fails → graceful generic
  fallback (no crash); just re-run the build script. (No more `C:\tmp` dependency.)
- **`jit_prelude.h` pulls all of `plan_executors.h`**, including the unused baked
  static executors (compiled then dead-stripped; `-Wno-unused-function`). A later
  cut can extract just `{struct, macros}` into a lean header — identical codegen.
- **Static-vs-generic gap is small in the measured regime** (~3–4% median, clean
  machine, large K=4 C2C — a memory-bound regime, per the project thesis). A
  definitive number needs CPU clock lockdown; the likely *sweet spot* is
  compute-bound (K=256), not yet measured.
- **fwd / DIT only** so far (the spike scope). bwd/DIF/other families per §10.
- **No eviction** in the in-process registry (fixed 256 slots) and no concurrency
  guard around the compile step — fine for single-threaded planning; revisit for
  multi-threaded JIT.

---

## 12. Status & provenance

1D C2C **forward** JIT: spike proven and runtime landed 2026-06-14 (gcc 15.2,
i9-14900KF). Bit-exact vs generic and vs baked static; persistent cache verified;
`execute_fwd` untouched (zero regression). **Cross-platform: Windows (.dll) and
Linux/WSL (.so) both proven bit-exact from one source.** Codelet link hardened
to in-repo objects. Clean-machine latency: static ~3–4% over generic in the
measured large-K=4-C2C regime (modest; memory-bound). See the project memory note
`jit_execution_path.md` for the running log.
