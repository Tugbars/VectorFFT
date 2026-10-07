# `core/` — the canonical FFT tree

This is the production core of the dag-fft-compiler library: the **public front
door** (`vfft.c`, implementing `include/vfft.h`) plus the engines, planners,
and transform layers it dispatches into. Organized into subfolders by role,
layered by dependency — each layer depends only on the ones above it.

```
core/
  vfft.c, vfft_execute.h, vfft_fingerprint.h
                THE front door (see "Front door" below) - sees both layouts.
                create and execute each fork on the layout ONCE.
  common/       shared by both layouts, depends on neither:
                abi/ (codelet ABI, route ids), support/ (ISA, CPU caches, pool,
                race body, clock, allocator), math/ (tw_exact), move/
                (transposes), policy/, wisdom/ (the wisdom2 store core),
                plan/ (vfft_internal.h: the plan struct, shared until D3)
  split/        the SPLIT (re/im planes) library: engine/, primes/, planning/,
                oop/, natorder/, rank1/, rank2/, rank3/, real/, trig/, wisdom/;
                split_create.h + split_execute.h are its side of the fork
  il/           the INTERLEAVED (z) library: isa/, planning/, rank1/, rank2/,
                rank3/, real/, wisdom/; il_create.h + il_execute.h are its side
  bridge/       the ONLY place besides the front door that sees both, and
                TEMPORARY: the 1D real dispatcher (real_bridge.h,
                real_bridge_exec.h); its one crossing left is the interleaved
                lane-major real BATCH on the split engines (D1)
  wisdom2/      front-side wisdom glue that spans both layouts: the legacy
                kind-3 reader and migration, the OOP codec aggregator, gates
```

**Architecture diagrams**: `docs/architecture/` (zones, folders, files and a text
map, generated from the includes by `python src/tools/archgraph.py`).

**Layout separation** (`docs/roadmap/layout_separation_plan.md`): phases 1-7
done. `split/` and `il/` include only `common/` and themselves; `bridge/`,
`wisdom2/` and the front door may include both. `python
src/tools/baseline/hygiene.py` checks the rule, and the step gate
(`src/tools/baseline/step_gate.py --enforce-deps`) fails on any violation.

## Front door: `vfft.c` (public API = `include/vfft.h`)

One create/execute/destroy surface over **four axes** committed at create time:
`transform` (C2C / R2C / C2R / DCT-I..IV / DST-I..III / DHT) ×
`placement` (in-place / out-of-place) ×
`layout` (SPLIT re/im planes / INTERLEAVED z, MKL DFTI_COMPLEX_STORAGE analog) ×
`order` (DEFAULT / NATURAL / SCRAMBLED, 1D/2D C2C). Plus `dims` 1..4,
`howmany` (K lanes), the opt-in padded-batch handle (`vfft_alloc_batch_for` →
`vfft_batch_planes` → `vfft_batch_stride` → `vfft_free_batch`), and rigor.

Contract: every unsupported or invalid cell is **refused at create with an
actionable stderr message**; `vfft_execute` validates the pointer signature
against the committed layout/placement/direction and computes NOTHING on a
mismatch. Never a silent no-op, never silent garbage. The support matrix lives
in `include/vfft.h` (the capabilities table + the SIGNATURE TABLE /
SUPPORT MATRIX blocks above `vfft_execute`); the machine proof is the gate
battery (`api_matrix_gate` (the serve/refuse table, benches/api_matrix_gate.c)).

Wisdom: the store is `src/wisdom/`, a root of per-CPU folders (`14900KF/`,
`Zen4/`, the empty `new/`), each holding the `wisdom2_*.txt` shards stamped with
the identity of the CPU that raced them. With no directory named, the library
serves from and saves into the folder stamped with this CPU's identity
(`common/support/cpu_identity.h`, `common/wisdom/wisdom2_folders.h`); nothing is
taken from another CPU's folder. `VFFT_WISDOM_DIR` or an explicit
`vfft_wisdom_load(dir)` names one directory as the store instead, unscanned
(gates and benches use a scratch copy of this CPU's folder). The frozen
bundle (`spike_wisdom.txt`, `bluestein_wisdom.txt`, `c2r_path.txt`) stays in
`src/dag-fft-compiler/generator/generated/` and is read from there. Misses race
at `config.rigor`, bank, and save the winner before create returns.

Runtime knobs (diagnostics/kill switches): `VFFT_NO_ZTURN` (fall back to the
legacy zsplit cascade), `VFFT_FORCE_ZROUTE` (pin the K=1 cascade route),
`VFFT_NO_IL2P` (disable the pure-IL 2-pass route), `VFFT_IL_PAD` (force the IL
padded arm), `VFFT_ZRACE_VERBOSE` (create-time race logging).

## Where things are

The per-folder READMEs (`oop/`, `planning/`, `support/`, `wisdom2/`,
`transforms/*/`) predate the separation and still describe the old folders;
their content is accurate, their paths are the old ones. Map:

| was | now |
|---|---|
| `engine/`, `primes/` | `split/engine/`, `split/primes/` |
| `support/*` (not `env.h`) | `common/support/` |
| `oop/tw_exact.h` | `common/math/` |
| `transforms/fft2d/transpose.h` | `common/move/` |
| `wisdom2/wisdom2.h`, `wisdom2_selftest.h` | `common/wisdom/` |
| split planners (`dp_planner`, `exhaustive_plan`, `measure`, `pad_calibrate`, `adopt_wisdom`, `wisdom_reader`, `dp_planner_split_oop`) | `split/planning/` |
| `oop/oop_{auto,dp,execute,mt}.h`, `support/strided_codelets.h` | `split/oop/` |
| `transforms/natorder/` | `split/natorder/` |
| `transforms/fft2d/` split 2D (`fft2d*`, `strided_tw.h`) | `split/rank2/` |
| `transforms/fft3d/`, `fftnd.h`, `fftnd_r2c.h` | `split/rank3/` |
| `transforms/real/` split engines | `split/real/` |
| `transforms/trig/` | `split/trig/` |
| `wisdom2_fftnd.h`, `wisdom2_stride_reader.h`, `wisdom2_real_reader.h` | `split/wisdom/` |
| `vfft_batch.h` | `split/rank1/` |
| `oop/il_isa.h`, `oop/avx512/`, `oop/ztt_qw16384.h` | `il/isa/` |
| `oop/il2p.h`, `il_flatdit*.h`, `il_prime.h`, `ztt*.h`, `k1_fourstep*.h` | `il/rank1/` |
| `dp_planner_il.h`, `il_slot_probe.h` | `il/planning/` |
| `il2d_*`, `fft2d_real_il.h`, `oop/il2d_proto.h` | `il/rank2/` |
| `fftnd_il.h` | `il/rank3/` |
| `zr2c.h`, `zr2c_build.h` | `il/real/` |
| `zrp.h`, `zrp_build.h`, `zttr.h`, `zttr_mt.h`, `zrm.h`, `zfsr.h`, `zrf.h`, `zrb.h`, `zrb_lanes.h` | `il/real/` |
| `transforms/fft2d/plane_queue.h` | `plane_queue.h` (front door) |

Deleted in phase 1 (dead): `transforms/conv/`, `fftnd_natorder.h`,
`fftnd_planner.h`, `fftnd_wisdom.h`, `engine/compat.h`.

Still mixed (to be cut): `oop/{oop_plan,oop_leaf_registry,k1_commit,c2c_ip_create,c2c_oop_create}.h`,
`planning/policy.h`, `support/env.h`, `transforms/{fft2d/fft2d_create,fftnd/fftnd_create,real/real_create}.h`,
`wisdom2/{wisdom2_oop,wisdom2_oop_reader,wisdom2_2d_reader,wisdom2_migrate}.h` and the two wisdom2 gates.

## Include convention — BARE includes, the build provides `-I`

Headers cross-reference each other **bare**: `#include "executor.h"`, not
`#include "engine/executor.h"`. The build system puts **every** `core/`
subfolder on the `-I` search path (`gauntlet/build.py:build_includes()`
walks `core/` recursively), so a bare include resolves regardless of which
subfolder the target lives in. Consequences:

- **Moving a file between subfolders needs no edit to a BARE include.** Some
  includes are path-qualified (`"common/support/race.h"`, `"../rank2/fft2d.h"`);
  `src/tools/baseline/relayout.py` moves files by a map and rewrites those.
- **Header basenames must stay globally unique** across all of `core/` —
  otherwise a bare include is ambiguous (first `-I` wins).
- Consumers (benches, the public build) also use bare includes:
  `#include "vfft.h"`, `#include "r2c.h"`.

SIMD codelets are **not** here — they live under `dag-fft-compiler/codelets/`
(generated by the OCaml emitters in `dag-fft-compiler/generator/`) and compile
as linked `.c` files; they include no core headers.

## Key entry points

- **Public API** (use this unless working on internals): `vfft_create` /
  `vfft_execute` / `vfft_destroy` in `vfft.c` — everything below is reached
  through it, chosen by wisdom.
- **c2c in-place, split**: `split/engine/planner.h` (`vfft_proto_auto_plan`) →
  `split/engine/executor.h`. MT via the `common/support/threads.h` pool (K-split).
- **c2c out-of-place, split**: `split/oop/oop_auto.h` champions.
- **c2c K=1, interleaved**: `il/rank1/` (il2p pair / il3p chain3, the flat DIT,
  ZTURN-T, the four-step, IL primes), raced by `il/planning/dp_planner_il.h`.
- **r2c/c2r**: `split/real/r2c_dispatch.h` / `c2r_dispatch.h`; interleaved
  K=1: even N in the real door `il/real/zrp_build.h` (zr2c, the pair, ZTT-r,
  the mono, the four-step), odd N in the door's odd race `il/real/odd_build.h`
  (the mono, the real flat DIT `il/real/zrf.h`, the real Bluestein
  `il/real/zrb.h`; gated against an IL c2c reference).
- **trig/DSP**: `split/trig/{dct,dct1,dct4,dst,dht}.h`.
- **2D/3D/4D**: split `split/rank2/`, `split/rank3/`; interleaved `il/rank2/`
  (il2d tier), `il/rank3/fftnd_il.h`.
- **prime N**: split `split/primes/prime_dispatch.h` → Rader / Bluestein
  (in-place); interleaved `il/rank1/il_prime.h`.
- **natural order (split)**: `split/natorder/` (1D), `natorder_2d.h` (2D).

## Gates

API surface: `api_matrix_gate` (the serve/refuse table, benches/api_matrix_gate.c)
(session scratchpad; walk the full support matrix + misuse diagnostics + the
header's compiled QUICK START). Feature gates live in `build_tuned/benches/`
(`zsplit_wis_gate`, `zsplit_api_gate`, `gate_vfft_rz`, `gate_4d`,
`gate_fndr_q1`, natorder/natmt tests, `regression_vs_mkl`). Run them with
`VFFT_WISDOM_DIR` pointed at a scratch dir so banked wisdom stays untouched.

## Migration headers at the top level

These files sit directly in `core/` (the front door) rather than in a module, because each is
about the library as a whole rather than about one transform family.

| file | role |
|---|---|
| `vfft_internal.h` | the three private structs — `vfft_plan_s`, `vfft_wisdom_s`, `vfft_batch_s`. Lifting these out of `vfft.c` is what let every later module header exist |
| `vfft_execute.h` | **THE execute entry point — every transform, BOTH layouts**, plus the execute-side helpers and `vfft_destroy`. 🔴 `vfft_execute` has EXTERNAL linkage, so the body is guarded by `VFFT_EXECUTE_IMPL` and exactly one TU defines it |
| `split/rank1/vfft_batch.h` (moved) | the owned-batch allocator behind `config.owned_buffers` / `config.batch`. Three descriptor shapes (c2c in-place, real, OOP 4-plane); a mismatched handle is refused, never reinterpreted |
