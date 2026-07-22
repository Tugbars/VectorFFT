# IL & OOP adapters — post-P1a API reference (6a16–6a18)

*What this covers: every entry point, contract, and mechanism added AFTER
`roadmap/il_architecture.md` (P1a as-built, 2026-07-14). P1a documented
`fwd_ilin[_jit]`, `bwd_ilout[_jit]` and the il_in/il_out codelet families;
the narrative for everything below lives in
`performance/mkl_geometry_contracts.md` §6a16–6a18. This file is the
reference card — signatures, aliasing rules, rejection matrix, resolution
recipes — for the parts that until now existed only in code comments and
session logs.*

All functions live in `src/core/oop/il_execute.h` unless marked
`oop_execute.h`. All return `int`: 0 = executed, −1 = rejected untouched
(resolve-before-touch: no partial writes on rejection).

## 1. New entry points

| fn | signature (after `const stride_plan_t *plan`) | semantics |
|---|---|---|
| `fwd_ilout_core` | `double *re, double *im, double *out_z, size_t slice_K` | split→z; stages [0, n−1) generic in-place on split, last stage folded into interleaved store |
| `fwd_ilout_jit` | `…, vfft_proto_exec_range_fn range_fn` | same, interior via `range_fn(plan,re,im,K,plan->K,0,n−1)`; NULL → core |
| `bwd_ilin_core` | `const double *z, double *re, double *im, size_t slice_K` | z→split; top stage folded from interleaved load (n1_bwd + cf_all conj), stages [n−2..0] generic |
| `bwd_ilin_jit2` | `…, range_fn` | jit tier: fused `t1s_dit_bwd_il_in` + verbatim leg0-conj entry, interior `range_fn(…,0,n−1)`; NULL or t1sb radix gap → whole core |
| `fwd_il2il_core` | `const double *z_in, double *work_re, double *work_im, double *z_out, size_t slice_K` | z→z: entry fold, generic interior [1, n−1), exit fold |
| `fwd_il2il_jit` | `…, range_fn` | interior via `range_fn(…,1,n−1)` |
| `bwd_il2il_core` | same shape as fwd_il2il_core | entry gen (n1_bwd + cf_all conj), interior [n−2..1], exit fold |
| `bwd_il2il_jit` | `…, range_fn` | fused t1s entry + `range_fn(…,1,n−1)` + **plain-flavor** exit (see §4) |
| `fwd_oop_jit` *(oop_execute.h, pre-existing, listed for the pair)* | `const double *src_re, const double *src_im, double *dst_re, double *dst_im, size_t slice_K, vfft_proto_exec_fn stages1_jit` | split→split OOP: stage 0 via 7-arg OOP n1 codelets src→dst, stages 1.. in-place on dst (classic jit `start_stage=1` or generic) |
| `bwd_oop_jit` *(oop_execute.h, NEW 6a18)* | same | pointer-swap identity `swap(DFT(im,re))` composed with the **forward** executor |

`bwd_oop` now delegates to `bwd_oop_jit(…, NULL)`.

## 2. Tier model and resolution recipes

Two executor function types:

```c
typedef void (*vfft_proto_exec_fn)(const stride_plan_t*, double*, double*,
                                   size_t, size_t, int);            /* classic: start_stage */
typedef void (*vfft_proto_exec_range_fn)(const stride_plan_t*, double*, double*,
                                   size_t, size_t, int, int);       /* + stop_stage */
```

Range semantics (both directions): run stages S with
`start_stage ≤ S < stop_stage` in the direction's natural walk order.
The range typedef is duplicated identically in `il_execute.h` and
`jit_runtime.h` (C11-legal) so the adapter header stays jit-independent.

| need | resolver (`src/dag-fft-compiler/jit/jit_runtime.h`) | tier order |
|---|---|---|
| classic fwd/bwd | `vfft_proto_plan_jit_fwd(plan)` / `_bwd(plan)` | baked lookup → process registry → runtime compile |
| range fwd/bwd | `vfft_proto_plan_jit_fwd_range(plan)` / `_bwd_range(plan)` | registry → runtime compile (**skips baked** — baked fns have no range flavor) |

NULL from any resolver → pass NULL → adapters run their `_core` twin. Tier
purity rule: a jit wrapper with any internal resolve gap (e.g. t1s-bwd-il_in
radix hole) falls back to the **whole** core path, never a mixed-tier
pipeline — this keeps every wrapper bit-identical to a single-tier reference.

**The bwd_oop_jit trap (contract, gated):** the swap identity lives in the
data pointers, not the direction. Pass the **FORWARD**-resolved executor.
`plan_jit_bwd`'s fn (fused t1s + leg0-conj) applied to swapped data is a
different transform.

Orchestrator auto-population (`h.exec_fwd/bwd`) sits behind
`#ifdef VFFT_USE_JIT` in `plan_orchestrator.h`; the adapters never require
it — explicit resolution above is the supported path.

## 3. Aliasing, scratch, ordering

| contract | rule |
|---|---|
| `z_in == z_out` | sanctioned for `*_il2il_*` only (entry fold fully consumes z before exit fold writes it); gated BIT incl. jit tier |
| z ↔ work split overlap | forbidden, ungated |
| ilin/ilout z ↔ split overlap | forbidden |
| oop src/dst | distinct; **src preserved** (gated) |
| in-place-z reality | API-level only: il2il needs the caller-provided NK-complex split work pair. No scratch-free z path exists by design — z is a boundary format; interior stages are split-native |
| z indexing | elementwise pairing: `(re[i], im[i]) ↔ (z[2i], z[2i+1])` over the plan's absolute `group_base + j·stride + k` lattice |
| ordering | il adapters inherit the engine's scrambled contract unchanged. `bwd_oop*` is the swap-identity inverse: **forward** ordering semantics, ≠ the stride bwd executor's contract |
| normalization | all backwards unnormalized, as ever |

## 4. Rejection matrix (−1) and flavor rules

| condition | affected |
|---|---|
| override plans (Rader/Bluestein) | all il/oop adapters |
| `num_stages < 2` | `*_il2il_*` (boundary folds would overlap); jit il2il wrappers with NULL range_fn delegate to core which rejects |
| DIF orientation | `fwd_oop*`, `bwd_oop*` (stage 0 carries twiddles — NO_DESTROY_INPUT physics) |
| DIT plan, twiddled stage-0 group | fwd entry resolvers (impossible in practice; belt and braces) |
| K % 8 | caller's pre-existing engine constraint, unchanged |

Flavor rules (both gate-caught, §6a17):
- **fwd folds are variant-bound** (T1S/LOG3/FLAT il_out per `st->t1s_fwd`/`use_log3`) — matches both generic and jit fwd tiers.
- **jit bwd exit fold is plain-flavor unconditionally** (`_vfft_il_resolve_bwd_exit_jit`): `STAGE_DIF_BWD` never binds log3. Core bwd exit stays variant-bound to match the generic tier. Perf consequence → selection rule: **bwd z→z: DIT→jit, DIF→core; fwd z→z: core** (jit interior adds nothing measurable; see 6a17 bench).

## 5. Fold-helper internal API (for future consumers: fftnd-IL, rm folds)

```c
typedef struct { vfft_il_n1f_fn n1, tw; } vfft_il_infold_t;   /* z -> split */
typedef struct { vfft_il_n1b_fn n1, tw; } vfft_il_outfold_t;  /* split -> z */
```

Six resolve/apply pairs, all `resolve` never touching data:
`_vfft_il_{resolve,apply}_fwd_entry`, `_fwd_exit`, `_bwd_exit`
(+ `_resolve_bwd_exit_jit` plain-flavor twin), `_bwd_entry_gen`
(n1_bwd + cf_all conj), `_bwd_entry_jit` (fused t1s + `_vfft_il_bwd_leg0_conj`,
a **verbatim** copy of `VFFT_PROTO_BWD_LEG0_CONJ_avx2`'s FMA grouping — do not
"simplify" to `_stride_cmul_scalar_inplace`; bit parity with the jit reference
depends on the exact grouping). The five `_core` adapters are compositions of
these; `fwd_ilin`/`bwd_ilout` DIF branches keep inline copies (accepted
duplication, capped refactor blast radius).

## 6. Codelet inventory delta

`t1s_dit_bwd il_in` extended from {4,5,8,16,25,32} to the full resolver set:
+{2,10,20,64}, both ISAs, files
`codelets/il/{avx2,avx512}/r{R}_t1s_dit_bwd_ip_il_in_{isa}.c`, generator flags
`--twiddled --in-place --t1s --bwd --ip-il-in`. Gated 24/24 BIT vs originals.
`VFFT_IL_POW2_ONLY` excludes {5,10,20,25} from tables as elsewhere.

## 7. Runtime-jit + gate/bench reproduction (container recipe)

`build_tuned/benches/` inventory: `gate_tw2.c gate_1020.c gate_dbl3.c` (codelet-level),
`gate_adapt.c` (adapter gate: T1–T11, wisdom `wad.txt`), `probe_jit.c`
(runtime-jit smoke), `bench_fwdfold.c bench_bwdfold.c` (spike wisdom,
same-process medians).

Runtime jit needs three `-D` overrides (Linux default `VFFT_PROTO_JIT_REPO`
is a WSL path — the historical silent-failure cause):

```
-DVFFT_PROTO_JIT_REPO='"<tree>/src/dag-fft-compiler"'
-DVFFT_PROTO_JIT_DIR='"/tmp/jitcache"'
-DVFFT_PROTO_JIT_CODELETS='"@/tmp/jitcache/codelets_linux.rsp"'
-I src/dag-fft-compiler/jit  -ldl
```

where the rsp lists `-fPIC` avx2 codelet objects (radices
{2,4,5,8,10,16,20,25,32,64} × families n1_{fwd,bwd}, t1_dit_fwd[_log3],
t1s_dit_fwd, t1_dif_fwd[_log3], t1s_dit_bwd, t1_dif_bwd). Emitted plan TUs
export dual symbols since `VFFT_PROTO_JIT_VERSION 4`:
`vfft_proto_jit_exec_range` (per-line `if (S < stop_stage)` guards around
untouched STAGE macros) + the classic 6-arg symbol as a saturated-stop
wrapper. Pre-v4 cached libs lack the range symbol → resolvers return NULL →
core fallback (graceful).
