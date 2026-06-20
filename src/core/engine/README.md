# `core/engine/` — the C2C kernel

The irreducible complex→complex FFT engine: the plan model, the executors that
walk it, the planner that builds it, and the twiddle machinery. Everything else
in `core/` (real transforms, trig, 2D, primes, OOP) builds on top of this.

Layout: **split-complex** (separate `re[]`/`im[]` planes), **lane-batched**
(`data[n*K + lane]` — bin *n* of the K independent transforms is contiguous over
lanes). **DIT forward / DIF backward**, zero-permutation roundtrip (fwd→bwd = ×N,
no reorder). The batch dimension K is the unit of both SIMD vectorization and
multithreading.

---

## Files

| file | role |
|------|------|
| `plan.h` | re-export of the plan types + portable aligned alloc (`vfft_proto_posix_memalign`/`_aligned_free`) |
| `planner.h` | build a `stride_plan_t` from (N,K): factorize, choose variants, wire codelets, compute twiddles |
| `executor.h` | single-thread dispatch (`vfft_proto_execute_fwd/bwd`) — Tier-1 lookup → generic fallback |
| `executor_generic.h` | cold-cell correctness baseline: per-stage function-pointer loop |
| `stride_executor.h` | the **multithreaded** executor (`stride_execute_fwd/bwd`) — K-split / group-parallel |
| `twiddle.h` | per-stage group layout + twiddle table computation (Method C, DIT & DIF) |
| `proto_stride_compat.h` | serial bridge: `stride_execute_fwd` → plan's execute; slice helpers for r2c fused stages |
| `compat.h` | misc shims |

> The plan **struct** (`stride_plan_t` / `stride_stage_t`) is physically defined in
> `generated/…/plan_executors.h` (emitted by the compiler) and re-exported through
> `plan.h`. The Tier-1 specialized executors are emitted into that same file.

---

## The plan model

A plan is a list of **stages**; stage *s* applies radix `R_s` with
`stride = (product of remaining radixes) · K`. Each stage is split into **groups**;
group 0 (`needs_tw=0`) is twiddle-free (the no-twiddle leaf), the rest carry twiddles.

`stride_stage_t` (key fields):
- `radix`, `stride`, `num_groups`, `group_base[]` — geometry
- `needs_tw[]`, `cf0_re/im[]` — per-group common factor (applied to leg 0)
- `tw_scalar_re/im[]` — **T1S** scalar twiddles (codelet broadcasts internally)
- `grp_tw_re/im[]` — **FLAT/LOG3** per-element twiddle arrays
- `cf_all_re/im` — K-replicated combined twiddle (backward conj path)
- `use_log3`, `use_n1_fallback`, `tape` — variant/dispatch flags + the (B)+(A) walk
- codelet fn-ptrs: `n1_fwd/bwd` (no-twiddle butterfly), `t1_fwd/bwd` (FLAT/LOG3),
  `t1s_fwd` (T1S), `n1_scaled_bwd` (fused scaled bwd for r2c)

`stride_plan_t`: `N`, `K`, `num_stages`, `factors[]`, `stages[]`, `use_dif_forward`,
and the **override backend** (`override_fwd/bwd/destroy/data`) — how Rader, Bluestein,
DCT/DST, 2D etc. hang their own execute fn off a plan (`num_stages=0`).

---

## In-place, permutationless, transposeless — by design

The stride executor runs the whole transform in **one buffer** (the caller's `re`/`im`),
multi-pass, with **no scratch, no ping-pong, no bit-reversal, and no transpose**. This
is the architectural core, not an optimization bolted on — everything else (wide-batch
SIMD, zero-copy K-split MT, the memory-bound win) follows from it.

**Transposeless.** Textbook / four-step (Bailey) FFTs move data between passes — a
bit-reversal permutation and/or transposes — so each sub-FFT sees contiguous memory.
This engine never moves data. Stage *s* operates **at its natural stride**
`= K · Π(remaining radixes)` directly on the single buffer: it walks `num_groups`
groups by pointer offset (`group_base[g]`) and each codelet strides over its R legs.
The passes change *which elements a codelet touches*, never *where they live*. And
because the layout is **lane-batched** (`data[n·K + lane]`), SIMD vectorizes across the
K batch — contiguous in memory — so no transpose is needed to feed the vector units
either (unlike interleaved engines, which shuffle re/im to vectorize).

**Permutationless.** A standard in-place FFT must undo the digit/bit-reversal its
butterflies induce — an extra permutation pass. Here the forward runs **DIT** and the
backward runs **DIF**: the forward leaves its output in digit-reversed order, and the
backward is written to *consume that exact scrambled order* and invert it. The two
scrambles cancel, so **fwd→bwd is the original ×N with zero reorder passes** — the
roundtrip is permutation-free by construction (see `vfft_proto_execute_bwd_generic`,
which simply walks the stages in reverse).

**What it buys / the one caveat.**
- **Minimal memory traffic** — no copy, no transpose buffer; the executor streams the
  plane through cache once per stage (the basis of the memory-bound thesis).
- **Trivially-correct K-split MT** — lanes are independent and never share a transpose
  buffer, so threading is a pure pointer-offset split (see `stride_executor.h`).
- **Caveat: forward output is digit-reversed (scrambled) order.** Pure c2c roundtrips
  don't care (the inverse un-scrambles). Consumers that need *natural* spectral order
  (2D, r2c, the DCT/DST/DHT family) absorb the reorder inside their own
  pre/post-processing rather than paying a standalone permutation pass. A
  natural-order in-place c2c mode is a planned feature — see
  [natural_order_inplace.md](natural_order_inplace.md).

---

## Twiddle strategy — Method C + the codelet variants

**Method C (fully fused):** at *plan time*, bake `common_factor × per_leg_twiddle`
into the stage's twiddle table; at *execute time* the common factor `cf0` hits only
leg 0 (K multiplies), then the codelet applies the rest.

A stage's inter-stage twiddle is applied by one of **four codelet variants** — same
ABI (`radix{R}_{variant}_dit_fwd_{isa}`), different *internal* twiddle handling. `n1`
is unconditional at **stage 0** (DIT's no-twiddle leaf); the other three are
alternatives at **stages 1+** (exactly one per stage). The choice is measured per
`(R, K, ios)` and persisted in wisdom — **the optimal plan mixes them across stages.**

| variant | wisdom code | what the codelet does with twiddles |
|---------|:-----------:|--------------------------------------|
| **n1** | — (stage 0) | no twiddles — pure radix butterfly |
| **t1** (a.k.a. **FLAT**) | 0 | reads the **full materialized `(R-1)·K` twiddle table** — most memory traffic, the default/baseline twiddled CT codelet |
| **log3** | 1 | **derives** the per-leg twiddles **in registers** via a cmul / radix-decomposition ladder from a small base set — *eliminates* the table from L1/DTLB pressure |
| **t1s** | 2 | **broadcasts** one scalar twiddle *per leg* across the K lanes from registers — eliminates the `(R-1)·K` table and the per-K twiddle loads |
| **BUF** | 3 | not implemented → falls back to T1S |

**Selection precedence at stages 1+ (plan-build & cost model, mirrored): `log3` → `t1s` → `t1`.**
`log3` is highest priority *when it wins* because its win is structural (no twiddle
table at all), bigger than t1s's (no per-K loads). In practice they're near-exclusive.
Measured share across the 198-cell production wisdom (735 stage-1+ stages):

- **t1s wins ~84%** (the Raptor Lake workhorse)
- **log3 wins ~10%** (mostly R=13/17/25/32/64 — large radixes where the table dominates)
- **t1 / FLAT wins ~6%** (mostly R=12/16/20 composites)

(The generic fallback in `executor_generic.h` realizes FLAT by K-blocked broadcasting
the scalar twiddles into an L1 buffer before calling the `t1` codelet — that staging is
an executor detail, not what makes a stage "FLAT"; the defining trait is the codelet
consuming a materialized `(R-1)·K` table.)

DIF supports `t1`(FLAT) + `log3` only (T1S→FLAT in DIF). `twiddle.h` computes both
orientations (`compute_twiddles_dit` / `_dif`).

### Performance: what each variant buys, and why plans mix them

Measured cycles-per-butterfly (AVX2, Raptor Lake, from `generator/cost_model/generated/radix_cpe.h`),
swept over the batch width `me=K ∈ {256, 4096, 65536}`. The **K-scaling** column is the
whole story:

| R | n1 | t1 (FLAT) 256→64k | t1s 256→64k | log3 256→64k |
|---|----|-------------------|-------------|--------------|
| 4 | 1.17 | 1.49 → **3.02** | 1.46 → **1.58** | 1.67 → 2.16 |
| 8 | 3.48 | 4.85 → **6.87** | 4.29 → **4.34** | 5.46 → 5.93 |
| 16 | 12.8 | 25.8 → 23.6 | **13.5 → 13.8** | 24.3 → 18.8 |
| 32 | 87.7 | 78.9 → **118.0** | **74.2 → 73.6** | 77.0 → 77.5 |

- **`n1`** — no twiddle at all, the cheapest stage and K-flat. This is why DIT pins the
  no-twiddle leaf to **stage 0**: spend the free variant where no twiddle is owed.
- **`t1` (FLAT)** — the materialized `(R-1)·K` table makes it **K-traffic-bound**: cost
  *climbs with K* (R4 doubles 1.49→3.02; R32 +50% 79→118) as the table grows and thrashes
  L1/DTLB. It's the baseline you want to *avoid* at high K.
- **`t1s`** — broadcasting scalars from registers makes it **essentially K-flat** (R32
  74.2→73.6 vs t1's 79→118). Removing the per-K twiddle loads kills the high-K penalty —
  that's why it's the workhorse.
- **`log3`** — register-derived twiddles, also K-flat but **compute-heavier**: at small/mid
  radix it loses to t1s (R16 18.8 vs 13.8). Its edge is *structural* — zero twiddle table →
  no DTLB pressure — which pays off on the **largest radixes** and on **AVX-512 / server**
  (more vector regs to hold the derivation). On RPL/AVX2 t1s usually still edges it, so it's
  the rarer pick here; on SPR it wins far more (host-dependent — see `codelet_selection_findings`).

**Why a plan mixes them.** A plan is `N = R0·R1·…`; stage 0 is `n1`, and each later stage
picks `t1`/`t1s`/`log3` *independently*. Every stage sees the same `K`, but a **different
`ios`** (inter-output stride `= K · Π(later factors)`), so each stage's twiddle-access
pressure differs — and the best variant differs with it. The planner scores each stage on
its own `(R, K, ios)` with the CPE above (precedence `log3→t1s→t1`), so the chosen plan is a
**per-stage mix** that no uniform choice matches.

Measured mix-vs-all-FLAT, single-thread, paced (`build_tuned/benches/bench_variant_flat_vs_mix.c`):

| cell | factors | mix (stages 1+) | mix vs all-FLAT |
|------|---------|-----------------|:---------------:|
| `N=100000 K=4` (low-K hi-N) | 10·16·25·25 | T1S,T1S,**LOG3** | **1.26×** |
| `N=60060 K=4` (low-K hi-N) | 12·11·13·5·7 | T1S,T1S,T1S,**LOG3** | **1.27×** |
| `N=65536 K=32` (K-bound) | 4⁶·16 | all-T1S | **1.19×** |

The two low-K composites win ~25–27% because a **LOG3 on the final, largest-radix stage**
(R=25, R=13) eliminates the biggest twiddle table while T1S handles the mids — that last-stage
swap is the whole delta. The K-bound pow2 cell is already optimal as uniform T1S (every stage
is twiddle-load-bound). Takeaway: **uniform-t1s is the right default; the wins come from a
LOG3 tail on the largest-`ios`/largest-radix stage** when K is small enough that the table —
not per-K loads — is the bottleneck.

---

## Executors

There are **two executor families**, and they are *mutually exclusive at include
time* (both define `_stride_cmul_*`, slice helpers, `STRIDE_MAX_STAGES`, …). A
translation unit includes one or the other:

### 1. `executor.h` (+ `executor_generic.h`) — single-thread, Tier-1 path
`vfft_proto_execute_fwd/bwd(plan, re, im, slice_K)`. Two-tier dispatch:
1. **Override** — if `plan->override_fwd` set, call it (Rader/Bluestein/DCT/2D…).
2. **Tier-1 specialized** — `vfft_proto_lookup_fwd_avx2/512(plan)` returns an emitted
   **(B)+(A) plan-shaped** executor for known shapes (≈5–6% faster than the loop). It
   walks a pre-baked "tape" of group invocations.
3. **Generic fallback** (`executor_generic.h`) — the per-stage, per-group
   function-pointer loop that handles *every* plan shape (the correctness baseline).
   Four twiddle paths per group: `n1` (no-tw) / LOG3 / T1S / FLAT.

`slice_K ≤ plan->K` lets a caller run the engine on a contiguous lane sub-range —
this is what the manual K-split MT wrappers and the r2c block path use.

### 2. `stride_executor.h` — the multithreaded executor
`stride_execute_fwd/bwd(plan, re, im)` (no slice arg; threads internally over `plan->K`).
Strategy is chosen at execute time on the thread count `T = stride_get_num_threads()`:

- **`K/T ≥ STRIDE_KSPLIT_THRESHOLD` (256) → K-split**: each thread owns a contiguous
  lane slice (rounded to a multiple of 8 = one cache line → no false sharing). **No
  barriers, no copies, no per-thread plans** — the embarrassingly-parallel path.
- **else → group-parallel**: every stage, each thread takes a contiguous slice of *that
  stage's groups* (`[ng·tid/T, ng·(tid+1)/T)`) and runs them at **full K**, with a
  **barrier between consecutive stages** (stage s+1's groups depend on s). Better when K
  is too small for clean K-split — keeps full codelet utilization, no false sharing on K.
- `T≤1` or `K<4` → serial slice. **DIF runs single-threaded** (v1.1).

Thread pool (`support/threads.h`): persistent workers created by
`stride_set_num_threads(n)`; worker *i* is pinned to **core i+1**, so the **caller
must be on core 0** (P-cores 0..7 on the 14900KF). Thread 0 = caller (no dispatch
overhead). Spin-wait completion; no OpenMP/TBB.

---

## The planner (`planner.h`)

Three entry points, all producing a ready-to-execute `stride_plan_t`:

| fn | behavior |
|----|----------|
| `vfft_proto_auto_plan(N,K,reg,wis)` | wisdom-first (honors its factors+variants+orientation), else greedy factorize |
| `vfft_proto_wise_plan(N,K,reg,wis)` | strict wisdom — NULL on miss |
| `vfft_proto_estimate_plan(N,K,reg)` | cost-model path — greedy factorize, T1S everywhere |

Pipeline (`plan_create_ex`): **factorize** (greedy largest-first over the available
radixes, then `reorder_pow2_innermost` so pow2 radixes sit innermost for SIMD) →
**wire codelets** per stage from the registry by variant (`t1_dit_fwd` / `t1_dit_log3_fwd`
/ `t1s_dit_fwd`, DIF analogues) → **compute groups + twiddles** → **pre-walk the (B)+(A)
tape** for Tier-1 lookup. `vfft_proto_plan_destroy` honors `override_destroy` first.

### Plan search (`../planning/`) — how the good plans get chosen

`planner.h` only *builds* a plan from a given factorization+variants. *Choosing* the
best one is a separate family of search strategies in `core/planning/`, spanning a
**fidelity ↔ cost** spectrum. All but `estimate` **measure**, so they're
**calibration-time** tools: search once, persist the winner to wisdom, and the runtime
does a pure lookup (`auto_plan`). `estimate` is the model-only exception, usable at
runtime for cold cells.

| strategy | file | how it picks | cost |
|----------|------|--------------|------|
| **estimate** | `estimate_plan.h` | V4 **cost model only**, no measurement — enumerate factorizations, score (fewer-stages + wide-radix-outer penalty + per-stage buffer-pass), pick min | µs, 0 benches |
| **DP** | `dp_planner.h` | FFTW-style **recursive measured** search: try each radix as first stage, recurse on N/R with **sub-problem memoization**, bench full candidates, permute the winning set | ~150 benches |
| **screened exhaustive** | `exhaustive_screened.h` | enumerate all (multiset × permutation), **rank by V4 cost model first**, then bench only the promising ones (V4 scores the *full* plan at parent (N,K) → preserves tail-heavy `[…,32,16]` patterns) | mid |
| **flat exhaustive** | `exhaustive_plan.h` | bench **every** (multiset × permutation) at the real parent (N,K) context; 3 warmups, best-of-3, 1.5× quick pre-screen | ~500–1500 benches |
| **patient exhaustive** | `exhaustive_patient.h` | flat exhaustive with **no pre-screen**, 5 warmups / best-of-7, **inter-candidate thermal pacing** (~200 ms sleep), optional top-N second-pass re-bench | slowest, highest-fidelity |

Why measured search beats pure recursion/model: a plan's cost is **context-dependent**
(cache/TLB interactions of the *whole* plan at the real (N,K)), so benching at the
**parent context** — what flat/patient/screened do — captures signal that isolated
sub-plan measurement (DP's memoization) or a model (estimate) can miss, especially
tail-heavy decompositions. That's the accuracy/cost trade across the table. (Detail
lives in the forthcoming `planning/` README; this is the engine-side pointer.)

---

## Build-config values (defaults)

| macro | value | meaning |
|-------|:-----:|---------|
| `STRIDE_MAX_STAGES` | 9 | max factorization depth (size of `factors[]`/`stages[]`) |
| `STRIDE_MAX_RADIX` | 32 | largest radix the stride executor's tables size for |
| `STRIDE_KSPLIT_THRESHOLD` | 256 | `K/T ≥ this` → K-split, else group-parallel |
| `STRIDE_TW_BLOCK_K` / `VFFT_PROTO_TW_BLOCK_K` | 64 | FLAT twiddle broadcast L1 block (doubles/lane-block) |
| `VFFT_PROTO_VARIANT_{FLAT,LOG3,T1S,BUF}` | 0,1,2,3 | per-stage twiddle variant codes (match the wisdom file) |
| available radixes | `{64,32,25,20,19,17,16,13,12,11,10,8,7,6,5,4,3,2}` | greedy factorization set (must match the registry) |

---

## Optimizations at a glance

- **Method-C fused twiddle** — combined table baked at plan time; execute touches
  leg 0 only.
- **Tier-1 (B)+(A) specialization** — emitted plan-shaped executors skip the
  per-group dispatch overhead (~5–6%).
- **SIMD preprocessing helpers** (`_stride_cmul_*`, `_stride_broadcast_2`) — AVX2/AVX-512
  versions of the executor's cf0-multiply / twiddle-broadcast bookkeeping (~5–15% on
  cells where that path dominates); codelets do their own SIMD.
- **Zero-permutation roundtrip** — DIT fwd + DIF bwd cancel; no bit-reversal pass.
- **K-split with cache-line-aligned slices** — MT with no barriers and no false sharing.
- **Per-stage variant mixing** — the planner can assign FLAT/LOG3/T1S independently per
  stage; the measured optimum is usually a mix.

---

*Note: `estimate` is the **intended zero-measurement path** for users who don't want to
run any calibration at all (no DP, no exhaustive) — it picks a plan from the V4 cost model
alone, in microseconds. It is **designed-but-not-yet-wired**: its cost-model inputs
(`factorizer.h`, `radix_memboundness.h`) still reference the deleted `prototype/` tree and
need **re-homing into `core/`** before it builds. Until then the live planning path is
wisdom + DP; estimate gets wired next.*
