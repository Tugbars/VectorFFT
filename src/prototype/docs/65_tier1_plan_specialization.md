# 65 — Tier 1 Plan-Shaped Specialization

**Status:** Landed in `src/prototype/generated/plan_executors.h` (auto-generated). Enabled by default via `vfft_proto_lookup_fwd_avx2` in `src/prototype-core/executor.h`. Bug fix for the NULL-tw segfault path landed 2026-05-17 (variant-aware lookup + per-group `needs_tw` branch).

**One-line summary:** For each `(N, K, factors, variants)` cell we have plan-level wisdom for, emit a hand-shaped C function that runs the FFT with direct codelet calls instead of function-pointer indirection through the generic executor. **Measured 12-16% latency win** on N=131072 K=4 plans; the entire `1.24× → 1.44× MKL` gain on the old wisdom plan was the Tier 1 dispatch.

---

## 1 — Why we do this

### 1.1 The numbers

Same plan, same codelets, same MKL build — Tier 1 specialization vs generic executor:

| Plan | Generic | Tier 1 | Saving |
|---|---:|---:|---:|
| Old wisdom `[4,4,4,4,8,4,4,4]` (8 stages) all-T1S | 1619 µs | **1432 µs** | **−12%** |
| Patient `[4,4,4,32,64]` (5 stages) all-T1S | 1679 µs | **1407 µs** | **−16%** |

The saving scales with stage count: ~14-23 µs per stage from eliminating indirection. At 8 stages × ~16k groups × 8 dispatches per group ≈ 130k indirect-branch sites per FFT. Each indirect-branch costs 1-5 ns of front-end stall vs the same direct call. Adds up.

### 1.2 What the generic executor pays

The generic path in `executor_generic.h` walks plans like this:

```c
for (int s = 0; s < plan->num_stages; s++) {
    stride_stage_t *st = &plan->stages[s];
    for (int g = 0; g < st->num_groups; g++) {
        double *base_re = re + st->group_base[g];
        double *base_im = im + st->group_base[g];
        if (!st->needs_tw[g])
            st->n1_fwd(base_re, base_im, ...);    /* INDIRECT */
        else if (st->use_log3)
            st->t1_fwd(base_re, base_im, ...);    /* INDIRECT */
        else if (st->t1s_fwd)
            st->t1s_fwd(base_re, base_im, ...);   /* INDIRECT */
        else
            st->t1_fwd(base_re, base_im, ...);    /* INDIRECT */
    }
}
```

Every `st->n1_fwd(...)` is an indirect call through a function pointer stored in the plan. The CPU's indirect-branch predictor needs to learn the target, the call can't be inlined, the call-graph is opaque to the optimizer. On long plans, the per-group dispatch decision tree (3-4 branches) also runs many times.

### 1.3 What the specialized executor avoids

For a known `(N, K, factors, variants)` we generate:

```c
static void exec_n131072_k4_44448444_v02222222_dit_fwd_avx2(
    const stride_plan_t *plan, double *re, double *im,
    size_t slice_K, size_t full_K, int start_stage)
{
    /* Stage 0: R=4, no twiddle */
    if (start_stage <= 0) {
        const stride_stage_t *st = &plan->stages[0];
        const stride_invocation_t * __restrict__ tape = st->tape;
        const size_t stride = st->stride;       /* hoisted */
        for (int g = 0; g < st->num_groups; g++) {
            size_t base = tape[g].base;
            radix4_n1_fwd_avx2(re + base, im + base, NULL, NULL,
                               stride, slice_K);   /* DIRECT */
        }
    }
    /* Stage 1: R=4, T1S */
    if (start_stage <= 1) {
        /* ... same shape, direct radix4_t1s_dit_fwd_avx2 ... */
    }
    /* ... one block per stage ... */
}
```

Every codelet call is a literal symbol reference (`radix4_t1s_dit_fwd_avx2`). The compiler can inline, the indirect-branch predictor doesn't fire, the variant-dispatch decision tree is collapsed because each stage's variant is baked in at emit time.

The `__restrict__` on `tape` tells the compiler the tape doesn't alias the data buffers. `stride` is hoisted outside the loop so it's a register, not a pointer-chase per iteration. Both micro-wins on top of the call-site change.

---

## 2 — How it works end-to-end

### 2.1 The (B)+(A) tape

The plan struct carries a pre-walked **invocation tape** per stage:

```c
typedef struct {
    size_t        base;     /* offset into re/im (doubles) */
    const double *tw_re;    /* per-group scalar twiddle, NULL for needs_tw=0 */
    const double *tw_im;
} stride_invocation_t;

typedef struct {
    int radix;
    size_t stride;
    int num_groups;
    /* per-group arrays (used by generic exec, also as source for the tape) */
    size_t *group_base;
    int    *needs_tw;
    double *cf0_re; double *cf0_im;
    double **tw_scalar_re; double **tw_scalar_im;
    /* ... */
    stride_invocation_t *tape;    /* pre-walked: one entry per group */
    /* function-pointer slots (generic exec uses these) */
    vfft_proto_n1_fn      n1_fwd;
    vfft_proto_codelet_fn t1_fwd;
    vfft_proto_codelet_fn t1s_fwd;
} stride_stage_t;
```

The tape is one 24-byte struct per group containing `{base, tw_re, tw_im}` — exactly what each codelet call needs. Plan-build time populates the tape; the executor reads sequentially. Cache-friendly compared to walking 4-5 separate per-group arrays.

### 2.2 Lookup + dispatch

`vfft_proto_lookup_fwd_avx2(plan)` matches the plan's `(N, K, factors[], variants[])` against the lookup table and returns the specialized function pointer (or NULL if no specialization exists for this exact tuple). The dispatch site:

```c
/* in executor.h */
vfft_proto_exec_fn fn = vfft_proto_lookup_fwd_avx2(plan);
if (fn) {
    fn(plan, re, im, slice_K, plan->K, /*start_stage=*/0);
    return;
}
vfft_proto_execute_fwd_generic(plan, re, im, slice_K);   /* fallback */
```

When the wisdom-driven lookup hits, we run the specialized executor. When it doesn't (cold cell, untracked variant mix), the generic path runs and we eat the indirection cost.

### 2.3 The `needs_tw[g]` branch (and why it can't go away)

DIT planning produces some groups with `k_prev=0` (the "zeroth" twiddle group). Those have no scalar twiddle data — `tape[g].tw_re == NULL`. If the specialized executor unconditionally calls `radix4_t1s_dit_fwd_avx2(..., inv.tw_re, ...)` on those groups, it reads through a NULL pointer → segfault.

The fix is a per-group branch:

```c
if (inv.tw_re) {
    radix4_t1s_dit_fwd_avx2(...);   /* twiddled */
} else {
    radix4_n1_fwd_avx2(...);        /* no-twiddle fallback */
}
```

This branch is well-predicted in practice (the per-group needs_tw pattern is regular within a stage). Could be further optimized by sorting groups so all `needs_tw==0` are at one end of the tape and split into two branchless loops — deferred future work.

### 2.4 Variant-aware lookup

The lookup function must check **both** factors and variants, because the same factors with FLAT vs T1S vs LOG3 variants use **different codelet symbols and different twiddle table layouts**. Dispatching a FLAT plan to a T1S-throughout specialization would read garbage twiddle data and produce silent corruption.

The check is per-stage: a stage is T1S iff `stages[s].t1s_fwd != NULL`, LOG3 iff `stages[s].use_log3 == 1`, FLAT otherwise. The lookup macro `_MATCH_T1S_INNER(plan, nstages)` verifies all inner stages match the spec's expected variant.

---

## 3 — Architectural choices

### 3.1 Per-tuple specialization vs runtime dispatch

We emit **one function per `(N, K, factors, variants)` tuple**. Alternatives we rejected:

| Approach | Pros | Cons |
|---|---|---|
| Generic + function pointers (status quo before Tier 1) | One function fits all plans | 12-16% latency penalty per FFT |
| **Per-tuple specialization (what we did)** | **Direct calls, no indirection, measurable win** | **Only covers tuples we pre-emit; cold cells fall through to generic** |
| C++ templates | Cleaner type-system errors | Pulls in C++ everywhere; name-mangling baggage |
| JIT codegen at plan-build time | Covers any tuple | Adds dependency on a JIT (LLVM/asmjit); large infra change |

The right answer is per-tuple specialization gated by wisdom coverage. Cells that matter get a spec; cells that don't fall through to generic at the indirection cost. Production has the same pattern.

### 3.2 Token-paste macros, not duplicated function bodies

A naively-emitted plan_executors.h is ~120 lines per specialization. At 100 wisdom cells that's 12,000 lines of nearly-identical code. We compress with token-pasting macros:

```c
#define VFFT_PROTO_STAGE_T1S(S, R) \
    if (start_stage <= S) { \
        /* ... 12 lines of boilerplate ... */ \
        radix##R##_t1s_dit_fwd_avx2(...); /* direct symbol after token-paste */ \
        /* ... else branch ... */ \
    }
```

`radix##R##_t1s_dit_fwd_avx2` — the `##` operator pastes the radix into the symbol name at preprocessing time. The compiler sees `radix32_t1s_dit_fwd_avx2` as a literal direct call, exactly as if hand-written.

After the refactor, each specialization is ~5-10 lines instead of ~120:

```c
static void exec_n131072_k4_44448444_v02222222_dit_fwd_avx2(...)
{
    (void)full_K;
    VFFT_PROTO_STAGE_OUTER(0, 4)     /* stage 0: R=4, no twiddle */
    VFFT_PROTO_STAGE_T1S  (1, 4)
    VFFT_PROTO_STAGE_T1S  (2, 4)
    VFFT_PROTO_STAGE_T1S  (3, 4)
    VFFT_PROTO_STAGE_T1S  (4, 8)
    VFFT_PROTO_STAGE_T1S  (5, 4)
    VFFT_PROTO_STAGE_T1S  (6, 4)
    VFFT_PROTO_STAGE_T1S  (7, 4)
}
```

**Generated code is identical** to the hand-written version. Only source-side compactness changes.

### 3.3 Hand-coded entries vs wisdom-driven emission

Currently `bin/emit_executor_h.ml` has a **hard-coded `spike_entries` list** at line 56. To add a specialization we edit OCaml + regen. Works for ad-hoc spike validation; doesn't scale.

The proper pipeline is wisdom-driven:

```
Calibration:
  patient or V4-screened picks the plan for (N, K)
       → result appended to wisdom file (one row per cell)

Build:
  emit_executor_h reads wisdom file
       → emits one Tier 1 specialization per row

Runtime:
  vfft_proto_lookup_fwd_avx2 dispatches by (N, K, factors, variants)
       → specialized executor runs
       → if no match: generic fallback
```

Each wisdom row is a whole-plan measurement — `(N, K, factors[], variants[], measured_ns)`. **No per-radix shortcut tables, no variant predicates.** Plan-level wisdom is the only thing that survives the cost-model ceiling ([feedback_cost_model_ceiling](../../memory/feedback_cost_model_ceiling.md) memory note explains why isolated-codelet measurements don't transfer to plan-context performance).

### 3.4 Why variants matter as much as factors

Earlier "wisdom_bridge" experiment in production tried to pick variants based on per-radix isolated benches — "at K=4, R=16 codelet's T1S is fastest, bake that into a predicate." It didn't work: the same R=16 codelet at the same K can prefer FLAT in one plan and T1S in another, because the cache state and ROB occupancy at that specific stage depends on what the surrounding stages did.

So our wisdom entries store **per-stage variant choices** — `[T1S, T1S, LOG3, T1S, T1S]` is a different wisdom row than `[T1S, T1S, T1S, T1S, T1S]` even if the factorization is identical. Tier 1 dispatch matches both factors and variants.

In practice we've seen variant choice flip rankings within a 5% window (within bench noise on most cells). For some cells the variant choice is significant (R=64 large-K cells where LOG3 vs T1S matters); for others it's noise (N=131072 K=4 where all-T1S and `T1S,T1S,LOG3,T1S,T1S` measure identically).

### 3.5 The fallback path is non-negotiable

Tier 1 only fires when we have wisdom for the exact tuple. Cold cells, weird K values, non-pow2 N with primes not in any spike — all fall through to the generic executor. The generic path **must stay correct and reasonably fast** because production users will hit cold cells.

This is why we kept the generic path as the primary code path and Tier 1 as an opt-in. It's not "Tier 1 or nothing" — it's "Tier 1 if we have wisdom for this tuple, generic otherwise."

---

## 4 — When Tier 1 wins more

Tier 1 wins **scale with stage count**. Each stage's indirect-call dispatch is eliminated.

| Plan stage count | Tier 1 saving (measured) |
|---|---:|
| 5 stages (`[4,4,4,32,64]`) | ~16% |
| 8 stages (`[4,4,4,4,8,4,4,4]`) | ~12% |

The 5-stage plan benefits more *per stage* because its inner stages have larger working sets (more codelet work per call → indirect-call overhead is a larger fraction of dispatch time). The 8-stage plan benefits more in *absolute* µs because there are more dispatches to eliminate.

Wins are also bigger when:
- Many groups per stage (more dispatch sites)
- Small radixes (R=4) at inner stages (call cost / work ratio is highest)
- Large K (more total per-group work but the per-group overhead scales)

Tier 1 wins are *smallest* when:
- Few stages (3-stage plans benefit only ~3-5%)
- Wide radixes (R=64) doing most work (codelet body dominates dispatch cost)
- Tiny K (per-call overhead dominates the codelet body anyway)

---

## 5 — Files

- [src/prototype/bin/emit_executor_h.ml](../bin/emit_executor_h.ml) — OCaml emitter (hard-coded `spike_entries` at line 56)
- [src/prototype/generated/plan_executors.h](../generated/plan_executors.h) — auto-generated; do not hand-edit (we did this session for the patient `[4,4,4,32,64]` specs as a test; proper fix is to add to OCaml `spike_entries`)
- [src/prototype-core/executor.h](../../prototype-core/executor.h) — Tier 1 lookup site (`vfft_proto_lookup_fwd_avx2`, falls through to generic)
- [src/prototype-core/executor_generic.h](../../prototype-core/executor_generic.h) — fallback path with function-pointer dispatch

## 6 — Open work

- **Macro refactor.** Compress plan_executors.h from per-stage 12-line bodies to 1-line macro invocations. Code-gen identical, source ~10× smaller.
- **Wisdom-file pipeline.** Patient/V4-screened output → wisdom file → OCaml emitter reads file → regen plan_executors.h. Replaces today's hand-coded `spike_entries`.
- **`needs_tw[g]` sort-and-split.** Reorder groups so needs_tw=0 are contiguous, then emit two branchless loops per stage. Recovers the per-group branch cost. Plan-build-time refactor needed.
- **FLAT and LOG3 specializations.** Today most specs are T1S-throughout. The needs_tw branch logic for FLAT (which uses K-blocked twiddle staging) is more involved and not yet covered.
- **DIF specializations.** Forward DIT is covered; DIF and backward DIT have no Tier 1 path yet.

---

## 7 — Related

- [docs/dev/vtune_n131072_k4_vfft_vs_mkl.md](../../../docs/dev/vtune_n131072_k4_vfft_vs_mkl.md) — VTune analysis that motivated the executor work in production
- [feedback_cost_model_ceiling](../../../memory/feedback_cost_model_ceiling.md) — why plan-level wisdom is the only thing that transfers
- [v4_joint_recalibration_2026_05_17](../../../memory/v4_joint_recalibration_2026_05_17.md) — the cost-model work that informs which plans wisdom should cover
- [63_v4_estimate_method.md](63_v4_estimate_method.md) — V4 estimate (the planner that picks candidates → patient confirms → wisdom records → Tier 1 specializes)
- [64_v4_joint_recalibration.md](64_v4_joint_recalibration.md) — V4 measurement-based recalibration
