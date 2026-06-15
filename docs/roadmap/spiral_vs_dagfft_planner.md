# SPIRAL's planner vs. dag-fft-compiler's: what's actually upgraded, and what transfers

*Roadmap study, 2026-06-15. Produced from the SPIRAL literature (web-researched + adversarially verified against primary sources) and a read-through of the current `src/dag-fft-compiler/core/` planner. File references are relative to this doc.*

## 1. What SPIRAL's planner actually is

"Well-upgraded DP planner" is slightly misleading. SPIRAL's DP recursion is **algorithmically the same idea as yours** — memoized recursion that builds the best plan for a transform bottom-up from the best plans of its sub-transforms (the same heuristic FFTW uses). The DP itself is not the upgrade.

What's upgraded is **the space the DP runs over**. SPIRAL doesn't search "factorizations of N." It searches **ruletrees**: each node is `(transform, breakdown-rule, instantiation)`. Recursively applying breakdown rules until all leaves are terminal yields one fully-expanded **SPL formula** (a product of structured sparse matrices — Kronecker products `A⊗I` = loops, twiddle diagonals `T`, stride permutations `L`). DP, exhaustive, random, and evolutionary search **all manipulate ruletrees, not code** — so adding a transform or rule needs zero change to the search engine. The DP table is keyed by `(transform, size)` (+ a context tag on the vector path) and stores the min-cost ruletree per key; each sub-transform is solved/timed once and reused everywhere.

Two optimality facts that bear directly on the memory-bound thesis:

- **DP is provably optimal only for arithmetic/op-count cost.** SPIRAL states the assumption verbatim: *"the best code for a transform is independent of the context in which it is called."* Holds exactly for op count → DP yields the optimal formula under that metric.
- **DP is a heuristic for measured runtime.** SPIRAL is explicit it *fails* for runtime: *"the left smaller transform (child) in the DFT rule is applied at a stride, which may cause cache thrashing and may impact the choice of the optimal formula."* Same sub-formula is fast or slow depending on its stride/position in the parent.

For where that breaks, SPIRAL keeps a **portfolio**: an "n-best per level" DP variant (relaxes context-independence at extra timing cost), a **context-keyed DP for the SIMD path**, **STEER** (evolutionary search over ruletrees — crossover swaps same-transform subtrees; ~20% faster than 1-best DP on DCT-II 2⁷ where DP's locality assumption fails), and a **learned-DP** over a regression-tree runtime model whose features include node *context*. Hill-climbing/random exist but are dominated/training-only.

Scale: for a 2ⁿ DFT under Cooley-Tukey only, formula count grows ~`O(4ⁿ/n^{3/2})` (Catalan), while **DP visits `O(n²)`** and matches exhaustive search up to N=2¹⁰. That polynomial collapse is the whole payoff of memoization.

The real differentiator: SPIRAL's pipeline is **layered — algorithm search → Σ-SPL expansion → loop-merging → index simplification → codegen** — and the search never reasons about loops/SIMD/threads. Backend specialization (SIMD width ν, threads, FPGA) is a *swappable tag-driven rewrite stage*. It's **offline library generation** (search+codegen once at install), which is why heavyweight search is affordable.

## 2. The specific upgrades over a factorization-level DP

**(a) Richer rule space — peer breakdown rules, not special cases.** A factorization planner knows only Cooley-Tukey splits. SPIRAL treats **CT, Good-Thomas/PFA (permutation-only, no twiddle), Rader (prime → size-(p−1) cyclic convolution), Bluestein (n → two *larger* DFTs), split-radix, and real-DFT rules** as *peer rules over the same DFT non-terminal* — so it can **mix strategies at every node** (CT down to a prime, then Rader, whose inner DFT_{p−1} is again CT/PFA). Same N → a combinatorial multiplicity of *structurally distinct* algorithms, not just factor orderings.

**(b) Sub-problem memoization across decompositions.** DP table keyed by `(transform, size[, context])`, shared across all decompositions containing that sub-transform — solve/time once, reuse everywhere. Exactly "recursive solver tree with sub-problem memoization."

**(c) A search-method portfolio for when DP isn't safe.** n-best DP, context-keyed DP, STEER, learned-DP — used precisely in the vectorization/cache regime where size-keyed DP becomes a heuristic.

**(d) Algorithm search decoupled from codegen.** Σ-SPL makes loops/index-arithmetic explicit (gather/scatter/perm/diag + iterative-sum), and **loop merging** fuses a permutation into a neighboring loop's gather by composing index functions — *eliminating the separate shuffle pass*. Crucially, **FFTW hardcoded loop fusion only for the CT shape**; Σ-SPL generalizes it to *arbitrary* formulas (Rader/Bluestein/PFA/mixed). That's what makes the rich rule set **fast, not merely expressible** — a new rule inherits the whole fusion + retargeting machinery free.

## 3. Concrete map to dag-fft-compiler

| Upgrade | Do you have it? | Verdict |
|---|---|---|
| (a) Richer rule space as peer rules | **Partial / No.** [rader.h:442](../../src/dag-fft-compiler/core/rader.h#L442) and [bluestein.h:582](../../src/dag-fft-compiler/core/bluestein.h#L582) exist as standalone builders, but [auto_plan (planner.h:220)](../../src/dag-fft-compiler/core/planner.h#L220) returns **NULL** for prime/non-smooth N — invoked only by gate/bench callers doing their own smoothness test. The exhaustive port **explicitly stripped** the Rader/Bluestein fallback ([exhaustive_plan.h:18-20](../../src/dag-fft-compiler/core/exhaustive_plan.h#L18-L20)). No node-level rule mixing. | Biggest *coverage* gap |
| (b) Sub-problem memoization | **Yes — the real thing.** [_vfft_proto_dp_solve_topk (dp_planner.h:472)](../../src/dag-fft-compiler/core/dp_planner.h#L472) is FFTW-style memoized recursion: cache keyed by **(N, K_eff)**, top-3 sub-plans/row, cache-hit returns without re-search. | Mostly done; refine |
| (c) Search-method portfolio | **Partial.** DP, flat/patient/screened exhaustive, V4/V5 estimate. No STEER, no learned-DP. But your **screened exhaustive (V4-rank → bench top-M)** covers the high-value middle. | Low priority |
| (d) Algorithm/codegen decoupling | **Yes, by construction.** Your OCaml DAG generator *is* the codegen layer; the C planner searches factorization/order/variant and [plan_create (planner.h:151)](../../src/dag-fft-compiler/core/planner.h#L151) wires codelets. | Already have it |

**Ranked by value-vs-effort on AVX2-only i9-14900KF:**

**(b) is your strongest existing asset — and it directly *is* your roadmap item.** Two things make your version *better-suited to a memory-bound objective than vanilla SPIRAL/FFTW DP* — keep them:
- The cache key carries **K_eff = K_outer × ∏(prefix radixes)** ([dp_planner.h:559](../../src/dag-fft-compiler/core/dp_planner.h#L559)), so a sub-size measured in one composition context can't pollute another. This is **exactly SPIRAL's "put context into the DP key" fix** — you derived the same medicine independently. Load-bearing.
- Every assembled `[R, sub]` candidate is **still full-benched at the parent (N, K_eff)** ([dp_planner.h:364](../../src/dag-fft-compiler/core/dp_planner.h#L364)) rather than trusting isolated sub-costs — your defense against the exact DP-context-dependence failure SPIRAL warns about. The cost: you've partly given up the polynomial-collapse memoization is supposed to buy.

The concrete refinement: you already keep `VFFT_PROTO_DP_TOPK_MAX=3` per row (that *is* the n-best relaxation) and a `believe_subplan_cost` axis ([dp_planner.h:217](../../src/dag-fft-compiler/core/dp_planner.h#L217)) that re-benches top-1 fresh in patient mode. So (b) needs **tuning, not new architecture**: raise TOPK toward SPIRAL's n-best where parent-context re-bench is cheap, and validate the `K_eff` key collapses enough sub-problems to stay sub-exponential at large pow2.

**(a) is the highest-value *new* work — but scope it to node-level rule mixing, not "add 100 rules."** Your factorization enumerator ([exhaustive_plan.h:114](../../src/dag-fft-compiler/core/exhaustive_plan.h#L114)) only reaches `remaining==1` through composite radixes; a prime N yields zero factorizations → NULL ([exhaustive_plan.h:365](../../src/dag-fft-compiler/core/exhaustive_plan.h#L365)). The minimal SPIRAL-style upgrade: **wire a smoothness check + Rader/Bluestein selection into `auto_plan`** so a prime/non-smooth N produces a plan whose *inner* DFT (`N−1` for Rader, `M≥2N−1` for Bluestein) is itself dispatched back through your DP/screened planner. That's exactly "mix CT and Rader per node," with code you already have — you just aren't routing to it. Treat Rader/Bluestein as **peer rules in `auto_plan`'s dispatch**, not caller-side special cases. Effort: moderate; value: high (closes the `mkl_bench_readiness` gap, makes you competitive on the exact non-smooth N where MKL/FFTW are closest).

**(c) is low priority for your target.** STEER pays off where DP's locality assumption fails *and* the space is too big for exhaustive. On AVX2 pow2/smooth N with depth-5 enumeration, your screened exhaustive already lands within 0–3% of the true winner at ~50× fewer benches — the same "structural prior shrinks the space, then measure survivors" move SPIRAL makes with its vector-DP. Skip STEER; if you want the learned-DP idea, the cheaper version is to keep improving the **V4/V5 estimate as the screen** (you already use measured `radix_memboundness.h` for R≥16).

**(d) you already have.** The one SPIRAL idea you *don't* exploit is **generalized loop merging across non-CT shapes**: if/when you wire Rader/Bluestein into the planner (a), make sure the DAG generator fuses their permutation/chirp passes the way it fuses CT twiddle passes — or you'll get the "expressible but not fast" failure SPIRAL calls out about FFTW.

## 4. Honest caveats

**Where SPIRAL's philosophy *misfits* your thesis — the important one.** SPIRAL's DP is provably optimal only under **arithmetic/op-count cost**, and even its runtime search inherits an arithmetic/locality-shaped worldview. Your measured win is **memory-pass count + dispatch/generic-tax elimination** — a different objective. The danger is concrete and is the same DP-context-dependence problem SPIRAL admits, sharpened for you:

> In a memory-bound parent, a sub-plan fast **in isolation** (good op count, fits L1 when measured alone) can be **slow in context** because it forces an extra memory pass or a stride that thrashes at the parent's resident tier. Op-count-optimal sub-plans are exactly the ones most likely to mislead a memory-bound planner.

This is why full-benching every `[R,sub]` at the parent context is correct and should *not* be optimized away for SPIRAL's cleaner memoization. It's also why your own memory (`v4_joint_recalibration`) already records that V4's `mb_factor` measurement context ≠ plan context and mis-ranks patient verdicts — that *is* SPIRAL's context-dependence breakdown, seen in your numbers. **Borrow SPIRAL's fixes (context in the DP key = your K_eff; n-best = your TOPK=3; measure-the-survivors = your screened exhaustive), not its objective (op-count optimality).**

**Unverified / flagged in the source findings:**
- The "DP matches exhaustive up to 2¹⁰ / times very few formulas" wording is SPIRAL-doc-attributed but the exact phrasing wasn't found in primary papers (reads FFTW-FAQ-style); the *substance* is solid.
- DP visit count is **`O(n²)`, not `O(n)`** (one survey misstated this).
- Verified SPIRAL cost measures: op count, instruction count, cache misses, accuracy. **"FMA count" / "code size" as search costs are not documented** — plausible but unverified.
- STEER DCT speedup ~20% over 1-best DP at DCT-II 2⁷.

**Net:** SPIRAL's "upgrade" isn't a better DP recursion — it's (a) a rule-generated formula space with peer breakdown rules mixable per node, and (d) a decoupled rewriting backend that makes that space fast. You already hold (b) and (d). The one move with real ROI on your target is **(a) narrowed to: route primes/non-smooth N through Rader/Bluestein as peer rules in `auto_plan`, recursing the inner DFT back through your existing DP/screened planner** — and make the DAG generator fuse their non-CT passes.
