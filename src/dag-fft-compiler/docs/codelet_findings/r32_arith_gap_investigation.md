# Investigating the R=32 Op-Count Gap: What's Actually Slow

## The question

R=32 generated has 600 vector instructions in source, hand-coded has 556. Source-level gap: 8%. Can we reduce it?

## What we found by looking at assembly

The source-level "8%" misrepresents where time actually goes. Looking at the compiled assembly side-by-side:

| Op type | Generated | Hand | Δ |
|---|---|---|---|
| FMA (fmadd/fnmadd/fmsub) | 169 | 130 | +39 |
| vmulpd | 97 | 101 | -4 |
| vaddpd / vsubpd | 332 | 347 | -15 |
| vxorpd (sign flips) | 12 | 11 | +1 |
| **Arithmetic total** | **610** | **589** | **+21 (3.6%)** |
| | | | |
| **vmovapd reg-reg** (FMA dest copies) | **133** | **28** | **+105** |
| **vmovapd reg-mem** (stack spill stores) | **235** | **55** | **+180** |
| vmovapd mem-reg (stack spill loads) | 138 | 112 | +26 |
| vmovupd load (input + twiddle) | 126 | 64 | +62 |
| vmovupd store (output) | 64 | 64 | 0 |

If we count each FMA as 2 scalar ops (because it does a multiply + add), the arithmetic gap is **8.3%** in scalar-equivalent terms (779 vs 719). So the source-level 8% reflects real arithmetic work, not just FMA-fusion accounting.

But the bigger gap is in the **memory operations**: 311 extra reg-reg copies and 180 extra stack-spill stores. That's **373 extra L1/move operations**. Each takes ~1 cycle at L1; we're paying ~373 extra cycles per iteration vs hand-coded.

## Where the runtime gap actually comes from

At R=32, K=128, hand=3933 ns and generated=6629 ns. Per inner-loop iteration (16 iters): hand=246 ns, gen=414 ns. Gap = 168 ns ≈ 600 cycles at 3.5 GHz.

Decomposing:
- 21 extra arithmetic ops (~10-20 cycles)
- 180 extra spill stores (~180 cycles direct, more if pipelined poorly)
- 105 extra reg-reg copies (~0-50 cycles, mostly free with rename)
- 26 extra spill loads + 62 extra unaligned loads (~80 cycles, partially overlapped)

The spill stores dominate. **The arithmetic gap (~5-10% of the total) is real but minor compared to the memory-traffic gap (~50-80% of the total).**

## Why the spill traffic is so different

Hand-coded uses 56 explicit spill stores + 56 explicit spill loads, declared as `spill_re[128]` and `spill_im[128]` stack arrays. These are organized:

- All happen at the natural boundary between PASS 1 and PASS 2
- Predictable stack stride (cache-line aligned)
- Prefetcher-friendly access pattern
- Same locations across all inner iterations

GCC's automatic spilling is opportunistic: it spills when the allocator runs out of registers, in an order driven by the SSA structure. Result: 235 stack stores scattered through the function, not all at one boundary, possibly with bad spatial locality.

That's the structural difference, and it's exactly the case for the spill variant we've been discussing. R=32 is where this matters most because register pressure (38+ live values) forces spilling regardless of how clever scheduling is.

## What we tried for the arithmetic gap

I added a peephole to algsimp:

```
Add(Mul(a, k), Mul(b, k)) → Mul(Add(a, b), k)   for constant k
Sub(Mul(a, k), Mul(b, k)) → Mul(Sub(a, b), k)
```

This is the algebraic factoring that hand-coded uses for twiddles where `cr = ci` (e.g., `(1±i)/√2` rotations). In theory it saves one mul per occurrence.

**It made the op count worse, not better.** R=32 went from 662 to 817 scalar ops (+23%).

The failure mode: when `Mul(xr, k)` is shared between the Re and Im parts of a complex multiply, factoring it kills the sharing. Re becomes `Mul(Sub(xr, xi), k)` — and now Im needs `Mul(xr, k)` again, recreated as a new node. Net: more ops, not fewer.

**Lesson:** local algebraic peepholes that restructure expressions are dangerous on a hash-consed DAG. They change which subexpressions exist, and when those subexpressions had multiple consumers (which is what hash-consing was preserving), the peephole creates extra work.

To do this correctly, the peephole needs **use-count awareness**: only factor when both Mul nodes have unique consumers (the Add/Sub being constructed). At construction time, we don't yet know all the consumers — they'll appear later in DFT recursion. So this would require a post-pass on the constructed DAG, after all consumers are known.

Reverted. The peephole is documented in algsimp.ml as future work.

## What would actually reduce the gap

Three real levers, in order of impact:

**1. Spill variant emission (most impact, ~30-50% gap reduction).**

Add a `~spill:bool` parameter to `emit_c.ml`. When true, identify cross-pass values (PASS 1 outputs whose last_use is past a "boundary") and emit explicit `__m512d spill_re[N]; ... _mm512_store_pd(&spill_re[i], t<tag>);` plus matching reloads at use sites.

The boundary detection can use the same lifetime analysis annotate.ml does, just with a different action: instead of opening a nested block, force a stack store/load through a buffer. ~150 lines.

This addresses the 235→55 spill store gap directly. Empirically expected to recover most of the 13–69% Hand–Topo gap at R=32.

**2. Use-count-aware factoring peephole (~5-8% gap reduction).**

Build the DAG normally, then run a post-pass that:
- Computes use counts for every node
- Finds Add/Sub nodes whose operands are both Mul(_, k) with k constant AND both Muls have use-count = 1
- Rewrites to Mul(Add/Sub(_, _), k)

This is the algebraic identity from earlier, but only applied when it doesn't break sharing. Would catch the ~10-20 muls in R=32 from twiddles with cr=±ci.

**3. Better FMA-variant selection at emit time (~2-3% gap reduction).**

GCC picks 9 different FMA variants in our code; hand-coded uses only 4. The difference is that hand-coded knows which operand to "reuse" as destination (via specific intrinsic forms), avoiding the 105 reg-reg copies.

We could direct GCC by structuring our intrinsic calls more carefully — emitting `_mm512_fmsub_pd(...)` patterns that prefer `vfmsub231pd`. ~50 lines of emit-layer work.

Of these, **(1) is by far the highest leverage** and what we should do next. (2) is interesting but complex and saves less. (3) is small and would be obsoleted by a future direct-assembly emitter anyway.

## Honest summary

The 8% source-level gap is mostly real (8.3% in scalar-equivalent ops). But it's the wrong target: at R=32, **memory traffic (spill management) dominates the runtime gap, not arithmetic.** Reducing the arithmetic from 779 to 719 scalar ops would save 5-10% of the gap; building the spill variant would close 30-50%.

The peephole experiment was instructive: it showed that arithmetic-level optimizations on a hash-consed DAG need use-count awareness to avoid breaking CSE. That's a real result for the algsimp design, even if the peephole itself didn't ship.

Recommended next step: build the spill variant. It's the right lever for R=32, and validates the variant-dispatcher framing the v2.0 architecture rests on.
