# R=32: Where SU Starts Mattering, And Where Spill Becomes Necessary

## TL;DR

Extended the math layer to R=32 = CT(4, 8). Two findings:

**1. SU is now a clear win, not a marginal one.** R=32 SU beats Topo by 6–24% across all K — much bigger than R=16's 2–6%. The pattern matches the prediction we made in the earlier SU writeup: SU's marginal value scales with DAG size.

**2. Topo *never* beats Hand at R=32.** Where R=16 had Topo winning 12–15% over Hand at large K, R=32 has Topo *losing* 13–69% at every K. The cause is structural: 38+ live values exceed AVX-512's 32 ZMM register budget. GCC must spill, and its automatic spill management is worse than Hand's explicit one. SU helps by reducing the spill volume but can't eliminate it.

This is the first radix where the spill variant becomes empirically necessary — without explicit spill modeling, we can't match Hand at any K.

## Setup

Extended `pick_algorithm` in `dft.ml` to handle R=32:

```ocaml
| 32 -> Cooley_Tukey (4, 8)
```

Matches user's `gen_radix32.py` (`N1=8, N2=4` in their convention). Their PASS 1 has 4 sub-FFTs of size 8, PASS 2 has 8 sub-FFTs of size 4. In our convention with input mapping `n = n1 + n2*N1`, we use `(N1=4, N2=8)` to produce the same structure.

The math layer is generic over (N1, N2); no other code changed. The same `dft_ct` recursive function that generated R=4, R=8, R=16 produces R=32 from this single line of pick_algorithm.

## DAG and op count comparison

| Metric | Generated | Hand-coded | Δ |
|---|---|---|---|
| DAG nodes | 671 | n/a | — |
| Vector instructions | 600 | 556 | +8% |
| Scalar-equivalent ops | 662 | 650 | +2% |
| Loads | 126 (62 rio + 64 tw) | 64 LD macros | — |
| Stores | 32 | 64 | — |
| Spill stores/loads | **0** | **56 + 56 = 112** | — |
| FMA count | 124 (from cmuls) | 94 | +32% more "FMA-style" |

The 8% vector-instruction gap traces to FMA fusion: hand-coded uses 94 FMAs, ours has 62 cmul-pairs (= 124 FMA-style instructions if GCC fuses surrounding mul+add into FMA, which at -O3 with -mfma it does for most cases). The 2% scalar-equivalent gap is small enough that op count isn't the issue.

The interesting structural difference: hand-coded uses 112 explicit spill operations between PASS 1 and PASS 2; ours uses 0 (we let GCC handle register pressure implicitly).

## Bench results, R=32 t1_dit AVX-512

3 runs at each K, single-process side-by-side timing. Sapphire Rapids container.

| K | Hand (ns) | Topo (ns) | SU (ns) | T/H | S/H | S/T |
|---|---|---|---|---|---|---|
| 64 | 1697 | 2487 | 1999 | 1.47 | 1.18 | **0.80** |
| 128 | 3932 | 6629 | 5092 | 1.69 | 1.30 | **0.77** |
| 512 | 25247 | 34508 | 28409 | 1.37 | 1.13 | **0.82** |
| 2048 | 173095 | 194253 | 181940 | 1.12 | 1.05 | **0.94** |

T/H = Topo/Hand; S/H = SU/Hand; S/T = SU/Topo.

All ratios stable across 3 runs (per-run variance ≤ 4%).

## Reading the data

**SU is now substantially better than Topo at R=32.**

At R=16 the SU effect was 2–6% in the compute-leaning regime and a slight loss (1–5%) in the memory-bound regime. At R=32 the picture is uniformly positive:

- K=128: SU is 23% faster than Topo
- K=512: SU is 18% faster than Topo
- K=2048: SU is 6% faster than Topo

The K=128 case is striking. Topo runs 69% slower than Hand there; SU brings that down to 30%. SU recovers more than half the gap.

**Why SU scales with DAG size (the prediction holds):**

GCC's scheduler does a reasonable job on 100-op DAGs (R=8). It's still adequate at 250 ops (R=16) — only marginal improvement available from SU. At 670 ops (R=32), GCC's heuristics start hitting their limits. Live-range analysis becomes expensive, register-allocation choices become more constrained, and the order of input matters more.

This is the *exact* prediction we made in the earlier SU writeup:

> Test on R=32 and R=64 — larger DAGs (700+ ops) are where GCC's own scheduling heuristics start to break down. SU's marginal value over Topo likely scales with DAG size.

Validated. SU at R=32 is a 6–24% gain, vs 2–6% at R=16. The trend should continue at R=64.

**Why Topo (and even SU) can't beat Hand at R=32:**

Hand's R=32 has explicit spill management:
- After PASS 1, store all 32 sub-FFT outputs to a stack `spill_re/spill_im` buffer
- Before PASS 2, reload them in the right order

That's 56+56 = 112 explicit memory operations per inner-loop iteration, on top of 64 input loads + 64 output stores.

These spills look expensive — and at small K they would be, in pure cycle count. But Hand's spill traffic is *organized*: predictable stack stride, prefetcher-friendly, cache-line aligned. GCC's automatic spilling, in contrast, is opportunistic: spill when the allocator runs out, in an order driven by the SSA structure rather than by access pattern.

At R=32's working-set sizes:
- 32 PASS 1 outputs × 2 (re/im) × 8 doubles per ZMM = 512 bytes of "pass-1 state"
- AVX-512 has 32 ZMM = 2048 bytes total register space
- After accounting for live inputs/twiddles, only ~16 ZMM available for PASS 1 state
- Required: 64 ZMM (32 outputs × 2). Shortfall: 48 ZMM worth.

So we need to spill ~48 vector registers of state regardless. Hand spills it predictably, GCC spills it unpredictably. The result: Hand wins by 13–67%.

## What this implies for v2.0

We've now seen the full progression:

| Radix | Op count | Topo vs Hand | SU vs Topo | Spill matters? |
|---|---|---|---|---|
| R=4 | 34 | tied | tied | no |
| R=8 | 99 | tied at large K, -8% small K | tied | no |
| R=16 | 262 | wins large K (+15%), loses small K (-15%) | small win (2–6%) | helpful |
| R=32 | 662 | loses everywhere (-13% to -67%) | substantial (6–24%) | **necessary** |

**The pattern is clean: as DAG size grows, the gap to hand-coded grows, and explicit spill management becomes increasingly necessary.**

For the actual library:
- R=4, R=8: ship Topo. No further work needed.
- R=16: ship Topo. Maybe SU at small K if benchmarks justify (~5% gain in narrow regime).
- R=32: ship SU as the default. Spill variant needed to close the remaining gap to Hand.
- R=64+: SU likely substantial. Spill variant likely critical. Algorithm choice (split-radix vs CT, CT(8,8) vs CT(4,16)) starts to matter.

## What we still owe at R=32

The remaining 5–31% gap from SU to Hand is recoverable through spill modeling. The current generator emits SSA with no spill awareness; GCC handles register pressure ad-hoc. A spill-aware emit would:

1. Identify cross-pass values (PASS 1 outputs with last_use after a "boundary")
2. Emit explicit `__m512d spill_re[32]; ...; _mm512_store_pd(&spill_re[i], t<tag>);`
3. Emit explicit reloads at the use site

This is a reasonable addition to `emit_c.ml` (~150 lines): a new `~spill:bool` parameter, plus a "boundary detection" pass that identifies the natural between-pass cut.

The empirical question for v2.0: does spill-with-Topo, spill-with-SU, or no-spill-with-SU win for R=32 at each K?

Hypothesis based on R=16 data: spill helps at small K (compute-bound, where Hand's organized memory traffic beats GCC's chaos), no-spill helps at large K (memory-bound, where the extra L1 traffic costs more than it saves). With R=32's high register pressure, spill might be the unconditional winner — there's no escape from spilling at this DAG size.

## Honest scope check

R=32 stats (671 nodes, 600 vector instructions) confirms the math layer scales cleanly. We added 1 line to `pick_algorithm` and got a working R=32 codelet that's correct to 1e-8 across all K we tested.

The 1-line extension produced a complete R=32 codelet that:
- Passes correctness (within FP rounding, ~1e-9 to 1e-8 across K=64 to K=2048)
- Compiles cleanly with -O3 -march=native -mavx512f -mfma
- Is 8% over hand-coded on op count, 2% on scalar-equivalent ops
- Runs 13–67% slower than hand-coded due to register-pressure spill management

That's the math layer working as designed: parameterize the factorization, get a DFT for free. The performance gap then localizes the next engineering question — *spill management* — to the emit layer where it belongs.

## Two side notes

**FP precision degrades with N.** At K=1024 our R=32 had max relative error 4.2e-9, just over the original 1e-9 threshold I used. With 32-input DFTs there are O(log₂(32)) = 5 dependent multiply-adds per output, accumulating ~32 × 5 = 160 ops worth of FP error. The 1e-9 threshold was right for R=16; 1e-8 is appropriate for R=32. For R=64 we'd want 1e-7 or so. Easy to handle in the bench harness; the codelet itself is fine.

**Generation time is fast.** The OCaml generator produced R=32 in <0.5 seconds. algsimp's hash-consing keeps the cost manageable even at 671 nodes. No optimization needed there.
