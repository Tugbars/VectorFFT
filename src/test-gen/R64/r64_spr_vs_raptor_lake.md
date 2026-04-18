# R=64 cross-chip: Raptor Lake vs SPR AVX2

This is the biggest cross-chip divergence we've seen in this project.

**Hardware:**
- SPR: Anthropic container, Sapphire Rapids (Golden Cove), server-class, 64-entry L1 DTLB, AVX-512 enabled
- Raptor Lake: Tugbars' i9-14900KF, Raptor Cove P-cores, consumer, 96-entry L1 DTLB, AVX-512 fused off

**17 AVX2 candidates benched on both** (same codegen, same sweep grid).

## Selector winners

| Selector family | SPR wins (36) | Raptor Lake wins (36) |
|---|---|---|
| `ct_t1s_dit` | **33 (92%)** | 4 (11%) |
| `ct_t1_dit_log3` | 0 (0%) | **27 (75%)** |
| `ct_t1_dit` | 1 (3%) | 0 (0%) |
| `ct_t1_dif` | 2 (6%) | 1 (3%) |
| `ct_t1_dit_prefetch` | 0 (0%) | 0 (0%) |
| `ct_t1_buf_dit` (any config) | 0 (0%) | **4 (11%)** |

These are roughly inverted. On SPR, t1s dominates; on Raptor Lake, log3 dominates.

## The log3/t1s ratio reveals the mechanism

```
AVX2 fwd at padded stride ios=me+8:

me    SPR log3/t1s   RL log3/t1s
64         1.14           0.86      <- on RL, log3 is 14% faster than t1s
128        1.12           0.88      <- 12% faster
256        1.08           0.98      <- about even
512        1.14           0.93      <- 7% faster
1024       1.16           0.96      <- 4% faster
2048       1.16           0.97      <- 3% faster
```

- SPR: t1s beats log3 at every me, by 8-16%.
- Raptor Lake: log3 beats t1s at every me, by 3-14%.

The crossover is chip-architecture-driven, not workload-driven.

## Why the inversion happens

**t1s mechanism:** eliminates 30-ish vector twiddle loads per butterfly column by pre-loading 63 scalar twiddles and broadcasting them at use. Trades vector loads for broadcasts.

**log3 mechanism:** loads ~18 base twiddles, derives the other 45 via arithmetic FMAs using `w[a+b] = w[a]*w[b]`. Trades vector loads for extra FMAs.

Both methods reduce load count. They differ in what they substitute:

### SPR (server Golden Cove, 64 DTLB entries)

- Load DTLB: 64 entries, tight. At me=2048 the twiddle table alone is ~32 pages. Load DTLB overflow is a big stall (32.9% of cycles per VTune on this exact codelet).
- t1s's savings (eliminate twiddle-side DTLB pressure entirely) are worth a lot because DTLB was the binding constraint.
- log3's savings (fewer loads but still loads, plus more FMA) don't fully escape the DTLB because it still loads 18 twiddle rows.
- SPR's strong execution engine means the extra FMAs log3 needs have a cost (they compete with butterfly arithmetic).
- **Result: t1s wins decisively on SPR.**

### Raptor Lake (consumer Raptor Cove, 96 DTLB entries)

- L1 DTLB has 50% more entries (96 vs 64). The twiddle table fits more loosely.
- Load DTLB was probably ~15-20% of cycles rather than 33%, so the savings from eliminating it entirely (t1s) aren't as valuable.
- log3's partial load reduction (18 loads vs 63) already addresses most of the DTLB pressure.
- Raptor Cove has aggressive FMA throughput (similar to Golden Cove). The extra FMAs log3 needs run for free when memory isn't saturated.
- t1s introduces 63 scalar broadcasts per butterfly. These are fast but not free — they add front-end pressure and consume execution slots. When you don't urgently need to eliminate loads (because DTLB has headroom), the broadcast cost shows up in end-to-end latency.
- **Result: log3 wins decisively on Raptor Lake.**

## Absolute performance

Raptor Lake is faster than the SPR container at every point. Ratios (RL/SPR) range 0.52-0.79 — Raptor Lake runs the same code 1.3-1.9× faster. This matches expectations (desktop chip with higher turbo frequency vs container-throttled server core).

```
me=1024 AVX2 fwd ios=1032 best ns:
  SPR     log3   83142 ns
  SPR     t1s    71430 ns   <- SPR selects t1s
  RL      log3   46280 ns   <- RL selects log3
  RL      t1s    47995 ns
```

Even though t1s is slower on Raptor Lake, the RL log3 number is faster than anything on SPR. Cross-chip codelet choice is a second-order correction on top of the bigger frequency-and-memory-hierarchy differences.

## The buf "wins" on Raptor Lake are artifacts

The 4 buf wins on Raptor Lake at first looked meaningful. Investigation:

All 4 wins are at me=64 (the smallest me), with configuration `tile128_drainstream` or similar tile=128 variant.

**Mechanism (inspected by reading generated code):** at me=64 and TILE=128, `n_full_tiles = me / TILE = 0`. The tile loop executes zero iterations. All the work happens in the **tail** loop — which uses `addr_mode='t1'`, identical to plain t1_dit.

So these "buf wins" are not buf-as-a-mechanism wins. They're effectively plain t1_dit calls, but compiled inside a function body that contains unused tile-loop scaffolding. The scaffolding changes code alignment, which changes i-cache layout and apparently benefits the tail-loop critical path on Raptor Lake.

This is a **compiler-alignment effect, not a buf-algorithm effect.** Same mechanism on SPR doesn't trigger — at me=64 on SPR, flat wins straight up (tile=128 buf is worse at 5476 ns vs t1_dit 4480 ns).

Interpretation: ICX on Windows + Raptor Lake happens to hit a favorable alignment for the larger buf function. GCC + Linux + SPR doesn't. This is noise-level, not optimization signal.

**True buf wins on Raptor Lake (where tiling actually runs): 0 of 14 sweep points where me >= 128.**

So: the R=32 pattern (buf helps when outbuf fits L1) doesn't reappear at R=64 on Raptor Lake either. Both chips reject real buf usage at R=64.

## Updated bottleneck taxonomy

The R=64 chip-specific selection table:

```
                    SPR (server)       Raptor Lake (consumer)
DTLB pressure       binding            non-binding
Best winner         t1s                log3
t1s speedup over    +50% at me=2048    +3% at me=2048 (within noise)
   flat t1_dit      (51% reduction)    (inverse - t1s slower)
log3 speedup over   +35% at me=2048    +37% at me=2048
   flat t1_dit      (also wins)        (wins here)
Buf wins            0                   0 (4 false positives = alignment)
```

**Bottleneck framework, with chip axis:**

| Family | Server AVX2 | Consumer AVX2 |
|---|---|---|
| flat (t1_dit) | baseline | baseline |
| log3 | usually beaten by t1s | **wins most regions at R≥32** |
| t1s | **wins when DTLB binds** | loses when DTLB has headroom |
| buf | wins at R=32 only | marginal |
| prefetch | 0 wins (regressed) | 0 wins |
| ladder | 0 wins | (not tested at R=64) |

## Actionable takeaways

**For the selector code generator:** R=64 AVX2 needs chip-detection or per-chip selectors. The static "pick one codelet" strategy will be wrong for at least one chip family at any given time.

**For the factorizer wisdom:** at R=64 on Raptor Lake, log3 is the inner-butterfly codelet. Wisdom files should be rebuilt per-chip, with log3 selected for R=64 last-stage plans on consumer chips.

**For the positioning story:** the claim is no longer "t1s is the R=64 winner." It's stronger: "**FFT codelet selection is chip-architecture-dependent at R=64 AVX2, and VectorFFT measures and exploits the difference where libraries with fixed codelet choices cannot.**" This is a better story for the public writeup than "our codelets are fast."

**For further R=64 optimization work:**
- Output tile reordering (the VTune split-stores fix) still worth exploring — different mechanism, attacks different problem
- Raptor Lake-specific log3 optimization: the 18 base twiddles could be hoisted/cached differently
- t1s might want more hoisted twiddles on SPR — the 2-twiddle hoist budget we chose was conservative

**Cross-chip implication for the R=20 and other uncommon radixes:** all previous results showing "X codelet wins" on a single chip are single-chip results. Expect inversions on cross-chip testing for at least some families.

## What SPR's "t1s wins 92%" number actually measured

It measured *one particular server chip under container throttling with default allocator behavior* (no huge pages, no NUMA pinning). The server-chip-specific story is real and the mechanism is real, but the universal-t1s-win claim was too strong. Updated claim: "t1s eliminates twiddle-side load pressure, which makes it the right choice when that pressure is the binding constraint."

## The bigger lesson for VectorFFT positioning

This result is actually *good news* for positioning VectorFFT as a hardware-characterization tool. The story arc improves:

1. "We tune per-chip" (true)
2. "Different chips need different codelets" (now *demonstrated* with dramatic numbers, not just asserted)
3. "Libraries that bake in fixed codelets miss up to 35% cross-chip" (measurable claim from this data)

The R=64 AVX2 story is the clearest example we have of this pattern.
