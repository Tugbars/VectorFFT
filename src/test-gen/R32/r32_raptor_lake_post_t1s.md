# R=32 with t1s on Raptor Lake — analysis

After adding `ct_t1s_dit` (scalar-broadcast twiddle) to the R=32 candidate
matrix, the Raptor Lake AVX2 selector changed dramatically. This confirms
the memory-bandwidth hypothesis from the bottleneck-coverage doc.

---

## The headline numbers

36 decisions, AVX2 only (Raptor Lake has AVX-512 fused off).

| Family | Before t1s | With t1s | Δ |
|---|---|---|---|
| `ct_t1s_dit` | 0 | **24** | **+24** |
| `ct_t1_buf_dit` | 19 | 12 | −7 |
| `ct_t1_dit_log3` | 16 | **0** | **−16** |
| `ct_t1_dit` (flat) | 1 | 0 | −1 |

**t1s wins 67% of regions.** Log3 goes from 16 wins to zero. Buf loses
~7 wins at mid-me to t1s but retains dominance at small me.

## The split by me regime is surgical

| me | Before | With t1s |
|---|---|---|
| 64 | buf:5, log3:1 | **buf:6** |
| 128 | buf:6 | **buf:6** |
| 256 | log3:3, buf:2, flat:1 | **t1s:6** |
| 512 | log3:4, buf:2 | **t1s:6** |
| 1024 | log3:4, buf:2 | **t1s:6** |
| 2048 | log3:4, buf:2 | **t1s:6** |

**Perfectly clean bimodal selection:**

- **me ≤ 128**: buffered wins all 12 regions. Small working set; buffer
  overhead worth paying for the locality win from access-pattern control.
  t1s doesn't help here because the vector twiddle loads were already
  cheap — the bottleneck wasn't load-port pressure at these sizes.

- **me ≥ 256**: t1s wins all 24 regions. The vector twiddle table at
  these sizes is `31 × me × 16 bytes`, which exceeds L1D at me=128,
  exceeds L2 at me=4096. Eliminating those loads entirely (via scalar
  broadcast) is the dominant win.

The transition at me=256 isn't a fuzzy boundary — it's a hard threshold.
All 6 regions at me=256 go to t1s; all 6 at me=128 go to buf.

## What this tells us about the actual bottleneck

### log3 going from 16 wins to zero is the biggest signal

Log3 was never winning because it was the "best" codelet. It was winning
because it was the **second-best** option in a regime dominated by
memory-load pressure, with t1s unavailable. Once the genuine solution
(t1s) is in the matrix, log3 evaporates.

This recontextualizes our earlier interpretation:

- Previously I said log3 addresses "instruction-bandwidth when the
  frontend is the ceiling." That was partially right, but underspecified.
  The reason log3's fewer loads helped on Raptor Lake wasn't because the
  frontend was the ceiling — it was because **load-port pressure** was
  the ceiling. log3 reduced loads 5:1 in exchange for more FMAs; those
  extra FMAs had spare port capacity, so it was a net win.
- t1s doesn't just reduce loads — it **eliminates vector twiddle loads
  entirely** and replaces them with scalar-broadcasts (which go through
  different hardware paths and amortize across all lanes of the SIMD
  register). Much stronger effect than log3's partial reduction.

So: **Raptor Lake R=32 is load-port-bound at mid-to-large me**, exactly
as the R=16 data suggested. We now have the right codelet family for it.

### Buf's remaining wins at small me are a different bottleneck

buf keeps all 12 wins at me=64 and me=128. The mechanism there is
access-pattern control (converts stride-ios loads to stride-1 scratch
buffer loads). At small me:

- The working set fits in L1D comfortably.
- Twiddle table is also L1-resident (twiddle_bytes at me=64 = 31 × 64 × 16
  = 32 KB — fits in 48 KB L1D).
- Load pressure isn't saturating ports yet.
- What matters is **stride aliasing** on power-of-2 strides, and buf
  fixes that specifically.

This is the memory-locality bottleneck, which the coverage doc already
identified correctly. t1s doesn't address memory-locality — it
addresses memory-bandwidth. The two codelets solve different problems
and the bench correctly picks the right tool for each regime.

### The distribution is clean enough to state as a rule

On Raptor Lake at R=32, AVX2:

- **me ≤ 128 → use buf** (memory-locality bottleneck)
- **me ≥ 256 → use t1s** (memory-bandwidth / load-port bottleneck)

That's a two-codelet decision tree. Eleven other codelets in the
selector are there for the small-me buf variations (different tile
sizes × drain modes × prefetch distances), but the family decision is
trivially simple.

## Per-element cost (fwd, median at padded ios for L2 conflict avoidance)

| me | median ns | ns/element |
|---|---|---|
| 64 | 1100 | 0.537 |
| 128 | 2663 | 0.650 |
| 256 | 5291 | 0.646 |
| 512 | 9997 | 0.610 |
| 1024 | 20690 | 0.631 |
| 2048 | 43090 | 0.658 |

Per-element cost is **remarkably flat from me=128 to me=2048** — about
0.6-0.66 ns/element across the entire range. That's almost perfect
linear scaling — no log2(me) factor creeping in from cache-hierarchy
transitions. Indicates the codelet selector is doing its job: at every
me, we're picking the codelet that fits the bottleneck at that size.

## What changed from the bottleneck coverage doc

The coverage doc identified t1s-for-R=32 as a gap. Data confirms:

1. **The gap was real.** 24 of 36 decisions flipped to t1s when made
   available. Not a marginal improvement — a dominant rewrite.

2. **The bottleneck model needs a small correction.** I said log3 solves
   "instruction bandwidth." More precisely: log3 partially addresses
   load-port pressure by trading loads for FMAs. It's second-best when
   t1s isn't available. This makes the log3 analysis less interesting
   than I made it sound.

3. **buf's bottleneck was correctly identified** — memory-locality via
   access-pattern control. The 12 buf wins at small me confirm this
   remains distinct from what t1s fixes.

4. **Port pressure is the real mid-to-large-me bottleneck** on consumer
   chips. Not instruction bandwidth, not frontend, not even memory-
   latency (which SW prefetch mostly handles). It's the load ports
   saturating with vector twiddle loads.

## Priorities changed by this result

1. **t1s for R=64 is now higher priority.** The gap identified at R=32
   is likely present at R=64 too (actually larger, since R=64 has 63
   twiddle rows). Porting t1s to R=64 will likely flip similar
   selections.

2. **t1s for R=16 is already there** — no action needed. This was
   already dominant on R=16 Raptor Lake (69%).

3. **SPR comparison when/if you run it:** predict t1s wins fewer
   regions on SPR than on Raptor Lake. SPR's stronger prefetcher
   partially hides the load pressure, so flat+tpf should remain
   competitive at more me values. Maybe 8-15 t1s wins on SPR instead
   of 24.

4. **Log3 is now borderline-removable from the candidate matrix** for
   R=32 on Raptor Lake. Not zero value (might still win on SPR), but
   zero regions on Raptor Lake. Worth keeping in the bench to confirm
   cross-chip behavior but don't expect it to appear in selectors.

5. **The bottleneck coverage doc needs a small update:** replace
   "log3 → instruction bandwidth" with "log3 → partial load-port
   reduction (second-best when t1s unavailable)". Will edit next.
