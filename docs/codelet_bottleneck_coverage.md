# Codelet Bottleneck Coverage Analysis

A systematic look at what computational bottlenecks FFT codelets can hit,
which codelet families address each one, and where our candidate matrix
currently has gaps or overlap.

---

## Taxonomy of bottlenecks

FFT codelet performance is gated by one of five things at any given
(chip, me, ios) point:

1. **Instruction-bandwidth-bound** — front-end decode/rename/issue
   throughput is the ceiling. Compute units and memory are both fed fast
   enough; the chip can't dispatch µops quickly enough.

2. **Memory-bandwidth-bound** — demand loads/stores saturate the load/store
   ports or the L1↔L2 bandwidth. Execution units have arithmetic work but
   are stalled waiting for data.

3. **Memory-latency-bound** — the working set spills L1 (sometimes L2),
   and individual cache-miss latency becomes visible. HW prefetcher
   helps here but not always enough.

4. **Memory-locality-bound (stride aliasing, DTLB)** — access pattern
   itself is pathological even if total bandwidth/working-set size looks
   fine. Power-of-2 strides hit the same L2 sets, or sparse access
   touches many pages.

5. **Register-pressure-bound** — intermediate values spill to stack
   because the register file is too small for the unrolled butterfly.
   Compute has enough work, memory has capacity, but every butterfly
   stage is serialized through stack fills.

Most real codelets hit two or three of these simultaneously. The codelet
family is chosen to target whichever is dominant at a given (chip, me,
ios).

---

## Mapping codelet families to bottlenecks

### `ct_t1_dit` (flat)

**The baseline.** Full unroll, explicit twiddle loads per butterfly,
no buffering, no broadcast trick.

Primary benefit: **simplest codegen, smallest code footprint per codelet,
lowest µop count when no tricks are needed.**

Wins when:
- Front-end has headroom (cores with wide decoders or small butterflies)
- HW prefetcher keeps twiddles flowing
- Working set fits comfortably in L1 with no aliasing trouble
- Stride is friendly (padded, not power-of-2)

Primary bottleneck it addresses: **instruction bandwidth** — fewest
instructions wins when nothing else is wrong.

When it loses:
- Stride aliases → L2 conflicts → becomes memory-latency-bound → buf wins
- Working set exceeds L1 → twiddle loads become latency-bound → log3 or
  SW prefetch helps
- Consumer chip with weak prefetcher at mid/large me → t1s (on R=16)
  eliminates the loads that flat can't hide

### `ct_t1_dit_log3` (derived twiddles)

Instead of loading each of R−1 twiddles per column, load log₃(R) base
twiddles and derive the rest via complex multiplications.

Primary benefit: **reduced twiddle memory footprint** (fewer twiddle
bytes per butterfly) and **reduced twiddle load count** (fewer `_mm*_load`
intrinsics in the inner loop).

But: log3 substitutes memory with arithmetic. It does more FMAs (to
derive twiddles on the fly) in exchange for fewer loads. It's not
memory-free — it's memory-for-compute traded.

Primary bottleneck it addresses: **instruction bandwidth when the
frontend is the ceiling and load count is a contributor to that
ceiling.** On chips with strong HW prefetch (SPR), the loads weren't
stalling anyway, so log3's real win is reducing µop count for the
frontend.

Wins when:
- Frontend-bound on a chip where memory is well-fed
- Code size matters (less I-cache footprint can help indirectly)
- Large me where twiddle table spills L1 — but prefetcher was hiding
  this anyway on server chips

When it loses:
- Very small me where log3's extra FMAs aren't amortized — flat wins
- Chips where FMA throughput is the real bottleneck, not load count
- Chips weak enough at HW prefetch that memory is the real bottleneck
  (consumer chips): then t1s or buf wins instead because they fix the
  actual problem

### `ct_t1s_dit` (scalar-broadcast)

Load each twiddle as a scalar once, broadcast to vector lanes, reuse
across multiple butterflies.

Primary benefit: **eliminates vector twiddle loads almost entirely.**
Instead of `_mm*_load_pd(&W[j*me+m])` per butterfly, one scalar load
followed by broadcast, reused for all lanes.

Primary bottleneck it addresses: **memory-bandwidth-bound** when twiddle
loads saturate load ports. On weak-prefetcher consumer chips, this is
the actual bottleneck at mid-me, and t1s destroys it.

Wins when:
- Weak HW prefetcher (consumer chips, Raptor Lake, older Intel client)
- Mid me where L1D pressure from twiddle table is real
- Power-of-2 strides (which alias the twiddle access pattern too)

When it loses:
- Server chips where prefetcher already hid the loads
- Very small me where broadcast setup costs aren't amortized
- Very large me where memory access pattern is the bottleneck, not
  memory bandwidth

**Currently: R=16 only.** Not in R=32 generator. This is a gap.

### `ct_t1_buf_dit` (buffered)

Copy `tile` columns of data to a local scratch buffer (changing access
pattern from stride-ios to stride-1), do the butterfly in-place on the
scratch, copy back.

Primary benefit: **converts pathological access patterns to sequential
ones.** The scratch buffer is unit-stride, no aliasing, fully in L1D
(for small-enough tile). The copy-in and copy-out are prefetcher-
friendly sequential loads/stores.

Primary bottleneck it addresses: **memory-locality** — stride aliasing,
DTLB pressure, irregular access patterns. And secondarily **register
pressure** because working out of a scratch buffer reduces how many
intermediate values need to stay in registers at once.

Wins when:
- Power-of-2 stride (stride aliasing is real)
- Working set exceeds L1 (buf makes the inner loop's working set a
  tile-sized slice that fits)
- Consumer chips where explicit access-pattern control beats the weaker
  HW prefetcher's guesswork

When it loses:
- Padded strides (no aliasing problem → nothing to fix → buf's copy
  overhead is pure cost)
- Very small working sets (fits in L1 anyway, no aliasing, buf just
  adds work)

### `ct_t1_ladder_dit` (5-base-twiddle ladder, AVX-512 only)

A log3-style derivation with a specific 5-base-twiddle structure
optimized for AVX-512's 32 ZMM registers. Uses more registers to cache
more derived twiddles in-flight, less memory traffic than flat.

Primary benefit: **keeps more twiddle values live in registers** than
log3 can. Reduces both load count AND FMA count at the cost of more
register pressure.

Primary bottleneck it addresses: **instruction bandwidth on wide
architectures**. With AVX-512's 32 ZMM, ladder can hide more than log3
by keeping 5 base twiddles + derivatives live without spilling.

Wins when:
- AVX-512 with abundant registers
- Specific me regimes where the ladder structure aligns with µarch ports

**Currently: R=32 only.** No R=16 ladder. This is fine — R=16 is small
enough that ladder's extra register use doesn't pay off.

---

## The knob dimension: prefetch

Orthogonal to family. Every family can have `twiddle_prefetch` enabled:
issue a `_mm_prefetch` to pull twiddle cache lines into L1 before
they're needed.

Primary bottleneck it addresses: **memory-latency** when twiddle table
spills L1 (me ≥ ~128 for R=32 on 48 KB L1D) AND HW prefetch isn't keeping
up. SW prefetch buys you deterministic fetch ahead of demand.

Wins when:
- Twiddle table spills L1 (our gating condition)
- HW prefetcher doesn't handle the access pattern well
- Especially on consumer chips where HW prefetch is weaker

Knob values:
- Distance 4, 8, 16, 32: how many iterations ahead
- Rows 1 vs 2: how many cachelines fetched per prefetch

**Both chips use SW prefetch heavily** (78% SPR, 92% Raptor Lake).
Raptor Lake's higher reliance is the "weak HW prefetcher needs more
help" signal.

## The knob dimension: drain

For buffered codelets only. After the butterfly is done, write the
scratch buffer back to `rio`. Can be:

- **Temporal:** normal stores, go through cache
- **Stream:** non-temporal stores, bypass cache

Primary bottleneck stream addresses: **cache pollution** when the output
won't be re-read soon. Saves cache space for data that will be re-used.

Wins when:
- Output bytes exceed L2 (otherwise you want it cached for next pass)
- Chip has efficient NT-store implementation
- Next computation stage won't touch these bytes

**Currently gated off** because our sweep never exceeds L2. This is
correct for now but unverified for larger-me workloads.

## The knob dimension: drain_prefetch (prefw)

For buffered codelets only. Issue `PREFETCHW` (prefetch-for-write) on
output pages before writing.

Primary bottleneck it addresses: **DTLB miss on output pages** — when
writing to `rio` touches pages whose PTEs aren't in the DTLB. Prefetch-
for-write warms DTLB and marks the page as owned-for-write, avoiding
write-miss delays.

Wins when:
- Output pages haven't been recently touched
- Chip has tight DTLB (consumer chips, or large working sets)
- Stride pattern crosses many 4 KB boundaries

Wins ~17-22% of buffered decisions on both chips — a solid but not
dominant benefit.

---

## Bottleneck coverage matrix

| Bottleneck | Family that addresses it | Knob that helps |
|---|---|---|
| **Instruction bandwidth** (frontend) | log3, ladder (AVX-512) | — |
| **Memory bandwidth** (load ports) | t1s (R=16 only) | SW prefetch (partial) |
| **Memory latency** (L1 spill) | — | SW prefetch (`tpf*`) |
| **Memory locality** (aliasing, DTLB) | buf (via tile size) | `prefw` (DTLB-specific) |
| **Register pressure** | buf (scratch frees regs) | ladder (AVX-512 cache in regs) |
| **Cache pollution** (output won't re-read) | — | `drain=stream` |

## Gap analysis

### Gap 1: No t1s for R=32

**Significance: medium-high.** On Raptor Lake R=16, t1s dominated at
69% of wins. It solves the memory-bandwidth bottleneck that hits consumer
chips hardest. R=32 doesn't have this — and on Raptor Lake, R=32 is left
with flat+SW-prefetch or buf as the memory-bandwidth fixes.

Is flat+SW-prefetch an acceptable t1s substitute? Partly. SW prefetch
reduces latency visibility but doesn't reduce port pressure. t1s reduces
port pressure by eliminating loads entirely. They aren't equivalent.

**Action:** extend R=32 generator to emit t1s variants, or document
that the bench has a known gap in R=32 memory-bandwidth coverage.

### Gap 2: No coverage for "compute-bound"

We assume FFT codelets are never purely compute-bound. This is mostly
true for small radixes where the butterfly is FMA-dense and memory-
dense in roughly equal measure. It may not be true at very large radix
where FMA count scales as R² and loads scale linearly.

**Significance: low for R ≤ 32.** Probably matters for R=64+.

### Gap 3: No intermediate between temporal and stream

Some chips (notably Zen) benefit from **weakly-ordered stores** — a
middle ground between temporal (fully cached) and stream (bypass cache).
x86 doesn't expose this directly, but the compiler can emit `movntpd`
only for specific stores via pragmas. Currently we have binary
temporal/stream; no middle option.

**Significance: low.** Adding this would add a 3-way knob × 2 (prefw) ×
N-tile × N-prefetch candidate explosion. Not worth it until we have
evidence a middle option would win.

### Gap 4: Buf with `drain_prefetch` on non-buf codelets

DTLB pressure is a problem for non-buf codelets too (writing back
directly to `rio` at large me). We could emit `prefw` in flat/log3
before writing output rows. Currently `prefw` is only on buf_dit.

**Significance: medium.** Could squeeze wins at mid-me where flat+tpf
wins but DTLB misses still hurt. Testable with one extra knob dim on
flat and log3.

### Gap 5: No coverage for "prefetcher-trained" regime

Some codelets could benefit from **deliberately training the HW
prefetcher** with a warmup pass. Rare and chip-specific; not a priority.

---

## Are we covering every scenario?

**For R=16**: yes, modulo the "compute-bound" gap (not relevant at R=16).
Families cover frontend (flat/log3), memory-bandwidth (t1s), memory-
locality (buf). Knobs cover latency (tpf), DTLB (prefw), cache-pollution
(drain=stream, gated).

**For R=32**: gap on t1s-equivalent for memory-bandwidth. Otherwise
covered. Ladder (AVX-512-specific) fills the very-wide-register niche
that flat/log3 can't exploit. Frontend via log3 and ladder. Memory-
locality via buf with tile-size knob. Latency via tpf. DTLB via prefw.

**For R=64**: unknown until we build. Will need to re-run this analysis
because:
- R=64 codelets may become register-pressure-limited in ways R=32 isn't
- Some families may not fit the 32-ZMM budget on AVX-512
- Tile sizes will need bigger L1 headroom consideration
- Compute:memory ratio shifts toward compute

---

## Priorities for next work

1. **Add t1s to R=32** if Raptor Lake / other consumer chips benefit
   from it the way they did on R=16. The memory-bandwidth bottleneck
   at mid-me is real on consumer parts and we don't have a great
   answer for it.

2. **Test drain_prefetch on non-buf codelets** (flat + prefw, log3 +
   prefw). Low-cost experiment; might reveal DTLB wins we're missing.

3. **Don't extend stream drain until we hit large me** where it might
   actually win. Current gating is correct for our sweep.

4. **Don't add more exotic knobs** (weakly-ordered stores, prefetcher
   training) without evidence they'd matter. Candidate count scales
   multiplicatively; adding knobs without payoff is expensive.

The coverage is pretty solid. The one meaningful gap is **t1s-for-R=32**,
and that's worth fixing before R=64 because the same issue will apply
there.
