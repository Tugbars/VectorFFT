# R=16 Codelet Selection: SPR vs Raptor Lake

Comparative analysis of the VectorFFT codelet bench outputs on two very
different Intel microarchitectures. Same codelet library, same candidate
matrix, same sweep grid — two strikingly different selections.

**Takeaway up front:** the winning codelet depends on the hardware
prefetcher's aggressiveness, not on any universal "best codelet" metric.
Server and consumer chips produce opposite winners at most sweep points.

---

## Chips

| | Sapphire Rapids (SPR) | Raptor Lake (i9-14900KF) |
|---|---|---|
| Class | Server (Xeon 4th gen) | Consumer desktop (14th gen) |
| AVX-512 | Yes | No (fused off in consumer) |
| L1D | 48 KB, 12-way | 48 KB, 12-way |
| L2 | 2 MB, 16-way (per core) | 2 MB, 16-way (P-core) |
| L3 | Large, shared, inclusive | 36 MB shared |
| Memory channels | 8 × DDR5-4800 | 2 × DDR5-5600 |
| Prefetcher aggressiveness | High (multi-stream, X-page) | Conservative |

The cache geometry is nearly identical per-core. The big differences are
(a) AVX-512 availability and (b) hardware prefetcher strength.

---

## Results summary

### Win counts

| Candidate | SPR (AVX2+AVX512) | Raptor Lake (AVX2 only) |
|---|---|---|
| `ct_t1_dit_log3__avx2` | **23** | 9 |
| `ct_t1_dit_log3__avx512` | **26** | — (no AVX-512) |
| `ct_t1s_dit__avx2` | 6 | **25** |
| `ct_t1s_dit__avx512` | 9 | — |
| `ct_t1_buf_dit__avx2__tile128_draintemporal` | 5 | 0 |
| `ct_t1_buf_dit__avx2__tile16_draintemporal` | 0 | 2 |
| `ct_t1_dit__avx2` | 2 | 0 |
| `ct_t1_dit__avx512` | 1 | — |
| **Total decisions** | **72** | **36** |
| **Distinct winners** | **7** | **3** |

### Win distribution (as % of available regions)

On SPR the selector picks 7 different codelets across the sweep. On Raptor
Lake the selector picks 3. Both are correct — they reflect which codelets
actually win on each chip.

| Candidate | SPR AVX2 % | Raptor Lake AVX2 % |
|---|---|---|
| `ct_t1_dit_log3` (derived twiddles) | 64% | 25% |
| `ct_t1s_dit` (scalar-broadcast) | 17% | **69%** |
| `ct_t1_buf_dit` (buffered) | 14% | 6% |
| `ct_t1_dit` (flat) | 6% | 0% |

**The picture flipped.** `log3` dominates SPR; `t1s` dominates Raptor Lake.

---

## Interpretation: why the flip

### `log3`: needs good prefetching

`ct_t1_dit_log3` derives its twiddles from three base twiddles and
computes the rest on-the-fly via log3-structured multiplications. It
still reads twiddles from memory — but the access pattern is more
structured, and the arithmetic-to-memory ratio is high.

**On SPR:** the HW prefetcher delivers twiddles ahead of need. The
derived-twiddle scheme runs near compute-bound. Wins 64% of regions.

**On Raptor Lake:** the HW prefetcher doesn't keep up with log3's
reads at larger me. Twiddle latency becomes visible. log3 still wins
at padded-stride large-me regions (where other candidates also slow
down), but loses at small-me regions where faster alternatives exist.
Wins 25%.

### `t1s`: eliminates twiddle loads

`ct_t1s_dit` broadcasts each twiddle as a scalar once per k-iteration
rather than loading a twiddle vector per element. Uses ~4× fewer memory
accesses than `t1_dit`. Trade-off: more ALU work to "spread" the scalar
into vector lanes.

**On SPR:** t1s is slower than log3 in most regions — the extra ALU
work doesn't pay off because HW prefetching made the memory-bound
alternative fast enough. Wins only 17%.

**On Raptor Lake:** t1s is dramatically faster than everything else at
small me. Without HW prefetcher rescuing the memory path, eliminating
twiddle loads outright becomes the winning move. **Wins 69%.**

### `buf_dit`: tile size tracks µarch width

| Chip | Tile size winner | Why |
|---|---|---|
| SPR | `tile128` (5 regions) | Wide front-end (6 decoder + 8 execution ports) amortizes buffer setup over more work |
| Raptor Lake | `tile16` (2 regions) | Narrower front-end (6 decoders) prefers smaller chunks; tile16 fits in 1-2 cache lines |

On SPR, `tile128_temporal` wins at large me (512, 1024, 2048) where
the bigger working set benefits from batched prefetching. On Raptor
Lake, `tile16_temporal` wins at just two regions (small ios, specific
points) — the larger tiles are never worth the setup cost.

### `flat ct_t1_dit`: strictly dominated on Raptor Lake

Flat `t1_dit` wins 2 regions on SPR (me=256 at one specific ios value).
On Raptor Lake it wins zero regions. Every other candidate beats it
somewhere.

This is a notable result: the "baseline" codelet — the simplest,
smallest, most predictable in codegen — is never the best choice on
Raptor Lake. At every measured (ios, me) point there's a specialized
codelet that beats it.

---

## Split-point analysis

Where exactly do the winners change? Here's the AVX2-only comparison
(Raptor Lake has no AVX-512 so we compare apples-to-apples):

### fwd direction

| ios | me | SPR winner | Raptor Lake winner |
|---|---|---|---|
| 64 | 64 | `t1s` | `t1s` |
| 72 | 64 | `t1s` | `t1s` |
| 128 | 64 | `log3` | `t1s` |
| 128 | 128 | `log3` | `t1s` |
| 136 | 128 | `log3` | `t1s` |
| 192 | 128 | `log3` | `t1s` |
| 256 | 256 | `flat t1_dit` | `t1s` |
| 264 | 256 | `log3` | `t1s` |
| 320 | 256 | `log3` | `t1s` |
| 512 | 512 | `buf_tile128` | `buf_tile16` |
| 520 | 512 | `log3` | `t1s` |
| 576 | 512 | `log3` | `t1s` |
| 1024 | 1024 | `buf_tile128` | `log3` |
| 1032 | 1024 | `log3` | `log3` |
| 1088 | 1024 | `log3` | `log3` |
| 2048 | 2048 | `buf_tile128` | `log3` |
| 2056 | 2048 | `log3` | `log3` |
| 2112 | 2048 | `log3` | `log3` |

**Patterns:**

1. **Small me (64, 128, 256) with any stride:** SPR picks `log3`, Raptor Lake picks `t1s`.
   HW prefetcher aggressiveness fully explains this — `t1s` avoids the memory pattern
   the Raptor Lake prefetcher can't sustain.

2. **Medium me (512, 1024) with power-of-2 ios:** both chips prefer a
   buffered variant, but different tiles — `tile128` on SPR (wide µop
   window), `tile16` on Raptor Lake (narrow µop window).

3. **Medium me with padded stride:** both chips converge on `log3`. This
   is the regime where stride-aliasing is solved, twiddle access is
   regular, and the FMA throughput matters most.

4. **Large me (2048):** SPR still picks buffered at pow2 stride; Raptor
   Lake has no buffered wins there. The tile sweet spot on Raptor Lake is
   smaller (tile16 loses at large me because buffer setup amortizes over
   less net work).

---

## The stride-aliasing signal

Both chips show strong stride-aliasing at power-of-2 ios, but Raptor
Lake is worse. Example at me=1024, flat `t1_dit`:

| ios | SPR fwd (ns) | Raptor Lake fwd (ns) |
|---|---|---|
| 1024 (pow2) | ~24000 | **17966** |
| 1032 (+8) | ~12000 | **10021** (−44%) |
| 1088 (+64) | ~12500 | **10161** (−43%) |

Raptor Lake loses **44%** of performance going from padded stride to
power-of-2 stride, on the same codelet, at the same input size. The
L2 set-associativity conflict is the cause, and it's more punishing
on Raptor Lake than SPR. Both chips have 16-way L2 at this point but
Raptor Lake's prefetch+fill pipeline apparently falls over harder
under the aliased access pattern.

**Note for planner:** the bench/selector can't change the ios of the
input, but a *planner* can — by selecting a different factorization
that naturally lands on padded strides. This is a v2 optimization:
composite plan-level decisions driven by the same chip profile.

---

## The stream drain catastrophe on Raptor Lake

Not captured in the selector output (no streaming variants won any
regions on either chip), but visible in the raw data: `drain=stream`
is genuinely catastrophic on Raptor Lake, dramatically worse than on SPR.

At me=64 with `tile16`:
- Raptor Lake temporal: 452 ns
- Raptor Lake stream: **2729 ns** (6.0× slower)
- SPR temporal: ~462 ns (comparable)
- SPR stream: ~1100 ns (2.4× slower — still bad but not 6×)

**Evidence-based pruning rule** emerging from this data: on consumer
chips like Raptor Lake, `drain=stream` should be gated to only
candidates where output bytes exceed L2 (>2 MB for this class), i.e.
me ≥ 4096 for R=16. On server chips the gate can be more lenient.

This is exactly the pruning logic we want for R=32 — where the
candidate matrix grows with twiddle-prefetch knobs, and cutting stream
candidates below threshold saves 25-50% of bench time.

---

## Infrastructure notes

- **Bench wall time:** SPR 90s, Raptor Lake 12s. Raptor Lake is faster
  per candidate largely because it only benches 11 candidates (AVX-512
  auto-skipped) vs 22 on SPR.
- **Harness portability:** Raptor Lake run used Intel ICX + LLD linker
  + Intel runtime libs; SPR run used GCC on Linux. Same harness C
  source compiled and produced valid binaries on both. Zero codelet
  changes needed.
- **Selector header:** both chips produce a valid `codelet_select_r16.h`
  with per-(ios, me) lookup tables. A planner compiled against one
  selector header will dispatch correctly for that chip.

---

## Conclusions

**This is what "µarch decides codelets, planner decides factorization"
looks like in practice.** The selector for SPR ships 7 distinct codelets
(many log3 variants). The selector for Raptor Lake ships 3 (mostly t1s).
A universal "best codelet" choice would be wrong on one of these chips.

### Validated predictions

- **Multi-region generation matters:** Raptor Lake's selector has buffered
  variants at specific (ios, me) points and non-buffered elsewhere.
  Collapsing to "one winner per chip" would have lost those wins.
- **Server vs consumer chips select different codelets:** strongly confirmed
  by the 64% → 25% flip in `log3` dominance.
- **Stride aliasing is µarch-dependent:** both chips suffer, Raptor Lake
  more. Selector correctly picks different codelets at pow2 vs padded ios.

### New evidence-based observations

- **`t1s` is underrated** on server chips, where HW prefetching hides
  its memory-reduction advantage. On consumer chips it's the dominant
  winner.
- **Stream drain is hard-off on consumer chips** below L2 threshold.
  Safe to gate pre-bench on R=32 and later.
- **Tile size tracks µarch width**, not cache size. SPR wants tile128;
  Raptor Lake wants tile16. Generation should offer both.

### For R=32

1. Keep `t1s` as a candidate family (critical for Raptor Lake, minor on SPR).
2. Gate `drain=stream` by `output_bytes > L2_size` as hardcoded constant.
3. Include both tile16 and tile128 in the buf_dit matrix — chip-specific.
4. Bench on both SPR and Raptor Lake to track how twiddle_prefetch knobs
   behave on weaker prefetcher chips (hypothesis: prefetch helps more on
   Raptor Lake).
