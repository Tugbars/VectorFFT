# R=64 buf port: negative result

**Summary: buf wins zero regions at R=64 on SPR. The mechanism that made it useful at R=32 breaks down as R grows.**

The R=64 buf port (ct_t1_buf_dit) was completed and benched. Correctness was verified bit-exact against t1_dit across 16 test cases on both AVX2 and AVX-512. 12 buf configurations were added to the candidate matrix (3 tiles × 2 drain modes × 2 prefw flags × 2 ISAs = 24 new candidates; total candidate count 34).

## Selector summary

72 decisions. All of them went to **t1s** or (on AVX-512 small me) log3. Zero buf wins.

```
34 wins  ct_t1s_dit__avx2        (was 33, now 34 — picked up one more small-me region)
21 wins  ct_t1s_dit__avx512
11 wins  ct_t1_dit_log3__avx512
 2 wins  ct_t1_dit__avx512
 2 wins  ct_t1_dif__avx512
 1 wins  ct_t1_dit__avx2
 1 wins  ct_t1_dif__avx2
 0 wins  ct_t1_buf_dit (any config)
```

## Why buf fails at R=64

### Memory traffic analysis (R=64, me=2048, AVX2)

| Variant | Traffic per call | Notes |
|---|---|---|
| t1_dit (flat) | **6 MB** | 2 MB input + 2 MB twiddle + 2 MB output |
| t1s | **4 MB** | 2 MB input + 2 MB output (twiddles in regs) |
| t1_buf | **10 MB** | 2 MB input + 2 MB twiddle + 2 MB outbuf-wr + 2 MB outbuf-rd + 2 MB rio-wr |

buf does *more* memory traffic than flat, because it stages outputs through the scratch buffer and then drains them. The hope is that:

1. The scratch writes hit a locality-friendly pattern (one page of outbuf per stream vs 64 scattered pages in rio)
2. The drain reads the scratch sequentially (warm cache)
3. The drain writes are page-dense (one stream's full TILE of k-positions written contiguously)

The first two hopes materialize. The third partially does. But the extra 4 MB of traffic (the staging + drain pass) has to be paid somewhere. t1s's 2 MB reduction (eliminating twiddle reads) buys what buf's extra 4 MB costs and more.

### The outbuf sizing problem at R=64

Outbuf size = 2 × R × TILE × 8 bytes (re + im, doubles).

| R | tile=32 | tile=64 | tile=128 |
|---|---|---|---|
| R=16 | 8 KB | 16 KB | 32 KB |
| R=32 | 16 KB | 32 KB | 64 KB |
| R=64 | **32 KB** | 64 KB | 128 KB |

With 48 KB L1D on modern consumer/server chips:

- **R=32 tile=64**: outbuf = 32 KB, fits L1 comfortably → drain reads hit L1
- **R=64 tile=32**: outbuf = 32 KB, *just barely* fits L1 (needs to evict other L1 residents) → drain reads often hit L2
- **R=64 tile=64**: outbuf = 64 KB, exceeds L1 → drain reads hit L2

The drain-reads-L2 case means buf adds a full 2 MB of L2→L1 traffic that didn't exist in flat. At R=32 this was avoided; at R=64 it's unavoidable.

The bench confirms this. AVX2 fwd, ios=me+8 (padded stride):

| me | t1_dit | t1s | best_buf (tile=32 temporal) | buf config |
|---|---|---|---|---|
| 256 | 20687 | 17852 | 35973 | tile32_temporal |
| 1024 | 129381 | 71430 | 181090 | tile32_temporal_prefw |
| 2048 | 251703 | 145691 | 348621 | tile32_temporal |

**At every me, buf loses to both t1s AND t1_dit.** Worst case it's 2.4× slower than t1s at me=2048.

### The one case where buf was competitive

At ios=me (power-of-2 stride, worst DTLB case), buf DID outperform naive flat t1_dit:

| me | t1_dit (ios=me) | buf tile=32 (ios=me) | buf vs t1_dit |
|---|---|---|---|
| 512 | 90783 | 75017 | **+17%** |
| 1024 | 247707 | 179010 | **+28%** |
| 2048 | 527446 | 353139 | **+33%** |

But t1s at the same conditions is still faster:

| me | t1s (ios=me) | buf tile=32 (ios=me) | t1s vs buf |
|---|---|---|---|
| 512 | 65797 | 75017 | **+12% in favor of t1s** |
| 1024 | 127709 | 179010 | **+29%** |
| 2048 | 261083 | 353139 | **+26%** |

t1s wins even in buf's best case.

## Stream drain mode is a disaster

Stream (NT) stores hurt at every me. At me=256 AVX2, buf_stream is **2.4×** slower than buf_temporal. This matches known behavior — NT stores are expensive for outputs that stay within L2, which is the working-set range we're in.

## prefw is neutral

Drain prefetch was sometimes fractionally faster, sometimes fractionally slower. Within noise. The store DTLB entries it's trying to warm are evicted too quickly under the outbuf traffic pressure for prefetch to matter.

## Combined t1s + buf not benched

I did not emit a "t1s + buf" combined variant. Mechanically it's straightforward — make the emitter apply t1s's scalar-broadcast twiddles inside the buf variant's inner kernel. The prior is this wouldn't win either, because even if you combine the two:

- t1s saves 2 MB twiddle traffic → now at 2 MB (input) + 2 MB (output) + 2 MB (outbuf-wr) + 2 MB (outbuf-rd) = 8 MB
- t1s alone is 4 MB

Still 2× the traffic. Unless the store-DTLB relief at DTLB-hostile strides is worth 4 MB of extra traffic, this doesn't help. And the data above shows t1s alone beats buf at DTLB-hostile strides anyway.

A "t1s + buf" port is worth ~3 hours. Low priority given the analysis above suggests it won't help.

## What this means

**buf is a R-dependent optimization.** At R=16/R=32 it wins some regions because the outbuf fits L1 comfortably and the output-side DTLB relief is proportionally larger. At R=64 the outbuf is too large to stay in L1, and the twiddle-side bottleneck (which t1s already solves) dominates the store-side bottleneck that buf attacks.

This is a useful cross-radix pattern for the taxonomy doc:

| Radix | buf wins | Why |
|---|---|---|
| R=16 | 0 regions on SPR (was tested, didn't win either on consumer chips AFAIK) | Compute-bound, no memory relief needed |
| R=32 | 12 regions (17%) | Outbuf fits L1, store DTLB relief outweighs staging overhead |
| **R=64** | **0 regions** | Outbuf spills L1 or is too large; twiddle DTLB (now solved by t1s) dominated store DTLB anyway |

The buf family has a sweet spot at medium R (R=32). Beyond that, the bottleneck balance shifts to favor approaches that reduce total work (t1s) over approaches that reorganize output access patterns (buf).

## What remains

Three takeaways from the buf bench:

**(a) VTune's "store DTLB 43.7% still on the table" prediction was wrong in implication.** That 43.7% existed in the flat codelet, but it's not actually bottlenecking us — when we eliminate the load side via t1s, the store-DTLB cycles overlap with other compute and don't show up as end-to-end delay. The 2× t1s speedup is already close to the compute-limited ceiling.

**(b) buf doesn't generalize to R=64.** We should document this as a radix-dependent family (useful R ≤ 32, not useful R ≥ 64).

**(c) Next optimization direction should target t1s itself, not try to replace it.**
   - Tighter twiddle-hoisting budget (we use 2 hoisted at R=64 AVX2; could be 4-6 if we push it)
   - Software pipelining across me (latency hiding)
   - Better drain/tile heuristics for small me where log3 still beats t1s on AVX-512

For further R=64 gains the highest-value move is probably profiling t1s with VTune to find the new bottleneck — we've only guessed.

## Artifacts

- `gen_radix64.py`: buf port in generator (fully working, correct)
- `candidates.py`: 34-candidate matrix including 24 buf configs
- `test_buf_correctness.c`: bit-exact validation (passed on AVX2 and AVX-512)
- `spr_results/measurements.jsonl`: 1224 measurements
- `spr_results/selection.json`: selector output (0 buf wins)
