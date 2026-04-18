# R=16 and R=32 on SPR AVX-512: the story

Separate from the AVX2 analysis, this is what happens on the same chip
(SPR-class container, AVX-512 full with AMX tiles) when the vector width
doubles from 4 doubles (AVX2) to 8 doubles (AVX-512).

All numbers are 36-decision SPR selectors at each radix.

---

## R=16 on SPR AVX-512

### Family wins (36 decisions)

| Family | AVX2 | AVX-512 |
|---|---|---|
| `ct_t1_dit_log3` | 22 (61%) | **22** (61%) |
| `ct_t1s_dit` | 8 (22%) | 6 (17%) |
| `ct_t1_dit` | 0 | **8** (22%) |
| `ct_t1_buf_dit` | 6 (17%) | **0** |

**Log3 dominates both ISAs at R=16 on SPR.** That's the surprising part —
the bottleneck-coverage doc later framed log3 as "second-best when t1s
isn't available," but at R=16 on this chip, log3 beats t1s head-to-head.
Log3's derived-twiddle trick wins here.

### The buf-disappears effect

On AVX2, buf wins 6 regions (all at large me=512+). On AVX-512, buf wins
**zero**. This is the cleanest single-variable flip in the data.

Why buf evaporates when the vector width doubles:

- Buf's job is fixing **memory-locality** (stride aliasing, DTLB pressure)
  by copying data to a scratch buffer.
- At AVX2 with VL=4, the butterfly does 4 columns per iteration.
  Aliased strides hit the same L2 sets ~4× per iteration.
- At AVX-512 with VL=8, the butterfly does 8 columns per iteration in
  one pass. The HW prefetcher sees wider strides, keeps up better.
- AVX-512's larger register file (32 ZMM vs 16 YMM) means the same
  butterfly code holds more intermediate values in registers, reducing
  the memory pressure that buf was compensating for.

The **same chip, same twiddle tables, same sweep** — just doubling vector
width makes the memory-locality bottleneck either go away or get absorbed
by the more capable OOO engine's wider speculation window.

### t1s wins less on AVX-512 than AVX2

AVX2: 8 regions. AVX-512: 6 regions.

At R=16, AVX-512's wider vectors reduce the load count per iteration
(fewer butterflies needed to cover the same me). The load-port pressure
t1s was relieving on AVX2 isn't as acute on AVX-512. So t1s's win-rate
shrinks.

### Plain `ct_t1_dit` (flat) comes back on AVX-512

Zero wins on AVX2. **8 wins** on AVX-512. These are mostly at me=256 and
me=1024 where the chip has enough headroom for naive straight-through
code to just work. AVX-512's wider execution means the work-per-iteration
is big enough that memory latency overlaps with compute naturally — no
codelet-level tricks needed. The straightforward code wins.

---

## R=32 on SPR AVX-512

### Before t1s was added (early bench)

| Family | Regions |
|---|---|
| `ct_t1_dit_log3` | 20 (56%) |
| `ct_t1_dit` | 11 (31%) |
| `ct_t1_buf_dit` | 3 (8%) |
| `ct_t1_ladder_dit` | 2 (6%) |

### After t1s was added

| Family | Regions |
|---|---|
| `ct_t1s_dit` | **18** (50%) |
| `ct_t1_dit` | 12 (33%) |
| `ct_t1_dit_log3` | 6 (17%) |
| `ct_t1_buf_dit` | 0 |
| `ct_t1_ladder_dit` | 0 |

**t1s takes over. Log3 gets eviscerated. Buf and ladder disappear
entirely.**

### The log3 → t1s reassignment is the main story at R=32 AVX-512

Log3 went from 20 wins to 6. Where did the 14 it lost go?

Looking at the per-me data before/after:
- me=256-512: log3-heavy before, now t1s/flat mixed
- me=1024-2048: log3 owned this before, now t1s dominates

This matches the R=16 pattern **but shifted**. On R=16 AVX-512, log3
won 22/36 with t1s present. On R=32 AVX-512, log3 wins only 6/36 with
t1s present.

What's different at R=32?

- **Twiddle table is 2× bigger** (31 rows vs 15). Load pressure scales
  linearly with R.
- **Butterfly compute is 2.5× heavier** (400 FMAs vs 160). Memory
  latency has more time to hide, but ALSO more loads per butterfly.
- **Bottom line:** the ratio of load work to compute work stays roughly
  the same, but the absolute loads per iteration increase. Load ports
  saturate sooner. t1s's zero-load approach wins more.

### Ladder gets zero on R=32 AVX-512

Ladder was designed for AVX-512's 32-ZMM budget — using more registers
to cache more derived twiddles. It won 2 regions before t1s. With t1s
present, zero.

Why: ladder still does vector twiddle loads, just fewer of them (5 base
twiddles instead of 31, with derivations). t1s does zero vector loads.
The magnitude of the fix matters: "fewer loads" loses head-to-head
against "no loads" whenever load-port pressure is the dominant factor.

Ladder's register-caching story is real but needs a different scenario
to win — **register pressure has to actually be binding**, which at R=32
it isn't (32 ZMM is enough for R=32's butterfly without ladder's tricks).

### Buf gets zero on R=32 AVX-512 too

Same reason as R=16: AVX-512's wider vectors + bigger ROB + stronger
prefetch absorb the memory-locality problems buf was fixing.

### SW prefetch stays at 44% usage

On AVX-512 R=32, 16 of 36 winners use SW prefetch. On AVX2 R=32 it was
50%. Similar-ish. The distances preferred are shifted: AVX-512 winners
cluster on tpf32 (distance 32), AVX2 clusters around tpf16.

Why different distances:
- AVX-512 processes 8 columns per iteration (VL=8). To prefetch data
  for iteration m+32, we need 32 × 15 cycles = ~480 cycles ahead.
- AVX2 processes 4 columns per iteration (VL=4). To cover same memory
  latency, want ~half the distance.
- So the autotuner finds these settings naturally.

---

## The pattern across radixes on SPR AVX-512

| Radix | Log3 wins | t1s wins | Flat wins | Other |
|---|---|---|---|---|
| R=8 | — (not benched) | — (not benched) | DIT:11 DIF:25 | — |
| R=16 | **22** | 6 | 8 | buf:0 |
| R=32 | 6 | **18** | 12 | buf:0, ladder:0 |

The selection migrates **from log3 to t1s** as R grows — because the
bottleneck that log3 partially solves (load count) becomes severe enough
at R=32 that you need the complete solution (t1s, which eliminates
vector twiddle loads entirely).

At R=16 log3 is *good enough* because the load count per butterfly
(15 loads per iter) is manageable. At R=32 (31 loads per iter) it isn't,
and t1s takes over.

## What's "meh" on SPR AVX-512

- **buf**: 0 wins at both R=16 and R=32 on AVX-512. The wider vector
  width kills its value on SPR.
- **ladder**: 2 wins at R=32 without t1s, 0 with t1s. Register-pressure
  story never lands on AVX-512 at these radix sizes.
- **prefw (drain-prefetch)**: 0 wins on AVX-512 at both radixes. DTLB
  warmth isn't a bottleneck on SPR's large DTLB.
- **stream drain**: not tested (gated by cache-spill rule).

## What works on SPR AVX-512

- **log3** at R=16 specifically
- **t1s** from R=32 upward
- **flat t1_dit** as the fall-through when nothing else is needed
- **SW twiddle prefetch** (`tpf*`) at ~44% of winners — genuinely useful
  on AVX-512 at mid-to-large me

## How this differs from the AVX2 story

Same chip, same bench, just different ISA. Yet the family preferences
shift substantially:

| Knob | AVX2 R=16 | AVX-512 R=16 | AVX2 R=32 | AVX-512 R=32 |
|---|---|---|---|---|
| log3 % | 61 | 61 | 39 | 17 |
| t1s % | 22 | 17 | 31 | 50 |
| flat % | 0 | 22 | 39 | 33 |
| buf % | 17 | 0 | 25 | 0 |

The key observation: **buf is an AVX2-only feature on SPR**. It wins
nothing on SPR AVX-512 at any radix we've tested. Buf's value proposition
(access-pattern control for a narrow vector with a weaker OOO window)
doesn't apply when the vector is 2× wider and the OOO engine has more
to work with.

## Predictions for R=64 SPR AVX-512

The migration from log3 → t1s across increasing R suggests t1s should
dominate even more at R=64. Predict:
- t1s wins 25+ regions on SPR AVX-512 R=64 (up from 18 at R=32)
- Log3 wins near-zero (maybe 1-2 at very small me)
- Flat/t1_dit keeps wins only at very small me
- **Ladder might finally appear**: R=64 has 63 twiddle rows + 64 inputs.
  Register pressure at R=64 AVX-512 is real (2× over the 32-ZMM budget).
  Ladder was designed for this case. If it wins anywhere, R=64 is it.
- Buf still zero (SPR AVX-512 consistently doesn't use it).

If ladder wins zero on R=64 too, drop it from the candidate matrix
entirely and save the maintenance burden.
