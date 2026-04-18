# R=32 with t1s: SPR vs Raptor Lake cross-chip comparison

Both chips benched with the same 163-candidate matrix including the new
`ct_t1s_dit` family. SPR is the Anthropic container (Sapphire Rapids-class
with AMX tiles, AVX-512 full). Raptor Lake is i9-14900KF consumer
(AVX-512 fused off, 8 P-cores).

---

## The three-way picture

**AVX2 (comparable across both chips):**

| Family | SPR AVX2 | Raptor Lake AVX2 | Δ |
|---|---|---|---|
| `ct_t1s_dit` | **11** (31%) | **24** (67%) | +13 on RL |
| `ct_t1_dit` (flat) | **14** (39%) | 0 (0%) | −14 on RL |
| `ct_t1_dit_log3` | 11 (31%) | 0 (0%) | −11 on RL |
| `ct_t1_buf_dit` | **0** (0%) | 12 (33%) | +12 on RL |
| `ct_t1_ladder_dit` | 0 | — | |

**AVX-512 (SPR only; Raptor Lake has AVX-512 disabled):**

| Family | SPR AVX-512 |
|---|---|
| `ct_t1s_dit` | **18** (50%) |
| `ct_t1_dit` (flat) | 12 (33%) |
| `ct_t1_dit_log3` | 6 (17%) |
| `ct_t1_buf_dit` | 0 |
| `ct_t1_ladder_dit` | 0 |

## The predictions held

I had predicted:

1. **t1s wins fewer regions on SPR than Raptor Lake** because SPR's
   stronger HW prefetcher partially hides load pressure. **Confirmed:**
   SPR AVX2 at 31%, Raptor Lake AVX2 at 67%. Roughly half the dominance.

2. **flat+tpf stays competitive on SPR.** **Confirmed:** SPR AVX2 flat
   wins 14 regions (39%), the plurality. On Raptor Lake it won 0.

3. **log3 still has a role on SPR.** **Confirmed:** SPR AVX2 log3 wins
   11 regions (31%), specifically at large me (me=512, 1024).

4. **Small-me buffered wins are Raptor Lake-specific.** **Confirmed:**
   SPR picks zero buffered variants. Raptor Lake picks buffered 12 times
   (all at me ≤ 128).

## Per-me breakdown — the transitions tell the story

### SPR AVX2 by me regime
| me | winners |
|---|---|
| 64 | flat:4, t1s:2 |
| 128 | **flat:6** |
| 256 | flat:4, t1s:2 |
| 512 | log3:4, t1s:2 |
| 1024 | log3:5, t1s:1 |
| 2048 | t1s:4, log3:2 |

### Raptor Lake AVX2 by me regime
| me | winners |
|---|---|
| 64 | **buf:6** |
| 128 | **buf:6** |
| 256 | **t1s:6** |
| 512 | **t1s:6** |
| 1024 | **t1s:6** |
| 2048 | **t1s:6** |

### SPR AVX-512 by me regime (for reference)
| me | winners |
|---|---|
| 64 | flat:4, log3:2 |
| 128 | **flat:6** |
| 256 | t1s:4, flat:2 |
| 512 | t1s:4, log3:2 |
| 1024 | t1s:5, log3:1 |
| 2048 | t1s:5, log3:1 |

## The three regimes in one framework

Looking at both chips together, I can now state the bottleneck regimes
cleanly:

**me ≤ 128 — memory-locality regime**

Both chips: working set fits in L1. Whoever best handles stride aliasing
wins. SPR picks flat (HW prefetch handles loads well enough that no fix
is needed). Raptor Lake picks buf (weaker prefetcher → explicit access-
pattern control via scratch buffer wins).

**me = 256-1024 — the crossover regime**

- SPR AVX2: a mix. flat+tpf wins when L1 still fits, log3 wins when
  it doesn't (me=512, 1024). The HW prefetcher on SPR can keep load
  latency mostly hidden through me=1024 with SW prefetch help.
- Raptor Lake: t1s everywhere. Load-port pressure is the dominant
  bottleneck, and t1s eliminates vector loads entirely.
- SPR AVX-512: t1s starts winning at me=256 because AVX-512's doubled
  vector width doubles the load count per butterfly (each load pulls
  8 doubles instead of 4), saturating ports sooner.

**me ≥ 1024 — SPR crosses over too**

At me=2048 on SPR AVX2, t1s finally overtakes log3 (4 vs 2). Even with
SPR's strong prefetcher, the twiddle table at me=2048 is
`31 × 2048 × 16 = 1 MB` — larger than L2 per core, so prefetcher is
now working against cache pressure. t1s's zero-twiddle-loads becomes
the win.

## What this tells us about the chip differences

### SPR has headroom Raptor Lake doesn't

- SPR AVX2 flat winning 14 regions means the HW prefetcher + L2
  bandwidth + load port throughput are all sufficient at those sizes.
  "Just emit the straightforward code" works.
- Raptor Lake can't afford that. Every region needs an explicit
  bottleneck-specific fix (buf for locality, t1s for bandwidth).

### The log3 dominance on SPR isn't about log3 being "good"

Same pattern as Raptor Lake: log3 fills a gap between flat's winning
regime and t1s's winning regime. At SPR me=512-1024, flat loses to log3
because twiddle loads start to saturate ports, but t1s wasn't winning
yet (only 2 of 12 regions). With t1s available, those regions go to
t1s in larger chips; on SPR they stay with log3.

Prediction for Granite Rapids / Diamond Rapids (even stronger prefetchers
than SPR): log3 wins even larger me regions because flat stays
competitive longer. Probably flat dominates through me=256-512, log3
through me=1024-2048, t1s only winning at me=4096+. Untested; speculation.

### Raptor Lake's buf dominance at small me is chip-specific

SPR picks zero buf. The tile-size machinery that was so important on
Raptor Lake is **not a general answer** — it's specifically addressing
consumer-chip weaknesses. On a chip with a strong enough prefetcher
and wide enough execution, the buf overhead isn't worth paying.

This means for multi-chip deployment:
- Ship buf variants for consumer chips only
- Ship flat+log3 for server chips
- Ship t1s for both (wins half the regions on SPR, 67% on RL)

## Prefetch dominance differences

| | SPR AVX2 | Raptor Lake AVX2 | SPR AVX-512 |
|---|---|---|---|
| tpf used | 18/36 (50%) | 17/36 (47%) | 16/36 (44%) |
| prefw used | 0/36 | 6/36 (17%) | 0/36 |

**`prefw` (drain-prefetch) is Raptor-Lake-only** in these results. Zero
on both SPR ISAs. This makes sense: SPR's DTLB is large enough that
explicit prefetch-for-write on output pages doesn't buy anything. On
Raptor Lake with a smaller DTLB and more pressure, it wins 17% of its
buf decisions. prefw is a consumer-chip-only knob per this data.

SW prefetch dominance is **roughly balanced** across both chips and
both ISAs (44-50%). Whatever SW prefetch is helping with, it's not
chip-specific.

## Implications for the selector design

1. **t1s is a universal win.** Wins on both chips, both ISAs. Should
   be in every R=32 codelet library.

2. **buf is a consumer-chip feature.** SPR gets no value from it at
   our sweep sizes. If code size / maintenance burden matters, buf
   variants could be stripped from server builds. But they don't hurt
   (SPR just picks other winners), so ship them everywhere by default.

3. **flat stays relevant on server chips.** Don't strip it. SPR wins
   14 regions with flat AVX2.

4. **log3 stays relevant on server chips.** Same logic.

5. **ladder wins nothing** (0 regions on SPR AVX-512). This is
   interesting — we thought ladder addressed a different bottleneck
   (instruction bandwidth + register pressure, AVX-512-specific). But
   t1s beats it in every head-to-head. Probably because ladder still
   does vector twiddle loads (just fewer of them); t1s does zero.
   Consider: ship ladder only if a specific chip (not yet seen) picks
   it, otherwise the matrix can drop it for R=32 going forward.

## Predictions for R=64

If the pattern scales:

- **t1s will dominate even more on Raptor Lake R=64** — 63 twiddle rows
  means the flat-load version has even more port pressure. Maybe 80%+
  wins on Raptor Lake.
- **SPR R=64 t1s wins will be higher than R=32** too — the me threshold
  where t1s beats flat/log3 will shift to smaller me, because the
  twiddle table grows as `63 × me × 16` instead of `31 × me × 16`.
  Maybe 15-20 SPR AVX-512 wins instead of 18 on R=32.
- **buf wins on Raptor Lake** will still exist at small me, but the
  transition point may shift down to me=64 only (vs me≤128 on R=32).

## Per-element cost comparison

Can't directly compare absolute nanoseconds across chips (different
clocks: Raptor Lake 5.68 GHz boost, SPR container unknown but likely
slower), but the **ratios within each chip** are informative:

| me | Raptor Lake ns (fwd) | SPR AVX2 ns (fwd) | SPR AVX-512 ns |
|---|---|---|---|
| 64 | 1100 | — (no data extracted yet) | — |
| 2048 | 43090 | — | — |

For a proper comparison I'd need the SPR ns numbers exported from the
report. If you want that, run `emit_selector.py` on both chips' data
and I'll cross-tabulate.

## Summary

The t1s port was a clear win on both chips. On Raptor Lake it took over
the selector (67%). On SPR it took 30-50% depending on ISA. The
bottleneck-coverage hypothesis is validated: t1s addresses
memory-bandwidth by eliminating vector twiddle loads, and this
bottleneck is most acute on consumer chips (weak prefetcher) and at
larger me (twiddle table spills L1/L2).

R=64 t1s is now clearly next priority. Expected to produce similar or
larger effects.
