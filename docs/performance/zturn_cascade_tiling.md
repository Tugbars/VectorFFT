# ZTURN cascade tiling — mechanism, legality, and the tile width

Follow-on to [../roadmap/cascade_load_path_restructure.md](../roadmap/cascade_load_path_restructure.md)
(the ZTURN restructure campaign) and companion to
[../wisdom/10_zturn_calibration_flow.md](../wisdom/10_zturn_calibration_flow.md)
(how a tile width is searched and banked).

Everything here concerns the K=1 SCRAMBLED cascade in `src/core/oop/zturn.h`.
Measured figures are benchmark-derived on one host (i9-14900KF, P-core, 48 KB
L1d) on 2026-08-02, warm and single-threaded, and are labelled where they appear.

---

## 1. Tiling is a loop interchange

The cascade runs several passes over the plane. Today each pass sweeps the whole
array before the next begins. Tiling changes only the **order the work is
issued in** — same butterflies, same twiddles, byte-identical output.

Take the grid of (pass × memory). Stage-major traverses it row by row:

```
             tile0  tile1  tile2  tile3
   pass 1  →   1      2      3      4        finish a pass across all memory,
   pass 2  →   5      6      7      8        then start the next one
   pass 3  →   9     10     11     12
   pass 4  →  13     14     15     16
   pass 5  →  17     18     19     20
```

Tile-major traverses it column by column:

```
             tile0  tile1  tile2  tile3
   pass 1  →   1      2      3      4        pass 1 still sweeps — its groups
   pass 2  →   5      9     13     17        span a whole section
   pass 3  →   6     10     14     18
   pass 4  →   7     11     15     19        one tile is pulled in once and
   pass 5  →   8     12     16     20        pushed through passes 2-5 in L1
```

Both issue the same 20 kernel calls. On the left, by the time pass 2 returns to
`tile0` that data left L1 long ago, so every call pays a trip to L2. On the
right, three of every four calls are cache hits.

---

## 2. Why the reorder is legal — this is the part that matters

A pass does not roam. It chops the plane into equal independent **groups**, each
a run of consecutive complex points, and **a group reads and writes only inside
its own window — nothing outside**. Not mostly-local: nothing.

Inside a group of width `W` the pass takes four values spaced `W/4` apart,
combines them, and writes the results back to those same slots. So a group's
width is four times the reach of one butterfly.

Each pass splits the previous pass's groups into four, so group widths **shrink**
across the cascade and the groups **nest perfectly** — every pass-3 group sits
wholly inside one pass-2 group, and no group ever straddles another pass's
boundary.

Therefore a window that holds *whole* groups of passes `k…n` contains all of
that work, sealed. It depends on nothing happening in any other window, so the
passes may be run window-by-window instead of pass-by-pass: same operations,
same dependencies honoured, bit-for-bit identical output.

> 🔴 **The width test — "the group must fit the tile whole" — is not a
> performance heuristic. It is the condition that makes the reorder legal.**
> Get it wrong and the answer is wrong, not slow. That is why the gate is
> `memcmp` equality and not a tolerance.

### 2.1 The two operations that are not groups

- The **ingest** reads the caller's input and rearranges it before any pass
  runs. Its reach is the whole array, so it can never fold into a tile — it
  must finish first, always.
- The **terminator** writes the output and reads across all four sections at
  once, so it is not sealed inside a tile either. Folding it in (`tfuse`)
  required its own argument.

> ⚠️ **Recorded hazard.** Nest the fused terminator's loops the wrong way round
> and the *forward* transform still comes out correct — later writes paper over
> the earlier wrong ones — while the *inverse* is silently corrupted. **A test
> that checks only the forward direction passes clean.** Gate both directions.

---

## 3. Why the fused set is a tail, and why one number names it

Because widths only ever shrink, the passes that fit a given tile are always a
run of consecutive passes at the **end**. There is never a case where pass 2
fits, pass 3 does not, and pass 4 fits again. So no per-pass flag is needed —
one number says where the fitting ones begin.

**Pass 1 is not permanently excluded.** Its groups are as wide as a whole
section, so fusing it means the tile *is* the section — which is fine while a
section still fits L1. Benchmark-derived: at 4096 the best configuration fuses
*every* pass, one tile per section. At 8192 it still works. Only at 16384 has
the section grown too large and pass 1 drops out. It is the first thing to fall
off as N grows, not a structural bar.

---

## 4. The tile width is a free parameter

The spec's rule is general — a pass fits a window `w` iff its group width
divides and is ≤ `w`. The implementation adopted the special case `w = D[tcut]`
because that makes the tail property automatic, and **every campaign figure
before 2026-08-02 was measured under that special case.**

Consequences:

- Pinning `w` to the chain's running products restricts the tile to that
  ladder, and the ladder steps by 4 or 8 — the only mid radices. From a 32 KB
  section you reach 8 KB but **never 16 KB**. Enumerated over all chains:
  at N=8192, **0 of 9 chains can express a 16 KB tile**; at 4096, 7 of 7 can.
- The legality predicate never required the pin — it already checks the general
  conditions. One assignment did.
- 🔴 **The width alone determines the cut.** The group widths form a divisibility
  chain, so the set that fits is a tail for free. It is **one knob, not two**;
  the cut is derived.

---

## 5. The working set is about twice the tile

```
working set = 16·w  +  Σ (w / D[s-1]) · (radix_s − 1) · 64      bytes
                       over the fused passes
```

First term the tile, second the twiddle records the fused passes read. The sum
is dominated by the **last** fused pass — narrowest groups, so its records cover
the least data each — and for an all-radix-4 chain the series comes to almost
exactly one tile.

**So sizing a tile as "L1 ÷ 16 bytes per point" is wrong by ~2×.** This is the
least guessable number in the whole mechanism.

It is an exact byte count over structures we control, not a fitted model. It
reproduces the occupancies recorded independently by the campaign:

| cell | tile | twiddles | working set | % of 48 KB L1 |
|---|---|---|---|---|
| 4096 `4⁶` cut 0 | 16 KB | 15.9 KB | 31.9 KB | 66.5% |
| 8192 `4.4.4.4.4.8` cut 0 | 32 KB | 15.9 KB | 47.9 KB | 99.9% |
| 16384 `4.8.4.4.4.8` cut 0 | 64 KB | 31.9 KB | 95.9 KB | 199.8% |

**Where the radix-8 passes sit changes the twiddle half by 2×.** At 8192 cut 0,
`4.4.4.4.4.8` gives 15.9 KB of twiddles; `4.4.8.4.4.4` gives 31.9 KB for the
identical 32 KB tile. A chain is not only an op-count choice.

---

## 6. Measured

Benchmark-derived, this host, 2026-08-02. Internal A/B, arms grouped by the plan
they actually engaged, per-width duplicate controls. Same-plan spread was
0.03–2.7% against an untiled A/A floor of 1.7% — **nothing under ~3 points is a
result.**

**N=16384, banked chain `4.8.4.4.4.8`:**

| width | occupancy | passes fused | measured | reachable how |
|---|---|---|---|---|
| 32 KB | 99.5% | 3 | **−13.78%** | width axis only |
| 16 KB | 49.7% | 3 | **−13.20%** | width axis only |
| 8 KB | 24.9% | 3 | −9.93% | ladder |
| 64 KB | 199.9% | 4 | −1.26% | ladder |

The width axis is worth ~3.3–3.9 points on this chain, replicated across three
runs. Joint chain × width in one run: `4⁷` measured **+15.26% slower untiled**
and **−15.52% tiled at 32 KB** — the fastest configuration measured, and one the
ladder cannot express for that chain either.

**N=8192 — the width axis does not help here.** Best is 32 KB at −18.3%, which
is the *existing* ladder option. The reason is visible in the model: 32 KB fuses
**four** passes at 99.9% while 16 KB fuses **three** at 49.7%.

> 🔴 **Occupancy only ranks candidates when passes-fused is held constant.**
> At 16384 all three widths fuse three passes and occupancy shows a clean
> interior optimum. At 8192 the pass count differs and dominates. A
> target-occupancy heuristic picks wrong at 8192 — which is why the occupancy
> band is a **filter** and the measured time is the **chooser**.

**N=2048 shows no win** (+3.3%, rising monotonically as tiles narrow — the
per-call cost on a plane already L1-resident). The occupancy model independently
returns *zero* in-band widths there, so the calibrator spends no benchmark time
on it.

---

## 7. Orientation: ours tiles a tail, MKL tiles a head

Tileability is a width test in both orientations, but stride runs opposite ways:

| | group widths across passes | narrow passes are | fuses a |
|---|---|---|---|
| **ours (DIF)** | shrink | late | **tail** |
| **MKL (DIT)** | grow | early | **head** |

So "tile the first k stages" describes their plan correctly and is a
*destructive* instruction for ours — our first passes are the widest, exactly
the ones that can never be tiled. **What transfers is the predicate, not the
index.**

They are not mirror images, in three ways:

1. **Which boundary folds in.** Our ingest can never fuse; the terminator can.
   Under DIT that inverts — the leaf folds in and the terminator cannot. The two
   designs delete traffic at opposite ends of the pipeline.
2. **Where the sacrificed pass lands.** Each orientation gives up exactly one
   full-width pass — ours first, theirs last, so one ends hot and the other ends
   with a full-array sweep immediately before the output write.
3. **The boundaries are not equal work.** Our ingest does the corner-turn and
   our terminator re-interleaves; their leaf does digit-reversal plus the split
   conversion. "You may fuse one boundary" is not the same prize on each side.

Reasoned in isolation, point 2 favours us — and MKL is the faster engine, so the
model is missing a term. That gap is the argument for **building both and racing
them**, once tiling itself is settled. Racing them untiled measures the wrong
thing: tiling is what makes orientation matter.

---

## Reproducing

```sh
# correctness — memcmp vs untiled, both directions, every legal width
python build_tuned/build.py --src build_tuned/benches/zturn_tcut_gate.c --vfft --compile
zturn_tcut_gate.exe --wisdir $SCRATCH/wisdir --no-dft

# the legal width space per cell + working sets — arithmetic only, no timing
python build_tuned/build.py --src build_tuned/benches/zturn_tile_census.c

# internal A/B (NOT a vs-MKL number — that only comes from bench_1d_vs_mkl.c)
python build_tuned/build.py --src build_tuned/benches/zturn_tcut_ab.c --vfft --compile
zturn_tcut_ab.exe --wisdir $SCRATCH/wisdir --cell 16384 --rounds 21 --cool 200
```

Reading the A/B output: group rows by the `ENGAGED AS` column, never by the arm
label — different labels engage identical plans, and those pairs are the only
honest same-plan noise floor. Compare only within one run.
