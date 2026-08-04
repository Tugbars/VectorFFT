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
> target-occupancy heuristic picks wrong at 8192 — which is why occupancy is a
> **diagnostic** and the measured time is the only **chooser**: the calibrator
> benches **every legal width** (an earlier occupancy filter was removed — an
> excluded width leaves no trace, so a wrong filter would be undetectable from
> its own output; see `docs/wisdom/10_zturn_calibration_flow.md` §3).

**N=2048 shows no win** (+3.3%, rising monotonically as tiles narrow — the
per-call cost on a plane already L1-resident). The calibrator benches every
legal width there too and banks **untiled by measurement** — the verdict is
earned per cell, never assumed.

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

### 7.1 The race ran — DIF-natural stays canonical (benchmark-derived, 2026-08-03)

The DIT-forward family was built as exact conjugates of the backward kinds
(`dts`/`dtsn`/`dtt`/`msd`, plus the store-side-permuted ingest `dtso`;
emitter machinery retained in `codelet_zsplit.ml`), gated conj-EXACT at the
kernel level and ~1e-14 against a naive DFT at the pipeline level, then
raced against the DIF-natural cascade (stfn terminator) under the paced
protocol (17 rounds, alternated order, control arm; raw runs in
`docs/research/dit_race_run{1,2}.txt`, `dit_race_v2_run1.txt`).

Measured on this host, warm:

- **DIT-natural / DIF-natural = 1.05–1.18 at every cell, three runs** —
  DIT loses everywhere; against *tiled* DIF-natural, 1.26–1.49.
- Stage decomposition: the mids are at parity (msd ≈ msg within noise);
  the loss sits in the boundary pair. The DIT finisher `dtt` *beats* the r8
  stfn terminator by ~19% in isolation, but the ingest gives it all back.
- **The scatter side is not the cause**: `dtso` (contiguous user reads,
  rho-scattered stores into the L2-resident plane) measured ≈ `dtsn`
  (D2/D = 0.92–1.06 across seven cells) — moving the permutation between
  the ingest's load side and store side changes nothing measurable.
- r8-tail chains (never picked by the DIF-calibrated bank) do not save it:
  DIT/DIF-natural ≈ 1.20 on 4.8.4.4.4.8 and 4.8.4.4.8.8.

Reading: the ZTURN-S plane geometry is co-designed with the DIF
orientation — corner-turn absorbed in the ingest's stores, terminator
reading contiguous section taps with no load shuffles. The conjugate-derived
DIT inherits that geometry mirrored to the wrong places (transpose in the
finisher, the w¹ workload in the ingest), and pays it regardless of which
side carries the permutation. A competitive DIT needs its own plane
geometry co-designed for the head-tiling orientation (what MKL builds) —
a new-geometry campaign, not additional wiring around these kernels.
The family stays in the tree at ≈1.0× kernel parity vs its donors, so other
microarchitectures can race it through the same machinery.

---

## 6.1 Through the front door — the closing state

Benchmark-derived, this host, 2026-08-02, warm, one run per cell/arm
(indicative until repeated on a cool machine; each vs-MKL ratio is same-run).
Canonical bench (`bench_1d_vs_mkl.c`), cell-per-process, core 2, widths arriving
**via wisdom replay only — zero environment variables**, exactly as a user's
`vfft_create` gets them. Ratio > 1 = vfft faster than MKL:

| N | pre-width baseline | same chains, tiling off | **tiled via banked wisdom** |
|---|---|---|---|
| 2048 | 1.05× | 1.15× | 1.07× (banked untiled) |
| 4096 | 0.80× | 0.84× | **0.95×** |
| 8192 | 0.78× | 0.81× | **1.01× — beats MKL** |
| 16384 | 0.88× | 0.82× | **1.03× — beats MKL** |

The front-door tiling deltas (−17…−21% against the same chains untiled)
reproduce the internal A/B. At campaign close the standing was
1.07/0.85/0.89/1.03; with banked tile widths it is **1.07/0.95/1.01/1.03**.

## 6.2 The in-place cell — same architecture, same standings

The cascade is alias-safe in-place by construction (ingest reads the user
buffer, the interior lives on an internal plane, the terminator writes the
user buffer — the same shadow-plane shape MKL uses, whose measured in-place
tax is ≈0). As of 2026-08-02 the front door serves it: `VFFT_INPLACE` +
interleaved + scrambled K=1 gets the tiled cascade with its banked width,
**byte-identical to the OOP output both directions** (gate: all five banked
cells memcmp-EXACT through `vfft_create`/`vfft_execute`; call form is
`(z, NULL, z, NULL)` — aliased destination).

Benchmark-derived, this host, 2026-08-02, warm, one run per cell,
cell-per-process, same-run ratios (canonical bench `--k1zip`: our cascade
in-place vs MKL `DFTI_INPLACE`, both interleaved; indicative until a
cool-machine session):

| N | vfft in-place | MKL in-place | ratio (>1 = vfft) |
|---|---|---|---|
| 2048 | 2194 ns | 2059 ns | 0.94 |
| 4096 | 3979 | 4050 | **1.02** |
| 8192 | 9793 | 8714 | 0.89 |
| 16384 | 17678 | 18940 | **1.07** |
| 32768 | 37503 | 37979 | **1.01** |

Because the in-place output is byte-identical to OOP, the work is identical —
so deltas between this table and §6.1 are thermal jitter, not an in-place tax.
Note 32768: its first valid measurement (an earlier routing defect silently
served the classic path there; fixed the same day), and a win.

⚠️ **Order asymmetry, stated plainly:** our arm delivers SCRAMBLED output (the
class's contract); MKL delivers NATURAL order — strictly more work, and MKL
offers no cheaper scrambled mode (`DFTI_BACKWARD_SCRAMBLED` is documented as
not implemented for most configurations). These ratios are honest for the
scrambled contract's consumers (pointwise/convolution pipelines, where a
matched-permutation inverse cancels the order for free) and must NOT be quoted
as natural-order parity: at +1%/+3% margins, paying for natural order would
likely turn the 8192/16384 wins into ties or narrow losses.

## 6.3 Natural order through the front door — the asymmetry resolved (2026-08-03)

The warning above is now answered by measurement instead of projection.
`order=NATURAL` for K=1 interleaved ≥2048 routes to the cascade with the
**load-permuted stfn terminator — natural output with NO reorder pass**
(`VFFT_NAT_ZCASC` in the natorder verdict; the create-time race measured the
cascade 4.7–7.1× faster than the tape incumbent at every cell, and the
consume path replays the banked verdict with zero race). The chain, route,
t2q, and tile width all replay from the same kind-4 line the scrambled path
uses — one calibration serves both orders.

🔴 **Vintage trap, recorded because it burned a table:** the first
natural-vs-natural run quoted 0.79–0.85 at 4096–16384 — those cells were
replaying a stale `oop_wisdom.txt` (no tile widths banked, 8192/16384 on
old chains ending in 8, the ~19 %-slower stfn form, and no 32768 kind-4
line at all). **Check the kind-4 vintage — chains AND width fields —
before quoting any front-door table.** All five cells were re-banked
through `calibrate_zchain` (711 benchmarks total; every winner ends in 4,
tiled w1024 @ 48 KB L1 at ≥4096), then both order modes re-ran
same-vintage, same-day.

**CITABLE (cool-machine session, 2026-08-04).** Benchmark-derived, this
host, calm machine, 3 runs/cell, cell-per-process, pinned core 2
(canonical bench `--k1zip` / `--k1nat`; both engines in-place
interleaved; per-cell spreads 1–3 %, cross-engine error bit-stable
across runs — this table supersedes the warm 2026-08-03 ranges):

| N | scrambled (`--k1zip`) | natural (`--k1nat`) | natorder tax | cross-engine err |
|---|---|---|---|---|
| 2048 | **1.15–1.18** | **1.09–1.16** | ~2 % | 6.9e-16 |
| 4096 | 1.02–1.04 | 0.96–0.99 | ~4 % | 7.3e-16 |
| 8192 | **1.05–1.06** | 1.00–1.03 | ~3 % | 6.2e-16 |
| 16384 | **1.05–1.08** | **1.02–1.03** | ~4 % | 6.6e-16 |
| 32768 | 1.00–1.02 | 0.94–0.97 | ~5 % | 7.2e-16 |

Readings:

- **In MKL's own best discipline — in-place, interleaved, NATURAL order —
  the cascade is at parity or ahead at every cell.** §6.2's projection
  ("wins become ties or narrow losses") landed almost exactly: 8192/16384
  went from +6–9 % scrambled to −2…+6 % natural.
- The measured natural tax (scrambled→natural delta, same run pair) is
  **2–6 %**, matching the B4 falsifier's kernel-level prediction of
  +2.5–5.7 % — where the incumbent reorder-pass design paid +13–27 %.
- The cross-engine column is an ELEMENTWISE compare against MKL's spectrum
  (~7e-16): both engines provably compute the same transform in the same
  order. The scrambled-vs-natural caveat above is dead for this mode —
  scrambled is now a 2–6 % *option*, not an asymmetry.
- 32768's middle natural run hit a thermal event that slowed BOTH engines
  equally (51/52 µs; ratio held) — the machine-noise protocol's control
  logic working as intended.

### 6.4 The sub-2048 in-place tier (il_coverage_plan.md Phases A+B, 2026-08-03)

Below the cascade tier, in-place interleaved K=1 used to convert to split
planes and run the proto engine (+ the reorder tape for natural; the il_me
A/B could even pick a PADDED K=8 plan — seven zero lanes computed for SIMD
width). Two changes, both raced through the same `@nat` verdict machinery:

- **Routing (Phase A):** an explicit-SCRAMBLED request below 2048 now
  reaches the native K=1 IL engines (identity permutation — the scrambled
  contract admits any self-consistent order). Gate: scrambled output
  memcmp-EXACT vs the natural handle at 0.95–1.05× its speed.
- **The ILP tier (Phase B):** `VFFT_NAT_ILP` — il2p/il3p attach in-place
  (aliased, alias-gated; two-stage engines, zout written only by the last
  stage), raced end-to-end at create vs the convert incumbent:
  **ILP won 9.1× / 7.2× / 5.7× / 4.0× at 128/256/512/1024.** Scrambled
  in-place rides the banked verdict hit-only (single wisdom writer).

vs MKL (both in-place, both natural, cross-engine elementwise ~2–4.5e-16).
**CITABLE (cool-machine session, 2026-08-04, with the blocked-mid race
live — supersedes the warm 2026-08-03 pre-promotion ranges):**

| N | vfft ns | MKL ns | ratio (>1 = vfft) |
|---|---|---|---|
| 128 | 74–75 | 68 | 0.91 (×3, tight) |
| 256 | 171–178 | 136 | 0.76–0.79 |
| 512 | 397–415 | 290–291 | 0.70–0.73 |
| 1024 | 1042–1107 | 825–833 | 0.74–0.80 |

(The pre-promotion warm table read 0.59–0.74 at 1024 and 0.68–0.71 at
512 — the blocked-mid promotion moved both cells and killed 1024's slow
tail; 128 is the unchanged control cell, now ×3 bit-tight.)

Read both numbers together: end users at these cells got ~4–9× faster than
the previous shipping path, and the remaining 0.6–0.9× stands on MKL's
strongest ground — small-N batched interleaved natural is where their
investment is deepest (mkl_blind_spot positioning). Closing that residue is
the K-across-SIMD question (il_coverage_plan Phase C3), not routing.

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

# §6.3 — refresh the kind-4 vintage FIRST (chains + widths; the burned-table rule),
# then the same-vintage order pair, cell-per-process, pinned core 2
python build_tuned/build.py --src benches/calibrate_zchain.c --vfft --compile
calibrate_zchain.exe <dir-of-oop_wisdom.txt> 1 2048 4096 8192 16384 32768
python build_tuned/build.py --src benches/bench_1d_vs_mkl.c --mkl --vfft --compile
bench_1d_vs_mkl.exe --k1zip <dir>/oop_wisdom.txt out_zip.csv 200 <N> 1 400 <flip> 2
bench_1d_vs_mkl.exe --k1nat <dir>/oop_wisdom.txt out_nat.csv 200 <N> 1 400 <flip> 2

# order=NATURAL front-door correctness (cold-start, scratch wisdom, both dirs
# vs naive DFT IN ORDER — roundtrip cannot gate ordering)
python build_tuned/build.py --src benches/vfft_natural_front_gate.c --vfft --compile
vfft_natural_front_gate.exe --wisdir $SCRATCH/natwis
```

Reading the A/B output: group rows by the `ENGAGED AS` column, never by the arm
label — different labels engage identical plans, and those pairs are the only
honest same-plan noise floor. Compare only within one run.
