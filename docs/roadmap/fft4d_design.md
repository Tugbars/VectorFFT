# 4D FFT design — the rank-d pass taxonomy, FFTW's rank-split fusion, and what it back-ports to 3D

*Roadmap design doc, 2026-07-13. Follows `fft3d_design.md`; grounded in the FFTW rank-geq2
solver source (fftw-3.3.x `dft/rank-geq2.c` / `rdft/rank-geq2-rdft2.c`, read in full), the
Franchetti–Püschel tensor formalism, and Akin–Franchetti–Hoe's hypercube-layout remark.*

> **TL;DR** — Adding a fourth axis is mechanically trivial (the 3D passes generalize by
> formula), but 4D is where two structural facts bite that 3D let us ignore: **(1) middle
> axes lose their accidental cache residency** (pass B's sub-volume is a whole cube, 32 MB
> at 128⁴ — the L2-blocking that pass B got for free in 3D must become an explicit shared
> primitive), and **(2) the pass-fusion axis opens up**: FFTW's rank-geq2 solver splits the
> dims into two contiguous groups at a *searched split point* and runs the trailing group as
> a loop over the leading indices — which, executed per-block while the block is
> cache-resident, collapses multiple whole-volume DRAM sweeps into one. At 64⁴ the fully
> fused form runs the entire 4D transform in **~2 volume sweeps instead of ~4**. The same
> move back-ports to 3D (fuse axis-1 + tiled rows per plane: ~3 sweeps → ~2), refining the
> `fft3d_design.md §12` dead-end (which rejected naive per-plane *plan-object* reuse, not
> fused per-block *execution*). The right implementation is rank-general (`fftnd`), with the
> split point `s` replacing the d! pass-order search as the calibrated knob.

---

## 1. The rank-d pass taxonomy — nothing new mathematically

Row-major `N1×…×Nd` split-complex, `re[((…(i1·N2+i2)…)·Nd)+id]`. For axis m (0-indexed):
outer count `O_m = ∏_{i<m} N_i`, length `N_m`, inner lanes `K_m = ∏_{i>m} N_i`. The engine
shape per axis is fully determined by the layout:

| axis | tensor factor | O_m | K_m | engine primitive |
|---|---|---|---|---|
| 0 | `DFT_N1 ⊗ I_{K_0}` | 1 | N2·N3·N4 | one native call (blocked or flat) — 3D pass A verbatim |
| 1 | `I_{N1} ⊗ DFT_N2 ⊗ I_{K_1}` | N1 | N3·N4 | loop of native calls per cube — 3D pass B with plane→cube |
| 2 | `I_{N1N2} ⊗ DFT_N3 ⊗ I_{N4}` | N1·N2 | N4 | loop of native calls per plane — 3D pass B shape again |
| 3 | `I_{N1N2N3} ⊗ DFT_N4` | N1·N2·N3 | — | tiled row pass over O_3 flattened rows — 3D pass C verbatim |

General rule: **axis 0 is one native call; the last axis is the tiled row pass; every middle
axis is the same loop-of-native-K shape with different (O, K)**. All factors commute. The
3D code already contains all three shapes; a literal `fft4d.h` would be a copy of `fft3d.h`
plus one more middle pass — which is precisely the signal to stop copying (§7).

Practical 4D shapes (the workloads that exist): lattice-QCD momentum space (32³×64 = 32 MB
split, 48³×96 ≈ 170 MB, 64³×128 ≈ 536 MB — anisotropic long time axis is the norm),
space-time turbulence spectra (kx,ky,kz,ω), k-t MRI, light fields. `plan->N` int guard:
128⁴ = 268 M fits; anything larger exceeds RAM anyway.

---

## 2. What FFTW's rank-geq2 solver actually does (primary source)

From the source (structure identical in `dft/rank-geq2.c` and `rdft/rank-geq2-rdft2.c`):

- `picksplit` → `X(tensor_split)(p->sz, &sz1, spltrnk, &sz2)` — the size tensor is split
  into two **contiguous groups** at split rank `spltrnk`. Registered split candidates are
  the "buddies" `{1, 0, -2}` (after the first dim, an alternative first pick, and
  second-from-last), each a separate solver instance the planner races — with the in-source
  `FIXME: Should we try more buddies?` admitting the set is a pragmatic truncation.
- Child 1 (`cldr`): solves the **trailing** group `sz2`, with vector dims = original vector
  ∪ `sz1` — i.e. *the leading dims become vector loops around the trailing-group transform*.
- Child 2 (`cldc`): solves the leading group `sz1` with vector = vecsz ∪ `sz2` — the
  converse, which in the lane-batched engine is exactly the native pass at K = ∏(trailing).
- Locality heuristic in `applicable()`: *"if the vector stride is greater than the transform
  size, don't use (prefer to do the vector loop first with a vrank-geq1 plan)"*.

Mapping: our unfused d-pass architecture is the fully-expanded leaf of FFTW's recursion
(every split taken down to rank-1). The thing we have **not** taken from it is the
intermediate stop: keeping the trailing group as a *unit executed per leading index* —
which is where the memory win lives (§4). FFTW searches `spltrnk`; we should calibrate it.

SPIRAL at rank 4 adds nothing structurally new: `DFT_{N1×N2×N3×N4} = DFT_{N1} ⊗ … ⊗ DFT_{N4}`
with the same Table-1 identities; the Akin–Franchetti–Hoe block-layout paper explicitly
notes the framework "can easily be extended to higher dimensional FFTs using higher
dimensional hypercube data layouts" — the same verdict as 3D §4b applies (we get the
block-transfer property in row-major for free; persistent hypercube layout costs re-layout
sweeps a drop-in library can't hide). 4D vector-radix stays parked with 3D's.

---

## 3. Where 4D actually differs — middle-axis cache residency dies

Per-iteration working set of axis m's pass = `16·N_m·K_m` bytes (the sub-volume spanning
axes m..d-1). In 3D this was: pass A = whole cube (blocked), pass B = one plane (1 MB at
256³ — accidentally L2-resident), pass C = one tile (L1). In 4D:

| pass | working set | 64⁴ | 128⁴ | 32³×64 (t last) | consequence |
|---|---|---|---|---|---|
| A (axis 0) | whole volume | 256 MB | 4 GB | 32 MB | blocked, as in 3D |
| B (axis 1) | one **cube** 16·N2N3N4 | 4 MB (L3 ok) | **32 MB (≥L3)** | 1 MB | needs lane-blocking too |
| C (axis 2) | one plane 16·N3N4 | 64 KB (L2) | 256 KB (L2) | 32 KB | fine |
| D (axis 3) | one tile 16·N4·B | 8 KB | 16 KB | 8 KB | fine |

So the 3D `a_block` field generalizes: **every pass gets an optional lane-block**, and the
block loop becomes a shared helper (`_fftnd_exec_lanes(plan, base, K, block, is_bwd)`) used
by pass A at O=1 and by any middle pass whose sub-volume outgrows cache. Same DIT/no-override
gate as 3D pass A. This is bookkeeping, not new machinery — the K-split slice primitive
already proved out.

---

## 4. The headline — fused trailing-group execution (and its 3D back-port)

The unfused architecture pays one full DRAM read+write sweep per pass at DRAM-resident
sizes. FFTW's rank-split, executed per-block while the block is cache-resident, merges the
trailing passes' sweeps:

**Fused group from split point s**: for each of the `O_s = ∏_{i<s} N_i` leading-index
blocks (a contiguous `∏_{i≥s} N_i`-element sub-volume): load once (implicitly, by
touching), run axis-s native (lane-blocked if needed), axis-(s+1) native per sub-plane, …,
tiled last axis — all while the block sits in L2/L3 — write back once. One sweep for
(d − s) passes.

Approximate volume-sweep counts (blocked-A assumed; transposes ride along):

| configuration | 4D sweeps | 3D sweeps | fused-block size | parallel grain |
|---|---|---|---|---|
| unfused (current 3D style) | ~4 | ~3 | — | lanes/planes/tiles per pass |
| s=2: fuse C+D per (i,j)-plane | ~3 | s=2 ≡ current 3D | 16·N3·N4 (64 KB at 64⁴) | N1·N2 blocks |
| s=1: fuse B+C+D per i-cube | **~2** | fuse B+C per plane → **~2** | 16·N2N3N4 (4 MB at 64⁴) | N1 blocks |

Two things to internalize. First, **the 3D back-port is the higher-priority item**: fusing
3D's axis-1 pass with the tiled row pass per i-plane (plane stays in L2 across both) removes
a whole volume sweep — a larger lever than blocked-A's measured +6–10%, on already-shipping
code. Second, this *refines, not contradicts,* `fft3d_design.md §12`: what was rejected
there was reusing `stride_plan_2d` **objects** per plane (scratch ×N1, MT collision). The
fused form shares one plan set + the existing per-thread tile scratch, block-parallel across
planes/cubes — none of the rejected costs apply.

Constraints that make s a **measured, per-shape verdict** rather than "always fuse max":
(a) the fused block must fit the target cache tier (s=1 at 128⁴ → 32 MB block ≥ L3 → the
fusion buys little; drop to s=2); (b) parallel grain `O_s` must cover T (s=1 with N1=4,
T=8 starves — either hierarchical block×lane splitting or s=2); (c) axis-s inside the fused
group may itself need lane-blocking (the §3 helper composes). This is FFTW's buddy search
with a physical prior: candidates s ∈ {1, …, d−1}, scored end-to-end by the calibrator —
a far better-structured knob than the d! pass-order sweep, which it subsumes and replaces.

---

## 5. Threading

Unfused passes keep the 3D modes. The fused loop is **block-parallel** over the `O_s`
leading indices, each worker running the whole trailing group on its blocks with its own
tile scratch — the 3D plane-parallel pattern with a bigger body. The one new wrinkle:
small `O_s` (anisotropic long-first-axis shapes, or s=1 with modest N1) needs a
**hierarchical split** — distribute (block × lane-range-within-axis-s) work items instead
of blocks alone; the flattened work index is `block_idx · n_lane_ranges + range_idx`,
barrier-free as ever since blocks are disjoint and lane ranges within a block are disjoint.

---

## 6. `fftnd` vs `fft4d` — the refactor decision

Writing `fft4d.h` as a third near-copy is the wrong move; the taxonomy is closed-form. The
proposal: **`fftnd.h`** with

```
typedef struct {
    int rank;                        /* 2..VFFT_ND_MAX_RANK (4 for now) */
    int N[VFFT_ND_MAX_RANK];
    int split;                       /* s: axes < s unfused, axes >= s fused per block */
    struct { stride_plan_t *plan; size_t lane_block; } axis[VFFT_ND_MAX_RANK];
    size_t B;                        /* last-axis tile height */
    /* per-thread tile scratch, JIT exec fns, nat hooks — as fft3d */
} stride_fftnd_data_t;
```

execution = unfused blocked-native passes for axes < s, then the fused per-block loop
(axis s native/lane-blocked → … → tiled last axis). `fft2d` is (rank=2, s=1);
current-`fft3d` is (rank=3, s=2 behaviorally ≡ unfused-with-tiled-last... strictly s=2
fuses nothing beyond the tiled pass; the new 3D win is s=1). Migration policy per the
codebase culture: **fftnd lands beside fft2d/fft3d, has to *beat or match* them cell-by-cell
before anything is retired** — the 2D/3D specializations stay the shipping paths until the
generic one wins on the bench, not on aesthetics.

API: this is the moment to stop growing `n[2]→n[3]→n[4]`. One final `vfft_config_t` change:
`int rank; int n[VFFT_ND_MAX_RANK];` (FFTW's `(rank, dims)` shape), `dims` kept as a
deprecated alias for one release. Wisdom cell key = (rank, N[0..rank-1], s, blocks).

---

## 7. r2c 4D

Reduce along the last axis (N4 even): the fused trailing group naturally hosts the tiled
r2c row pass per block; leading axes run c2c at K′ = pad(N4/2+1)-scaled inner products, with
the same column digit-reversal bookkeeping as 2D/3D r2c. Inherits the split-layout real-FFT
tax; sequenced after 4D c2c numbers exist, same as 3D.

---

## 8. Dead ends & carried verdicts

| idea | verdict |
|---|---|
| full 4D rotations (six-step between passes) | dead — 2D/3D verdicts compound; fusion attacks the same traffic without rotations |
| persistent hypercube (tesseract) data layout | dead for the drop-in API — 3D §4b argument, one dimension worse |
| per-plane/per-cube reuse of `stride_plan_2d/3d` *objects* | dead as objects; **alive as fused execution** sharing plans + scratch (§4) |
| d! pass-order calibration | replaced by split-point s ∈ {1..d−1} — subsumes the orders that matter |
| 4D vector-radix codelets | parked with 3D's, OCaml-compiler research item |

---

## 9. Phased checklist

1. **3D back-port** — ✅ **mechanism landed + measured 2026-07-13** via `fftnd.h` rank=3,
   s=1 (no separate fft3d flag needed). Container A/B, identical auto inners, all variants
   **bit-EXACT** vs `fft3d.h`:

   | 128³ (32 MB) | cyc | vs fft3d | 192³ (108 MB) | cyc | vs fft3d |
   |---|---|---|---|---|---|
   | fft3d flat (ref) | 132.1M | 1.000× | fft3d flat (ref) | 363.8M | 1.000× |
   | fftnd s=2 flat | 131.1M | 1.007× | fftnd s=2 flat | 365.4M | 0.996× |
   | fftnd s=1 fused | 124.6M | **1.059×** | fftnd s=1 fused | 349.3M | 1.041× |
   | s=2 + blkA 512 | 127.8M | 1.033× | s=2 + blkA 512 | 337.0M | 1.080× |
   | s=1 + blkA 512 | 127.7M | 1.034× | **s=1 + blkA 512** | **324.6M** | **1.121×** |

   Verdicts: generic machinery is free (s=2 ≡ fft3d within noise); fusion and blocked-A
   **stack at DRAM-resident sizes** (+12.1% at 192³) but not at semi-L3-resident 128³ on
   this (large-L3 cloud) host — per-cell wisdom verdict, as designed. 256³/512³ on the
   14900KF is the remaining decisive run.
2. **`fftnd.h` core** — ✅ landed (`src/core/transforms/fftnd/fftnd.h`): taxonomy executor,
   shared `_fftnd_exec_lanes`-style per-axis lane blocking, fused trailing group,
   hierarchical (outer × lane) MT. Correctness (`test_fftnd_roundtrip.c`): rank-3
   **bit-match vs fft3d** at s∈{1,2} incl. prime middle axis; rank-4 split equivalence
   (s=1/2/3 memcmp-identical); 100+ cells of roundtrip/Parseval/DC over 9 shapes × s ×
   T∈{1,2,4}, primes at every axis position, forced lane-blocked fused middles, O_s<T
   stress, rank-2 sanity. ALL PASS.
3. **Bench** — ✅ `bench_fftnd.c` (fuse3d + mkl4d modes), container results (AVX2 codelets,
   pip MKL, ST pinned — indicative only): 16⁴ **3.58×** over MKL DFTI rank-4 (s=1 best);
   QCD-shape 32³×64 (32 MB): s=3 2.85×, s=2 3.22×, **s=1 3.41×** — monotonic in fusion
   depth, s=1 is **19.6% faster than unfused**; rt ≤ 6.8e-12, sorted-|X| vs MKL ≤ 8.4e-15.
   48³×96 (170 MB) and MT columns pending the 14900KF.
4. **Calibrator/wisdom** — ✅ **landed + validated 2026-07-13** (`fftnd_planner.h` /
   `fftnd_wisdom.h`): per-axis DP recipes, {s × lane-block} structural sweep,
   roundtrip-gated end-to-end timing, one-line-per-cell text wisdom, calibrate-on-miss
   append, warm rebuild **bit-EXACT** vs cold (0.26–36 ms vs 9–35 s cold). Container
   verdicts: 64³ → s=1 (+8.4%); 32³×64 → s=2 by 1.8% under DP inners (s=1 under auto
   inners) — the s-verdict is per-cell AND per-inner. MT fully wired 2026-07-14: starved-grain
   fused mode (sequential blocks × parallel-within, per-pass joins) + T-keyed wisdom;
   `test_fftnd_mt.c` bit-EXACT across T∈{1,2,4,8} in all modes. Remaining: `_build_nd`
   in `vfft.c` + rank-ified `n[]` config; MT scaling on the 14900KF.
5. **r2c 4D** — ✅ landed 2026-07-14 (`fftnd_r2c.h`, rank 2–4; 7/7 incl. per-bin
   external validation at 2.6–5.5 eps; fused-group hosting of the r2c rows per §7's
   sketch stays a follow-up — v1 runs the row pass unfused). Natural order — ✅ maps
   (`fftnd_natorder.h`). The fftnd-vs-fft2d/fft3d retirement decision — the parity
   + fusion data above says fftnd is the future path; retirement still gated on the
   14900KF cell-by-cell sweep per the codebase rule.
