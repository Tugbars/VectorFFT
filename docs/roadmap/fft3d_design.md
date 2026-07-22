# 3D FFT (c2c, then r2c) — design from the 2D machinery + FFTW/SPIRAL study

*Roadmap design doc, 2026-07-13. Produced from a read-through of `src/core/transforms/fft2d/`,
`src/core/engine/`, `src/core/vfft.c` (dims=2 dispatch), and the FFTW3 / SPIRAL literature
(Frigo & Johnson, Proc. IEEE 93(2) 2005 + "Implementing FFTs in Practice" 2008; Franchetti &
Püschel, "Fast Fourier Transform" encyclopedia chapter; SPIRAL Proc. IEEE 93(2) 2005).*

> **TL;DR** — 3D is three tensor factors, and the engine already implements all three shapes.
> For a row-major `N1×N2×N3` split-complex cube: **axis 0 is the existing native column pass at
> K=N2·N3**, **axis 2 is the existing tiled transpose row pass with row count N1·N2**, and the
> only new code is **axis 1: a plane loop of native K=N3 batched FFTs** (one `vfft_proto_execute_*`
> call per plane — the same ST execution path `_fft2d_tiled_range` already uses inside workers).
> ~90% of `fft3d.h` is `fft2d.h` with different parameters. The one genuinely new performance
> lever, taken from FFTW's buffered/vector-recursion machinery and validated by the SPIRAL
> multicore-FFT formula: **L2-blocked execution of the axis-0 pass** via the existing K-split
> slice helpers, which collapses `stages(N1)` DRAM sweeps into ~1 and directly attacks the known
> high-K TLB wall. Pass order and pass-A blocking become calibrated axes in a
> `fft3d_c2c_planner.h` that mirrors `fft2d_c2c_planner.h`.

---

## 1. What the 2D code gives us (read-through summary)

The 2D module is a complete template. What each piece contributes to 3D:

| file / mechanism | what it is | role in 3D |
|---|---|---|
| `fft2d.h : stride_fft2d_data_t` | N1,N2, plan_col (K=N2), plan_row (K=B), B, per-thread scratch pool, JIT-resolved exec fns, `nat_col_list` | template for `stride_fft3d_data_t` (adds plan_axis1, N3) |
| `fft2d.h : column pass` | axis-0 FFT run natively — lane-batched layout `re[i*K+lane]` makes axis 0 *be* the batch, K-split MT built in | **pass A verbatim**, K = N2·N3 |
| `fft2d.h : _fft2d_tiled_range/_mt` | tiled row pass: gather B rows via `stride_transpose_pair` → FFT(N2, K=B) in L1 scratch → scatter; tile-parallel, per-thread scratch, no barriers | **pass C verbatim**, row count N1·N2, row length N3 |
| `transpose.h` | cache-oblivious line-filling SIMD transpose (4×4/8×4 AVX2, 8×8 AVX-512), beats `mkl_domatcopy` | unchanged — pass C's substrate |
| `proto_stride_compat.h : vfft_proto_execute_fwd/bwd(plan,re,im,K)` | the ST slice execution path callable from worker threads (the row pass already does this) | **pass B's per-plane call** |
| `stride_executor.h : _stride_execute_*_slice(plan,re,im,slice_K,full_K)` | run all stages of a K-baked plan on a contiguous lane sub-slice (the K-split primitive) | **pass A L2-blocking** (§6) |
| `stride_plan_2d_from(...)` | wisdom-driven builder, caller-owned inner plans, avoids exhaustive-at-create | template for `stride_plan_3d_from(...)` |
| `fft2d_c2c_planner.h` | top-K seed per axis × cross-product, roundtrip-gate, end-to-end timing, natural-aware, order-neutralized interleaved sweep | template for `fft3d_c2c_planner.h` (adds pass-order + pass-A-mode axes) |
| `fft2d_c2c_wisdom.h` + `vfft.c _build_2d` | (N1,N2) cells, calibrate-on-miss, persisted `fft2d_c2c_wisdom.txt` | template for `fft3d_c2c_wisdom.txt`, (N1,N2,N3) cells |
| v1.0 safety note in `stride_plan_2d_wise` | wisdom-driven plan_col + K-split silently corrupts at intermediate T (1024², err ~1e6 at T=2/4) — cols forced non-wisdom | **inherit for pass A and pass-A blocking** until the K-split/variant bug is diagnosed |
| `fft2d_r2c.h` | tiled r2c row pass + column c2c at K=(N2/2+1), reverse-tile-order c2r | template for 3D r2c phase (§10) |

Known constraints that carry over unchanged: DIF plans run ST in the top-level executor (v1.1);
`FFT2D_MAX_THREADS`-bounded scratch pool; N-even constraint only for the r2c axis; output is
digit-scrambled per axis (roundtrip-definitive correctness, the dag convention).

---

## 2. The core identity — why 3D is nearly free here

SPIRAL's tensor formalism (Franchetti & Püschel, eq. 20–22 of the encyclopedia chapter) writes the
3D DFT as three commuting factors:

```
DFT_{N1×N2×N3} = (DFT_N1 ⊗ I_{N2·N3}) · (I_{N1} ⊗ DFT_N2 ⊗ I_{N3}) · (I_{N1·N2} ⊗ DFT_N3)
```

The formalism's two primitive shapes are `A ⊗ I_K` (vector parallelism: every scalar op of A
becomes a K-wide vector op on contiguous data) and `I_n ⊗ A` (block parallelism: n independent
contiguous copies of A). **VectorFFT's lane-batched engine is a hardware realization of
`DFT_N ⊗ I_K`** — element i of lane k at `data[i*K+k]`, every butterfly a full-SIMD contiguous
vector op. That is exactly why the engine wins 1D at high K, and it means the three 3D factors
map onto existing primitives with no new butterfly math:

| tensor factor | memory shape (row-major `re[(i·N2+j)·N3+k]`) | engine primitive |
|---|---|---|
| `DFT_N1 ⊗ I_{N2·N3}` (axis 0) | element i of pencil (j,k) at stride N2·N3 → lanes ARE contiguous | native plan, K = N2·N3 — the 2D column pass with a bigger K |
| `I_{N1} ⊗ (DFT_N2 ⊗ I_{N3})` (axis 1) | within plane i: stride N3 → native K=N3; planes offset by N2·N3 | one plan at K=N3, executed N1 times with base `re + i·N2·N3` |
| `I_{N1·N2} ⊗ DFT_N3` (axis 2) | contiguous length-N3 rows, N1·N2 of them | tiled row pass: per tile, `A⊗B = L·(B⊗A)·L` at tile granularity — the SIMD transposes *are* the stride permutations `L`, done blockwise so they stay cache-resident |
| commutativity | factors act on disjoint axes → all 3! = 6 pass orders are mathematically valid | pass order is a **calibration axis**, not a correctness constraint (§7) |

The 2D module is the N1×N2 restriction of exactly this table. Nothing in the 3D factorization
requires a codelet, twiddle, or executor change.

---

## 3. FFTW's rank-3 machinery — what it does, what transfers

FFTW represents a problem as `dft(N, V, I, O)` with I/O tensors: `N` = transform dimensions
(rank), `V` = vector loops (vrank), each dimension a `(n, istride, ostride)` triple. A 3D FFT is
rank-3/vrank-0, and the planner *searches* over reductions rather than hardcoding row-column:

| FFTW mechanism | what it does | VectorFFT translation |
|---|---|---|
| rank-geq2 splitting | rank-3 → {rank-1 + rank-2, rank-2 + rank-1}, recursively, trying axis groupings | our fixed 3-pass split is one point in that space; the *searchable residue* is pass order + pass-A mode → put both in the calibrator, wisdom-cache the verdict (same philosophy, offline instead of per-plan) |
| vrank loop extraction (§4.2.3) | peel one vector loop, recurse on the rest; loop order free | pass B *is* the extracted `I_{N1}` loop around a native-K problem |
| buffered plans | copy a few strided butterflies to a small contiguous buffer, compute, copy back | the tiled row pass is precisely a buffered plan with B=8 — already ours, already beats MKL |
| indirect plans (§4.2.4) | shuffle first (rank-0), then solve contiguously | Bailey. 2D measured verdict: tiled beats Bailey at every size 32²–1024² → don't resurrect full-volume rotations for 3D (§12) |
| **vector recursion** | push a strided vector loop *into* the CT decomposition toward the leaves, so large-stride passes never walk the whole array per stage | the important steal. Our analog: **lane-blocked pass A** — run *all stages* of the axis-0 plan on one L2-sized lane block before the next block (§6). The slice helpers make it a loop, not new machinery |
| in-place rank-0 transposes | cache-oblivious square transpose solvers | already have a better one (`transpose.h` beats `mkl_domatcopy`) |
| planner + wisdom | measure compositions, DP-memoize, persist | the `fft2d_c2c_planner` pattern is already the FFTW planner specialized to this decomposition — extend, don't redesign |

One FFTW detail worth copying into the calibrator design: FFTW observes plan choice flips with
**anisotropy** (256×256×32 wants a different composition than 32×256×256). The wisdom key must be
the ordered triple, and the bench matrix must include anisotropic shapes (§11).

---

## 4. SPIRAL's rank-3 machinery — what it does, what transfers

SPIRAL derives all of §2 by rewriting `DFT_k ⊗ DFT_m ⊗ DFT_n` with the Table-1 identities
(`A⊗B = L(B⊗A)L`, `A⊗B = (A⊗I)(I⊗B)`, stride-permutation factorizations), then maps the resulting
SPL formula to loops/SIMD/threads in a separate tag-driven stage. Relevant results:

| SPIRAL result | content | verdict for us |
|---|---|---|
| row-column derivation (eq. 22 → 3D) | §2's factorization is *the* canonical shared-memory form | adopted — it's what the 2D code already is |
| multicore FFT (eq. 17) | all inter-core exchanges at cache-block granularity, parallel compute blocks, no fine sharing | validates the existing barrier-free model: K-split slices, plane blocks, and row tiles are all ≥cache-line-granular; keep it |
| Korn-Lambiotte / long-vector form | maximize `A ⊗ I` vector shape throughout | the lane-batched engine is this form natively — the reason pass A/B need no transpose at all |
| four-step / six-step (eq. 15–16) | transpose-bracketed forms for machines where transposes are cheap relative to strided compute (vector machines, GPUs, distributed) | shared-memory verdict already measured in 2D: tiled > Bailey. Keep `use_bailey`-style plumbing for completeness, don't default it |
| **vector-radix (eq. 23)** | fuse butterflies *across* dimensions — one stage processes an r₁×r₂(×r₃) sub-block of all axes at once, fewer total passes | the one idea that is *not* reachable by composition: it needs new fused codelets. This is an OCaml dag-compiler research direction (emit `radixR1xR2` volumetric butterflies; e.g. fuse the last stage of pass B with the first stage of pass C inside the L1 tile). Explicitly out of scope for v1 — codelet-matrix explosion, unproven win — but it is the honest answer to "what would SPIRAL do that we can't by reuse" |
| vector recursion (eq. 19) | same locality trick as FFTW's, in formula form: replace element-stride permutations with vector-stride ones at the cost of an extra pass | same conclusion as §3 → lane-blocked pass A |

The repo's own `docs/roadmap/spiral_vs_dagfft_planner.md` conclusion holds here too: SPIRAL's
value is the *space*, not the search. For 3D the space collapses to a handful of measured axes
(§7), which the existing calibrate-on-miss pattern covers.

---

## 4b. SPIRAL's dedicated 3D paper — cubic block layouts (Akin, Franchetti & Hoe, ICASSP 2014)

*"FFTs with Near-Optimal Memory Access Through Block Data Layouts" (+ FCCM'12 2D predecessor,
JSPS 2016 journal version). Read in full 2026-07-13.*

**The method.** Machine model = two-level hierarchy where main memory is made of SB-sized blocks
(DRAM rows, 8 KB on their platform) and touching a *different* block costs `A_miss = A_hit + C`.
On their DDR2-800 board, element-strided access across rows delivered **1.16 GB/s vs 11.87 GB/s**
contiguous (of 12.8 peak) — a ~10× gap. Their fix: store the n³ cube as k×k×k sub-cubes, each
sub-cube one contiguous DRAM row ("cubic layout"), and restructure the three passes via rewrite
rules (their Table 2, rules 11–16) so that x-, y-, and z-pass all read/shuffle/compute/write
**whole cubes** — every DRAM transaction a full row-buffer hit, all strided permutation work
demoted to local memory. Measured (Altera DE4 FPGA, Spiral-generated hardware pipelines):
**up to 6.5× over the naive strawman** (same platform, standard layout, element-strided DRAM),
**83% of bandwidth-bound theoretical peak, 97.5% of realistic peak, 5.5× DRAM energy**. Note the
paper also *dismisses vector recursion* as needing "impractically large local storage for data
block sizes dictated by the main memory" — true when SB = 8 KB DRAM rows and local store is
tiny, **not true on a CPU** where the effective block is a cache line/page and caches are MB-scale;
blocked pass A (§6) is exactly the vector-recursion-family move made practical by that difference.

**Does it transfer? DRAM-transaction audit of the three passes** (the useful lens the paper
provides — apply their `A_miss = A_hit + C` model to our design on RPL):

| pass | DRAM access shape in our design | row-buffer / prefetcher behavior | their critique applies? |
|---|---|---|---|
| A flat | R concurrent streams, each a *contiguous K-vector leg* (K·16 B ≈ 1 MB at 256³), legs at stride s·K·16 | streams are sequential → row-buffer hits + prefetcher lock-on; the cost is stages(N1) full-volume sweeps + TLB spread, **not** per-element row misses | no — the lane-batched layout already makes every leg a block transfer |
| A blocked | same streams confined to a 16·N1·C window, all stages before moving on | ~1 volume sweep, window-local TLB | their thesis, achieved without a layout change |
| B | plane (16·N2·N3 B: 1 MB at 256³, 16 MB at 1024³) read contiguously once; stages run cache-resident (L2 ≤ 256³, L3 ≤ 1024³ on RPL) | one sequential stream per plane | no |
| C | tile gathers = B=8 concurrent sequential row streams; scatter same | row-buffer friendly; measured near-free in 2D | no |

**Verdict.** The property the paper buys with a *custom persistent layout* — "main memory only
ever sees contiguous block transfers" — our decomposition already has in plain row-major, because
(i) the lane-batched engine makes butterfly legs contiguous by construction, (ii) pass B is
accidentally plane-blocked, (iii) pass C is L1-tiled, and (iv) blocked pass A closes the one gap.
Crucially, our passes have **no inter-stage permutation passes at all** — the strided
`In⊗L / L^{n³}_{n²}` permutations their cubic layout exists to tame simply do not appear in the
three-pass in-place formulation. Adopting a cubic layout in a library with a row-major in-place
API would additionally cost two full re-layout sweeps (in→cubic, cubic→out) that cancel the
savings; the paper never pays this because on their platform the data *lives* in cubic layout for
the application's lifetime (they own the layout; a drop-in library does not). Residual candidates
worth a note, not a workstream: (a) a cubic-layout *internal* scratch for a future ≥512³
out-of-place path where we own the intermediate buffer anyway; (b) their per-transaction cost
model as the vocabulary for the pass-A flat-vs-blocked wisdom verdict. Also for the record: the
paper's headline is 6.5×-vs-naive on FPGA hardware — a strawman baseline a cached CPU never
resembles; no ~1.6× figure appears in it.

---

## 5. Proposed architecture — `fft3d.h`, three passes

Split-complex `re[(i·N2 + j)·N3 + k]`, `im[...]`, in-place, unnormalized
(`bwd(fwd(x)) = N1·N2·N3·x`).

```
FWD:  pass A: axis-0 — native plan_axis0 (N1-point, K = N2·N3)      [§6: blocked or flat]
      pass B: axis-1 — for i in 0..N1-1:
                 vfft_proto_execute_fwd(plan_axis1, re + i·N2·N3, im + i·N2·N3, N3)
              (plan_axis1: N2-point, K = N3; plane-parallel across threads)
      pass C: axis-2 — _fft3d_tiled_mt ≡ _fft2d_tiled_mt with rows = N1·N2, rowlen = N3
                 (plan_row: N3-point, K = B; gather/scatter via stride_transpose_pair)

BWD:  pass C' → pass B' → pass A'   (reverse order by convention; §7 notes order is free)
```

`stride_fft3d_data_t` = `stride_fft2d_data_t` + `{int N3; stride_plan_t *plan_axis1;
vfft_proto_exec_fn exec_ax1_fwd, exec_ax1_bwd;}`; `tile_sz = N3·B`; scratch pool unchanged
(T copies). `_fft3d_jit_resolve` extends `_fft2d_jit_resolve` to the third inner plan. The whole
thing wraps as an override plan via `_fft3d_wrap` exactly like `_fft2d_wrap`
(`plan->N = N1·N2·N3, K = 1, override_fwd/bwd/destroy`).

Per-pass notes:

- **Pass A** is the 2D column pass with K = N2·N3. Prime N1 falls through
  `vfft_proto_auto_plan_dispatch` to Rader/Bluestein exactly as in 2D (override plans dispatch
  fine — the column feed is a contiguous stride-K batch, native to both). Inherit the v1.0
  safety: **non-wisdom plan_axis0** (exhaustive → auto fallback) until the K-split + variant-code
  corruption bug (intermediate-T, 1024²) is diagnosed — pass A leans on K-split/slices harder
  than 2D ever did.
- **Pass B** builds *one* plan at (N2, K=N3) and reuses it across all N1 planes — same base-offset
  trick the tiled pass uses for its scratch. The call is the compat ST path, so it is legal from
  worker threads (this is exactly how `_fft2d_tiled_range` runs its row FFTs today). Prime N2 →
  Rader/Bluestein at K=N3, also fine.
- **Pass C** is `_fft2d_tiled_range` verbatim with `N1 → N1·N2` (the flattened `(N1·N2)×N3` view
  is exact: rows are uniformly strided, tiles never straddle anything meaningful). B stays
  `_fft2d_choose_tile(N3, N1·N2)` → 8. The `nat_col_list` hook carries over unchanged for the
  eventual axis-2 natural-order tape.
- **Arbitrary sizes** come free: every axis is the 1D engine + prime dispatch, so N1,N2,N3 are
  unconstrained for c2c (r2c will constrain N3 even, as 2D constrains N2).

---

## 6. Pass A at big cubes — the blocked variant (the one new perf idea)

The honest math first. Split-complex cube footprint = 16·N1·N2·N3 bytes:

| cube | footprint | where it lives (RPL: 32K L1 / 2M L2 / 36M L3) |
|---|---|---|
| 64³ | 4 MB | L3-resident |
| 128³ | 32 MB | ~L3, marginal |
| 256³ | 256 MB | DRAM every pass |
| 512³ | 2 GB | DRAM + TLB pain |

DRAM sweep count per pass, flat execution:

| pass | flat behavior | sweeps (volume read+write) |
|---|---|---|
| A (K = N2·N3) | each stage of plan_axis0 touches the whole cube | **stages(N1)** (2 for 256=16×16, 3 for 512) |
| B (plane loop) | plane = 16·N2·N3 bytes (1 MB at 256²) → all stages(N2) run L2-resident after first touch | **~1** |
| C (tiled) | tile L1-resident; one gather + one scatter | **~1** (+ transpose traffic, near-free per 2D measurements) |

Pass B and C are already optimal-shaped — the plane loop is accidentally a perfect L2-blocking,
and this asymmetry is worth internalizing: it's *why* pass order might matter at some sizes, and
why pass A is the only pass worth new engineering.

**Blocked pass A.** The axis-0 butterflies mix elements only along axis 0 at a fixed lane, so a
contiguous lane block `[c, c+C)` is a fully independent K=C sub-problem of the K-baked plan.
The K-split slice helpers already execute exactly this. Running them *sequentially per block,
all stages per block*:

```
for (c = 0; c < K; c += C)          /* C chosen so 16·N1·C ≤ L2-ish, e.g. C=4096 at N1=256 */
    _stride_execute_fwd_slice(plan_axis0, re + c, im + c, min(C, K-c), K);
```

collapses pass A from stages(N1) DRAM sweeps to ~1, and shrinks the per-butterfly page working
set from R pages at stride 16·N2·N3 spread across the cube to R pages within a 16·N1·C window —
i.e., it attacks the measured TLB wall (`docs/performance/high_k_real_fft_architecture_wall.md`)
at the algorithm level, complementary to the hugetlbfs/1G-page route. This is FFTW's vector
recursion and SPIRAL's eq. 19, realized with zero new executor code. MT is the natural superset
of K-split: distribute blocks across threads, each thread runs its blocks sequentially
(block count ≥ T; same 8-lane rounding as the existing K-split dispatch).

Two cautions, both inherited: (1) the slice path shares whatever the intermediate-T K-split
corruption bug is — same plan-class restriction as §5 until diagnosed; (2) DIF-oriented
plan_axis0 needs `_stride_execute_fwd_dif_slice` and stays ST-per-block for now (executor v1.1
note). Blocked-vs-flat is a **measured verdict per cell** in the calibrator, not a heuristic —
at 64³/128³ flat may well win (cube ~L3-resident, blocking is pure overhead), and the wisdom
should say so per size.

---

## 7. Calibrated axes (`fft3d_c2c_planner.h`)

Mirror `fft2d_c2c_planner.h` (top-K seed per axis at in-context batch → cross-product → roundtrip
gate → end-to-end best-of-trials timing → wisdom entry), with the axis set extended:

| axis | candidates | expectation |
|---|---|---|
| inner plan per axis | 1D planner top-K at (N1, K=N2·N3), (N2, K=N3), (N3, K=B) | as 2D — dominant axis |
| pass-A mode | flat vs blocked(C ∈ {L2-fit, 2×L2-fit}) | blocked wins ≥128³–256³, flat below |
| pass order (fwd) | A→B→C (default) vs C→B→A; full 3! only at PATIENT | matters mainly for anisotropic shapes; factors commute so all are correct |
| tile B | 8 default; sweep {4,8,16} only at PATIENT | 8, per 2D |

Cross-product size stays tractable: 2D uses top-K per axis with K≈3; 3 axes × 2 pass-A modes ×
2 orders ≈ 108 end-to-end timings per cell at PATIENT, fewer at MEASURE (top-1 seeds, default
order, both pass-A modes). Wisdom file `fft3d_c2c_wisdom.txt`, cell key = ordered (N1,N2,N3),
same auto-load/calibrate-on-miss/persist flow through a `_build_3d` in `vfft.c`.

---

## 8. Threading — three modes, still no barriers

| pass | mode | mechanism |
|---|---|---|
| A flat | executor K-split | built-in; K = N2·N3 ≫ 256·T always → K-split regime, never group-parallel |
| A blocked | block-parallel | blocks distributed across threads, each thread sequential-per-block (K-split superset) |
| B | plane-parallel | contiguous plane ranges per thread; each worker calls the ST compat executor per plane; no shared writes across planes |
| C | tile-parallel | `_fft2d_tiled_mt` verbatim, tiles = ⌈N1·N2/B⌉ (huge → good load balance, better than 2D's) |

Same pool (`threads.h`), same caller-runs-share pattern, same per-thread scratch. Pass boundaries
are the only synchronization (pool wait), as in 2D. The 2D MT result (dag 2.19–6.08× over MKL at
T8, largely because MKL's 2D threading self-sabotages at ≤512²) should *strengthen* in 3D: more
tiles, more planes, and MKL's 3D threading has the same structure it had in 2D.

---

## 9. API / integration

- `vfft.h`: `dims` gains 3; `int n[2]` → `int n[3]`. **Struct-size/ABI change** — acceptable
  pre-1.x but call it out in the header changelog; every `vfft_config_t` zero-init site is
  unaffected (calloc-style init already the documented pattern).
- `vfft_create` dims==3 branch → `_build_3d(transform, N1, N2, N3, rigor, reg, W, recalib, order)`
  mirroring `_build_2d`; persist `fft3d_c2c_wisdom.txt` alongside the 2D tables in the same
  wisdom dir.
- Gates: `order != DEFAULT` rejected for dims==3 in v1 (scrambled only — same
  no-silent-wrong-order contract; the 3-tape natural machinery is a clean follow-up since
  `nat_col_list` (axis 2, in-scratch) and the whole-row mechanism (axes 0/1) both generalize).
  `batch`/padding rejected for dims==3 as it is for dims==2 today.
- Raw builders exported like 2D: `stride_plan_3d(N1,N2,N3,reg)` (exhaustive inners — small N
  only, same warning), `stride_plan_3d_from(N1,N2,N3,B, plan_axis0, plan_axis1, plan_row)`
  (wisdom path, takes ownership), `stride_plan_3d_bailey` deliberately **not** provided (§12).

---

## 10. r2c / c2r 3D — phase 2

FFTW convention, reduce along the innermost axis: `N1·N2·N3` reals → `N1×N2×(N3/2+1)` complex
(N3 even). Direct lift of `fft2d_r2c.h`:

- **fwd:** phase 1 = tiled **r2c** row pass over the flattened N1·N2 rows (existing
  `_r2c_worker_fwd` tile machinery, row count swapped); phase 2 = axis-1 c2c per plane at
  K′ = K_pad(N3/2+1); phase 3 = axis-0 c2c at K = N2·K_pad — plus the column digit-reversal perm
  bookkeeping the 2D version already carries.
- **bwd:** axis-0, axis-1 IFFTs, then the tiled c2r row pass in **reverse tile order** (same
  longer-scatter in-place hazard as 2D).
- Expectation setting: 2D r2c runs 0.63–0.69× MKL for structural reasons (split-layout
  pack/recombine tax + K_pad + perm — the same fused-kernel workstream as 1D r2c). 3D r2c
  inherits it; two of three passes are c2c, so the blended number should land *better* than 2D
  r2c but still behind MKL until the fused real path exists. Ship c2c first; it's where the
  architecture wins.

---

## 11. Correctness protocol + benches

- **Definitive:** roundtrip `bwd(fwd(x)) == N1·N2·N3·x`, tol ≤ ~2e-14 at 256³ (2D precedent:
  ≤1.8e-14 at 512²), random + adversarial inputs.
- **Analytic spot checks (order-agnostic):** DC input → single nonzero total energy;
  Parseval `Σ|x|² · (N1N2N3) == Σ|X|²`; separable input `a(i)b(j)c(k)` → spectrum is the outer
  product of the three 1D spectra as *multisets* per axis (checks axis independence without
  needing natural order).
- **Cross-library elementwise:** vs MKL/FFTW after a per-axis unscramble in the *test harness
  only* (the three per-axis digit-reversal tapes from the inner plans' chains — the
  `vfft_natorder_2d_build_axis` machinery already builds these from a chain).
- **Benches:** `bench_fft3d_vs_mkl.c` (ST), `bench_fft3d_mt_vs_mkl.c` (T8), sizes
  32³–256³ (512³ wisdom-path only) **plus anisotropic** {256×256×32, 32×256×256, 256×32×256,
  1024×64×64} — anisotropy is what exercises pass order and pass-A mode; isotropic-only
  benching would leave those calibration axes untested.
- MKL harness gotcha to pre-empt (2D lesson): size the MKL 3D r2c CCE buffer at the full
  complex footprint, not the reduced one — the 2D undersizing segfault cost a day of
  false-positive "port bug".

---

## 12. Dead ends — decided now, not re-derived later

| idea | why not |
|---|---|
| Bailey-style 3D (volume rotations between passes) | full-array transposes touch the cube twice per axis; 2D measured tiled > Bailey at *every* size 32²–1024², and 3D volumes are strictly worse for full rotations. Distributed-memory codes need it (no shared address space); we don't |
| per-plane `stride_plan_2d` reuse for passes B+C | N1× scratch-pool blowup, per-plane plan overhead, and its internal MT collides with plane-level MT. The flattened pass C is strictly better (N1·N2 tiles in one pool) |
| vector-radix (fused multi-axis codelets) | real but expensive: new codelet families from the OCaml compiler, matrix explosion (r₁×r₂ per ISA per direction), unproven win vs the L1-tiled composition. Park as a dag-compiler research item: fuse pass-B-last-stage × pass-C-first-stage inside the tile |
| wisdom-driven plan_axis0 in v1 | the known intermediate-T K-split corruption (2D v1.0 note). Same ~3–5% per-stage tuning cost accepted, same diagnosis unblocks both dims |
| heuristic pass-A blocking threshold | measured verdict per wisdom cell, like every other verdict in this codebase |

---

## 13. Implementation checklist (phased)

1. **`fft3d.h` core** — ✅ **landed + validated 2026-07-13** (`src/core/transforms/fft3d/fft3d.h`).
   `stride_fft3d_data_t`, `_fft3d_tiled_range/_mt` (rows=N1·N2), plane-loop pass B
   (+ plane-parallel MT), lane-range pass A, `_fft3d_wrap`, `stride_plan_3d[_from]`.
   Validation (`test_fft3d_roundtrip.c`, gcc -O2 -mavx2 -mfma, avx2 codelets + registry):
   49/49 cells — {32³, 16×32×64, 64×32×16, 60×20×12, 61×16×16, 16×61×16, 16×16×61, 8³}
   × {FLAT, BLOCKED(64)} × T∈{1,2,4} + exhaustive-builder smoke. Roundtrip ≤1.6e-12 rel,
   Parseval ≤1.2e-14, DC exactly 1 bin; prime-61 on every axis position exercised the
   Rader/Bluestein override on all three pass paths. FLAT/BLOCKED and all T produced
   identical error values (consistent with bit-exact outputs across modes/threads).
2. **Blocked pass A** — ✅ landed inside step 1 (unified: FLAT = one block per lane range).
   Remaining: 128³/256³ A/B *timing* on the target host (this container is not the 14900KF),
   and exercising the DIF-forward gate with a wisdom-emitted DIF inner (auto plans here
   were DIT; the override gate is proven, the DIF branch is not yet).
3. **Planner + wisdom** — DP-planned inners ✅ **proven end-to-end 2026-07-13** via
   `bench_fft3d_vs_mkl.c` (per axis: `vfft_proto_dp_init(K,N)` → `vfft_proto_dp_plan_measure`
   → `vfft_proto_plan_create_ex`, prime/non-smooth falling back to `auto_plan_dispatch`).
   Container smoke results (unknown cloud CPU, AVX2 codelets, MKL via pip `libmkl_rt`, ST
   pinned, cycles best-of-7 — **indicative only, not the 14900KF**):

   | size | roundtrip | sorted-\|X\| vs MKL | flat (cyc) | blocked (cyc / lanes) | MKL (cyc) | best vs MKL |
   |---|---|---|---|---|---|---|
   | 32³ | 9.8e-15 | 6.0e-15 | 482,835 | — (heur flat) | 1,363,959 | **2.88×** |
   | 48³ | 1.0e-14 | 5.2e-15 | 2,134,289 | — (heur flat) | 7,592,407 | **3.56×** |
   | 64³ | 1.4e-14 | 7.5e-15 | 5,625,998 | — (heur flat) | 12,045,066 | **2.14×** |
   | 128³ | 1.2e-14 | 6.7e-15 | 77,327,071 | **72,372,031 / 512** | 162,944,188 | **2.25×** |
   | 256×64×32 | 1.3e-14 | 7.6e-15 | **12,004,281** | 12,921,988 / 256 | 28,846,624 | **2.40×** |
   | 32×64×256 | 1.3e-14 | 7.5e-15 | 13,229,780 | **12,000,451 / 2048** | 34,431,796 | **2.87×** |

   Observed: (a) sorted-magnitude multiset agrees with MKL to ~7e-15 everywhere — the scrambled
   spectrum is elementwise MKL's spectrum as a multiset, so the transform is validated *against
   MKL*, not just self-consistent; (b) blocked pass A wins at the two K=16384 cells (+6.4% at
   128³, +10.2% at 32×64×256) and loses slightly at K=2048/N1=256 — exactly the "measured
   verdict per cell" prediction; (c) DP chose DIF single-stage rows twice ([32], [64]) so the
   DIF path in pass C is now exercised live, and DIT ax0 chains kept the lane-split legal;
   (d) DP ax0 winners flipped between runs at 128³ ([8×4×4] vs [4×4×2×4]) — container noise
   flipping near-ties, the known DP-noise sensitivity; wisdom caching is the remedy.
   Remaining for phase 3 proper: `fft3d_c2c_planner.h` (end-to-end 3D cross-product, not
   per-axis DP alone), `fft3d_c2c_wisdom.h`, `_build_3d` in `vfft.c`, `n[3]` API change.
4. **Benches + docs** — the §11 matrix vs MKL/FFTW on the 14900KF; write `docs/fft3d_port.md`
   in the `fft2d_2d_port.md` format with the measured tables.
5. **r2c/c2r 3D** — §10, after c2c numbers are banked.
6. **Later:** 3-tape natural order; the K-split corruption diagnosis (unblocks wisdom-driven
   pass A in both 2D and 3D); vector-radix study in the OCaml compiler.
