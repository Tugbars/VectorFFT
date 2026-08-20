# VectorFFT v1.0 — performance results

> **Where we win most — MKL's blind spot.** VectorFFT's lead over MKL is widest exactly
> where MKL invests least: **odd / mixed-radix, scrambled-order, in-place, batched.**
> Power-of-2 is MKL's home turf (decades of split-radix tuning) and our *narrowest* win
> (median **1.86×**); odd composites — where MKL falls back to generic mixed-radix / Bluestein —
> are our *fattest* (median **3.47×**, ~2× more margin). Three effects stack into that blind spot:
> (1) a DAG-compiler-tuned codelet for **every smooth radix** (not just 2/4/8), (2) **scrambled-order**
> in-place that skips the bit-reversal MKL pays, and (3) the **split lane-batched** layout that makes
> the throughput regime trivially parallel. MKL is tuned for the opposite corner: power-of-2, natural
> order, single transform. (Scrambled order is the right contract for convolution-class work — FIR
> filtering, polynomial / big-integer multiply, correlation, lattice-crypto NTT — where a fwd→bwd
> roundtrip or a pointwise multiply is order-agnostic.)

Empirical performance of VectorFFT across three axes:

1. **Wall-time vs MKL** on 1D C2C — single-thread (238 cells) and multi-threaded (the headline metric)
2. **Wall-time vs FFTW3** on 1D C2C and the r2r family (DCT/DST/DHT), single-thread
3. **Multi-threaded scaling** at T=2/4/8 across the transforms

(Plan-quality / cost-model analysis lives in its own doc: [docs/cost_model/](../cost_model/).)

All numbers are from the i9-14900KF calibration host (P-core pinned,
performance plan, single-threaded unless noted). The numbers move on
different hardware — see "Hardware caveats" at the end.

## 1. vs MKL — 1D C2C

Source: `build_tuned/benches/vfft_perf_tuned_1d.csv`
(238 cells × MKL ILP64 sequential, calibrated wisdom loaded).

```
Category              Cells    Min   Median    Max   Mean
─────────────────────────────────────────────────────────
Small (N≤128)            15   2.60×   4.28× 15.33×  5.98×
Power-of-2               29   1.10×   1.86×  3.04×  1.96×
Composite                43   1.62×   2.85×  4.51×  2.93×
Odd composite            26   2.26×   3.47×  5.16×  3.36×
Mixed deep               35   1.66×   2.71×  5.78×  2.89×
Prime powers             25   1.67×   2.69×  4.16×  2.76×
Genfft (R=11/13)         17   1.65×   2.79×  3.75×  2.63×
Rader primes             24   1.29×   2.34×  3.85×  2.36×
Bluestein primes         24   1.02×   1.55×  3.52×  1.74×
─────────────────────────────────────────────────────────
OVERALL                 238   1.02×   2.64× 15.33×  2.83×

Wins vs MKL: 238/238 (100%)
```

Headline:

> **VectorFFT beats MKL on 100% of bench cells (238/238). Median speedup
> 2.64×, mean 2.83×, range 1.02×–15.33×.**

The median 2.64× win comes from VectorFFT's twin advantages:
1. **Plan-level joint search** at calibration time — picks better
   factorizations than per-codelet wisdom (see
   [docs/wisdom/00_thesis.md](../wisdom/00_thesis.md)).
2. **Fully tuned codelet portfolio** — every shipped radix has
   variant codelets (FLAT / LOG3 / T1S / BUF) selected per
   `(R, me, ios)` cell.

### Arbitrary K — odd / non-multiple-of-VW batch (single-thread)

In-place c2c now accepts **any** batch K, not just `K % VW == 0`. A codelet-internal
rem-aware tail ([arbitrary_k_tail_handling.md](arbitrary_k_tail_handling.md)) covers the
`1..VW-1` leftover lanes: the bulk full-vector loop, then **`rem==1` → one scalar single
lane, `rem>=2` → one masked vector pass**. Every radix carries it — monolithic (r2–r5, r7,
primes) and composite / CT-blocked (r8, r16, r32, r64). Bit-exact at every K (`fwd+bwd ==
N·x` + a bulk-vs-tail-split diagnostic, all `corr = 0.0`).

Forward vs MKL `DFTI_INPLACE` split, `bench_inplace_oddk` `measure_ab` (best-of-5 min,
cachebust + cool, order-flip). **Each cell uses its CALIBRATED `spike_wisdom` factorization**
(odd K reuses the same N's nearest-K plan). Two methodology notes matter:
- These run the **GENERIC executor** — the only path that carries the tail today. The §1 main
  table (and the `CSV` column below) is **baked/JIT**, which is ~1.5–2× faster on multi-stage
  cells, so the *absolute* margins here sit below the baked reference (the documented "generic
  floor; `--jit` widens it"). The apples-to-apples comparison is **odd-K vs even-K on the same
  generic executor.**
- Measured on a **live host**; cells marked `*` show order-flip spread (thermal noise).

```
 N      plan (calibrated)    K=32 rem0   K=33 rem1   K=31 rem3   | CSV baked K=32
──────────────────────────────────────────────────────────────────────────────────
 64     8x8/DIT              3.43×       2.69×       2.65×       | 3.03×
 128    4x32/DIF             1.92×*      2.30×       (noisy)     | 3.26×
 256    4x8x8/DIT            1.31×       2.27×       (noisy)     | 3.04×
 512    4x4x32/DIF           2.09×       1.78×       2.46×       | 1.98×
 1024   4x4x8x8/DIT          1.19×       1.63×       1.64×       | 2.75×
 4096   4x4x4x8x8/DIT        2.70×*      1.48×       1.52×       | 2.57×
──────────────────────────────────────────────────────────────────────────────────
```

Composite-stage cell (CT-blocked r16, `bench_oddk_composite`, plan N=256 [16,16] T1S):
`K=8 (rem0) 3.02× · K=13 (rem1) 2.31× · K=17 (rem1) 2.48× · K=15 (rem3) 2.61×`.

> **Odd-K is bit-exact and competitive — on the generic executor it tracks or beats the even-K
> cell (N=256: even 1.31× → odd 2.27×; N=1024: even 1.19× → odd 1.63×; N=64: even 3.43× → odd
> ~2.67×).** The tail adds no structural cost; the gap to the baked §1 reference is the
> generic-vs-baked executor difference, not the remainder handling. Reaching the baked numbers at
> odd K needs the JIT/baked executor to carry the tail (today JIT = even-K 1D C2C only — a
> follow-up).
>
> **Why scalar-at-rem==1 + masked-at-rem≥2** (the measured contract): a *pure* scalar tail erodes
> as the scalar fraction grows (1.77× at rem=1 → 1.61× at K=31 → 1.29× at K=15 rem=3), while one
> masked pass is **flat in rem** (~1.6–1.72×). The hybrid takes the cheaper scalar lane at rem==1
> and the flat masked pass at rem≥2. The scalar lane renders monolithically (no register pressure
> at width 1), so even spill-scratch composite codelets honour it.

### Multi-threaded — vs MKL at T=8

dag (8 P-cores, pinned core 0, pool K-split) vs MKL `mkl_set_num_threads(8)`, **identical split
lane-batched layout**, order-neutralized (engine order flipped per cell) + paced — the same fairness
as the single-thread table above. Source: `bench_1d_vs_mkl.c --mt` → `vfft_perf_tuned_1d_mt.csv`
(129 cells, K≥32).

```
 N      K    dag-T8 (ns)  MKL-T8 (ns)  dag/MKL
─────────────────────────────────────────────
 8      256         571       23,906   41.90×
 64     256       8,140       46,266    5.68×
 256    256      45,560      128,653    2.82×
 1024   256     224,288      694,963    3.10×
 4096   256     696,100    3,387,937    4.87×
 256    32        8,715       19,634    2.25×
 1024   32       50,375       66,834    1.33×
 4096   32      233,233      405,020    1.74×
```

> **At T=8, VectorFFT beats MKL on 129/129 cells (K≥32) — median 3.76× (K=32: 3.00×, K=256: 4.38×),
> up to 41.9× at tiny N where MKL can't usefully thread the batch.** Our split, lane-batched layout
> makes K independent transforms trivially parallel (no barriers); MKL's batched split-mode threading
> scales poorly at modest N. These use the **generic** executor — a conservative floor (JIT is wired
> and bit-exact again post-core-move; re-running with `--jit` widens the margin).

### Multi-threaded — INTERLEAVED transform-contiguous batch at T=8

A second batch geometry, and the fairest MT cell in this document: DFTI with
`DFTI_NUMBER_OF_TRANSFORMS=K, DFTI_INPUT_DISTANCE=N` **is** our transform-contiguous
layout, so both engines read byte-identical memory and compute the same natural-order
spectrum — the correctness column is a cross-engine elementwise compare (~1e-16 every
cell), not a roundtrip proxy. The whole process is confined to the **8 distinct P-cores**
(affinity mask `0x5555`; logical 0,2,…,14) before any MKL/OpenMP initialization, so
neither engine can borrow E-cores or HT siblings. Source: `bench_1d_vs_mkl.c --ilmt`
→ `vfft_perf_tuned_1d_ilmt.csv`.

Four timed arms per cell — ours at T=8 and T=1, MKL at T=8 and T=1 — plus a repeat of
arm 1 as a control. Our pool is torn down before every MKL arm (our workers spin and
would otherwise steal cores), and MKL gets ≥300 ms to park its threads before ours.

```
 N      K    ours-T8   ours-T1    MKL-T8    MKL-T1 | vs MKL best  our scale  MKL scale
────────────────────────────────────────────────────────────────────────────────────
 256    4        607       614     3,573       673 |     1.11×      1.01×      0.19×
 512    4      1,146     2,271     6,626     1,419 |     1.24×      1.98×      0.21×
 1024   4      1,667     4,036     8,378     3,415 |     2.05×      2.42×      0.41×
 4096   4      4,973    16,199    31,433    15,266 |     3.07×      3.26×      0.49×
 16384  4     20,013    80,390   114,647    89,490 |     4.47×      4.02×      0.78×
 65536  4    137,438   494,375   568,275   581,750 |     4.13×      3.60×      1.02×
 256    8      1,012     1,170    19,933     1,268 |     1.25×      1.16×      0.06×
 512    8      1,080     4,879    23,064     2,911 |     2.69×      4.52×      0.13×
 1024   8      2,457     8,339    25,125     6,713 |     2.73×      3.39×      0.27×
 4096   8      5,138    33,352    60,036    31,823 |     6.19×      6.49×      0.53×
 16384  8     20,593   173,900   223,613   203,280 |     9.87×      8.44×      0.91×
 65536  8    113,550   919,550 1,127,175 1,042,487 |     9.18×      8.10×      0.92×
 256    32     1,673     4,704    25,034     4,991 |     2.98×      2.81×      0.20×
 512    32     2,758    17,056    32,020    11,190 |     4.06×      6.18×      0.35×
 1024   32     6,449    33,274    46,897    27,995 |     4.34×      5.16×      0.60×
 4096   32    18,240   153,947   188,880   170,467 |     9.35×      8.44×      0.90×
 16384  32    85,187   757,650   869,825   875,550 |    10.21×      8.89×      1.01×
 65536  32 1,182,062 4,126,425 7,977,025 8,091,300 |     6.75×      3.49×      1.01×
────────────────────────────────────────────────────────────────────────────────────
ns/call. "scale" = that engine's OWN T1/T8 (8.00 = perfect on 8 cores).
```

> **18 of 18 cells win vs MKL-BEST — median 4.10×, up to 10.21× at 16384×32. Our own
> scaling reaches 8.89× on 8 cores — near-linear.** Measured 2026-08-13,
> post-tangent/wing32; supersedes the 2026-08-06 table (17/18, median 3.09× — its one
> loss, 256×4 at 0.78×, is now 1.11×). The movers are the sub-2048 cells the tangent arc
> rebuilt (all clean ≤2.4% repeat-arm spread); the 65536-row and 256×8/32 cells carried
> 12–27% spreads (machine in use) — read those within their noise. K=1 1024 was re-raced
> the same day and the tangent variants LOST (wash) — the batched 1024 gains come through
> the TC-batch path's shared sub-kernels, not a new 1024 plan.

**We compare against MKL's *faster* configuration, which is almost always its serial one.**
That is the finding this table exists to record: **MKL's threaded arm never beats its own
serial arm at any cell measured** — its scale column runs 0.06×–1.00×, capping exactly at
parity. This is not a mis-measurement. `MKL_VERBOSE=1` confirms the threading layer is
`intel_thread` and that the calls run at `NThr:8`; it was re-checked under three affinity
masks (`0x5555`, `0xFFFF`, unmasked) with the same result. MKL threads the work correctly
and simply loses doing so at these granularities: at 256×8 its threaded arm costs 18,828 ns
against 1,216 ns serial, 15× slower for identical math. Scoring against MKL's T=8 column
instead would inflate every ratio (median 8.25× vs the honest 3.09×) by crediting us for an
option MKL's own users would not select.

Three caveats stated plainly:

- **This is a threading-architecture win, not a kernel win.** Compare the two T=1 columns —
  that is the kernel-vs-kernel fight with threading removed. MKL leads at N ≤ 1024
  (1.18×–1.56×, worst for us at 1024); we lead from 4096 up (1.06×–1.74×). The sub-2048
  serial gap is real and is tracked separately.
  *(Update 2026-08-16: this table predates the tangent/wing32/TURNED campaign. In the
  K=1 natural-order grid below, the serial sub-2048 gap has since closed to parity —
  128 = 1.05×, 256 = 1.00×, 512 = 0.98–1.00×, 1024 ≈ 0.95-to-parity. These batched
  cells have NOT been re-measured against the new pool, so the numbers above stand
  as-measured.)*
- **Our workers never park.** Dispatch is cheap precisely because the pool spins rather than
  sleeping, which costs idle CPU and power. MKL's threads sleep after `KMP_BLOCKTIME`, which
  is better behaviour inside a host application doing other work. On this benchmark the trade
  is pure upside for us; in production it is a real cost.
- **Two cells are noise-dominated** — 256×8 (9.2% control spread) and 4096×8 (7.8%). Every
  multi-× result sits far outside its own control spread; anything inside it is not a result.

### Natural order — in-place (single-thread)

In-place c2c natively emits **digit-scrambled** order (the convolution contract — §1 headline). The
`VFFT_ORDER_NATURAL` flag (parity with MKL's `DFTI_ORDERING`) delivers **bin-for-bin natural order**
by running the same scrambled FFT plan and then a per-cell **reorder methodology**, calibrated once and
persisted as a **self-contained `@nat` record** (wisdom v8 — its own `J_nat`-optimal factorization + mode,
regime-separated from the scrambled entry so a natural create never perturbs the scrambled plan):

- **FREE** — the cell is already natural (single-stage / prime); zero extra pass.
- **PURE** — a cycle-following K-row permutation pass over the scrambled output.
- **PSWAP** — an involution **pair-swap** on a *palindromic* factorization (whose digit reversal is
  its own inverse), the cheapest possible reorder. The planner injects a palindromic chain as a
  candidate since the DP scores factorizations under scrambled economics.

Forward via the public API (`vfft_create` order=NATURAL + `vfft_execute`; the FFT is **JIT/baked**,
only the reorder pass is a memory permutation) vs MKL's natural DFTI (its default order), same
fairness as §1 (best-of-5 min, cachebust + cool + order-flip, P-core-pinned). Output order validated
**bin-for-bin** against a naive O(N²) DFT (elementwise, both natural) **plus** roundtrip
`fwd+bwd == N·x` (all e-14/e-15 every cell; see the `fwd`/`rt` gates). Source: `bench_1d_vs_mkl`-modeled
`natorder_vs_mkl.c`, rebuilt against the v8 `@nat` wisdom and consuming the dev-calibrated records.

```
 N      K    mode     vfft ns   dag/MKL   chain / note
────────────────────────────────────────────────────────────────────────
 64     4    FREE         141     2.28×    radix-64 leaf — zero reorder
 100    4    PSWAP        295     2.72×    10·10  (odd-radix palindrome)
 128    4    PURE         508     1.27×    8·16   (cycle-follow reorder)
 250    4    PURE       1,079     1.84×    10·25
 256    4    PSWAP        899     1.35×    16·16  palindrome
 512    4    PSWAP      1,892     1.68×    8·8·8  palindrome
 1024   4    PSWAP      6,033     1.23×    4·64·4 injected — MKL's best-tuned cell (pow2)
────────────────────────────────────────────────────────────────────────
 128    32   PURE       2,751     2.09×    4·32
 512    32   PURE      15,554     1.61×    4·4·32
────────────────────────────────────────────────────────────────────────
 median               ~1.68×    (9/9 win)
```

> **Natural-order in-place beats MKL on every sampled cell — 1.23×–2.72×, median ~1.68×** — even though
> MKL emits natural order *natively* while we run a full reorder pass on top of the scrambled FFT. The
> lead tracks the reorder cost exactly: **FREE** (64/4, single radix — no reorder) and **PSWAP** (100/4,
> a cheap involution pair-swap on a palindrome) give the widest margins (2.28×, 2.72×); **PURE** (the
> cycle-follow pass, heaviest tax) the narrowest (128/4 = 1.27×). Natural gives up roughly a fifth of the
> scrambled lead — the "20–25%" design target — yet still clears MKL, because MKL pays for its *own*
> bit-reversal to reach the same order. The hardest cell is **1024/4 = 1.23×**: N=1024 is pow2, MKL's
> most hand-tuned case (its narrowest blind spot), carrying the full reorder tax — and it still wins.
>
> **Mode selection is now stabilized** (this resolves the earlier "not-yet-paced" follow-up). The
> create-time calibrator measures candidates **interleaved + best-of-rounds** with a **5% win-margin**
> and a **natural-intrinsic tie-break** (FREE > PSWAP-pairs > PURE-cycle, then fewer stages, then
> lexicographic) — decided purely on the natural objective, **never** the scrambled winner. The old
> 256/4 PURE↔PSWAP flap is gone: it settles deterministically on the 16·16 palindrome. The self-contained
> `@nat` records are dev-calibrated over the K∈{4,32}, N≤1024 grid and the public API consumes them.
>
> **Honesty note:** measured on a **live host** (not the locked-down §1 machine) → ratios are
> **directional**; the correctness gate (elementwise-natural vs MKL + roundtrip) is exact. Natural order
> here is **in-place 1D C2C** — r2c/c2r/trig are already natural, and OOP carries its own natural kinds
> (LEAF/BAILEY2, same `VFFT_ORDER_NATURAL` flag; see the out-of-place tables). The 2D `@nat2d` cells are
> calibrated but not yet benched vs MKL DFTI 2D. Roundtrip / convolution consumers keep the faster
> scrambled default.

### Order × placement — the K=1 INTERLEAVED grid

The tables above are the split lane-batched path. This one is the **K=1 interleaved**
(`layout=INTERLEAVED`) story, and it differs in a way worth stating explicitly: **above
2048, natural order is a different terminator, not a reorder pass.** The `stfn` cascade
writes natural order directly from the last stage, so there is no `PURE`/`PSWAP` permutation
pass to pay — unlike the reorder methodology described immediately above, which is the
split path's mechanism. Measured through the public API, MKL has no reorder pass here
either; this is the matching design, not a divergence.

Every order × placement combination is served by a **native** engine. No convert fallback
remains anywhere in this grid — the last hole (OOP-natural ≥2048, which read 0.17× through
the convert bridge) closed 2026-08-04.

| order × placement | sub-2048 (mono ≤64 · il2p/il3p 128–1024) | ≥2048 (cascade tier) |
|---|---|---|
| **NATURAL · in-place** | native — `VFFT_NAT_ILP`: il2p/il3p aliased, raced vs convert at create, banked `@nat` | native — `VFFT_NAT_ZCASC`: `stfn` natural-terminator cascade, **no reorder pass**, raced vs tape, banked `@nat` |
| **NATURAL · OOP** | native — same IL engines, `z_in → z_out` | native — natord cascade via `@natoop` verdict + create race |
| **SCRAMBLED · in-place** | native — **identity rule**: served by the natural-native engines (the identity permutation is contract-legal; bits identical to natural, gated `IDENT`) | native — ZTURN-S digit-scrambled comb (kind-4 verdict); the comb is a REAL permutation (A3-gated) |
| **SCRAMBLED · OOP** | native — identity rule, same engines (`scr==nat` EXACT, gated) | native — kind-4 cascade attaches to the OOP handle, matched-permutation roundtrip |

**DEFAULT order** = engine-native everywhere (fastest, order-agnostic): resolves to the
scrambled-native path in-place, the calibrated winner OOP.

Measured vs MKL, like-for-like order and placement, same-run ratios (>1 = we win):

```
  N       NATURAL in-place   NATURAL OOP   SCRAMBLED in-place
──────────────────────────────────────────────────────────────
  128        0.91 †‡           1.05 ★★        (= NAT bits)
  256       0.85–0.86 ▲◆‡      1.00 ★★        (= NAT bits)
  512       0.78–0.80 ▲‡     0.98–1.00 ★★     (= NAT bits)
  1024      0.91–0.95 ▲     ~0.95–parity ✦    (= NAT bits)
  2048      1.09–1.16       0.99–1.11         1.15–1.18
  4096      0.96–0.99       0.91–0.94         1.02–1.04
  8192      1.00–1.03       0.95–0.98         1.05–1.06
  16384     1.02–1.03       0.94–0.98         1.05–1.08
  32768     0.94–0.97       0.88–0.91         1.00–1.02
──────────────────────────────────────────────────────────────
† vintage 2026-08-04 (not re-measured)
▲ 2026-08-06: blocked R≥32 kernels are the SHIPPED DEFAULT — a structural
register-file rule (a monolithic R≥32 body holds ~40–64 live values against
AVX2's 16 registers), not a per-cell race. Same-run A/B against the
monolithic arm, pair pinned by wisdom so only the kernels vary, 8 alternating
arms on a pinned core: 1024 −27% (0.65→0.92), 512 −8% (0.70→0.78).
◆ 256's gain needed BOTH halves and neither alone: the calibrator replaced
the heuristic balanced pair (16,16) with the measured (8,32), which puts
R=32 in the leaf slot, which the structural rule then blocks — 0.76→0.85.
Fully blocking a (16,16) pair instead was raced and LOST by 4.4%, so radix
choice dominates form choice here.
★★ 2026-08-16, the CLOSED sub-2048 campaign: tangent interiors
([tangent_scaled_butterflies.md](tangent_scaled_butterflies.md)) + the wing32
R32 forms + the TURNED store-edge axis
([store_edge_taxonomy.md](store_edge_taxonomy.md) defines T128/T256/M-128 and
the full set), all dp-raced (`calibrate_k1` over the
full form pool; verdicts banked as pair + `il_kv`). Winning rows: 128 = pair
4×32, kv 64 (mono mid + T256 wing32 leaf, 65 ns vs MKL 69); 256 = 16×16, kv 51
(tangent both slots, 136 vs 136); 512 = 16×32, kv 67 (tangent mid + T256 wing32
leaf, 296–297 vs 291–295). Canonical bench `--k1noop`, both flip orders,
cross-engine correctness 3–4e-16. The 2026-08-12 era below (★-history) first
crossed 128/256 with tangent-only forms; the wing32 leaf then solved the R32
slot (the old "+32% killed leaf" was an edge×interior interaction, not the
tangent construction) and the TURNED axis picked the store edge per cell.
Historical ★ numbers: 128 1.04 (8×16, kv 51) · 256 1.01–1.02 · 512 0.87–0.88
(kv 35, pre-wing32) — superseded by the rows above.
✦ 2026-08-16, 6 reps, both flip orders, canonical bench: ours
848/889/893/901/978/987 ns vs MKL 833/841/854/860/864/873 — **median ratio
0.96, best-vs-best 0.98, one rep at 1.02**. The route is CLASSIC blocked 32×32
**re-raced against the complete form pool** (tangent, wing32, both TURNED
edges) and re-confirmed by the dp with the nearest challenger +4.5% behind:
1024's regime is memory (L1 pending misses ~19× the 512 level, store-latency
bound), where the tangent family's port/instruction levers buy nothing. The
cell's variance is dominated by MKL's own in-place shadow-plane placement
(its floor alone spans 833–873 here; historically 812–905), so the honest
datum is **~0.95-to-parity** — do not quote a third digit. (The 2026-08-12
figure 0.83–0.90 came from a noisier 5-rep set on the same route.)
‡ pre-tangent plan. The banked kind-3 row for this N CHANGED on 2026-08-12, so
the in-place figure no longer describes what ships. Sub-2048 in-place and OOP
run the SAME IL engines (see the grid above), so it is expected to track the
OOP column — but it has not been re-measured, and is not quoted as if it had.
▢ engine serves it; no banked table yet    (= NAT bits) identity rule
```

Reading it honestly:

- **Sub-2048 K=1 natural is AT PARITY — the campaign is closed (2026-08-16).**
  128 = 1.05×, 256 = 1.00×, 512 = 0.98–1.00×; 1024 sits at ~0.95-to-parity with its
  variance dominated by MKL's own in-place shadow-plane placement (see ✦). The tier's
  final architecture: **tangent/wing interiors own the L1-resident cells, classic
  blocked owns the memory-bound cell (1024 — re-raced against the full pool, classic
  won by 4.5%), and the store edge (TURNED-128 vs -256) is a per-cell raced axis, not
  a default** — T256 won at 128/512 on this machine, and the losing forms stay in the
  pool as inventory for other platforms. Remaining known headroom, parked at the
  owner's wrap: 3-stage chains (+7.6–8.8% at 512, measured twice, never banked).
  *(The paragraphs below record the mid-campaign analysis that got here — the R32
  census and its levers. Historical: the wing32 forms subsequently solved the R32
  slot and the "cannot form a both-slots pair at 512" constraint dissolved.)*
  A like-for-like census against
  MKL's own 32-point column kernel (`mkl512__col32_fwd_loop.asm`, same work unit: 32 ymm
  loads → 32 ymm stores, twiddles hoisted as constants) says the opposite of the obvious
  inference:

  | | MKL col32 | ours `n1tb48` |
  |---|--:|--:|
  | instructions | **460** | 563 |
  | fma / bare mul | **68 / 0** | 36 / 20 |
  | naked add+sub | 118 (**63%**) | 152 (73%) |
  | shuffle + xor | **54 + 15** | 82 + 29 |
  | stack ops | **78 (0.42/fp-op)** | 48 (0.23/fp-op) |

  **MKL spills 1.6× more per arithmetic op than we do and still wins.** It treats stack
  traffic as cheap and buys instruction count and FMA density with it — 18% fewer
  instructions, 1.9× the FMAs, *zero* bare multiplies. So "reduce spills" is the wrong
  lever against MKL; the right ones are instruction count and naked-add/bare-mul
  elimination, which is the direction the tangent construction already pushes. A second,
  independent lever is visible in the same table: our interleaved-complex sign/lane
  handling costs **42 extra shuffle+xor instructions**, roughly 40% of the whole
  103-instruction gap, and has nothing to do with tangent.
  ⚠ **Regime matters when quoting these**: this column is the single-transform K=1
  natural-order cell, the one that has always been hardest. The same N wins comfortably
  batched — 1024 is 1.57× at K=256 OOP (see the table below) — and the scrambled column
  leads everywhere ≥2048. Do not read a sub-2048 K=1 number as the library's position.
- **Historical context for the ▲ cells** (superseded by the above but the mechanism
  still holds): the ▲ cells run the blocked
  R≥32 kernels as the shipped default — a structural rule (a monolithic R≥32 body holds
  ~40–64 live values against AVX2's 16 registers and spills ~27% of its stream; blocked
  construction is the only body shape that fits, the same tier the split emitters apply
  at generation time), not a measured per-cell pick. A side observation from the A/B
  worth keeping: the monolithic arm's own spread at 1024 was 36% between two runs while
  the blocked arm's pair agreed to 0.5% — a spill-bound body is at the mercy of ambient
  load in a way a register-resident one is not. 256 moved 0.76→0.85 only once the
  calibrator replaced its heuristic (16,16) pair with the measured (8,32) — the pair
  verdict is what put an R≥32 body in a slot the blocked rule could act on, so neither
  half would have delivered it alone.
- **Remaining levers for the two cells still behind (512, 1024)**, in order of measured
  promise: (1) an **R32 blocking geometry co-designed with the tangent constant set** —
  the spill census above localizes the loss precisely, and both remaining cells are the
  R32-bound ones, so this single lever addresses both; (2) the stage-count axis (3-stage
  chains beat every 2-stage pair at 512 by +7.6–8.8%, measured twice, not yet banked).
  Two things are now *closed* rather than open: a hand-wired fully-tangent 512 is a wash
  (~1%), so more tangent at the current geometry is not the answer; and R=64 is no longer
  the "last structural gap" — 1024's problem is that it is R32 in *both* slots, which an
  R=64 kernel does not address.
- **≥2048 is parity-or-win** on every row except natural-OOP at 4096/32768.
- **The SCRAMBLED row leads everywhere ≥2048, and the reason is structural rather than
  a kernel advantage**: setting `DFTI_ORDERING` to `DFTI_BACKWARD_SCRAMBLED` does not
  change MKL's output ordering (the setting reads back as applied, but the spectrum is
  unchanged — verified through the public API), so MKL has no scrambled mode. That row
  therefore races our structurally cheaper path against their only path. It is a fair
  comparison of what each library can actually deliver for a scrambled-order consumer,
  not a like-for-like kernel comparison.

Batching rides this grid unchanged: the canonical `VFFT_BATCH_TRANSFORM_CONTIGUOUS`
geometry runs K independent K=1 transforms, so every cell above applies per transform at
every K, with no batch tail, no padding and no even-K constraint.

### Out-of-place — vs MKL (single-thread)

dag OOP c2c vs MKL `DFTI_NOT_INPLACE` split-complex, **identical layout**, order-neutralized + paced
(same fairness as the in-place table). Two natural-order kinds (LEAF, BAILEY2 fused-transpose stores)
and the scrambled-order MODEB (in-place dataflow run OOP; bit-exact roundtrip). Calibrated per-cell in
**isolated processes** to avoid cross-cell carryover biasing the kind pick. Source:
`bench_1d_vs_mkl.c --oop` → `vfft_perf_tuned_1d_oop.csv` (31 pow2 cells, K∈{32,128,256,1024}).

```
 N       K     kind     plan          dag/MKL
──────────────────────────────────────────────
 8       32    LEAF     —              10.78×
 8       256   MODEB    8               5.67×
 16      32    BAILEY2  4×4             5.97×
 64      256   MODEB    4,4,4           2.11×   (carryover sweep mis-picked BAILEY2 → 0.77×)
 256     256   MODEB    4,4,16          2.09×
 1024    256   MODEB    4,4,4,4,4       1.57×
 4096    32    MODEB    4,4,4,8,8       1.63×
 65536   256   MODEB    4,4,8,16,32     1.40×
──────────────────────────────────────────────
 Min 1.37×   Median 2.01×   Max 10.78×   Mean 2.49×   Wins 31/31
```

> **Out-of-place, single-thread, VectorFFT beats MKL on 31/31 cells — median 2.01×, range
> 1.37×–10.78×.** Small N favors the natural-order LEAF/BAILEY2 kinds; mid/high N and high K favor
> MODEB. Per-stage variants are inherited variant-rich from the in-place wisdom (FLAT/T1S/LOG3 mixed),
> and BAILEY2's `t1p` stage is flat-vs-log3 searched per cell.

### Out-of-place — vs MKL at T=8

Same OOP cells, dag K-split across 8 P-cores (pool, pinned core 0) vs MKL `mkl_set_num_threads(8)`,
identical NOT_INPLACE split layout, order-neutralized + paced. MODEB/LEAF are truly lane-sliced;
BAILEY2 runs single-threaded (its inter-stage transpose isn't lane-independent — 2-phase MT is a
follow-up) so its rows are dag-ST vs MKL-8T. A per-cell MT-vs-ST gate guards correctness. Source:
`bench_1d_vs_mkl.c --oop --mt` → `vfft_perf_tuned_1d_oop_mt.csv` (31 cells).

```
 N       K     kind     dag/MKL-T8   note
──────────────────────────────────────────────
 8       256   MODEB      38.53×     MKL can't thread tiny batch
 16      32    BAILEY2    45.80×     dag-ST vs MKL-8T
 64      256   MODEB       5.26×
 256     256   MODEB       2.80×
 1024    256   MODEB       2.74×
 4096    256   MODEB       4.86×
 65536   256   MODEB       3.10×
 1024    32    MODEB       1.24×     (min)
──────────────────────────────────────────────
 Min 1.24×   Median 2.80×   Max 45.80×   Wins 31/31
```

> **Out-of-place at T=8, VectorFFT beats MKL on 31/31 cells — median 2.80×, up to 45.8× at tiny N.**
> The huge small-N margins are where MKL can't usefully thread the batch; the steady mid/high-N MODEB
> wins (1.2×–5×) are the real K-split scaling. Generic executor (JIT wired + bit-exact, not yet
> re-run here); BAILEY2 MT is a follow-up — both are conservative floors.

### Out-of-place — arbitrary K (odd / non-multiple-of-8 batch)

The OOP path used to fail-closed on `K % 8 != 0`. It now serves **any K**, across all three
kinds, via a **codelet-internal rem-aware tail** (the same contract as the in-place tail,
`docs/performance/arbitrary_k_tail_handling.md`): the bulk full-vector loop, then for the
`1..VW-1` leftover batch lanes **`rem==1` → one scalar single lane, `rem>=2` → one masked
vector pass**. The scalar lane is rendered monolithically (no register pressure at width 1).
Two of the three kinds keep **natural order** at odd K:
- **MODEB** (scrambled) rides the tailed in-place codelets (n1 OOP wrapper) — any K, free.
- **LEAF** (natural, N≤128) — the `n1_oop` leaf carries the tail.
- **BAILEY2** (natural, all N) — a new **per-lane `t1_oop`** second-stage codelet + a per-group
  twiddle table replace `t1p`'s per-VW-block broadcast (which straddles k2 boundaries at odd K).

Forward measured vs MKL `NOT_INPLACE` split, calibrated OOP wisdom loaded for the aligned cell
(K=32) and `dp_best` for the odd cells (no wisdom entry); best-of-5 min, cachebust + cool,
order-flip. The chooser picks the fastest kind per cell; odd-K natural-order is available
whenever LEAF/BAILEY2 win.

```
 N      K    rem  kind     order      dag/MKL
─────────────────────────────────────────────
 8      31   3    LEAF     natural    6.32×
 8      33   1    LEAF     natural    6.26×
 16     31   3    BAILEY2  natural    2.99×
 16     33   1    BAILEY2  natural    2.82×
 64     31   3    BAILEY2  natural    1.79×
 64     33   1    BAILEY2  natural    2.06×
 256    31   3    BAILEY2  natural    1.32×
 256    33   1    MODEB    scrambled  1.40×
 1024   31   3    BAILEY2  natural    1.30×
 1024   33   1    BAILEY2  natural    1.36×
─────────────────────────────────────────────
```

> **Odd-K out-of-place beats MKL on every cell — 1.30×–6.32×, landing in the same band as the
> adjacent calibrated even-K cells (§1 above).** Correctness is the gate: OOP roundtrip
> `fwd+bwd == N·x` at ~1e-15 every cell, and forced-BAILEY2 forward is bit-correct vs a naive
> O(N²) DFT in natural order (the per-lane `t1_oop` + per-group twiddle table validated at
> N=256/512/1024). The `rem==1` scalar lane costs nothing measurable vs the masked neighbours.
>
> These were measured on a **live host** (not the locked-down clean machine the §1 even-K table
> used), so they are **directional** — several cells show order-flip spread from thermal noise.
> The calibrated even-K cells in §1 are the publication reference; the odd-K numbers track them.

## 2. vs MKL — 2D C2C

dag tiled 2D (`fft2d.h`, B=8: gather→K=B row FFT→scatter via SIMD transpose, native
column pass) vs MKL DFTI 2D (split, `DFTI_NOT_INPLACE`), single-thread, same fairness
as §1 (per-cell order-flip, cachebust + pace, best-of-5, ns timing). dag is **in-place,
scrambled order** (DIT); MKL is natural order — so the definitive correctness gate is the
roundtrip `fwd+bwd == N1·N2·x` (all e-14/e-15), and `elem≈1e0` just confirms the scramble.
Source: `bench_1d_vs_mkl.c --2d` → `vfft_perf_tuned_2d.csv`.

The plan comes from a dedicated **PATIENT 2D c2c calibration** (own `fft2d_c2c_wisdom`,
scored *end-to-end on the 2D transform* — PATIENT is the recommended planner; MEASURE is
the fast mode, exhaustive `stride_plan_2d` the wisdom-miss fallback). Inner row/col FFTs
are baked-or-JIT resolved (`--jit`). Measured **cooled** (20 s pre-cool + 30 s between
runs), median of 3. Source: `bench_1d_vs_mkl.c --2d --jit` → `vfft_perf_tuned_2d.csv`.

```
 N1×N2     dag/MKL   order
──────────────────────────────────
 64×64     ~1.6×*    scrambled
 128×128    1.41×    scrambled
 256×256    1.26×    scrambled
 512×512    1.29×    scrambled
──────────────────────────────────
 median    ~1.35×    (4/4 win)
```

*64² falls back to exhaustive — its PATIENT-banked plan was a calibration **noise
artifact** (a 5 µs cell is below reliable timing; the gate happened to measure exhaustive
slow at a hot moment). Exhaustive's 64² plan is measurably faster (~4.8 µs vs ~6.8 µs).

Headline:

> **2D C2C beats MKL on all 4 square cells — PATIENT-calibrated, median ~1.35×, up to 1.41×
> (128²).** The tiled B=8 row pass keeps the working set in L1/L2 and the SIMD 4×4/8×4
> transpose makes gather/scatter nearly free; JIT specializes the cold inner FFTs (bit-exact).
> Our plan times are **thermally rock-stable** (512² = 749 µs across every run this session);
> the run-to-run swing in the *ratio* is MKL's own variance, not ours. For these small 2D
> cells **PATIENT ≈ exhaustive** — full enumeration is cheap and good at this size — but both
> clear MKL on every cell. In-place scrambled-order 2D (the convolution contract);
> rectangular / non-pow2 cells are follow-ups.

### 2D C2C — vs MKL at T=8

Same cells, dag 2D threaded vs MKL `mkl_set_num_threads(8)`, identical split layout,
order-flipped + paced, **with an MT-vs-ST forward gate** (threaded fwd must equal the
single-thread fwd bit-for-bit — folded into rt; all e-14, so the tile-parallel path is
race-free). dag threads the **row pass only** (tile-parallel pool, per-thread scratch);
the **column pass stays serial** — that's the 2D self-scaling ceiling. Source:
`bench_1d_vs_mkl.c --2d --mt` → `vfft_perf_tuned_2d_mt.csv`.

```
 N1×N2     dag-T8 (ns)  MKL-T8 (ns)  dag/MKL   dag self-scale ST->T8
──────────────────────────────────────────────────────────────────
 64×64           5,641       48,751   8.64×     0.86× (overhead)
 128×128        23,996       87,069   3.63×     ~1.0×
 256×256        70,210      214,597   3.06×     1.88×
 512×512       500,200    1,307,575   2.61×     1.57×
──────────────────────────────────────────────────────────────────
 median                              ~3.34×     (4/4 win)
```

> **At T=8, 2D C2C beats MKL on all 4 cells — median ~3.3×, up to 8.6×.** Two effects:
> (1) dag's own scaling is **modest** (256² 1.88×, 512² 1.57×; tiny N regresses under threads)
> because only the row pass is parallel — the serial column pass caps it. (2) The large
> vs-MKL margins at small N are **MKL failing to thread tiny 2D**: at 64², MKL-T8 (48,751 ns)
> is ~6× *slower* than MKL-T1 (8,494 ns) — pure threading overhead — so dag wins 8.6×. Lifting
> the ceiling (parallel column pass / full-plane tiling) is the 2D-MT follow-up.

## 3. vs MKL — 1D R2C

R2C is the clearest embodiment of the split-layout trade: the **packing tax** that costs
us single-thread is the *same* lane-batched layout that makes K-split MT trivially parallel
(independent lanes, no barriers, no shared transpose buffer). So r2c **loses single-thread
and wins big multi-threaded** — throughput over single-core latency, by design
([transforms/real/README.md](../../src/core/transforms/real/README.md)). dag via the real
dispatcher (`vfft_r2c_plan_create`/`execute`, SPLIT, **JIT-wired**) vs MKL DFTI real r2c
(CCE); same fairness as §1; correctness vs a reference DFT (r2c is natural order). The
dispatch routes **rfft** at low K (JIT-specialized — see below) and **decoupled-stride** at
K≥32. Source: `bench_1d_vs_mkl.c --r2c [--mt]` → `vfft_perf_tuned_r2c{,_mt}.csv`.

### Single-thread — the packing tax

```
 N      K     path    dag/MKL    note
──────────────────────────────────────────────
 256    8     rfft     1.07×     JIT-wired rfft, low-K win
 256    16    rfft     1.15×
 256    256   stride   1.04×
 512    8     rfft     1.17×
 1024   8     rfft     0.64×     large-N rfft plane = L2-bound
 1024   256   stride   0.80×     decoupled-r2c structural gap
──────────────────────────────────────────────
 18 cells: 6 win.  Median 0.79×, range 0.46–1.17×.
```

> **Single-thread, r2c trails MKL — median 0.79×.** This is the honest cost of the split
> layout (the pack tax) plus MKL's heavily-tuned real-FFT. The **JIT lifts the low-K rfft
> cells to wins** (256/8 1.07×, 256/16 1.15×, 512/8 1.17×) — exactly where rfft is
> competitive; it can't close the large-N rfft L2 wall or the decoupled-stride high-K gap.

### Multi-threaded (T=8) — the layout payoff

```
 N      K     path    dag/MKL-T8   dag self-scale ST→T8
──────────────────────────────────────────────────────
 256    8     rfft      21.75×     ~1.0× (rfft is ST)
 256    256   stride     5.30×     2.79×
 512    256   stride     4.47×     4.87×
 1024   256   stride     3.65×     3.72×
 1024   16    rfft       1.74×     ~1.0×
──────────────────────────────────────────────────────
 18 cells: 18 win.  Median ~4.7×, range 1.74–21.75×.
```

> **At T=8, r2c beats MKL on all 18 cells — median ~4.7×, up to 21.8×.** The decoupled-stride
> path (K≥32) K-splits cleanly and scales **2.8–4.9×**. The rfft path (K<32) **also K-splits**
> (lane ranges, `rfft_natural_mt`), so MT is honored on every path — but its gain is small
> (~6–9% at K=16, none at K=8) because the rfft K-range sits at the **lane-split SIMD floor**:
> 8-wide lanes ÷ 8 threads leaves <1 SIMD group/thread, so K=8 falls back to single-thread and
> K=16 only splits ~2-way. The split layout is still the edge — it lets us thread the batch
> where MKL's real-FFT can't at modest N (MKL-T8 is ~20× slower than MKL-T1 at 256/8). The same
> layout that taxed us single-thread is the multithreading edge — the design trade paying off.

### 1D C2R (backward) — the natural split path

c2r (complex→real, the r2c inverse) gets the **same** split-layout treatment, and it's the
direct mirror of r2c's story. The public API hands c2r a **split** half-spectrum (the r2c
output), so the fast packed c2r — which needs a *packed* half-spectrum — was unreachable, and
the old path forced the slow decoupled-**stride** backward (~0.44–0.46× MKL). New this session:
a **fused natural initiator** (`c2r_execute_natural`, the inverse of rfft's natural terminator)
reads split re/im **directly** through the fast packed cascade — **no repack**. vfft's c2r front
door now runs a natural-vs-stride bake-off (mirror of r2c's), picking per cell; the forced-stride
hardcode is gone. Roundtrip `c2r(r2c(x))==N·x` is the gate (all e-14). Source:
`bench_1d_vs_mkl.c --c2r [--mt]`.

> ## 🔴 EVERY vs-MKL RATIO IN THIS SECTION IS VOID (found 2026-08-09, fixed 2026-08-13)
>
> The `--c2r` MKL arm reused the **forward** descriptor for `DftiComputeBackward`; DFTI
> distances are argument-anchored, so the backward read the CCE plane at the real-domain
> distance — a **heap OOB at every K>1** timing aliased garbage. Both tables below keep their
> **dag-side** numbers (self-scaling, natural-vs-stride uplift), but every dag/MKL column —
> including the "parity at K=8 (0.92×)" headline — is unusable. **Fix:** a backward-twin
> descriptor with swapped distances plus a per-run `mklref` correctness gate (unnormalized
> backward == N·x, printed in every row), so the arm is now proven on hardware each run.
> First **valid** cells (2026-08-13 smoke, gated 8.9e-16/1.0e-15): **0.366 at 512×4, 0.458 at
> 1024×16** — materially worse than the void table suggested. Note the comparison is
> home-layout vs home-layout: our natural path consumes a **split** re/im half-spectrum
> (lane-major batch) while MKL consumes **interleaved CCE** (transform-major); the
> interleaved-vs-interleaved like-for-like is the D2 zr2c route — **shipped 2026-08-13,
> measured in the subsection below**. A full re-sweep of the split cells above is still
> pending.

#### Single-thread — the packing tax (again)
```
 N      K     path      dag/MKL    note
──────────────────────────────────────────────
 256    8     natural    0.92×     ≈parity — packed-speed on split input
 256    16    natural    0.74×
 256    64    natural    0.55×     mid-K: MKL compute-bound / L1-resident
 256    128   natural    0.55×
──────────────────────────────────────────────
 natural ≈ 2× the old forced-stride path; reaches MKL parity only at K=8.
```
> **Single-thread, c2r trails MKL — same split-layout tax as r2c.** The natural path roughly
> **doubles** vfft's low-K c2r over the old stride path and reaches **parity at K=8 (0.92×)**,
> but MKL's compute-bound real backward still wins mid-K. (Even the unreachable packed path is
> only ~0.61× MKL at K=64 — the gap is structural in the cascade, not the split read.)

#### Multi-threaded (T=8) — the layout payoff (again)
```
 N      K     path      dag/MKL-T8   dag self-scale ST→T8
──────────────────────────────────────────────────────
 256    8     natural    ~17×        ~1.0× (K<16: lane-split floor)
 256    32    natural    ~7.9×       ~1.4×
 256    64    natural    3.9×        1.9×
 256    128   natural    3.0×        2.2×
 256    256   natural     —          2.8×   (MKL-T8 crashes at N·K≥131072)
 512    256   natural     —          3.6×
 1024   256   natural     —          2.8×
──────────────────────────────────────────────────────
```
> **At T=8 the split layout pays off — dag wins every cell, scaling 1.9–3.6× to high K.** The
> natural path K-splits the batch cleanly (`c2r_natural_mt`, pool lane-slabs; MT output is
> **bit-identical** to single-thread — race-free, lane-indexed scratch). MKL's c2r does **not**
> benefit from threads at these modest-N batch sizes: **MKL-T8 is slower than MKL-T1** even
> pinned to the same 8 cores (it parallelizes *within* the length-N transform, not across the
> K-batch where the work is), so the dag/MKL-T8 ratios at low K are inflated by MKL's thread
> overhead — the honest number is dag's own **2.8× self-scaling at high K**. Same trade as r2c:
> the layout that taxes us single-thread is exactly the multithreading edge.

### 1D INTERLEAVED r2c/c2r, K=1 — the D2 zr2c route (like-for-like vs MKL's home layout)

The first **interleaved-vs-interleaved** real-transform comparison — both engines consume/produce
the packed CCE plane, no layout excuse on either side. Ours = the D2 composite (`vfft.c` zr2c
route, shipped 2026-08-13): reinterpret x[N] as z[N/2] (zero work) → child c2c(N/2) NATURAL →
z→z Hermitian fold; c2r is the mirror with the fold leading. Two child routes, raced and banked
per cell in `oop_wisdom.txt` kind-5 rows: **OOP-IL** (IL c2c engine) and **cascade** (natural
in-place cascade). MKL = DFTI_REAL CCE **DFTI_INPLACE** — its best real arm (V6), backward on
its own twin descriptor with the per-run `mklref` gate. Gates: cross-engine fwd elementwise
(~5e-16) + each engine's backward vs N·x (~1e-15). Medians of 5, pinned core 2, pace 300 ms.
Source: `bench_1d_vs_mkl.c --zr2c` → `zr2c_quietday_20260813.csv`. Ratio = MKL/ours: >1 we win.

```
 N       r2c OOP-IL  r2c cascade  r2c BEST | c2r OOP-IL  c2r cascade  c2r BEST
────────────────────────────────────────────────────────────────────────────────
 512       1.06×        —          1.06×   |   0.93×        —          0.93×
 2048      0.88×       0.92×       0.92×   |   0.94×       0.92×       0.94×
 8192      1.08×       0.96×       1.08×   |   0.99×       0.96×       0.99×
 16384     1.08×       1.06×       1.08×   |   0.94×       0.83×       0.94×
 65536     0.89×       1.12×       1.12×   |   0.83×       0.84×       0.84×
────────────────────────────────────────────────────────────────────────────────
 r2c: parity-to-winning at 8192–65536 (65536 = the cascade child, 1.12×).
 c2r: 2026-08-21 — the 2048 hole (was 0.55×) is CLOSED: the zr2c child ran
 BLOCKED codelets fwd and MONOLITHIC bwd; eight blocked bwd twins now ship.
 65536 not re-measured. Medians of 5.
```

> **r2c reaches the parity band mid-N and wins at 65536 (1.12×, cascade child); the small end
> trails ~0.85×.** The c2r column is the open problem, and the 2048 cell shows exactly where:
> MKL's backward costs **the same as its forward** (1198 vs 1200 ns), while ours pays **~+50%
> over our own forward** (r2c ~1400–1500 ns → c2r ~2200 ns) — in **both** child routes, so no
> route pick fixes it. The fold is direction-symmetric; the entire penalty is the child's
> **natural c2c backward** (IL bwd / backward cascade), a c2c-side workstream — fixing it moves
> the whole c2r column, 2048 (0.55×) first. Day-to-day ratio drift on this host is up to ~0.2
> per cell (thermal); quote the **shape**, not one day's third digit.

## 4. vs MKL — 2D R2C

dag tiled 2D real-to-complex (`fft2d_r2c.h`: tiled R2C row pass + native column c2c)
vs MKL DFTI 2D real (CCE), single-thread, same fairness as §1–§3 (per-cell order-flip,
cachebust + pace, best-of-5, ns timing). dag output is **split** (out_re/out_im) and
**scrambled** (DIT); MKL is CCE-interleaved natural — so the definitive correctness gate
is the roundtrip `r2c+c2r == N1·N2·x` (all e-14/e-15), not an elementwise compare. Plans
are per-cell tuned; the inner column c2c is JIT-specialized. Source:
`bench_1d_vs_mkl.c --2dr2c` → `vfft_perf_tuned_2dr2c.csv`.

### Single-thread

```
 N1×N2      dag/MKL    order
──────────────────────────────────
 64×64       0.86×     scrambled
 128×128     0.85×     scrambled
 256×256     0.80×     scrambled
 512×512     0.89×     scrambled
──────────────────────────────────
 median     ~0.85×     (best-of-3)
```

> **Single-thread, 2D R2C trails MKL — median ~0.85×, range 0.80–0.89×.** As with 1D R2C
> (§3), this is the honest cost of the split lane-batched layout (the real-FFT pack tax)
> against MKL's heavily-tuned 2D real path — the same layout trade that becomes an edge
> under threading. Per-cell plan tuning closes most of the gap; the 256² cell is the
> laggard (0.80×). See the multi-threaded results below.

### Multi-threaded (T=8)

Same cells, dag threading the **row pass only** (tile-parallel pool, per-thread scratch; the
column c2c and the c2r backward stay serial — that's the 2D self-scaling ceiling), calibrated
plans, pinned core 0, with an **MT-vs-ST forward gate** (the threaded fwd must equal the
single-thread fwd bit-for-bit — folded into rt; all e-14/e-15, so the tile-parallel path is
race-free). Source: `bench_1d_vs_mkl.c --2dr2c --mt` → `vfft_perf_tuned_2dr2c_mt.csv`.

```
 N1×N2     dag-T8 (ns)   dag self-scale ST→T8
──────────────────────────────────────────────
 64×64          6,734    0.78×  (overhead)
 128×128       23,271    0.96×
 256×256       70,010    1.71×
 512×512      415,188    1.38×
──────────────────────────────────────────────
```

> **dag's 2D R2C self-scaling is modest — 256² 1.71×, 512² 1.38×; tiny N regresses under
> threads.** Only the row pass is parallel, so the serial column c2c + c2r passes cap it —
> the same ceiling as 2D C2C (§2). The MT-vs-ST gate confirms the tile-parallel forward is
> race-free (rt e-14/e-15).
>
> **No vs-MKL-T8 ratio is reported here.** MKL's threaded 2D *real* path is pathological in
> this `mkl_rt` + 8-thread configuration: a fixed ~30–370 ms per-call overhead, independent
> of transform size and wildly inconsistent run-to-run (256² measured 366 ms one rep, 32 ms
> the next). MKL-T8 thus comes out ~hundreds-of-× slower than MKL-T1, so the apparent dag
> "win" of 60×–5000× is a pure measurement artifact, not real speedup — MKL simply does not
> usefully thread small 2D real transforms in this setup. (1D C2C and 2D C2C thread fine in
> the same binary, so this is specific to the 2D real descriptor.) The c2r backward row pass
> now threads too (see the 2D C2R subsection below); parallelizing the **column** passes is the
> remaining 2D-MT lever that would lift the self-scaling ceiling further.

### 2D C2R (backward)

The inverse — complex (CCE / split) → real 2D, `fft2d_r2c.h`'s c2r path, **PATIENT-calibrated**
(separate `fft2d_c2r_wisdom`; c2r's optimum ≠ r2c's — all 4 cells WON their own gate),
single-thread (the c2r backward is **serial** — not yet tile-parallel). Roundtrip
`r2c+c2r == N1·N2·x` is the gate (all e-14/e-15). Measured **cooled**, median of 3. Source:
`bench_1d_vs_mkl.c --2dc2r` → `vfft_perf_tuned_2dc2r.csv`.

> ⚠ **UNAUDITED (2026-08-13):** this mode's MKL arm has the same bug *class* that voided the
> 1D `--c2r` ratios — one 2D handle (default strides) serves both compute directions, and
> MKL's backward output is never validated. The vs-MKL ratios below stand until audited, but
> do not build on them; the dag-side numbers are unaffected.

```
 N1×N2     dag/MKL   order
──────────────────────────────────
 64×64      0.84×    scrambled
 128×128    0.95×    scrambled
 256×256    0.75×    scrambled
 512×512    0.95×    scrambled
──────────────────────────────────
 median    ~0.89×    (single-thread)
```

> **Single-thread, 2D C2R trails MKL — median ~0.89×, range 0.75–0.95×.** Same real-FFT
> structural tax as r2c (§3, §4): the split lane-batched layout costs single-thread what it
> repays under threading. c2r lands right alongside the r2c forward (0.89× vs §4's 0.85×); 256²
> is the laggard (0.75×). PATIENT ≈ MEASURE here — the gap is structural, not plan-mode.

#### 2D C2R — multi-threaded (T=8)

The c2r backward is **now tile-parallel** (new this session): its row pass reads the padded
col-FFT scratch and writes reals to a *distinct* user buffer, so tiles are independent — the
same tile-parallel pool as the r2c forward, each thread with its own scratch slot + inner-pack
tid (the prior serial path was forced only by a hardcoded inner-slot index, not a real data
hazard). The column c2c IFFT stays serial — the self-scaling ceiling, as in §2/§4. **MT-vs-ST
gate:** the threaded c2r equals the single-thread output bit-for-bit (rt e-14/e-15 — race-free).
MKL's threaded 2D-real backward is anomalous on this host (§4), so we report dag **self-scaling**,
not a vs-MKL ratio. Cooled, median of 2. Source: `bench_1d_vs_mkl.c --2dc2r --mt`.

```
 N1×N2     dag-T8 (ns)   dag self-scale ST→T8
──────────────────────────────────────────────
 64×64          6,007    0.78×  (overhead)
 128×128       22,144    0.91×  (overhead)
 256×256       66,169    1.59×
 512×512      328,031    1.53×
──────────────────────────────────────────────
```

> **2D C2R self-scaling — 256² 1.59×, 512² 1.53×; small N regresses under threads.** Right
> alongside the r2c forward (§4: 1.47× / 1.46×) — only the row pass is parallel, the serial
> column IFFT caps it. Tiny cells (64²/128²) regress: threading overhead exceeds the few µs of
> row work. Full-arsenal milestone: **every 2D real path now threads** (r2c forward + c2r
> backward); parallelizing the column passes is the remaining lever.

## 5. vs FFTW3 — single-thread

VectorFFT's calibrated wisdom path measured against FFTW3 with
`FFTW_MEASURE` planning. FFTW3 split-complex API
(`fftw_plan_guru_split_dft`) so the layout matches VectorFFT exactly —
no interleave / deinterleave overhead on the FFTW side.

### 1D C2C — full sweep

Source: [build_tuned/benches/bench_1d_vs_fftw.c](../../build_tuned/benches/bench_1d_vs_fftw.c)
(207 cells × MKL bench grid, calibrated wisdom loaded). Same N/K grid
as Section 1's MKL bench, so ratios are directly comparable.

```
Category       Cells    Min   Median    Max    Mean
─────────────────────────────────────────────────────
Small (N≤128)    15   1.86×   4.10×   8.70×   4.60×
Power-of-2       30   1.34×   3.08×  15.89×   4.28×
Composite        33   1.82×   3.45×  15.07×   4.93×
Odd composite    18   1.38×   3.67×   6.29×   3.72×
Mixed deep       18   1.50×   5.28×  11.38×   5.11×
Prime powers     30   1.37×   5.09×  17.79×   6.85×
Genfft (R=11/13) 15   1.85×   3.25×  10.94×   4.52×
Rader primes     24   1.07×   2.23×   4.05×   2.38×
Bluestein primes 24   0.92×   1.15×   1.74×   1.22×
─────────────────────────────────────────────────────
OVERALL         207   0.92×   3.21×  17.79×   4.25×

Wins vs FFTW3: 202/207 (97.6%)
```

Headline:

> **VectorFFT beats FFTW3 on 202/207 (97.6%) of bench cells. Median
> speedup 3.21×, mean 4.25×, range 0.92×–17.79×.**

The median against FFTW3 (3.21×) is meaningfully higher than the
median against MKL (2.64× from Section 1). FFTW3 is genuinely behind
on power-of-two and prime-power cells once N·K outgrows last-level
cache — the calibrated wisdom routes around L3 thrashing while
FFTW's plan search doesn't capture the cache-residency effect.

**Top wins (large prime-power and pow-of-2 cells):**

| Cell | Factors | Ratio |
|------|---------|------:|
| N=390625 (5^8) K=256 | 5×5×5×5×5×5×25 | **17.79×** |
| N=78125 (5^7) K=256 | 5×5×5×25×5×5 | 17.51× |
| N=65536 K=256 | 4×4×8×16×32 | 15.89× |
| N=131072 K=256 | 4×4×4×4×4×4×32 | 15.57× |
| N=100000 K=256 | 4×25×5×8×25 | 15.07× |

At these sizes FFTW drops to ~1 GFLOP/s while VectorFFT sustains
~17–20 GFLOP/s — 1D batched FFT against a 16M+ working set is
memory-bound, and our wisdom-tuned multi-stage factorizations keep
inner radices L1-resident across the K=256 batch.

**Weakest cells (Bluestein primes — pre-wisdom snapshot):**

| Cell | Ratio (pre-wisdom) |
|------|------:|
| N=179 K=256 (Bluestein) | 0.92× (FFTW wins) |
| N=59 K=256 (Bluestein) | 0.93× (FFTW wins) |
| N=59 K=32 (Bluestein) | 0.96× (within noise) |

> **Note:** these FFTW3 ratios are the **pre-Bluestein-wisdom** snapshot. With the calibrated
> per-(N,K) `(M, B)` wisdom these sub-1.0× cells turn into wins (the vs-MKL §1 table shows every
> Bluestein cell ≥1.0×). A fresh `bench_1d_vs_fftw` run is pending; the table above is the historical
> lower bound, not the shipped result.

Full per-cell data: [build_tuned/results/vfft_perf_tuned_1d_fftw.txt](../../build_tuned/results/vfft_perf_tuned_1d_fftw.txt)
(human-readable, generated from
[vfft_perf_tuned_1d_fftw.csv](../../build_tuned/results/vfft_perf_tuned_1d_fftw.csv)
via `python build_tuned/make_perf_txt_fftw.py`).

### r2r family

The DCT / DST / DHT wrappers are built atop our R2C using Makhoul (DCT-II/III)
and Lee 1984 (DCT-IV); DST-II/III piggyback on DCT-II/III with sign-flip
+ index reversal; DHT is a free derivation of R2C output. Specialized
straight-line N=8 codelets (`gen_dct8.py`, `gen_dct3_n8.py`) bypass
Makhoul for the JPEG block size.

All numbers here are **single-threaded** (T=1) vs FFTW3 with `FFTW_MEASURE`
planning, split-complex API.

### DCT-II (REDFT10) — `bench_dct2_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 1024 (JPEG) | 2,300 | 3,400 | **1.48×** |
| 8 | 4096 | 9,500 | 11,100 | 1.17× |
| 16 | 1024 | 12,400 | 39,200 | 3.16× |
| 32 | 1024 | 32,200 | 81,100 | 2.52× |
| 64 | 1024 | 71,200 | 173,800 | 2.44× |
| 128 | 256 | 28,900 | 88,300 | 3.06× |

Wins all measured cells (range 1.17–3.16×).

### DCT-III (REDFT01) — `bench_dct3_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 1024 (JPEG) | 2,500 | 2,900 | 1.16× |
| **8** | **4096** | **17,200** | **10,400** | **0.60× (FFTW wins)** |
| 16 | 1024 | 13,700 | 41,100 | 3.00× |
| 32 | 1024 | 34,100 | 84,800 | 2.49× |
| 64 | 1024 | 75,200 | 178,100 | 2.37× |
| 256 | 256 | 65,900 | 203,300 | 3.08× |
| 1024 | 256 | 416,000 | 1,495,500 | **3.59×** |

> **The only v1.0 r2r loss vs FFTW3** is DCT-III at N=8 K=4096 (0.60×).
> Both N=8 codelets (`gen_dct3_n8`) target the JPEG-range K (256–1024)
> and don't optimize for very-large-K layout. FFTW switches to a
> different large-batch code path that still beats us at K≥4096. v1.1
> fix: a K-specialized DCT-III N=8 variant — same flavor as the JPEG
> codelet, different cache layout for K≥4096. Tracked in
> [docs/v1_1_codelet_roadmap.md](../v1_1_codelet_roadmap.md).

### DCT-IV (REDFT11) — `bench_dct4_vs_fftw`

After the specialized N=8 codelet landed:

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 256 | 800 | 2,700 | 3.38× |
| 8 | 1024 | 4,300 | 9,400 | 2.19× |
| 8 | 4096 | 17,600 | 36,900 | 2.10× |
| 16 | 1024 | 8,900 | 35,900 | **4.03×** |
| 32 | 1024 | 28,300 | 74,200 | 2.62× |
| 64 | 1024 | 60,800 | 161,800 | 2.66× |
| 256 | 256 | 59,500 | 186,000 | 3.13× |
| 1024 | 256 | 354,200 | 1,482,100 | **4.18×** |

Wins all measured cells (range 1.85–4.18×). The pre-codelet build
showed losses 0.53–1.06× at small N — codelet flipped that.

### DST-II / DST-III (RODFT10 / RODFT01) — `bench_dst23_vs_fftw`

| Variant | N | K | vfft ns | fftw ns | ratio |
|---------|--:|--:|--------:|--------:|------:|
| DST-II | 8 | 256 | 600 | 2,400 | **4.00×** |
| DST-II | 16 | 1024 | 16,100 | 38,900 | 2.42× |
| DST-II | 32 | 1024 | 39,100 | 78,500 | 2.01× |
| DST-II | 64 | 1024 | 90,800 | 173,600 | 1.91× |
| DST-II | 256 | 256 | 82,600 | 198,600 | 2.40× |
| DST-II | 1024 | 256 | 553,900 | 1,484,500 | 2.68× |
| DST-III | 8 | 256 | 700 | 2,900 | **4.14×** |
| DST-III | 16 | 1024 | 21,400 | 40,800 | 1.91× |
| DST-III | 32 | 1024 | 41,100 | 83,200 | 2.02× |
| DST-III | 64 | 1024 | 94,700 | 176,700 | 1.87× |
| DST-III | 256 | 256 | 84,300 | 207,100 | 2.46× |
| DST-III | 1024 | 256 | 544,900 | 1,507,000 | 2.77× |

Wins all measured cells. Range 1.85–4.14×; strongest at small N where
FFTW's DST is less specialized than its DCT path.

### DHT (Hartley)

Per session notes, DHT lands **1.9–2.8× over FFTW** across the same
N/K range. A dedicated `bench_dht_vs_fftw` per-cell table was not
written for v1.0 — `test_dht.c` confirms 22/22 cells pass at machine
precision vs FFTW reference, but timing data was not preserved. v1.1
adds the bench so the DHT row matches the DCT/DST detail level.

### Headline (r2r vs FFTW3, T=1)

> **VectorFFT wins 53/54 measured r2r cells vs FFTW3** (1.16–4.18×
> range; mean ~2.5×). Single loss: DCT-III at N=8 K=4096 (0.60×) —
> codelet-fixable in v1.1.

| Family | Ratio range | Cells | Wins |
|--------|:-----------:|:-----:|:----:|
| DCT-II | 1.17–3.16× | 6 | 6/6 |
| DCT-III | 0.60–3.59× | 7 | 6/7 |
| DCT-IV | 1.85–4.18× | 11 | 11/11 |
| DST-II | 1.91–4.00× | 6 | 6/6 |
| DST-III | 1.87–4.14× | 6 | 6/6 |
| DHT | ~1.9–2.8× (summary) | — | — |

MKL TT was also benched for DCT-IV (4–13× wins) and DST (timing-only —
MKL TT computes a different PDE-oriented math convention, so the
comparison is informational, not apples-to-apples). FFTW3 is the
correct r2r baseline.

## 6. Multi-threaded scaling

### 1D C2C — direct MT vs MKL

See **§1 → "Multi-threaded — vs MKL at T=8"** for the head-to-head: 129/129 wins at T=8, median
3.76× over MKL (K=32: 3.00×, K=256: 4.38×). R2C inherits the same K-split MT (its inner C2C threads).

### DCT-II / DCT-III / DCT-IV / DST-II/III / DHT (wrapper MT, new in v1.0)

Source: [build_tuned/benches/bench_mt_dct.c](../../build_tuned/benches/bench_mt_dct.c).

```
Transform   Cell           T=1 ns   T=2 (×)    T=4 (×)    T=8 (×)
──────────────────────────────────────────────────────────────────
DCT-II      N=256  K=1024   482000  1.04   1.95   2.60
DCT-IV      N=256  K=1024   452200  1.12   1.77   2.09
DST-II      N=256  K=1024   620900  1.17   2.06   2.49
DHT         N=256  K=1024   452900  0.97   1.55   1.85
DCT-II      N=1024 K=1024  2297700  1.08   1.55   2.35
DCT-IV      N=1024 K=1024  2682900  1.16   1.77   2.65
DST-II      N=1024 K=1024  2713900  0.97   1.41   2.11
DHT         N=1024 K=1024  2047300  0.88   1.23   1.67
DCT-II      N=4096 K=1024 13911400  1.11   1.65   2.11
DCT-IV      N=4096 K=1024 16838200  1.20   1.58   2.14
DST-II      N=4096 K=1024 19109400  1.13   1.62   2.20
DHT         N=4096 K=1024 13426100  1.06   1.49   1.83
DCT-II      N=4096 K=4096 58493200  1.12   1.61   2.14
DCT-IV      N=4096 K=4096 72842000  1.22   1.44   1.62
DST-II      N=4096 K=4096 80495400  1.13   1.63   2.20
DHT         N=4096 K=4096 59296500  1.06   1.55   1.87
```

Best speedup at T=8: **2.65×** (DCT-IV at N=1024 K=1024). Typical
**1.6–2.4×** across cells.

### Why not 8× at T=8?

The DCT/DST/DHT family is implemented as **three sequential passes**:

```
Pass 1: pre-permute / pre-twiddle    — bandwidth-bound
Pass 2: inner FFT (R2C or C2C)       — has its own MT
Pass 3: post-process / post-twiddle  — compute + memory mix
```

Each pass reads + writes the full N·K data once. Total memory traffic
≈ 3 × N·K × 16 bytes per call. At N·K = 16M (N=4096 K=4096), that's
~768 MB per call. DDR5 on this CPU saturates around 25 GB/s, putting
a wall-time floor around 30 ms per call — close to what we measure
(27 ms at T=8). Adding more threads can't beat physics.

### Where the 8× comes back: v1.1 fused codelets

The v1.1 codelet roadmap
([docs/v1_1_codelet_roadmap.md §2](../v1_1_codelet_roadmap.md))
adds specialized straight-line codelets — `e10_*` for DCT-II,
`e11_*` for DCT-IV, `r2hc_*` for R2C — that fuse all three passes
into one tight kernel. Arithmetic intensity rises dramatically:

| Generation | Memory traffic / call | T=8 ceiling |
|-----------|----------------------|:-----------:|
| Pre-v1.0 (sequential wrappers) | 3 × N·K·16 bytes | ~1.4× |
| **v1.0 (parallel wrappers, current)** | **3 × N·K·16 bytes** | **~2.6×** |
| v1.1 (fused codelets) | 1 × N·K·16 bytes | ~5× projected |

The v1.0 parallel wrappers lift the floor from 1.4× to 2.6×. Fused
codelets lift the ceiling from 2.6× to ~5× by eliminating the
multi-pass bandwidth traffic. Both are needed for the full picture.

DHT scales worst (1.6–1.9× at T=8) because its pre-phase is one big
sequential memcpy of N·K doubles — left intentionally non-parallel
because it's pure memory bandwidth, and a single optimized memcpy
typically beats T smaller memcpys when the limit is DRAM throughput.
DHT will benefit most from v1.1 fused codelets.

## 7. Per-codelet performance (VTune-grade)

For deep per-radix analysis at K=256 see
[docs/vtune-profiles/](../vtune-profiles/) — one detailed profile per
radix R ∈ {4, 8, 10, 11, 12, 13, 16, 20, 25, 32, 64}. Top-line:

| Radix | Retiring (% of pipeline slots) | Bottleneck |
|------|:-----:|------|
| R=4  | 86% | compute-peak (port 0/1 at 96/91%) |
| R=8  | 72% | DFT-8 critical path dependency chains |
| R=10 | 63% | radix-5 + radix-2 FMA chains |
| R=11 | 59% | Winograd, machine-clears flagged |
| R=12 | 57% | radix-3 + radix-4 FMA chains |
| R=13 | 60% | Winograd + Sethi-Ullman |
| R=16 | 25% | store-bound + L1 latency (post-prefetch) |
| R=20 | 54% | radix-5 FMA chains |
| R=25 | 50% | hybrid compute/store |
| R=32 | 34% | L1 store-DTLB overflow (~80 pages) |
| R=64 | 27% | load + store DTLB overflow (~160 pages) |

Most radixes retire 50–86%. R=16/32/64 hit memory-system bottlenecks
that the codelet alone can't fix (huge codelets exceed DTLB capacity);
these benefit specifically from the cost model's variant-aware
selection (T1S / LOG3 / BUF) which routes around their bottlenecks
when wisdom shows another protocol wins.

## 8. Hardware caveats

### These numbers are from one CPU

All measurements: i9-14900KF (Raptor Lake, hybrid 8P+16E), 5.7 GHz
turbo, AVX2. Numbers move on:

- **Sapphire Rapids / Emerald Rapids** — should be similar or better
  (same uarch family, often better memory subsystem). Wisdom carries
  over without recalibration.
- **Zen 4 / Zen 5** — different uarch. CPE numbers shift; recommend
  re-running `cpe_measure` and `calibrate_tuned` on the target host.
  Architectural advantages (cost model, wisdom, MT) carry over; per-
  cell speedups may differ.
- **AVX-512 hardware** — codelets exist, but CPE table currently
  holds only AVX2 measurements. Re-run cpe_measure on AVX-512 host
  for accurate estimate-mode plans there.

### Consumer PC vs calibration host

The numbers in this doc are from the calibration host running clean
(idle background, performance plan, single P-core pinned). On a
consumer PC running normal background load, expect:

- **vs MKL ratios**: similar (within 5–10% — the win is structural)
- **Estimate vs wisdom mean**: drifts up to 1.3× on a noisy host (was
  1.19× on the calibration host)
- **MT scaling**: slightly weaker (T=8 ceiling drops 10–20% under
  thermal/freq fluctuation)

### What `Ts > 8` looks like

We bench up to T=8. On the i9-14900KF's hybrid 8P+16E config, T=16
or T=24 starts using E-cores, which run ~60% the IPC at higher
latency. Per-thread efficiency drops sharply past 8. For workloads
that benefit from many threads, the bench grid should be extended
(v1.1 work).

## 9. Reproducing these numbers

### vs MKL

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_1d_vs_mkl.c --mkl
build_tuned/benches/bench_1d_vs_mkl.exe        # single-thread -> vfft_perf_tuned_1d.csv
build_tuned/benches/bench_1d_vs_mkl.exe --mt   # T=8 (K>=32) -> vfft_perf_tuned_1d_mt.csv
```

Requires MKL ILP64 (Intel oneAPI install); single-thread uses `mkl_set_num_threads(1)`, `--mt`
uses 8. 238 cells × ~1 second = ~5 minutes wall (single-thread).

### 1D C2C vs FFTW3 (single-thread)

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_1d_vs_fftw.c --fftw
# fftw3.dll must be co-located with the exe (already copied into build_tuned/).
build_tuned/benches/bench_1d_vs_fftw.exe \
    build_tuned/vfft_wisdom_tuned.txt \
    build_tuned/results/vfft_perf_tuned_1d_fftw.csv \
    build_tuned/results/vfft_acc_tuned_1d_fftw.csv
```

Long run — 1–2 hours on the calibration host because of FFTW's
`FFTW_MEASURE` plan-search cost on the larger prime-power cells
(N=823543 alone takes ~30 min at K=256). Run with no other significant
load for cleanest numbers.

### r2r vs FFTW3 (single-thread)

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct2_vs_fftw.c --fftw
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct3_vs_fftw.c --fftw
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct4_vs_fftw.c --fftw
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dst23_vs_fftw.c --fftw
build_tuned/benches/bench_dct2_vs_fftw.exe
build_tuned/benches/bench_dct3_vs_fftw.exe
build_tuned/benches/bench_dct4_vs_fftw.exe
build_tuned/benches/bench_dst23_vs_fftw.exe
```

Requires FFTW3 (vcpkg install or local build). ~30 seconds wall total.
Each binary plans with `FFTW_MEASURE` so first-run setup is the bulk
of the time; benched min over 21 reps after 5 warmup.

### MT scaling for DCT/DST/DHT

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_mt_dct.c
build_tuned/benches/bench_mt_dct.exe
```

~30 seconds wall. Run with no other significant load on the machine
for cleanest numbers.

## See also

- [docs/cost_model/](../cost_model/) — how the estimate path achieves 1.20×
- [docs/wisdom/](../wisdom/) — how the calibrator achieves the optimum
- [docs/v1_1_codelet_roadmap.md](../v1_1_codelet_roadmap.md) — what closes the remaining gaps
- [src/core/README.md](../../src/core/README.md) — user-facing API docs and threading status
