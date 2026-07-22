# The row-major engine — K=1 single-transform, and the MKL ground truth behind it

*2026-07-22. Canonical design doctrine for native single-transform (K=1) FFT. Written after
reverse-engineering MKL's actual K=1 kernels by disassembly (not from our own perspective),
timing MKL interleaved-vs-split on the i9, and reading our own BAILEY2 executor. Everything here
is measured or read from code — so we don't re-derive it. Supersedes the uncertainty in
[k1_single_transform.md](../performance/k1_single_transform.md) about "is there a packing delay"
and corrects the earlier split-layout-wins framing.*

Host: i9-14900KF (Raptor Lake, AVX2), MKL 2025.3, sequential (`MKL_THREADING_LAYER=SEQUENTIAL`),
mingw 15.2. Trapping technique: gdb hardware watchpoint on the output buffer during
`DftiComputeForward` call #2 (steady state), `disassemble` the store site. MKL's real kernels are
internal statics; exported symbol names in gdb are the *nearest export* and are garbage — trust the
disassembled bytes, not the name.

---

## 1. The one-sentence problem

**Our engine is entirely col-major** (SIMD across the batch dimension K; every codelet vectorizes
across lanes). A single transform (K=1 — the most common FFT call in the world, and what an MKL/FFTW
user issues by default) has no batch to vectorize across, so it falls onto the arbitrary-K `rem==1`
**scalar tier**: bit-exact, ~2.6–2.7× off the batched ceiling. **We have no row-major engine** (SIMD
across the butterflies *within* one transform). That is the entire gap.

## 2. The organizing frame — the 2×2 (axis × complex layout)

| | **split** (`re[]`,`im[]` separate) | **interleaved / IL** (`[re,im,re,im]`) |
|---|---|---|
| **col-major** (SIMD across K lanes / transforms) | ✅ original engine — 2.64× median vs MKL | ✅ the IL campaign (ledger §6a16–§6a53) |
| **row-major** (SIMD across butterflies of ONE transform) | ❌ none (scalar tail + half-scalar BAILEY2) | ❌ **none — and this is the MKL-user default** |

We own both col-major quadrants. We have **zero** row-major. The missing quadrant that matters most
is **row-major × interleaved** — one interleaved transform, what an MKL/FFTW caller actually brings.

## 3. MKL ground truth — what its K=1 kernels actually are (disassembled)

### 3a. N=64, INTERLEAVED (default `DFTI_COMPLEX_COMPLEX`) — single monolithic in-place pass
One contiguous hot-kernel window, instruction census:

| class | count |
|---|--:|
| `vaddpd`/`vsubpd` (butterfly) | 93 |
| `vmulpd`+`vfmadd231pd` (twiddle) | 36 |
| `vshufpd`/`vshufps` (**re/im swap**) | 26 |
| `vperm2f128`/`vinsertf128` (inter-stage) | 27 |
| `vxorpd` (± i sign) | 10 |
| `vmovupd` (ld/st, many to `(%rsp)` = spill) | 79 |
| `vgather` | **0** |

- **2 complex per YMM** (interleaved packs one complex's re+im adjacent → only 2 points/vector).
- Data loads at `0x60/0x160/0x260/0x360(%rcx)` = **stride 0x100 = 16 complex = N/4** → radix-4 DIT
  stage reading its 4 inputs at N/4, plain packed loads (no gather).
- The 26 shuffles are the **interleaved-complex-multiply tax**: `vshufpd (swap re/im) → vmulpd →
  vfmadd → vxorpd (sign)` per complex mul. Forced by adjacency of re,im.
- The 27 perms are inter-stage cross-lane reshaping. Heavy stack spilling (N=64 blows past 16 YMM).

### 3b. N=64, SPLIT (`DFTI_REAL_REAL`) — the layout our benches use
FP butterfly region: **13 `vaddpd` + 12 `vsubpd` + 2 `vmulpd` = 27 FP ops, `vshufpd/vshufps` = 0,
all `%ymm` (95 refs, 0 `%xmm`).** **MKL's split kernel has ZERO re/im shuffles** — it does exactly
what a split codelet of ours would. The 26 shuffles in 3a were purely the interleaved tax; MKL ships
a separate split kernel that avoids them entirely.

### 3c. N=1024 — MULTI-PASS four-step through a SCRATCH buffer
Watchpoint on a mid element shows it written in ≥2 well-separated code regions (~1.2 MB apart in
code = distinct kernels), and the transform reads/writes a **scratch buffer** (`%r11`) alongside the
user buffer (`%rdx`):
- **Region B** (touched the element ~12×): **transpose-dominated** — 16 perms + 30 `vmov` + only ~8
  FP ops, moving data three-way scratch↔stack↔user. This is the four-step's data-reshape pass.
- **Region A**: pure scalar loop/address glue (0 vector ops).

So large-N K=1 in MKL is a four-step (column DFTs in scratch → **separate tiled transpose** → row
DFTs), NOT one giant kernel. Small N (≤~64) is the four-step collapsed to one in-place pass.

## 4. MKL timing — INTERLEAVED IS FASTER than split (K=1, i9, best-of-9)

| N | IL (ns) | split (ns) | split/IL |
|--:|--:|--:|--:|
| 64 | 30.3 | 31.9 | 1.05× |
| 256 | 135.8 | 187.7 | **1.38×** |
| 1024 | 813.1 | 776.1 | 0.95× |
| 4096 | 4085.2 | 4934.1 | 1.21× |
| 16384 | 19076.5 | 23309.7 | 1.22× |

**Interleaved is ~20–38% faster than split at most sizes.** The re/im shuffles hide under FMA
latency (port 5 vs ports 0/1); split's penalty is **two memory streams** (`re[]` + `im[]` = two
prefetch streams, two TLB footprints) vs interleaved's one contiguous stream. **Consequence: our
split-only benches have been beating MKL's *slower* kernel. A real MKL user (interleaved, single
shot) races a target ~20% quicker than we've measured against.**

## 5. Why MKL does it this way — the fence has a reason (do NOT go blind)

MKL is designed by experts who see every detail. Their two apparent "taxes" are **prices paid for
generality**, not oversights:

1. **Interleaved default = the universal ABI.** C `double complex`, C++ `std::complex`, Fortran
   `complex*16`, numpy, MATLAB — all interleaved. The default honors zero-copy interop with all of
   them. The in-register shuffles are the price, and it's the *right* price: deinterleave→split is a
   full **memory** pass, worse for a bandwidth-bound kernel than in-register shuffles that hide under
   FMA. And MKL **also ships the split kernel** (§3b, 0 shuffles) for split callers. Both bases
   covered; nothing left on the table.
2. **Separate tiled transpose at large N = generality + cache scalability.** A *fused*-transpose
   store is bound to one `(R1,R2,radix,dir,placement)`. MKL must serve arbitrary N (primes,
   mixed-radix, Bluestein), any placement, sizes past L3, threaded. A general **tiled** transpose:
   works for all factorizations, cache-blocks so it scales when a fused scatter would thrash TLB,
   supports in-place uniformly, keeps the codelet/i-cache small. Fusion wins only in a **narrow**
   band (small/mid pow2, cache-resident, OOP/scratch-OK, ST) — which is exactly why our BAILEY2
   fades at N=4096 and is K-split-excluded. MKL declines to specialize that narrowly on purpose.

**The trap to avoid:** building a K=1 kernel that only wins vs interleaved-MKL (mirage — they have a
shuffle-free split kernel too), or only at small-N-pow2-split (narrow demo). MKL does **not** leave
K=1 on the table — its split kernel vectorizes across butterflies natively (§3b). The gap is **ours**
(we drop to scalar), not theirs.

## 6. Where we honestly win — codelet quality, in a chosen regime

The re/im shuffle was never the story. Strip it away:

- **Row-major IL is MKL's own fast layout.** In it we pay the *same* shuffle tax MKL pays — no free
  win. We compete purely on **codelet quality**: DAG list-scheduling, factorization search, k-varying
  FLAT twiddles, register/spill discipline — the same muscle that wins 2.64× median at K≥8. Winnable,
  but on merit.
- **Row-major SPLIT is the open, interesting question.** It trades shuffles for two memory streams.
  In the *batched* world two streams lose (§4). But a single K=1 transform up to a few thousand
  points is **cache-resident**, so the two-stream/TLB penalty that hurts streaming work largely
  evaporates in L1/L2 — split might beat IL exactly where MKL's split loses to its own interleaved.
  **This is a measurement to run, NOT a claim to ship on.**
- We must support **both layouts, layout-parametric**, as the IL campaign did for col-major. IL first
  (what users bring, MKL's fast path); split second (the measure-it upside).

Our durable edge is the [MKL blind-spot](../performance/mkl_geometry_contracts.md) thesis: batched
throughput, convolution pipelines that stay in the frequency domain, odd/mixed radix, in-place
scrambled — regimes a general-purpose library structurally won't specialize into. K=1 is us
*extending our K≥8 advantage down to a shape we currently drop*, not exploiting an MKL hole.

## 7. What already exists (so we scope honestly)

- **The codelets exist for BOTH layouts.** Col-major already built the split family and the IL family
  (`codelets/il/`, ledger §6a16–§6a53). A codelet processes `me` lanes at stride `ios`; it does not
  know whether the lanes are batch elements (col-major) or butterflies of one transform (row-major).
  So a row-major stage = a col-major codelet driven with `me = N/r`, `ios = N/r`, and a **k-varying**
  twiddle fill.
- **BAILEY2 is a hand-rolled 2-stage row-major plan, half-vectorized** (read from
  [oop_plan.h:311](../../src/core/oop/oop_plan.h#L311)). At K=1: column pass = `for n1<R1:
  leaf(vl=K=1)` → R1 **scalar** size-R2 transforms; row pass = one `t1p(me=R2·K=R2)` → **vectorized**
  across R2 butterflies. One pass vectorized, one scalar → the measured −33% partial win is arithmetic
  (~half the passes). Measured K=1 (from k1_single_transform.md): scalar tier 1.86/10.19/54.74 µs
  (N=256/1024/4096); BAILEY2 1.24/6.84/53.75 µs (−33%/−33%/−2%); K=8÷8 ceiling 0.74/3.75/20.72 µs.
  Lane-padding K=1→8 is **dead** (+206…+287% — costs a full 8-lane batch).

## 8. The build — a row-major planner (not new codelets)

The genuinely new artifact is a **row-major planner**, layout-parametric, that drives the existing
codelets with butterfly-axis geometry:

1. **Butterfly-axis factorization** — chain radices so every stage's vectorized count `m = N/r ≥ VW`;
   isolate the `m < VW` region into a tail.
2. **k-varying FLAT twiddle tables** — fill `W[(j-1)*m + k]` with the *k-varying* per-butterfly
   twiddle (not the col-major lane-constant fill). **Only FLAT-family codelets transfer** — T1S /
   scalar-broadcast variants assume lane-constant twiddles and are structurally invalid row-major
   (same geometric law as [strided_twiddle_variants.md](strided_twiddle_variants.md): the admissible
   twiddle-application set depends on which axis SIMD runs along vs the twiddle index).
3. **Inter-stage transpose — scale it.** Fuse into stores when cache-resident (BAILEY2's trick), tiled
   separate pass when not (MKL's choice past L2). Do not lock into fusion, or we die at large N.
4. **Layout-parametric** — IL codelets for the IL path (leaf **deinterleaves** into the compute, à la
   FFTW genfft, so an interleaved caller pays no separate conversion pass), split codelets for the
   split path. Generalize BAILEY2's `t1p` to both.
5. **Arbitrary N** — mixed radix, and route primes to the existing Rader/Bluestein. Not just √N×√N.
6. `t1_strided` (a twiddle-applying strided stage codelet) is only needed if a strided intermediate
   beats the fused/tiled transpose — build on demand, hand-reference-first (§6a36 discipline).

North star = **FFTW's genfft shape** (general, split-internal-with-leaf-deinterleave OR IL-native,
per-stage codelets, any N) — which our DAG compiler already *is* — not a narrow BAILEY2 special case.

## 9. Guardrails (so a future session doesn't relearn §5 the hard way)

- **Benchmark row-major vs MKL-INTERLEAVED** (the fast path, §4), not vs MKL-split. No phantom wins.
- **IL first.** It's the user default and MKL's fast layout. Split is a measure-it hypothesis (§6).
- **Handle interleaved input natively** (leaf-deinterleave) — do not force split-only, or we lose to
  MKL on the conversion pass for the 90% of callers with interleaved data.
- **Scale the transpose** (fused small, tiled large). BAILEY2's N=4096 −2% wash is the fusion ceiling.
- **Arbitrary factorization**, prime-safe.
- Correctness gate: fwd vs naive DFT + roundtrip, **det AND rand input** (symmetric probes mask
  bugs — the July-6 DIF K-split lesson), both directions, both layouts.

## 10. Reproduce the MKL trap (tooling)

```
# interleaved probe: DftiCreateDescriptor(DFTI_COMPLEX,1,N); DftiCommit; ComputeForward x3
# split probe: + DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL)
gcc -O2 probe.c -I"$MKLROOT/include" -L"$MKLROOT/lib" -lmkl_rt -o probe.exe
MKL_THREADING_LAYER=SEQUENTIAL gdb -batch \
  -ex 'b DftiComputeForward' -ex run -ex c \        # call #2 = steady state
  -ex 'watch *(double *)$rdx' -ex c \               # trap first output store = the kernel
  -ex 'disassemble $pc-1600,$pc+300'
```

Probe sources kept in-repo: `build_tuned/benches/mkl_probes/` — `mkl_k1_probe.c` (IL),
`mkl_k1_split.c` (split), `mkl_k1_time.c` (IL-vs-split timing), + README with the exact gdb recipe.

## 11. THE BAKEOFF — both methods built from our codelets and raced (2026-07-22, i9)

Per the decide-by-measurement directive: both candidate K=1 architectures were built and raced
before committing to either. Spike: `build_tuned/benches/k1_fourstep_spike.c`
(`python build.py --src benches/k1_fourstep_spike.c --compile`; MKL columns from
`benches/mkl_probes/mkl_k1_time.c`). Same machine, same core (pin 2), same hot-loop best-of-N
methodology both sides. **All numbers below are the i9 re-baseline — they supersede the §7
container-era µs figures for this host.**

### 11a. The two arms — zero new production codelets

- **Arm A (ours, "four-step over the batch engine")**: the batch identity made literal. Stage 1 =
  ONE call of the *same* `n1_oop(R2)` codelet BAILEY2 already uses, but at `count=R1` (column c IS
  lane c — the identity from §7), OOP `x→s`. Then an explicit SIMD 4×4 transpose `s→d`. Then
  **verbatim** BAILEY2 stage 2 (`t1` + its own `Qr/Qi`; at K=1 the per-lane table
  `Qr[(l2-1)·R2+k2] = W_N^{l2·k2}` IS the four-step diagonal). Because stage 2 and the leaf are
  *shared with BAILEY2*, the A/B isolates exactly one variable: vectorized-stage-1+lump-transpose
  vs scalar-stage-1-with-fused-stores. **Arm A's output is bit-identical to BAILEY2's
  (cross-diff 0.0)** — same codelet DAG, same rounding.
- **Arm B ("MKL's method", fused mono)**: hand-written in-register 8×8 four-step at N=64, split
  layout — per t-half: 8 vector loads → DFT8 across elements → four-step twiddle cmul (k-varying
  vector constants); 4×4 *register* transposes (the in-register reshaping of §3a, paid once, not
  smeared); DFT8 across columns per m-half; contiguous natural-order stores. Zero re/im shuffles
  (split — §3b's advantage kept). ~110 lines of intrinsics. This is the §6a36-discipline hand
  reference for a future OCaml `--k1` emission mode.

Gates: fwd vs naive O(N²) DFT, natural order, det AND rand inputs — **all green** (1e-15…1e-12,
every pair, every N, both arms + mono + LEAF).

### 11b. Results (ns, best pair per arm, hot-loop; MKL best-of-9 same pinning)

| N | MKL-IL | MKL-split | **Arm A** (pair) | A / MKL-IL | A / MKL-split | BAILEY2 | scalar tier | old route |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 30.2 | 33.9 | **45** (8×8) — **mono 32** | 0.67× (**mono 0.94×**) | **mono WIN 1.06×** | 86 | 155 | LEAF 164 |
| 128 | 68.6 | 127.7 | **93** (8×16) | 0.74× | **WIN 1.37×** | 191 | 446 | LEAF 696 |
| 256 | 136.4 | 194.4 | **172** (4×64) | 0.79× | **WIN 1.13×** | 382 | 799 | B2 382 |
| 512 | 290.1 | 337.9 | **393** (8×64) | 0.74× | 0.86× | 866 | 1602 | |
| 1024 | 750.1 | 780.5 | **1248** (64×16) | 0.60× | 0.63× | 2442 | 3585 | |
| 2048 | 2109.6 | 2439.7 | **3326** (64×32) | 0.63× | 0.73× | 7067 | 7897 | |
| 4096 | 3979.5 | 5025.7 | **7250** (64×64) | 0.55× | 0.69× | 16220 | 18430 | |
| 8192 | 8739.5 | 10815.0 | **26759** (64×128) | 0.33× | 0.40× | 56824 | 63518 | |

### 11c. What the data says

1. **Arm A halves BAILEY2 at every size** (A/B2 = 0.45–0.52, all pairs, all N). Vectorizing the
   column pass pays for the explicit transpose several times over; the transpose itself is cheap
   (11 ns at N=64, ~1.3 µs at N=4096, roughly flat per byte).
2. **The mono tier is real and we can build it**: mono-64 = 32 ns vs MKL-IL 30.2 (0.94×), and it
   BEATS MKL's own split kernel (33.9). At N=64 the fused mono beats Arm A 1.4× — in-register
   reshaping wins exactly where the whole transform fits in registers, as §5 predicted.
3. **Like-for-like layout (split vs split): we WIN through N≈256** (1.37× at 128, 1.13× at 256)
   and hold 0.86× at 512 — with zero new codelets. The remaining gap vs MKL-IL at these sizes is
   substantially the *layout tax* (their one interleaved stream vs our 4–6 split streams, §4),
   which the existing `codelets/il/` family is designed to remove.
4. **Optimal pair drifts with N** (4×64 at 256 → 64×16 at 1024 → 64×64 at 4096): fat leaves win
   when t1's table stream is small; balanced pairs win as R1 grows. Per-cell wisdom calibration,
   as everywhere else in the engine.
5. **N≥8192 is the two-level ceiling** (0.33×): the per-lane t1 table alone streams
   (R1−1)·R2·16 B ≈ 129 KB *per execute* at 64×128, the split-OOP working set (~3 buffer pairs)
   leaves L2, and the unblocked 4×4 transpose thrashes. MKL recurses (§3c). Fixes are known, not
   mysterious: level-3 recursion (row pass = another four-step), tiled transpose, and/or
   twiddle recompute-in-register ([strided_twiddle_variants.md](strided_twiddle_variants.md) §5).

### 11d. VERDICT — commit direction (pending user review of this doc)

**Both methods enter the plan, tiered exactly like MKL tiers its own:** fused mono (Arm B shape)
for N ≤ ~128; four-step-over-batch (Arm A shape) for 256–4096. Neither arm refuted the other —
they own disjoint regimes, and the data drew the boundary at N≈128–256.

Productization order:

1. **P1 — BAILEY2V**: generalize the BAILEY2 executor to the Arm A shape (vectorized stage 1 +
   SIMD transpose; stage 2 unchanged), per-cell (R1,R2) from OOP wisdom, old path as kill-switch.
   Bit-identical by construction (§11a). Immediate ~2× on the routed K=1 answer.
2. **P2 — IL-native Arm A** (the flagship): drive the `il/` codelet family through the same
   4-step so an interleaved caller pays no conversion — attacks the 0.55–0.79× vs MKL-IL head-on.
3. **P3 — OCaml `--k1` mono emission** for N ∈ {16,32,64,128}, modeled on the validated mono-64
   hand reference (both layouts; IL variant pays §3a's shuffle tax, split variant doesn't).
4. **P4 — large-N (≥8192)**: level-3 recursion + tiled transpose + twiddle recompute. Separate
   workstream; do not let it block P1–P3.

Guardrail unchanged: the bar is **MKL-interleaved** (§9). Split-vs-split wins are reported as
secondary evidence only.

## See also
- [k1_single_transform.md](../performance/k1_single_transform.md) — the K=1 gap + BAILEY2 record.
- [strided_twiddle_variants.md](strided_twiddle_variants.md) — the twiddle-geometry law (§8.2).
- [mkl_geometry_contracts.md](../performance/mkl_geometry_contracts.md) — layout contracts / blind spot.
- vectorfft_feature_ledger.md §6a16–§6a53 — the IL campaign (col-major × IL).
