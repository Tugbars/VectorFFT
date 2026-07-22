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

**⚠ 2026-07-22 REVISION (deeper census, §12.1): the mid-N structure is exactly TWO passes, and
the "transpose-dominated region" reading above was wrong.** N=256/512/1024 all run: pass 1 =
twiddle-FREE radix-16/32 column kernel, user buffer → **stack-resident** scratch (~24KB frame);
pass 2 = **streamed-twiddle** radix-16/32 kernel reading scratch linearly and writing the user
buffer in R strided N/R **sections** — natural order fused into the final stores, NO separate
transpose/permutation pass, every output written exactly once. The mono tier stops at 64.

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

## 11e. WHY WE LOSE — attributed by experiment (2026-07-22, same session as MKL columns)

Three additions to the spike decomposed the gap: an **in-place** split arm (2 buffers, matching
MKL's DFTI_INPLACE config), a **native-IL** arm (z→z through the *unwired*
`radixN_n1_oop_fwd_avx2_UG_UG_il_in` codelets — il_derive.py twins of the split leaves that
already sit in the codelet lib with the same free-stride 11-arg ABI — plus a v1 exit
interleave-sweep), and an **IL mono-64**. All gate green det+rand (the deriver's known broken
tail never runs: our `me=R1` is always %4==0).

| N | MKL-IL | MKL-split | A-split-ip | A-IL-v1 (sweep) | notes |
|--:|--:|--:|--:|--:|---|
| 64 | 30.2 | 33.7 | 44 | 60 — **mono-IL 44** | mono-split 33 |
| 256 | 137.1 | 195.7 | **176** (4×64) | 251 | beats MKL-split 1.11× |
| 1024 | 753.3 | 776.4 | **1113** (64×16) | 1498 | in-place −15% vs OOP |
| 4096 | 4078.7 | 4932.8 | 7161 (64×64) | 7700 | in-place ≈ OOP (L2-bound) |

Attribution, in order of measured size:

1. **Input-side IL is FREE today.** A-IL-v1 ≈ A-split-ip + exit-sweep cost, exactly: the
   `il_in` leaf's in-register deinterleave hides completely under the column pass. The entire
   v1 IL premium is the un-fused OUTPUT sweep (+75 ns at 256, +385 at 1024) — removed by
   emitting a **`t1_oop` il_out twin** (assessed as recombination-only: the permute lattice
   exists in `emit_c.ml:1584-1663`/`emit_render.ml:86-131`; the OOP fold is local to
   `emit_load/store_unitgroup`, simpler than the ip path — touch `codelet_oop.ml` edge kind +
   `gen_main.ml` flags + registration).
2. **The t1 twiddle stream is structural**: `tw[(l2-1)·me+b]` is loadu-streamed →
   (R1−1)·R2·16 B/execute ≈ N·16 B — a table stream as large as the dataset itself, every
   execute ([radix16_t1_oop_avx2.c:116](../../src/dag-fft-compiler/codelets/oop/avx2/radix16_t1_oop_avx2.c#L116)).
   This is the dominant reason we trail MKL-split at N≥1024 *in the same layout*. Fixes:
   recompute-in-register (strided_twiddle_variants §5) or deeper factorization (P4 recursion).
3. **Fat-leaf spill traffic**: the radix64 `n1_oop` leaf is a BLOCKED two-pass body with
   `__m256d spill_re[64]` — 128+128 spill stores/reloads per iteration on top of 128+128 data
   moves. The leaf pass costs 2.9–5.2 µs at N=4096 partly for this reason.
4. **Footprint**: in-place (2-buffer) buys 10–15% at N=1024–2048, nothing at ≤512
   (L1-resident either way) or 4096 (L2-bound either way). Real but secondary.
5. **MKL's IL advantage at 256 is not just layout**: their IL kernel beats their own split by
   1.43× at 256 — the fused-mono tier plausibly extends to 256 in MKL. Matching them there may
   need OUR mono tier at 256 too (the `--k1` emitter, not hand-writing).

Mono-IL at 64: 44 ns vs MKL 30.2 — boundary shuffles cost +11 ns over split-mono 33. The hand
kernel uses no FMA in its cmuls yet; emitter-scheduled `--k1` codelets are the path to parity,
not further hand-tuning.

**Priority consequence**: P2 (t1_oop il_out twin → true IL-native four-step at split cost) and
P3 (--k1 mono tier, IL-first) are confirmed as the two build items; P4 (t1 table
recompute/recursion) is what's left of the MKL-split gap at large N.

## 11f. P2 CODELET LAYER SHIPPED — native IL emission on the OOP family (2026-07-22)

The emitter now has `--oop-il-in` / `--oop-il-out` (avx2; avx512 masked lattice explicitly
gated pending). Implementation: two mode refs + an IL branch in
`emit_load_unitgroup`/`emit_store_unitgroup` (`codelet_oop.ml`), signature swap to
(z, unused) keeping the 11-arg `vfft_oop11_fn` shape, flags + validation + `_il_in`/`_il_out`
name suffix in `gen_main.ml`. Because the tail passes re-enter the edge emitters at narrower
widths, the emitted twins are IL-correct at ALL widths — vector (unpack+perm lattice), SSE2
(`_mm_unpacklo/hi`), scalar (`z[2k]/z[2k+1]`) — verified in the generated C. This
definitively retires the il_derive.py mechanism: all 14 derived `radixN_n1_oop_il_*` files
(both ISAs) are DELETED; emitted `radix{4,8,16,32,64}_{n1_oop_il_in,t1_oop_il_out}_avx2.c`
live in `codelets/oop/avx2/` (lib now 717). Regen recipe:
`gen_radix.exe R --oop --oop-buffer-oop --oop-load UG --oop-store UG --isa avx2
[--twiddled] --oop-il-in|--oop-il-out --emit-c` (WSL, opam 5.2.0, DUNE_CACHE=disabled,
targeted `dune build bin/gen_radix.exe`).

Spike v2 (z→z, 3 passes: il_in leaf → transpose → t1_il_out; NO sweep) — all gates green
det+rand; within-session numbers (session runs ~10% warm vs §11e; MKL drifted up equally, so
compare ratios within-session only — machine lockdown for paper numbers):

| N | split-ip | IL v1 (sweep) | **IL v2 (twin)** | v2 premium over split |
|--:|--:|--:|--:|--:|
| 64 | 46 | 61 | **54** | +17% |
| 256 | 210 (32×8) | 276 | **254** (32×8) | +21% — but only **+6 ns at 4×64** |
| 1024 | 1368 (32×32) | 1697 | **1544** (32×32) | +13% |
| 4096 | 9456/10380 (64×64) | 10242 | **9688** (64×64) | ~+3% |

The residual IL premium = boundary shuffles, and it scales with the number of t1 store sites
(R1·R2/VW): fat-R2 pairs minimize it (4×64 at N=256: IL v2 ≈ split + 6 ns). So the IL pair
optimum can differ from the split pair optimum — one more reason the per-cell pair wisdom
must calibrate the layout axis jointly.

**P2 status**: codelet layer DONE (emit path + 10 twins + spike gates + bench). Remaining for
production: an OOP plan kind that drives the IL twins (BAILEY2V-IL), per-cell (pair × layout)
wisdom, and bwd (emit an il_out_sw (im,re)-swapped store variant — one more flag on the same
lattice — since the z-buffer swap identity can't swap pointer args). Then P3 (--k1 mono).

### 11f-2. Plan layer SHIPPED (same day)

- **Emitter**: `--oop-il-in-sw` / `--oop-il-out-sw` added (the (im,re)-swapped lattices; fwd
  DAG, fwd-named symbols per the established convention). 20 twins total in
  `codelets/oop/avx2/` (`radix{4..64}_{n1_oop_il_in[,_sw], t1_oop_il_out[,_sw]}_avx2.c`);
  codelet lib 727.
- **Registry** ([oop_leaf_registry.h](../../src/core/oop/oop_leaf_registry.h)):
  `vfft_oop_leaf_il_fn(R, sw)` / `vfft_oop_t1_il_fn(R, sw)`, avx2-guarded (return 0 on
  avx512 builds → callers degrade to split).
- **Plan kind** ([oop_plan.h](../../src/core/oop/oop_plan.h)): `VFFT_OOP_KIND_BAILEY2V` +
  `vfft_oop_plan_create_k1(N, R1, R2)` (K=1, R1/R2 % 4, reuses `_vfft_oop_fill_bailey`
  tables verbatim) + `_vfft_k1_transpose` + execute cases: split fwd (in the main switch —
  split bwd rides the existing swap-identity path untouched), and
  `vfft_oop_execute_fwd_il / _bwd_il(p, z_in, z_out)` (z→z, z_in==z_out safe; bwd = both
  swaps folded into the _sw lattices, output in normal (re,im) order).
- **Gates** (`build_tuned/test/test_k1_fourstep.c`): 11 cells × det+rnd, N=64..4096 —
  split fwd vs naive ✓, **cross vs BAILEY2 = 0.0 bit-identical on every row** ✓, split
  roundtrip ✓, IL fwd ✓ (same errors as split — same arithmetic), IL roundtrip via _sw ✓,
  R2=128 degrades to split-only without failure ✓. ALL GREEN.

**Still open on P2**: vfft.c front-door routing (K=1 → BAILEY2V; the interleaved buffer
contract `sim==dim==NULL` → the *_il entry points) + per-cell (pair × layout) wisdom/tuner.
Then P3 (--k1 mono emission, hand mono-64 as reference).

### 11g. Plan-API bench vs MKL (isolated cells, cooled, 2026-07-22 evening)

`build_tuned/benches/bench_k1_vs_mkl.c` (--mkl, same-process order-flipped best-of-5;
in-place arm = `execute_fwd(p, w, w)` — d==x is safe, the leaf drains x into scratch first):

| N | MKL-IL | MKL-sp | B2V-sp | B2V-ip | B2V-il | mono-sp | mono-il | best-sp/MKL-sp |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 31.3 | 35.4 | 43 | 42 | 53 | **30** | 44 | **1.18× (mono)** |
| 256 | 136.4 | 192.9 | 205 | 209 | 261 | | | 0.94× |
| 1024 | 874.4 | 802.7 | 1574* | **1319** (64×16) | 1828* | | | 0.61× |
| 4096 | 4077.0 | 4966.4 | 10305* | 9967* | 11363* | | | ~0.50×* |

`*` = pair pick order-biased (see below). Three lessons:

1. **Plan API adds no overhead** — B2V numbers track the spike's raw-call arm within noise;
   **mono-64-split = 30 ns is AT PARITY with MKL-IL (31.3)** and 1.18× over MKL-split. The
   P3 mono tier is fully justified as the small-N production route.
2. **The bench's inline pair sweep reproduces the cmp_old_new fixed-order bias**: candidates
   are tried R2-descending, first-tried runs coolest, so fat-R2 pairs win ties spuriously
   (16×64 / 32×128 picked over the spike-proven 64×16 / 64×64). Production pair choice =
   CALIBRATOR (per-candidate isolated, order-neutralized) writing per-cell
   **(pair × placement × layout)** wisdom — the axes are coupled (§11f), one joint search.
3. **Stability is itself diagnostic**: MKL holds ±3% across sessions; our mid-N numbers swing
   ±25% (172→205 at 256, 1113→1319 at 1024-ip across the day). Their compact working set is
   frequency-insensitive; our streamed t1 tables + multi-buffer footprint ride the L2. The P4
   structural fixes are also stability fixes. Paper numbers need machine lockdown.

Large-N addendum (planner-shaped, per user review comment): the column pass need not be a
monolithic leaf — letting it be a COMPOSED batch plan (DP planner domain, e.g. [16,8] chain at
K=64 for N=8192) decouples the pair space from the leaf ceiling and is the natural level-3
recursion (P4).

## 12. GAP-CLOSING RESEARCH (2026-07-22, five threads + probes) — the plan of record

### 12.1 MKL mid-N ground truth (re-disassembled; supersedes the §3c "four-step" reading)

N=256 (16×16), 512 (32×16), 1024 (32×32) are all **exactly TWO passes**: (1) twiddle-free
radix-16/32 column kernel, user→**stack scratch** (inside a ~24KB aligned frame); (2)
streamed-twiddle kernel, scratch→user, whose stores go to R strided N/R **sections** — the
reshaping rides the final stores; there is NO transpose pass and no third sweep. Twiddles are
**streamed, not recomputed** (~1:1 load:mult; one linear `%r8` cursor, 0x40 B/complex
splat-duplicated pairs — FFTW's "t2fv storage #2" layout). Byte roofline at 1024: **4 sweeps
× 16KB + one linear table stream.** Our 3-pass four-step moves 6 sweeps + 63 parallel strided
table rows. Mono tier confirmed to stop at N=64. (gdb recipe bugs found and fixed in the
probes README: `watch -l` required, `--args` required — without it gdb silently ran N=64 for
every "size".)

### 12.2 t1 decomposition probe (`build_tuned/benches/k1_t1_decomp.c`, measured)

t1 (twiddled combine) vs n1 (identical DFT, zero twiddles), same shapes:
64×16: 853 vs 640 (tw = 212, 25%) · 32×32: 529 vs 495 (tw ≈ 0) · 64×64: 4422 vs 3726
(tw = 696, 16%) · 64×128: 10666 vs 9066 (tw = 1600, 15%). **Twiddle streaming costs
15–29% of t1 at R1=64 and ~nothing at R1≤32; the dominant cost is the combine compute +
its doc-58 pass-boundary spills.** Note 32×32's total t1 (529) beats 64×16's (853): structure
and pair choice dominate twiddles.

### 12.3 FFTW ground truth (local fftw-3.3.10 source)

No execute-time recurrence anywhere — tables built per-entry at plan time in long-double with
octant reduction (accuracy-first; kernel/trig.c). The transferable mechanism is **t3-class
twiddle-log3**: load only {w¹,w³,w⁹,w²⁷} per row and reconstruct the rest via ≤2-term
in-register complex products (genfft `-twiddle-log3 -precompute-twiddles`) — 7.75× twiddle-byte
cut at radix-32; N=1024 total twiddle footprint 2KB vs ~16KB t1-style. Leg-axis products of
loaded b-vectors stay b-vectors, so this DOES transfer to the four-step t1 (unlike our
lane-engine LOG3/cf0). FFTW selects t1/t2/t3 per cell by measurement — same thesis as ours.

### 12.4 Ranked plan (expected gain at the 1024/4096 cells; every item gated per-cell)

1. **TWO-PASS restructure** — fuse the transpose, matching MKL's shape. Two routes, race
   them: (a) t1 with a **UnitLeg load edge** (4×4 in-register transpose in the load preamble
   — un-stub the M2-phase-2 extraction, `emit_c.ml:3310-3330`; the lattice exists inlined at
   610-807); (b) leaf with **transposed stores**. Removes the transpose sweep (180 ns@1024,
   1.3 µs@4096), one scratch buffer, and one L1/L2 round-trip → byte-roofline parity with
   MKL's 4-sweep structure. Est. 1319→~1050 at 1024.
2. **Calibrated structure choice** (pair × placement × layout wisdom, isolated methodology) —
   the probe shows R1≤32 combines are disproportionately cheaper (t1 529 vs 853 at 1024);
   the mispicks cost 10–20% today.
3. **--k1 mono tier for N≤128** — scoped at ~500–650 LoC (thread report: new driver module
   composing existing UG edges + blocked-R16 bodies + T4 register-transpose printer + rodata
   twiddle tables; milestones M1=N-64-parity-vs-hand-spike first). MKL two-passes at 128
   (68.5 ns) — a mono-128 plausibly WINS outright. Do NOT build on dft_expand_twidsq
   (batch-vectorized → scalar at K=1); do NOT add lane-varying const IR (rodata tables).
4. **Twiddle-stream variants for R1=64 shapes only** (the measured 15–29%): try in order
   (a) **linear table layout** — one cursor in codelet consumption order instead of 63
   parallel strided rows (fill + render tweak, cheapest, it's what MKL does); (b) **FFTW-t3
   log3 legs** (63→~5 loads); (c) **factored b-axis anchors** (t1p addressing exists,
   emit_render.ml:166-169). All become measured per-cell variants per the 3-twiddle-methods
   thesis. **Pure runtime recurrence: rejected** (126 ymm loop state → stack; breaks
   bit-exact gates; FFTW's documented accuracy stance).
5. **Composed column pass** (P4): runnable today — `vfft_proto_execute_fwd_oop` on a
   `plan_create(R2, R1, factors)` DIT plan + a **permuted** `_vfft_k1_transpose` absorbing the
   plan-time digit-reversal. Kills the leaf ceiling (N≥8192) and the 128+128 spill wall
   (structural per oop_stride_specialization.md — emitter knobs measured useless there).
6. **Bonus for the natural-order workstream**: MKL's sectioned-scatter final pass is the
   "fused scatter" made viable by scratch-OOP piping — feeds natural_order_inplace Phase-1
   (the SCRATCH mode's separate scatter sweep may be removable the same way).

### 12.5 Item 1 SHIPPED — two-pass restructure (same day)

Native **UnitLeg edges** in `codelet_oop.ml` (self-contained lattices, widths 4/2/1; the
`emit_c` M2-phase-2 stubs bypassed; UL+il / UL+post-tw combos failwith-guarded). 10 UL twins
emitted (`radix{4..64}_{t1_oop_ul, n1_oop_ugul}_avx2.c` — 3× smaller than flat siblings: UL
configs carry no rem-tail re-renders, `me%4` contract). Registry `vfft_oop_t1_ul_fn` /
`vfft_oop_leaf_ugul_fn` (avx2-guarded). Plan entry points `vfft_oop_execute_fwd_2pa`
(leaf-UG → t1-UL, transpose in the t1's loads) and `_2pb` (leaf-UG_UL transposed stores →
flat t1); 2 passes, 1 scratch pair, dst==src safe, **Qr/Qi unchanged** (group↔leg relabel).
Gates: **bit-identical (0.0) to the 3-pass path on every cell, det+rand**; 8×128 = 2pa-only
(no UL-leaf 128) as designed.

Isolated cooled bench (best two-pass route per cell, in-place):

| N | MKL-IL | MKL-sp | 3-pass ip | **best 2-pass** | Δ vs 3-pass | vs MKL-sp |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 31.4 | 35.7 | 42 | **37** (2pa 8×8) | −12% | 1.17× WIN (mono 30: 1.19×) |
| 256 | 136.0 | 200.0 | 207 | **174** (2pa 64×4) | −16% | **1.15× WIN** |
| 512 | 297.3 | 336.9 | 406 | **375** (2pb 8×64) | −8% | 0.90× |
| 1024 | 858.6 | 740.1 | 1421 | **1193** (2pb 32×32) | −16% | 0.62× |
| 2048 | 2067.3 | 2372.6 | 4528 | **3939** (2pb 64×32) | −13% | 0.60× |
| 4096 | 4121.1 | 5002.3 | 10111 | **8584** (2pa 64×64) | −15% | 0.58× |
| 8192 (hot run) | 8989 | 10896 | 28314 | **21868** (2pa 64×128) | −23% | 0.50× |

Two-pass wins **uniformly** (−8…−23%); routes trade wins per cell (2pa fat-leaf small-N,
2pb mid-N) → one more axis for the joint calibrator. Sanity: 2pb@32×32/1024 ≈ n1(32) 495 +
t1(32×32) 529 from the §12.2 probe + overhead ≈ 1193 ✓ — the pipeline is now the sum of its
kernels, no structural slack. Remaining mid-N gap vs MKL-sp (1.6× at 1024) is now PURELY
per-kernel quality: their straight-line radix-32 kernels vs our doc-58 spill-blocked bodies +
their linear twiddle cursor vs our 63 strided rows — i.e. §12.4 items 3 (mono ≤256), 4a
(linear twiddle layout), 5 (composed/leaner column kernels).

### 12.6 Item 4a MEASURED — linear twiddle layout is MARGINAL; banked as a calibrator variant

`--oop-tw-linear` emitter mode built (ref in emit_state, linear branch in the Twiddle render:
`tw_re[b*NLEGS + j*VW]`, one contiguous (R1−1)-vector block per quad; UL-only by validation —
no rem tail exists to mis-index the layout). 5 `_twl` twins + `Qlr/Qli` consumption-order
tables + `vfft_oop_execute_fwd_2pa_twl`. Gates bit-identical (0.0 — same values, different
order). **Measured (isolated, cooled): twl vs 2pa on the same pair = −5% (64) / +7% (256) /
−1% (512) / −4% (1024) / +13% (2048) / −1% (4096) — noise-level, pair-dependent.** The
scheduler emits twiddle loads in DAG order (not leg-ascending), so the layout is
quad-block-local rather than truly sequential; block locality evidently wasn't the
bottleneck — consistent with §12.2 (twiddle share ≈ 0 at R1≤32, and the best pairs mostly
avoid R1=64). VERDICT: keep `_twl` as a per-cell calibrator variant (free, bit-identical);
do NOT default it; do NOT build the log3/factored variants unless VTune later shows the
anchor stream at R1=64 pairs actually evicting L1. The 3-twiddle-methods thesis extends:
measured selection, not doctrine.

Best-of-today ladder after two-pass (this cooler evening session, isolated):
64 = **29** (mono, 1.22× MKL-sp) · 256 = **154** (2pa 4×64, **1.25× MKL-sp**, 0.89× MKL-IL)
· 512 = **360** (0.93×) · 1024 = **1016** (2pb 16×64, 0.75×) · 2048 = **3311** (2pa 64×32,
0.75×) · 4096 = **7220** (2pb 64×64, 0.68×). Day trajectory at 1024: 1574 → 1319 → 1193 →
**1016**. Next lever by size: --k1 mono (≤256, targets MKL-IL), leaner column kernels /
composed pass (≥1024).

### 12.7 Item 3 M1 SHIPPED — `--k1-mono` emitter, N=64 at MKL-IL parity

`Codelet_oop.emit_k1_mono` (~180 LoC driver, end of codelet_oop.ml): the whole K=1 four-step
as ONE emitted function — per column-chunk h [UG load (shadowed base ptr + `b=0` + baked
stride consts, existing edge emitter verbatim) → radix-R2 monolithic body (same
`prepare_butterfly` DAG as the OOP family, block-scoped per instantiation) → four-step
twiddle cmul against **emit-time rodata tables** (no runtime Qr/Qi, FMA cmuls) → park in
function-scope U vars] → per row-chunk [4×4 register transpose U→lanes → radix-R1 body → UG
store, natural order]. `gen_radix.exe 64 --k1-mono --isa avx2 --emit-c` →
`vfft_k1_mono64_fwd_avx2` (uniform 11-arg ABI, extra args ignored), 699 lines.
Registry: `vfft_k1_mono_fn(N)`.

Gates: det 1.07e-14 / rnd 2.55e-15 vs naive. **Parity race (isolated, cooled): emitted 31 ns
vs hand spike 29 vs MKL-IL 30.4** — the driver reproduces the hand kernel within noise on its
first measured run. The M1 milestone discipline (validate the driver on the size with a
hand-written oracle before touching register-pressure-heavy sizes) paid exactly as intended.

Remaining mono milestones: M2 = IL variants (compose the existing il_in/il_out lattices into
the driver's edges); M3 = N=128 (16×8) / 256 (16×16) — lifts the M1 N-restriction, radix-16
stages route through the validated blocked-body path, intermediate becomes a 2–4KB stack
array; M3 is where the 256 cell (today 154 ns four-step vs MKL-IL 136) gets its shot at
IL-parity. M4 = bwd (conjugated tables + _sw IL twins).

### 12.8 M3 MEASURED — the mono tier's boundary is N=64, same place MKL draws it

M3 generalized the driver to any (R1,R2) (`--k1-r1` override; 128 emitted BOTH pairs, 256 as
16×16; monolithic-rendered radix-16 bodies first cut). All gate green (1e-14, incl. the 256).
Race (isolated, cooled): **64 = 30 ns (parity reconfirmed)** · 128: monos 98/104 vs our OWN
two-pass 86 (2pb 4×32) vs MKL-IL 67.3 · 256: mono **272** vs four-step 175 vs MKL-IL 135.9.

**VERDICT — mono tier REFUTED above 64, banked at 64.** The 128 result is the clean proof:
a mono has zero pass overhead and still loses to the pass-structured route, so the loss is
pure register economics (8+ radix-16 bodies + N-sized function-scope state on 16 ymm = spill
storm; doc-58 predicted monolithic-R16's ~50% penalty). This empirically confirms MKL's own
tier schedule (§12.1: their mono stops at 64 too) on OUR codelets — the fence §5 warned
about, measured from the inside. A blocked-body mono-256 might close part of 272→~200 but
cannot beat the 175 two-pass; not worth the emitter surface. M2/M4 (IL/bwd mono variants)
now scope to N=64 ONLY.

**Where the K=1 campaign therefore stands / what remains**: tiering = mono-64 (30 ns, MKL-IL
parity, emitted) + two-pass four-step everywhere else. The ENTIRE remaining gap vs MKL-IL
(67 vs 86 at 128; 136 vs 175 at 256; 0.6-0.75× at ≥1024) is per-kernel quality of the
two-pass kernels: MKL's straight-line twiddle-free column kernels (FNA/FNB, no spill arrays)
and streamed sectioned-store twiddle kernels (W1/W2) vs our doc-58 blocked leaves and t1
combines. That is the leaner-column-kernels workstream (§12.4 item 5) + the joint calibrator
— no structural moves left to make.

## 13. PRODUCTIZATION (step 1+2 of 4 DONE, 2026-07-22 late)

**Step 1 — mono-64 tier completed (M2+M4)**: `--k1-il` / `--k1-sw` driver modes emit
`vfft_k1_mono64_8x8_il_{fwd,bwd}_avx2` (bwd = fwd DAG + _sw lattices, forward tables — the
swap identity folded into the boundaries; split bwd = caller pointer-swap, zero code). Gates:
IL fwd 1.07e-14, IL roundtrip 2.84e-14, split-bwd 4.35e-14.

**Bonus — TRUE 2-pass IL route**: the composition [il_in leaf → t1 UL-load + il_out store]
was already emittable (validation matrix allowed UL load + il_out store in one codelet — no
new emitter code). 10 twins (`radix{4..64}_t1_oop_ul_ilout[_sw]_avx2.c`) +
`vfft_oop_execute_{fwd,bwd}_2p_il` (z→z, 2 passes, 1 scratch pair, ZERO conversion/transpose
sweeps — the full MKL two-pass shape on an interleaved buffer, both directions). All gates
green.

**Step 2 — the four-axis calibrator** (`build_tuned/benches/calibrate_k1.c`): per-cell
process, candidates = route {3p, 3p-ip, 2pa-ip, 2pb-ip, twl-ip, 3p-il, 2p-il, mono[-alt,-il]}
× pair, per-trial candidate-order ROTATION (kills the cmp_old_new fixed-order bias by
construction), best-of-4, verdicts per axis (split-oop / split-ip / il). Ladder verdicts
(`benches/k1_calibration_verdicts.csv`):

| N | split-ip winner | il winner |
|--:|---|---|
| 64 | twl 8×8 · 37.5 | mono-il · 44.9 |
| 128 | twl 16×8 · 84.9 | 2p-il 16×8 · 107.2 |
| 256 | twl 4×64 · 153.6 | 2p-il 4×64 · 224.1 |
| 512 | 2pa 64×8 · 381.5 | 2p-il 64×8 · 479.7 |
| 1024 | **2pb 16×64 · 958.8** (best-ever) | 2p-il 16×64 · 1341.4 |
| 2048 | 2pb 32×64 · 3660.9 | **2p-il 64×32 · 3511.0 — IL beats split!** |
| 4096 | 2pa 64×64 · 8165.4 | 2p-il 64×64 · 8611.5 |
| 8192 | twl 64×128 · 19650 | — (no 128 IL leaf) |

Findings: (1) **twl wins 4/8 split-ip cells in-context** — the "marginal" §12.6 variant wins
when measured per-cell among mixed candidates, vindicating both the bank-don't-default call
and the measured-selection thesis; (2) **2p-il is within a few % of split everywhere and WINS
outright at 2048** — the IL boundary fold is essentially free in the 2-pass shape; (3) the
in-context mono-64 (40.9) loses to twl-ip (37.5) at 64-split while winning the IL axis —
context-realistic calibration differs from isolated single-arm benches; the front door honors
the calibrator. (4) 1024 day trajectory: 1574 → 1016 → **958.8**.

**Remaining (next session)**: step 3 = wisdom-format extension (K1VERDICT lines into the OOP
wisdom family) + vfft.c front-door routing (split → winner route; `sim==dim==NULL` IL
contract → *_il/mono-il; bwd via swap identity / _sw; old routes = kill-switch); step 4 =
canonical public-API regression ladder + natmt untouched + full-diff handoff for review.

### 13.1 The twiddle-method menu completed (LOG3) + calibration v2

Per user review: the K=1 t1 stage was FLAT-only while the scrambled 1D path mixes
FLAT/T1S/LOG3. Geometry (§strided_twiddle_variants §4): T1S and lane-LOG3 are INADMISSIBLE
here (W varies along the SIMD axis) — but **leg-axis LOG3** (FFTW-t3 class: load base-leg
twiddle VECTORS {w¹,w³,w⁹,…}, derive the rest by vector cmuls — b-vector products stay
b-vectors) transfers. Emitted via the EXISTING `--log3` flag on the t1 path (TP_Log3 was
already the substitution machinery; zero OCaml changes): 10 twins
(`radix{4..64}_t1_oop_{ugug,ul}_log3_avx2.c`), **252→24 twiddle loads at radix-64**, SAME
Qr/Qi tables (sparse subset — pure fn-pointer swap on existing plans; plan fields
t1_l3/t1_ul_l3). Gates green at 1e-14 vs naive (tol-gated: derived twiddles differ from
loaded in the last ulp, like FFTW's t3).

Calibration v2 (menu now FLAT / FLAT-linear(twl) / LOG3 × route × pair × placement ×
layout): log3 wins specific slots by 10–18% (3p-l3@1024 1316 vs 1608; 2pa-l3@4096 7426 vs
8272; 3p-l3@8192 26512 vs 29582) and loses others — a true measured-selection member. v2
split-ip verdicts (cooler run): 64=mono 30.8 · **128=mono-alt 72.3 (!)** · 256=2pa 170.4 ·
512=2pb 359 · 1024=2pb 1030 · 2048=twl 2488 · 4096=2pb 6262 · 8192=2pa 22290.

**⚠ Calibration stability finding**: v1→v2 winners FLIPPED at several cells (64: twl 37.5 →
mono 30.8; 128: 2p-86 → mono-alt 72.3; 2048/4096 absolutes swung ~20%+). Within-run ordering
is fair (rotation); across runs, frequency/thermal state changes the OPTIMA, not just the
numbers (L2-sensitive routes suffer when hot; monos don't). FRONT-DOOR REQUIREMENT: persist
wisdom from the MEDIAN/MAJORITY of ≥3 ladder runs (or a locked-down machine), and prefer an
incumbent unless the challenger wins by a margin (>5%). The mono-128 "refutation" (§12.8)
softens to: mono-128 is within the variance band of the two-pass — let the multi-run
calibrator decide per machine-state.

## See also
- [k1_single_transform.md](../performance/k1_single_transform.md) — the K=1 gap + BAILEY2 record.
- [strided_twiddle_variants.md](strided_twiddle_variants.md) — the twiddle-geometry law (§8.2).
- [mkl_geometry_contracts.md](../performance/mkl_geometry_contracts.md) — layout contracts / blind spot.
- vectorfft_feature_ledger.md §6a16–§6a53 — the IL campaign (col-major × IL).
