# VectorFFT vs FFTW genfft — CSE, Op Counts, and Why Fewer Ops ≠ Faster

> A grounded comparison of VectorFFT's algebraic simplifier
> ([`src/dag-fft-compiler/generator/lib/algsimp.ml`](../../src/dag-fft-compiler/generator/lib/algsimp.ml))
> against FFTW 3.3.10's `genfft/algsimp.ml`. Every number here is measured —
> from FFTW's own generated-codelet op-count headers and from VectorFFT's
> op-count journey docs — not asserted.

Both simplifiers descend from Frigo's genfft (PLDI'99): a DFT becomes a DAG of
adds/muls, and a simplifier collapses the redundancy. They reach **nearly
identical op counts** on the bread-and-butter sizes, by **different mechanisms**,
and — the punchline — where they *do* differ, the op-count winner is often the
runtime *loser*.

---

## TL;DR

- **CSE mechanism differs, results barely do.** FFTW uses a *semantic* CSE (a
  numeric oracle that calls two expressions equal if they evaluate equal at 20
  random points) plus *live network transposition*. VectorFFT uses *deterministic
  structural hash-consing* + explicit algebraic-rewrite passes + direct FMA-intrinsic
  emission.
- **Measured op counts are equal where it matters.** VectorFFT matches FFTW
  **exactly** on pow2 (R=8, R=16, both no-twiddle and twiddle codelets) using only
  generic structural CSE — no oracle, no transposition.
- **The prime "advantage" isn't a CSE advantage.** On R=5/R=7 the pre-hand-code
  gap was 4–6 ops, closed to **exact** by hand-coding the Winograd roots-of-unity
  identities — which is **exactly how FFTW gets them too** (hand-coded in
  `gen_notw`; FFTW's oracle doesn't discover them either).
- **The R=32/64 residual (+14 / +66 ops) is an algorithm choice, not CSE.** It
  comes from split-radix — and split-radix **spills its op-count advantage away**
  (SR-64 = 507 spills vs CT-64 = 307; loses up to 33%).
- **The only pure-CSE edge FFTW's oracle has** (auto-unifying bit-different but
  equal constants) VectorFFT **matched with a one-line 13-sig-fig rounding fix**,
  recovering 16 / 76 / 249 ops at R=64 / 128 / 256.

**Bottom line:** VectorFFT's generic CSE is op-count-equivalent to FFTW on pow2,
near-exact on primes, and FFTW's only remaining lead traces to a decomposition
(split-radix) that loses on real hardware. *They optimize the op count; VectorFFT
optimizes the machine.*

---

## 1. Two CSE architectures

| Aspect | VectorFFT `algsimp.ml` | FFTW genfft `algsimp.ml` |
|---|---|---|
| **CSE mechanism** | eager **structural hash-consing** at construction | **memoizing state monad** during a simplify pass |
| **Equality test** | structural (tag/pointer, after canonicalization) | **numeric oracle** (20-point random eval) ∨ structural |
| **Hash key** | structural `node_kind` | **value** (`Oracle.hash` = eval at a fixed random oracle) |
| **Finds** | structurally-identical (after canonicalization) | **anything numerically equal**, even different structure |
| **Determinism** | exact, deterministic, reproducible | probabilistic (false-positive prob ≈ 0) |
| **Cost / equality** | O(1) tag compare | ~20 evaluations |
| **Sum node** | binary `NK_Add`/`NK_Sub` (n-ary `NK_Plus` dormant) | native n-ary `Plus of expr list`; **no `Sub`** |
| **FMA** | first-class `NK_Fma` atom + cascade, **on by default** | **no FMA node**; `enable_fma=false` — delegates fusion to the C compiler |
| **Transposition** | implemented but **dormant** | **on by default**; driver loops `algsimp ∘ transpose` to a fixpoint |
| **Numbers** | IEEE double + `%.13e`/`1e-14` canonicalization | exact arbitrary-precision (`number.ml`) |

The substantive difference is the equality test. FFTW's `equalCSE`
(`genfft/algsimp.ml`) is:

```ocaml
let equalCSE a b =
  if (!Magic.randomized_cse) then               (* default: true *)
    (structurallyEqualCSE a b || Oracle.likely_equal a b)
  else
    structurallyEqualCSE a b
```

`Oracle.likely_equal` (`genfft/oracle.ml`) evaluates both expressions at 20
random inputs and proclaims them equal if all agree within `1e-8`. Because two
distinct linear functions agree at a random point with probability 0, this
reliably unifies expressions that are *algebraically* equal but not *syntactically*
identical. VectorFFT's `hashcons` instead unifies only structurally-identical
nodes (after tag-sorting, sign-hoisting, flatten/cancel canonicalization), in O(1).

See the full mechanism write-up in
[`algsimp.md`](../../src/dag-fft-compiler/docs/compiler_internals/algsimp.md).

---

## 2. Measured op counts (the apples-to-apples test)

No-twiddle complex DFT codelets (FFTW `n1_N`), FMA-branch op-count totals, from
FFTW's generated-codelet headers and VectorFFT's
[`62_winograd_5_and_7.md`](../../src/dag-fft-compiler/docs/62_winograd_5_and_7.md)
/ [`58_op_count_journey.md`](../../src/dag-fft-compiler/docs/58_op_count_journey.md):

| Radix | VectorFFT | FFTW genfft | Gap | How VectorFFT gets there |
|---|---|---|---|---|
| R=3 | **12** | 12 | **0** | generic CSE, no hand-coding |
| R=5 | 36 → **32** | 32 | +4 → **0** | exact after Winograd hand-code |
| R=7 | 66 → **~60** | 60 | +6 → **~0** | exact after Winograd hand-code |
| R=8 | **exact** | exact | **0** | generic CSE (n1 *and* t1) |
| R=16 | **exact** | exact | **0** | generic CSE (n1 *and* t1) |
| R=32 | 386 / 510 | 372 / 496 | **+14** (3.6%) | residual |
| R=64 | 978 / 1230 | 912 / 1164 | **+66** (7%) | split-radix algorithm, *not* CSE |

(R=32/64 shown as `n1 / t1`. FFTW reports two forms per codelet — a non-FMA
`adds, muls` count and an FMA-branch `adds, 0 muls, fmas` count; the table uses
the FMA-branch totals both emit. Example FFTW headers: `n1_5` = 14 adds + 18 fma
= 32; `n1_8` = 44 adds + 8 fma = 52; `n1_16` = 104 adds + 40 fma = 144.)

---

## 3. Reading the gaps

**Pow2 (R=8/16): exact, via generic structural CSE.** No oracle, no transposition.
For the codelets that dominate real workloads, VectorFFT's structural hash-consing
reaches FFTW's op count outright. This alone refutes "FFTW's CSE finds sharing you
miss."

**Primes (R=5/7): both toolchains hand-code the same identities.** The pre-fix
gap was 4–6 ops. VectorFFT's own diagnosis
([`62_winograd_5_and_7.md`](../../src/dag-fft-compiler/docs/62_winograd_5_and_7.md)):

> *"The root cause is that **generic algsimp cannot discover algebraic identities
> specific to roots of unity for a given prime N**. FFTW's `gen_notw -fma` emitter
> doesn't discover them either; **it has them hand-coded inside the codelet
> recipe**. The corresponding fix on our side is also to hand-code them."*

After hand-coding Winograd-5/7 (`lib/dft.ml :: dft_winograd5`), VectorFFT matches
FFTW exactly (e.g. DFT-5: 14 add/sub + 18 fma = 32 ops, 0 standalone muls). The
prime advantage was never FFTW's CSE being smarter — *neither* generator's CSE
discovers these; both hand-code them.

**R=32/64 (+14 / +66): a decomposition choice, not CSE.** FFTW reaches its R=64
count via **conjugate-pair split-radix** (22 unique constants). VectorFFT's current
`split_radix.ml` produces **worse** output — 1144 ops at R=64, more than its own
Cooley–Tukey (978) — so CT is used. Hash-consing vs oracle is irrelevant here; it's
which decomposition you run.

---

## 4. The oracle's one real edge — matched without one

VectorFFT's R=64 investigation found a genuine case of structural CSE missing
value-equal sharing: `sin(π/8)` and `cos(3π/8)` are mathematically equal but came
out **bit-different**, blocking hash-cons unification of the `tan(π/8)` ratios
across symmetric-angle rotations. **This is exactly what FFTW's numeric oracle
catches for free.**

VectorFFT closed it by rounding `cr`/`ci` to 13 significant figures *before*
dividing in `const_cmul`'s Path B
([`58_op_count_journey.md`](../../src/dag-fft-compiler/docs/58_op_count_journey.md)):

> *"This canonicalizes symmetric angles so the resulting inner ratios are
> bit-identical, letting hashcons unify downstream Muls. Saves 16 ops at R=64, 76
> at R=128, 249 at R=256 with zero algorithmic change."*

So the oracle's edge is **real but marginal**, and was recovered with a one-line
deterministic canonicalization — no 20-point random evaluator, no false-positive
surface, no exact-arithmetic number tower.

---

## 5. Why fewer ops ≠ faster — the split-radix register wall

The R=64 "+66 op gap" comes from split-radix, and split-radix is precisely the
algorithm whose low op count **does not survive register reality on wide SIMD**.

From [`sr_parity_plan.md`](../../src/dag-fft-compiler/docs/sr_parity_plan.md):

- At compute-bound K=8, CT beats SR by **−1% / +22% / +33%** for N=16/32/64 — the
  margin *grows* with radix.
- **SR-64 = 507 spills vs CT-64 = 307 spills** (no cut topology).
- *"REFRAME: the SR↔CT gap is SPILLS, not FMA. Forced-lift closed most of the fma
  deficit yet SR still loses by up to 33%. The real SR lever is CUT TOPOLOGY
  (`dft_split_radix_spill`) to bound peak-live like CT's recipe cut, NOT more
  fusion."*
- *"SR wins at high K = FALSIFIED"* — its only win is a narrow K=64 L2-residency
  band, gone by K=256.

Corroborated by [`audit_avx2_leaf_campaign.md`](../../src/dag-fft-compiler/docs/audit_avx2_leaf_campaign.md)
("Split-radix: spills flat, liveness peaks worse"). And it's not even an
FMA-gating artifact: `fma_lift` is disabled for `Split_radix`
([`pipeline.ml:198`](../../src/dag-fft-compiler/generator/lib/pipeline.ml#L198)),
but VectorFFT's own analysis says that's *secondary* — the monolithic topology and
its unbounded peak-live set are the problem.

The same effect shows up *within* Cooley–Tukey: at R=64, `CT(4,16)` has **950 ops**
vs `CT(8,8)`'s **978**, yet the 950-op version benched **13.8% slower** (register
pressure, dependency-chain depth) — so VectorFFT's default correctly picks the
*more-ops, fewer-spills* factorization.

### The CT-side mechanism: 2-pass register-budget blocking

VectorFFT's R=32/64 codelets don't just *avoid* split-radix — they win via an
explicit structural cut that bounds register pressure. At R ≥ 32, no-twiddle
codelets are emitted **2-pass**
([gen_main.ml:381](../../src/dag-fft-compiler/generator/lib/gen_main.ml#L381),
[dft.ml:1899](../../src/dag-fft-compiler/generator/lib/dft.ml#L1899) →
`dft_expand_n1_blocked`): one inlined Cooley–Tukey step split by a PASS 1 → PASS 2
spill seam ([58_n1_blocking.md:45–48, 150–152](../../src/dag-fft-compiler/docs/58_n1_blocking.md#L45)):

- **PASS 1** — the N1 inner DFT-N2s, outputs **spilled to slots**
- **PASS 2** — the N2 inner DFT-N1s, **reading the spilled slots**
- spill markers captured at the seam (the same SU+spill recipe the t1 codelets use)

The cut bounds **peak-live to the vector register file**, *not* L1. Monolithic R=64
had **peak_live = 994 — 31× over the 32-ZMM budget**
([58_n1_blocking.md:28, 137–138](../../src/dag-fft-compiler/docs/58_n1_blocking.md#L137)),
which triggers catastrophic spill cascades; blocking brings **each pass to ~35**,
inside the allocator's ~30-ZMM per-cluster working set. (R ≤ 16 stays monolithic —
it already fits, so blocking would be pure overhead.) The codelet's *data* (32–64
complex = 0.5–1 KB) is L1-resident either way; what didn't fit is the **register
file**. L1 enters only indirectly: the inter-pass spill slots are L1-resident, so
the 2-pass cut routes intermediates through L1 in a *controlled, orderly*
store→reload instead of the monolithic version's thrashing spill cascade.

This is the **exact "recipe cut" split-radix lacks.** SR-64 spills 507 vs CT-64's
307 because CT got the 2-pass seam and monolithic SR didn't — the proposed
`dft_split_radix_spill` is the SR equivalent of this same CT cut.

> FFTW's op-count metric (`Stats.complexity` weights muls ×20, adds ×10) optimizes
> peak-live into the ground at R=64. The "+14/+66 gap" costs VectorFFT little
> because its CT R=32/64 codelets are **2-pass, register-budget-bounded** —
> peak-live ~35 against a 32-ZMM file — whereas FFTW emits monolithic straight-line
> C and hands a 994-deep live set to gcc/clang's allocator. FFTW's lower op count is
> only a win if you ignore register allocation, which genfft does (it delegates
> regalloc to the C compiler) and VectorFFT doesn't. Another instance of
> controlling the machine vs counting the ops.

---

## 6. What VectorFFT's approach buys

The structural / deterministic / direct-emission design isn't merely "FFTW minus
the oracle" — it's a different objective, better matched to a memory-bound,
beat-MKL, fixed-target library:

- **Zero false-merge risk** — structural CSE can never unify two non-equal
  expressions; FFTW special-cases `Store`/`Uminus(Store)` precisely to stop its
  evaluator from aliasing them.
- **Bit-exact determinism** — same input → byte-identical codelet, every host.
  Validation is bit-exact; generation must be too.
- **Direct FMA-intrinsic control** — VectorFFT chooses the FMA variant, operand
  order, and whether to fuse (the `flatten_fma_mul_addend` density gate is a
  decision FFTW *can't* make — it delegated it to the C compiler).
- **The structural IR is the substrate for explicit scheduling / regalloc /
  spill** — stable tags let spill markers survive the FMA cascade via `tag_remap`.
  FFTW genfft does *no* register allocation; VectorFFT's whole spill-aware,
  n1-blocking codegen layer is built on structural identity.
- **Optimizes measured runtime, not a static op-count proxy** — exactly the axis
  §5 shows FFTW's metric gets wrong.

(Full benefit/cost ledger in
[`algsimp.md`](../../src/dag-fft-compiler/docs/compiler_internals/algsimp.md).)

---

## 7. Actionable residuals

1. **`dft_split_radix_spill` cut-topology** — bound split-radix's peak-live the
   way the CT recipe-cut does. *Then* SR's lower op count might finally pay off in
   the narrow K=64 L2-residency band where it already flickers a win. Not more FMA.
2. **The oracle gap is closed structurally** — the `sin/cos` bit-equality case is
   handled by 13-sig-fig rounding; no general numeric oracle is warranted. If new
   bit-different-but-equal constants surface, extend the same canonicalization.
3. **CSE is uniform across transforms** — r2c/c2r/DCT/DST run the *identical*
   `of_assignments` → cascade as c2c. The documented r2c-vs-MKL runtime gap is
   *executor/memory-bound* (L2-resident vs MKL's L1 cache-blocking), **not** a CSE
   deficiency. Don't chase CSE there; chase the executor.

---

## Sources

**VectorFFT** (in-repo):
[`algsimp.ml`](../../src/dag-fft-compiler/generator/lib/algsimp.ml),
[`pipeline.ml`](../../src/dag-fft-compiler/generator/lib/pipeline.ml),
[`compiler_internals/algsimp.md`](../../src/dag-fft-compiler/docs/compiler_internals/algsimp.md),
[`62_winograd_5_and_7.md`](../../src/dag-fft-compiler/docs/62_winograd_5_and_7.md),
[`58_op_count_journey.md`](../../src/dag-fft-compiler/docs/58_op_count_journey.md),
[`sr_parity_plan.md`](../../src/dag-fft-compiler/docs/sr_parity_plan.md),
[`audit_avx2_leaf_campaign.md`](../../src/dag-fft-compiler/docs/audit_avx2_leaf_campaign.md).

**FFTW 3.3.10 genfft** (external, `~/fftw-3.3.10/genfft/`): `algsimp.ml` (the
`AlgSimp`/`Transpose`/`Stats` modules), `oracle.ml` (numeric CSE + sign oracle),
`expr.ml` (n-ary `Plus`, no `Sub`), `magic.ml` (`randomized_cse=true`,
`network_transposition=true`, `enable_fma=false` defaults). Op counts from the
generated scalar codelet headers in `~/fftw-3.3.10/dft/scalar/codelets/n1_*.c`.

- Frigo, *A Fast Fourier Transform Compiler*, PLDI'99 — <https://www.fftw.org/pldi99.pdf>
- Frigo & Johnson, *The Design and Implementation of FFTW3*, Proc. IEEE 93(2), 2005 — <https://www.fftw.org/fftw-paper-ieee.pdf>

---

*All comparative numbers reflect VectorFFT's measured op-count docs and FFTW
3.3.10's generated-codelet headers at time of writing. Re-verify against the
generators if either toolchain moves.*
