# Lane-batched SIMD: where VectorFFT's parallelism comes from, and what that forces

Status: DESIGN RATIONALE (evergreen) · Companion to docs/roadmap/fft2d_v2_design.md §2

## 1. The fork is the parallelism source, not the layout

The layout question ("lane-major vs row-major") is downstream. The real
decision every SIMD FFT makes first is: **where do the 4 (or 8) values in a
vector register come from?** Two answers exist, and each then *forces* its
own memory layout, codelet style, and compiler architecture.

**Across-transform (VectorFFT):** a register holds the same sample position
from 4 *different* transforms — lane 0's x[n], lane 1's x[n], lane 2's
x[n], lane 3's x[n]. One load, one butterfly instruction, four transforms
advance in lockstep. For that load to be a single contiguous instruction,
those four values must be adjacent in memory — which is exactly lane-major
`x[n*K + t]`. The layout is the *consequence* of the parallelism choice.

**Within-transform (MKL/FFTW single-transform kernels):** one transform,
so the register must hold 4 *different points of it*. A butterfly pairs
points at distance m, and m shrinks every stage (N/2 → … → 1). Early
stages the partners live far apart — clean vertical loads. Late stages the
partners are adjacent — they sit in the *same* register and must be
shuffled apart, computed, shuffled back, with different marshalling per
stage. No lanes exist, so the natural layout is one transform's samples
contiguous — row-major. Again: layout follows parallelism.

## 2. What each choice costs and buys

| | across-transform (ours) | within-transform (MKL single) |
|---|---|---|
| register holds | same point, 4 transforms | 4 points, 1 transform |
| shuffles in the math | **zero** | every late stage |
| codelet is | a scalar recipe, trivially widened | hand-choreographed vector dance |
| twiddles | broadcast | permuted per stage |
| compiler must model | scalar dataflow only | shuffle nodes, register cohabitation, port pressure |
| requires | K ≥ vector width, lane-major | nothing but one transform |
| degrades when | K = 1 (¾ of the vector idles) | never — but is far harder to generate |

Assembly-line version: our factory runs 4 identical products side-by-side
through shared stations — every station hits all 4 at once, nothing is
ever rearranged, but the products must *arrive* side-by-side. The other
factory takes one product and works 4 of its parts simultaneously,
constantly rearranging parts on the bench so the tool can reach all four —
and the rearrangement pattern changes at every step of the recipe.

## 3. Why the corpus does not transfer

Everything in the DAG compiler — expr, algsimp, the scheduler, regalloc,
emit, every emitted codelet family, the jit, the wisdom — is built on one
premise: **a codelet is a scalar dataflow with no data movement between
lanes.** Vectorization is literally `double` → `__m256d`.

Within-transform kernels violate the premise at the root. Shuffles become
first-class dataflow nodes; the cost model needs shuffle-port pressure;
scheduling must reason about which points cohabit a register at each
stage; the emitter needs a marshalling vocabulary per stage shape.
Building that is writing a second compiler (FFTW's genfft SIMD layer,
from scratch). Nothing transfers. This is why fft2d v2 rejects the style
outright (v2 design §2) — not because it is worse, but because it prices
in the corpus we already own.

## 4. The 2D consequence: shuffles at the door, not in the math

The 2D row pass has B = 8 rows per tile — **the parallelism already
exists as a batch.** The rows are merely stored wrong for the engine
(end-to-end row-major, because that is what a 2D array is) instead of
side-by-side lane-major. v1's answer was a repacking crew before the line
(the transpose pass — measured ~101 µs of pure movement at 256², §6a33).
v2's answer teaches the line's first station a clever 8-at-a-time pickup
from the end-to-end stack: the 8×8 register-block transpose folded into
the leaf's *loads*, and its inverse folded into the terminator's *stores*.
The shuffles exist only at the memory boundary, once, on entry and exit;
every station between remains the untouched scalar-recipe machine.

One sentence: **MKL moves shuffles into the math because it must
manufacture parallelism; we keep them at the door because ours arrives
pre-made, just badly parked.**

## 5. Natural order: the presentation adapter for lane-major

The engine's native truth is *scrambled* (digit-reversed) *lane-major* —
that is what the cascade produces with zero extra work. Natural ordering
is not part of the math; it is the **presentation layer** that makes the
lane-major engine honor the external contract consumers expect
("frequency f is at position f").

The observation that makes it cheap — and this is the part that is easy to
miss: in lane-major, one "element" of the frequency axis is a **K-wide
contiguous strip** (a whole row). Unscrambling is therefore *row-strip
permutation* — memcpy-grade, vectorizable, K-amortized moves (the c2c
NATURAL tapes; the 2D pack's perm). A contiguous single-transform layout
pays element-granularity scatter for the same service. So the layout that
forced the ordering problem also makes its solution nearly free.

Two honest qualifications keep this precise:

1. Natural order fixes the **row axis**, not the lane axis. A single
   lane's spectrum is still strided at K after natordering; natorder makes
   the [H×K] bundle *read as* K interleaved natural spectra, not as K
   contiguous ones. Where a consumer needs one contiguous spectrum, a
   gather/pack still happens — in 2D, the pack pass is exactly that
   adapter's final step (pad rows → contiguous per-row user spectra), and
   in 1D the z contract's interleave is its sibling.
2. "Gimmick" undersells one hard requirement: bin addressability. Most
   consumers (filtering, spectral edits, the z/CCE contract itself) need
   "bin f at position f" without carrying a permutation table. The adapter
   is presentation-layer, but the contract it presents is load-bearing.

Net framing: **scrambled lane-major is the engine; natorder + pack/
interleave are the adapters that make it wear the industry's contiguous,
ascending-frequency clothes — and lane-major is what makes the clothes
cheap to put on.**
