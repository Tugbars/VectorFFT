# K=1 — the single-transform question, measured and answered (2026-07-19)

> **The one-paragraph truth.** The engine is lane-batched — SIMD lives
> across K — so a single transform (K=1, the most common FFT call shape in
> the world) runs its entire body through the anyk rem==1 **scalar tier**:
> bit-exact, but ~2.7× off the batched ceiling. Lane-padding K=1→8 is
> **measured dead** (+206…+287%: it costs exactly a full eight-lane batch,
> because zero lanes compute like anything). The real answer is
> intra-transform ("row-major") vectorization — and it turns out the tree
> **already ships it**: BAILEY2, the fused-transpose four-step OOP kind,
> is picked at K=1 on the natural path today and beats the scalar tier by
> **−33%** at N=256/1024. What remains is routing (default-order and
> in-place K=1 onto it, verdict-gated), not building.

Companions: `arbitrary_k_tail_handling.md` §8 (the K=1 pure-scalar
extreme, bit-exactness), `v1_0_results.md` (the BAILEY2 record),
`il_padding_tail_handling.md` (the verdict-lifecycle pattern the routing
will reuse). Benches: `build_tuned/benches/bench_k1_answer.c` (the four-arm padding
race), `build_tuned/benches/bench_k1_bailey.c` (BAILEY2 vs the scalar tier).

## 1. The gap, precisely

Layout is `x[n*K + k]`, lanes contiguous; every codelet vectorizes across
lanes. At K=1 the bulk loop (`b + VW <= me`) never fires and the whole
transform runs the rem==1 monolithic **scalar** rendering. Correct by the
§8 record; slow by architecture. Measured (split in-place, jit, med9):

| N | scalar tier | K=8 batch ÷ 8 (ceiling) | gap |
|---|---|---|---|
| 256  | 1.86 µs  | 0.74 µs  | 2.5× |
| 1024 | 10.19 µs | 3.75 µs  | 2.7× |
| 4096 | 54.74 µs | 20.72 µs | 2.6× |

## 2. Lane-padding: measured dead

The obvious "column-major answer" — pad K=1 to Kp=8, run full-width,
read lane 0 — was raced same-process through the §6a55 IL padded arm:

| N | scalar | pad→8 | delta |
|---|---|---|---|
| 256  | 1.86  | 5.70   | **+206%** |
| 1024 | 10.19 | 31.90  | **+213%** |
| 4096 | 54.74 | 211.62 | **+287%** |

The numbers explain themselves: pad→8 ≈ exactly the full K=8 batch cost
(31.9 vs 8×3.75=30.0 at N=1024). Zero lanes compute like real lanes; the
"equal instruction count" intuition (P vector ops vs P scalar ops) loses
to the empirical fact that the scalar tier is only ~2.7× off ceiling
while eight-lane work is 8× the ceiling's per-transform cost. **Padding
answers K∈{2..7}; it cannot answer K=1.**

## 3. The row-major inventory — already in the tree

Intra-transform vectorization needs three passes for N = N1×N2:

| pass | mechanism | status |
|---|---|---|
| column | the native lane engine at K=N2 — the four-step's column pass IS the engine's home shape | shipped |
| twiddle | the §6a41 twiddle-stage engine | shipped, dormant |
| row | the strided monos (SIMD along the row), N ∈ {4,8,16,32,64}, generator `--strided` extensible | shipped |
| transposes | **none on the primary path** — the strided family exists to delete them; `transpose.h` blocked transpose is only the six-step fallback's tool for uncovered row lengths | — |

And the fully-fused composition of exactly this idea already exists as an
executor:

## 4. BAILEY2 — "fused2" verified

The remembered name "fused2" is **BAILEY2** ("BAILEY2 fused-transpose
stores", v1_0_results.md): the four-step with its transposes fused into
the stage stores — zero movement passes, the §6a33 doctrine embodied —
plus a per-cell flat-vs-log3-searched t1p stage. It is the natural-order
1D OOP kind that posts the tree's largest MKL margins at small N (5.97×
at (16,32)); it is ST-only (the inter-stage transpose is not
lane-independent, so K-split MT is excluded at vfft.c:1353).

**At K=1 it is already the answer** (picker-chosen, naive-verified,
med9):

| N | BAILEY2 nat-OOP K=1 | scalar tier | delta | ceiling |
|---|---|---|---|---|
| 256  | 1.24 µs  | 1.86  | **−33%** | 0.74 |
| 1024 | 6.84 µs  | 10.19 | **−33%** | 3.75 |
| 4096 | 53.75 µs | 54.74 | −2%      | 20.72 |

It captures ~55% of the scalar→ceiling gap at small/mid N; at 4096 its
small-N design fades to a wash on this box, exactly per the v1_0 record.

## 5. What remains — routing, verdict-gated

1. **Default-order K=1 → BAILEY2.** Contract-legal under the
   chain-defined-order reading (the default order is "some consistent
   order per plan"; natural qualifies trivially). A §6a59-style per-cell
   A/B (BAILEY2 arm vs scalar arm, hysteresis, stamped) picks honestly —
   covering the N=4096-class cells where it's a wash.
2. **In-place K=1.** BAILEY2 is OOP-only; a scratch-bounce wrapper (OOP
   execute + one 2N-double copy back) is noise against −33%; the same
   verdict decides.
3. **The ceiling residue** (6.84 → 3.75 at N=1024): BAILEY2 pays for
   natural ordering a scrambled contract doesn't need, and runs ST. A
   scrambled variant and/or 2-phase MT is the deeper session — with a
   smaller prize now that §4 exists.

## 6. File map

| what | where |
|---|---|
| the scalar tier + K=1 bit-exactness | `arbitrary_k_tail_handling.md` §2, §8 |
| padding race | `build_tuned/benches/bench_k1_answer.c` |
| BAILEY2 K=1 race | `build_tuned/benches/bench_k1_bailey.c` |
| BAILEY2 record + kinds | `v1_0_results.md`; vfft.c OOP kinds (LEAF/MODEB/BAILEY2) |
| the dormant twiddle engine | §6a41, `strided_tw.h` |
| verdict-lifecycle pattern to reuse | `il_padding_tail_handling.md` §5, §6a59 |
