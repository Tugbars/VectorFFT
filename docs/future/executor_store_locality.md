# Executor Store-Locality: Closing the Pow2 DTLB Gap

**Status:** Investigation / design note.
**Motivation:** `docs/vtune-profiles/vtune_full_fft_n16384_k4.md` — VectorFFT's single biggest deficit vs MKL is **Store DTLB 32.4% vs MKL 2.7%**, a 30-point gap concentrated on large pow2 sizes.

---

## Observation

Our codelets run at high retiring rates (R=4 at 85.9%, R=8 at 72%, R=32 at 57%). The full-FFT profile at N=16384 K=4 shows we actually *retire* at 48.4% vs MKL's 38.2% — we win the compute fight. MKL retains a near-tie in wallclock (1.02x) only because it crushes us on store-side memory traffic.

```
                  VectorFFT   MKL
Store DTLB           32.4%    2.7%
Load  DTLB           14.5%   26.7%
Retiring             48.4%   38.2%
Instructions         167 B   128.8 B
```

MKL writes to "a handful of pages per pass"; we write across dozens. At stride 32k (large-N late passes in an in-place pipeline), every vector store lands on a new 4 KiB page and hits STLB. 2.7% store DTLB is only reachable if the store *destination* stays inside a small page working set.

## Root cause: in-place pass-by-pass executor

Our executor is strictly in-place per pass: pass k reads from the single working buffer at stride `s_k = N / (r_1 * ... * r_k)` and writes back to the same buffer at the same stride. Consequences:

- For large N, `s_k` quickly exceeds page size. A radix-R butterfly with output stride > 4 KiB spreads R stores across R distinct pages.
- The STLB write-side has only ~16 entries on Raptor Lake. R=32 with stride > 4 KiB exhausts it after one butterfly group.
- Pass k+1 then re-reads those pages in a different order, inflating load DTLB too — though Raptor Lake's 96-entry load STLB handles this better than the 16-entry store STLB.

## Two fixes MKL likely uses (one or both)

### A. Out-of-place staging buffer

Add a second cache-sized working buffer (≈256 KB for L2). Alternate pass direction A→B, B→A. Within one pass, *loads* come from arbitrary strides but *stores* are contiguous into the destination buffer. DTLB-store drops to near zero because stores walk pages sequentially.

**Pros:**
- Localized change. Planner untouched.
- Per-pass decision: in-place for early passes (small stride → fine), out-of-place for late passes (stride > page size).
- Existing codelet ABIs unchanged — only the wrapper that calls them changes.

**Cons:**
- 2x memory footprint during execution (one extra buffer of size N×K×16 bytes).
- Zero-permutation roundtrip property depends on DIT/DIF direction matching the buffer direction; needs care.
- Wisdom format gains an `oop_from_stage` field alongside `split_stage`.

### B. Bailey / six-step reshape at the cache boundary

For N > L2-capacity (roughly N ≥ 16k complex = 256 KB), the "six-step FFT" reframes the 1D transform as a 2D array:
1. Reshape N = N1 × N2 into an N1×N2 matrix (logical view, no copy if dims align).
2. N2 row-FFTs (small, cache-resident).
3. Element-wise twiddle multiply (the "diagonal twiddle").
4. Transpose (in-place or via our existing 8×4 tile kernel).
5. N1 column-FFTs (again small, cache-resident after transpose).
6. Final transpose back (often fuseable with step 4 if layout permits).

Every heavy step is now on cache-resident tiles. DTLB-store becomes a function of the tile, not the full array.

**Pros:**
- Algorithmically principled. Same approach MKL and FFTW use for large N.
- Reuses our existing transpose infrastructure (`transpose.h` 8×4 kernel) and small-N executor.
- Natural fit for 2D FFT internals we already have.

**Cons:**
- Planner must choose N1, N2 (a second factorization decision per large N).
- Twiddle table grows (the N1×N2 diagonal twiddles are distinct from the pass-local ones).
- Threshold tuning: at what N does Bailey beat the pass-by-pass executor? Likely around 16k–32k, but measured per machine.

## Relationship to the blocked executor

Our existing blocked executor (`executor_blocked.h`) is a partial answer: it already groups passes to keep working sets in L2. But it does not *change the store destination pattern* — stores still land at stride `s_k`. The blocked executor reduces load-side DTLB thrash (passes stay warm) but doesn't touch the store-side problem that dominates at large pow2.

**Relationship to DIT/DIF:** DIF intermediate passes have more page-local store patterns than DIT in large-stride contexts. Adding DIF variants of R=16/R=32 is a complementary lever — it shrinks the store-DTLB gap without changing the executor, and composes with either fix above.

## Suggested order of attack

1. **Measure first.** Add a VTune configuration that reports store-DTLB separately for (a) small pow2 cases where we win (confirms low baseline), (b) large pow2 cases where we tie MKL (confirms the gap). This anchors the hypothesis.
2. **R=32 DIF codelet.** Bounded work (~1–2 days generator + tests). Rerun the four tightest pow2 cases. If DTLB-store drops meaningfully, the DIT/DIF lever is confirmed.
3. **Out-of-place staging (Fix A).** If step 2 closes half the gap, do Fix A for the remainder. If step 2 closes all of it, skip Fix A.
4. **Bailey (Fix B).** Revisit only if A+DIF don't suffice, or if we want a principled algorithmic story for the paper. Bailey is the textbook answer for N > L2; worth implementing eventually regardless.

## Planner / wisdom implications

- Fix A requires one bit per pass (in-place vs out-of-place), or a single "oop_from_stage" index analogous to `split_stage`.
- Fix B introduces a new "use_bailey" flag + (N1, N2) factorization alongside the stride-FFT factorization.
- Both interact with the existing blocked-executor fields. A single `strategy` enum may be cleaner than stacking booleans.

## Extending the evidence: VTune MKL vs ours on mixed-radix close-calls

The existing full-FFT VTune (N=16384 K=4) covers one pure-pow2 case. To confirm the store-locality hypothesis generalizes — and to rule out that the DTLB-store gap is an artifact of pure pow2 specifically — we should profile MKL vs VectorFFT on mixed-radix factorizations where MKL still comes within ~20% of us.

**Candidate cases (from the recent 1D CSV run):**

| N | K | category | factors | vfft_vs_MKL |
|---|---|---|---|---|
| 256 | 256 | pow2 | 4x4x16 | 1.15x |
| 4096 | 4 | pow2 | 4x4x16x16 | 1.22x |
| 8192 | 4 | pow2 | 4x4x32x16 | 1.10x |
| 16384 | 4 | pow2 | 2x8x16x64 | 1.03x |
| 32768 | 4 | pow2 | 4x4x32x64 | 1.02x |
| 65536 | 4 | pow2 | 32x32x64 | 0.99x |
| 131072 | 4 | pow2 | 4x4x4x32x64 | 1.02x |
| 65536 | 256 | pow2 | 64x32x32 | 1.06x |
| 10000 | 256 | composite | 2x5x8x5x5x5 | 1.79x |
| 500 | 256 | composite | 20x5x5 | 2.18x |

The large-pow2 K=4 cluster is where the gap is tightest — exactly the regime where R=16/R=32/R=64 dominate the factorization. Mixed-radix composite cases (500, 10000, 20000) are wider margins but *include* R=5/R=8/R=25; if MKL's executor advantage is DTLB-store-specific, those should show a noticeably smaller DTLB gap despite also having strided late passes.

**Hypothesis to confirm:**
- Pure pow2 large N (32768–131072 K=4): DTLB-store ≥ 25% for us, ≤ 5% for MKL.
- Mixed composite (10000 K=256): DTLB-store materially lower for us (fewer large-stride passes because R=5 passes have smaller strides), confirming the gap is *stride-dependent*, not pow2-specific.
- If confirmed, Fix A / Fix B targeted at large-stride passes — which happen more often on pow2 because R=16/32/64 are larger than the mixed-radix alternatives.

**Counter-hypothesis worth ruling out:**
- If mixed-radix cases also show large store-DTLB gaps, the problem is broader than "large radix at high stride" and points more at a general out-of-place staging strategy (Fix A).
- If mixed-radix cases show small store-DTLB (VectorFFT close to MKL on store-DTLB there), but MKL still beats us compute-side, then split-radix / instruction-count is relatively more important than store-locality — ordering of fixes should flip.

**VTune run recipe:**
- Reuse existing `bench_vtune` harness pattern from `vtune_full_fft_n16384_k4.md`.
- 3 additional configurations: (N=65536, K=4), (N=65536, K=256), (N=10000, K=256). Each run VectorFFT + MKL separately.
- Collect uarch-exploration. Report the same metric table format as the existing full-FFT doc so results are comparable row-to-row.
- Save under `docs/vtune-profiles/vtune_full_fft_{N}_{K}_{backend}.md` per case.

Cheap to run (few minutes each), directly validates or invalidates the store-locality thesis before committing to Fix A or Fix B implementation.

## Evidence this is the right axis

- VTune data is unambiguous: **store DTLB is the single largest VectorFFT-vs-MKL gap** at 30 points, and concentrated on the regime (large pow2, mid-to-high K) where we're tightest against MKL.
- MKL's load-DTLB is actually *worse* than ours (26.7% vs 14.5%). MKL is not magical on memory overall — they've specifically engineered the store side. That tells us the fix is known and tractable, not a dark art.
- Our 23% higher instruction count suggests split-radix is also on the table long-term, but algorithmic rework is a larger bet. Fix A / DIF is the lower-risk first move.
