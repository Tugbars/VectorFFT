# 2D C2C natural order — status

**Status: implemented, correct, adversarially verified, perf characterized.** `VFFT_ORDER_NATURAL`
now covers 2D C2C in-place — the last order×dimension cell that was gated out. Uncommitted on
`dev/arbitraryTail`. This doc is the standing reference for what was built, why, how it performs, and
what remains.

Companion: the 1D design + measurements live in memory (`natural_order_inplace_design.md`) and
[docs/performance/v1_0_results.md](../performance/v1_0_results.md) §"Natural order — in-place".

---

## 1. What natural order means for 2D, and how it's produced

2D C2C is separable — two inner 1D plans on the split-complex row-major matrix `re[i1*N2 + i2]`
([fft2d.h](../../src/core/transforms/fft2d/fft2d.h)):

- `plan_col` — N1-point FFT down the columns (axis-0 / rows, baked `K = N2`)
- `plan_row` — N2-point FFT across the rows (axis-1 / within-row, baked `K = B` tiled), bracketed by a
  SIMD transpose (`stride_transpose_pair`)

Both inners are DIT/DIF, so each axis comes out **digit-scrambled**. The 2D output is scrambled in
**both** dimensions:

```
buffer[i1][i2] = natural[perm1_inv(i1)][perm2_inv(i2)]
natural[k1][k2] = scrambled[M1[k1]][M2[k2]]
```

where `M1` = digit reversal of `plan_col`'s chain, `M2` = `plan_row`'s. The two permutations act on
**orthogonal axes and commute**. Natural order = undo both, per axis. `DEFAULT`/`SCRAMBLED` leave the
scrambled output untouched (the fast convolution contract); `NATURAL` applies the reorder.

The order axis is the same public flag as 1D: `config.order ∈ {DEFAULT(0), NATURAL(1), SCRAMBLED(2)}`
([include/vfft.h](../../include/vfft.h)). For 2D, `SCRAMBLED` is a no-op alias of `DEFAULT` (the engine
is natively scrambled); only `NATURAL` builds a reorder.

---

## 2. The two-axis reorder — and why the axes are handled differently

The digit-reversal permutations are derived **per axis at plan time** by an impulse probe on each inner
plan (`vfft_natorder_detect`, orientation auto-detected — DIT vs reversed-factor DIF), then turned into
a reorder tape. The 1D natorder machinery (`natorder_perm.h` / `natorder_exec.h`) is reused verbatim
because the N1×N2 matrix *is* the `(N rows × K doubles)` shape those kernels were built for.

| axis | what it permutes | where it runs | kernel |
|---|---|---|---|
| **dim1** | whole matrix **rows** (M1, N1 rows of N2 contiguous doubles) | vfft.c `_natorder_2d`, on the user buffer after the FFT | `pair_pass` (PSWAP) if M1 is an involution, else `cycle_pass` (K=N2) |
| **dim2** | **within** each row (M2, the N2 axis) | fft2d.h `_fft2d_tiled_range`, **fused into the row-FFT scratch** | `cycle_pass` at K=B, full-SIMD, L1-hot |

**dim2 is "mechanism-2":** right after each tile's inner FFT the scratch holds the N2 spectrum as
exactly the `(N2 points × K=B)` layout `cycle_pass` wants (`sr[j*B]`), so one `cycle_pass(sr,si,B,…)`
unscrambles the N2 axis while the data is L1-hot, in the existing streaming pass — **it costs nothing
measurable** (§4). A single stack `rtmp[2*FFT2D_DEFAULT_TILE]` per call keeps it MT-safe.

**Forward vs backward** (the axes commute, but the code uses the unconditionally-correct reversed-order
inverse):
- Forward: FFT, then unscramble — dim2 in scratch (after inner FFT), dim1 whole-row (after the pass).
- Backward: re-scramble, then IFFT — dim1 whole-row (before `stride_execute_bwd`), dim2 in scratch
  (after gather, before the inner IFFT). `pair_pass` is self-inverse; `cycle_pass_inv` inverts
  `cycle_pass`.

**FREE axes:** an axis whose inner plan is single-radix (`num_stages<=1`) — including prime
Rader/Bluestein overrides — is *already natural*, so its tape is empty and its pass is skipped. The
adversarial review confirmed this is correct (Rader scatters output to natural bin g^{-q}, Bluestein
is natural), matching the validated 1D rule.

### Code touchpoints
- **Gate** ([vfft.c](../../src/core/vfft.c) ~L1150): `dims < 2` → `dims <= 2` for `NATURAL`/`SCRAMBLED`
  (still `transform==VFFT_C2C && !batch`, so 2D r2c/c2r/padded stay rejected).
- **Handle** (`struct vfft_plan_s`): `nat2d`, `nat2d_row_list`, `nat2d_row_is_pairs`, `nat2d_col_list`,
  `nat2d_tmp`.
- **Helpers**: `_natorder_2d_build_axis` (probe → detect → `mk_pairs` [dim1] or `mk_cycles`);
  `_natorder_2d` (dim1 apply).
- **Create** (`order==NATURAL`): build both tapes from `d->plan_col`/`d->plan_row`; borrow the dim2
  tape into `d->nat_col_list` (h owns the malloc; `_fft2d_destroy` must not free it); `nat2d=1`.
- **fft2d.h**: `nat_col_list` field (borrowed) + NULL-init in `_fft2d_wrap` + the fused pass in
  `_fft2d_tiled_range`.
- **Destroy**: frees the three `nat2d_*` pointers (NULL-safe).

---

## 3. Correctness

- **`natorder_2d_test.c`** (naive **separable** 2D DFT reference): 32×32 and 64×64 `NATURAL` match
  bin-for-bin at **1.6e-14 / 2.9e-14**; `DEFAULT`/`SCRAMBLED` are scrambled (1.2–1.8e0); roundtrip
  `fwd+bwd == N1·N2·x` at **<1e-14** every cell. 16×16 is the single-radix case (radix-16 per axis →
  FREE → scrambled==natural), handled correctly.
- **8-agent adversarial review** — **zero confirmed bugs**. All five lenses clean: order-correctness
  (M1/M2 from the right plans, no axis mixup), inverse-roundtrip (true inverse, not luck),
  memory-safety (tmp exactly sized, no leak/double-free/UAF), edge-assumptions (prime-dim FREE proven
  correct; Bailey moot — public path is always tiled), scope-regression (DEFAULT/SCRAMBLED
  byte-identical, 1D untouched, gate excludes 2D r2c/c2r).

---

## 4. Performance — the tax is entirely dim1, and it shrinks with N

The **reorder tax** = natural fwd / scrambled fwd on the *same* calibrated plan (`natorder_2d_tax.c`,
QPC best-of-5, core-pinned). Naively it looked like it grew with N (32² 1.21× → 64² 1.61×). Isolating
each axis (single-radix on the other axis makes it FREE) tells the real story:

| cell | elems | reorder measured | tax |
|---|---|---|---|
| 16×64 | 1024 | **dim2-only** | **0.99× (FREE)** |
| 64×16 | 1024 | **dim1-only** | **1.49×** |
| 256×16 | 4096 | dim1-only | 1.17× |

**dim2 is free** (mechanism-2 works); **the entire tax is dim1** — the whole-row reorder. And the dim1
tax **shrinks with N** (1.49× @1K → 1.17× @4K) because the reorder is O(N²) while the FFT is O(N²log N),
so reorder/FFT ~ 1/log N. Extrapolated to the documented **128²–512²** sizes it lands **~1.1×** — and
since scrambled 2D beats MKL 1.26–1.42× there, natural 2D would be **~1.1× over MKL: competitive**.
(Unconfirmed — 128²+ times out on *create-time* 2D MEASURE calibration, not the reorder; see §6.)

### Why dim1 costs what it does — and what PSWAP does

`natorder_reorder_micro.c` isolates the mechanism (`cycle_pass` vs `pair_pass` on the *same* involution
perm, no FFT, no calibration noise):

| cell | row width | cycle_pass | pair_pass (PSWAP) | speedup |
|---|---|---|---|---|
| 64×16 | 128 B | 396 ns | 85 ns | **4.68×** |
| 256×16 | 128 B | 1699 ns | 621 ns | **2.73×** |
| 64×64 | 512 B | 538 ns | 576 ns | 0.93× |
| 256×64 | 512 B | 2920 ns | 2894 ns | 1.01× |
| 1024×16 | 128 B | 6712 ns | 6318 ns | 1.06× |

Two regimes:
- **Narrow rows (small N2, ~128 B):** the reorder is **dependency/overhead-bound** — `cycle_pass` walks
  each cycle as a serial dependency chain. **PSWAP** (independent pair-swaps, no chain, SIMD/prefetch/OoO
  friendly) wins **3–5×**. Applies when M1 is an involution ⇔ the column chain is **palindromic**.
- **Wide rows (≥512 B):** **bandwidth-bound** (moving the row data dominates) — PSWAP and cycle are
  equal (~120 GB/s).

### Decisions taken

- **Mechanism-2: kept.** dim2 is free, and fusing it into the SIMD scratch also stops the mechanism-1
  scalar dim2 (N1 separate K=1 calls) from growing at large N.
- **Opportunistic PSWAP: kept.** `_natorder_2d_build_axis` tries `mk_pairs` for dim1; when the
  calibrated column chain is already palindromic (common for pow2: 8·8, 16·16, 4·4·4) dim1
  automatically uses `pair_pass` — a **free 3–5× on narrow rows**, neutral otherwise.
- **Palindromic-column injection: tried, measured a WASH, reverted.** Forcing a palindromic column
  chain to *guarantee* PSWAP is a wash: at 64×16 the injected `8·8` uniform-T1S column FFT is ~281 ns
  slower than the calibrated `16·4`, which almost exactly cancels PSWAP's ~311 ns reorder saving. The
  reason is fundamental — **no free lunch:** the calibrator already picked the fastest chain, so buying
  a palindrome costs ~as much FFT speed as PSWAP returns. PSWAP is free *only* when the fast chain is
  already palindromic (the opportunistic path).

---

## 5. What ships today

- 2D C2C `NATURAL` (in-place): correct, verified, roundtrip-clean, `DEFAULT`/`SCRAMBLED` byte-identical.
- dim2 reorder free (mechanism-2). dim1 reorder: opportunistic PSWAP where the column chain is
  palindromic, cycle-following otherwise; tax shrinks with N toward ~1.1× at the documented sizes.
- No regression to any DEFAULT/scrambled/1D/r2c/2D path.

---

## 6. Remaining work (bigger, lower-priority — the tax already shrinks with N)

1. **Large-N measurement + docs.** The real blocker is create-time 2D MEASURE calibration timing out at
   128²+, *not* the reorder. Pre-populating the 2D wisdom offline (calibrate 128²/256²/512² once) lets
   `natorder_2d_tax` measure natural-vs-scrambled there and confirm the ~1.1× extrapolation — which is
   what an honest `v1_0_results.md` 2D-natural section needs.
2. **Scatter-fusion for wide/square cells.** The wide-row dim1 tax is bandwidth-bound, so PSWAP can't
   help it — only *locality* can. Fuse the dim1 whole-row reorder into the row-FFT **scatter** (write
   each tile's B rows to their M1-permuted positions) so it rides existing traffic, L1-hot like dim2.
   Real work: it needs a per-row-base scatter that breaks the block-SIMD transpose kernel.
3. **OOP 2D natural.** 2D already uses a scratch plane; an OOP variant could land natural order in the
   final scatter-transpose to a distinct output buffer.

---

## 7. Reproduction

Untracked probes in `build_tuned/test/` (build with `python build_tuned/build.py --src <file> --vfft
--jit`; the tax probes need a P-core-pinned quiet host for stable numbers):

- `natorder_2d_test.c` — correctness (naive separable 2D DFT + roundtrip), DEFAULT/NATURAL/SCRAMBLED.
- `natorder_2d_tax.c` — reorder tax (natural fwd / scrambled fwd), incl. axis-isolation cells.
- `natorder_reorder_micro.c` — `cycle_pass` vs `pair_pass` on the same involution perm (no FFT).

> **Note on 2D calibration cost:** every distinct `(N1,N2)` cell pays a dedicated PATIENT 2D search on a
> wisdom miss; the small-square-pow2 cells calibrate in seconds, but 128²+ and non-square cells can time
> out at create time. This is orthogonal to natural order — it caps how large a cell the probes can
> reach without offline wisdom, not the reorder's correctness or the shipped runtime.
