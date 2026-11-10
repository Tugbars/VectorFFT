# The door and replay fixes of 2026-11-10 (commit 7fdd119c): what was checked

Branch `dev/FP32Experiments`, i9-14900KF (AVX2). The fixes are in core code both precisions compile:

1. An OUT-OF-PLACE C2C plan called with ONE buffer (`out == in`): a plan served by a solo kernel now runs that call on
   the alias-tolerant n1c kernel of its length (`k1_alias`); before, the call ran the plan's own `restrict` kernel on
   one buffer. Every other route already switched to its in-place form.
2. An IN-PLACE C2C plan called with TWO buffers: refused at `vfft_execute` / `vfftf_execute` (the header: in place,
   `out == in`); before, it ran as an out-of-place transform. The library's own composites still run an in-place child
   with two buffers through the internal door.
3. A four-step row that names the no-copy form (`il_kv=1`) replays its chain (`il_sb`) from the one-thread row of the
   request's own placement, the row the planner writes it on; before, always from the out-of-place row.

## Files

| file | what |
| --- | --- |
| `door_probe_under_gdb.md` | which kernel each call shape reaches, the tree before the fixes (5901946f) and the fixed tree: breakpoints on the radix-8 solo kernels, N = 8, FP64 and FP32 |
| `il_solo_gate.md` | the FP64 solo-kernel gate with its two new laws (the out-of-place plan on one buffer; the in-place plan with two buffers refused), cold scratch store |
| `vfftf_gate.md` | the FP32 gate with the same two laws on every cell, cold scratch store, 2..2^20 |
| `fp64_ab_summary.md`, `fp64_ab_rounds.csv` | FP64 timing A/B, main (fe0012c9) against the fixed tree: 15 rounds, 27 cells (`gauntlet/ab_fp64.py`) |
| `door_ab_both_precisions.md`, `door_ab_fp64_only_builds.md` | the cost of one public execute at N = 2..64, the tree before the fixes against the fixed tree, 15 rounds, on two independent pairs of builds |
| `real2d_odd_n2_ab.md` | 2D r2c at an odd N2, the tree before the fixes against the fixed tree, 9 rounds |
| `k1_fourstep_gate_before_and_after.md` | the FP64 four-step gate on a seeded scratch store, built from both trees |

## Results

- **The call shapes** (`door_probe_under_gdb.md`). Before: the aliased call of the out-of-place plan hit
  `radix8_z_n1_fwd/bwd` (restrict), and the in-place plan with two buffers was served. After: the aliased call hits
  `radix8_z_n1c_fwd/bwd` and is bitwise the out-of-place result; the in-place plan with two buffers is refused, its
  output buffer untouched. Both precisions, both directions.
- **Gates.** `il_solo_gate`: ALL PASS, 31 lengths (2..64), both new laws at every length. `vfftf_gate`: ALL PASS,
  80 cells (K = 1 and the contiguous batch of 3, T = 1 and 4), both new laws in every cell.
- **The replay** (not a file: two scratch stores). The FP32 in-place row at 2^23 edited to `il_kv=1 il_sb=8.16.16`, the
  out-of-place row left at `il_kv=0`: the build before the fix replays the in-place cell as form 0, the fixed build as
  form 1 (the door's log line `[k1fs] ... replay FOUR-STEP 2048x4096 form=...`); the form-1 plan agrees with MKL's
  double-precision result to 5.9e-7.
- **FP64 against main** (`fp64_ab_summary.md`): median new/old 0.998..1.004 on every one-thread cell, every route
  matched; the threaded cell `mt:65536` 1.011 (0.965..1.076; 0.997 with the same spread in the night's run).
- **The door's cost at N = 2..64** (two build pairs): medians 0.991..1.005. One cell, FP64 N = 4 out of place, read
  1.023 in the first pair and 1.004 in the second: the old executable of the first pair was the outlier (2.921 ns
  against 2.986..2.994 in the other three), a property of that link, not of the change.
- **2D r2c at an odd N2**: 1.004, 1.005 and 1.008 at 16x61, 24x31 and 128x225, inside the rounds' spread. The change
  moved the compiler's inlining in the 2D real row pass (`_il2d_rows_fwd_set`: one copy of the internal door is now a
  call), which is why these cells were measured.
- **AVX-512**: the CMake library with both precisions (`VFFT_ISA=avx512`, `VFFT_ENABLE_FLOAT=ON`) builds with 0 errors.

## Found on the way (the first two fixed the same afternoon, below)

- `k1_fourstep_gate` fails on the current 14900KF store with the tree before the fixes and with the fixed tree alike:
  four checks of its law "in place bitwise out of place" (524288 and 2097152, both order classes). The two placements
  are raced and banked as their own rows (2^21: 4096.512 out of place, 2048.1024 in place), and the in-place rows of
  2026-09-30 carry no child, so the gate's in-place create races on its scratch store.
- A contiguous batch of solo-length transforms, out of place, called with one buffer still runs the `restrict` kernel
  on it: the batch loop calls its one-transform plan's executor directly, not through the public door.
- `gauntlet/ab_fp64.py`'s default cells include `ip:8192`, `ip:16384` and `ip:32768`, which have no in-place row in the
  14900KF store: the default run stops at the first of them (a cell raced). The runs here name their 27 cells.

## The second round (the same afternoon)

| file | what |
| --- | --- |
| `batch_probe_under_gdb.md` | the out-of-place contiguous batch on one buffer, serial and threaded, before and after: which kernel it reaches |
| `k1_fourstep_gate_new_law.md` | the four-step gate with its in-place law against the reference |
| `vfft_tcbatch_gate.md` | the FP64 contiguous-batch gate after the batch fix |
| `fp64_ab_summary_round2.md`, `fp64_ab_rounds_round2.csv` | FP64 timing A/B, main against the final tree, 15 rounds, 27 cells |
| `real2d_odd_n2_ab_round2.md` | 2D r2c at an odd N2 on a second pair of builds, 15 rounds |

- **The batch.** An out-of-place contiguous batch over a solo plan, called with one buffer, runs every transform on the
  alias-tolerant kernel (`k1_alias`, asked once per call in the serial loop and once per slab in the threaded one).
  Under gdb: `radix8_z_n1_fwd` before, `radix8_z_n1c_fwd` after, serial (K = 3) and threaded (K = 8, T = 4), both
  precisions, bitwise the two-buffer result.
- **The four-step gate.** Its in-place law is now the forward check against the same reference (the natural spectrum's
  spot bins, or the natural spectrum as a multiset) and the in-place roundtrip; bitwise the out-of-place output is
  printed, never asserted. ALL PASS on the seeded scratch store (in place 9.0e-17..1.2e-16 on the spot bins, at most
  3.8e-16 on the multisets). `vfft_tcbatch_gate`, `il_solo_gate` and `vfftf_gate` (80 cells): ALL PASS again.
- **2D r2c at an odd N2 is not slower.** The inlining the first fixes moved in `_il2d_rows_fwd_set` is in a cold
  fallback (a child plan without a bound executor); the per-row loop is instruction for instruction the same. On a
  second pair of builds, 15 rounds: 0.996, 1.004 and 0.998 at 16x61, 24x31 and 128x225.
- **FP64 against main, the final tree**: median new/old 0.986..1.005 on the 25 one-thread cells (the lowest, 0.986, is
  the lane-major batch), every route matched; `tc:4096:4` threaded 0.995, `mt:65536` 1.009 (0.977..1.028).
- **AVX-512**: both precisions build with 0 errors.
