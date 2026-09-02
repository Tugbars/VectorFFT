# `core/transforms/natorder/` — ORDER_NATURAL

The scrambled engines emit digit-reversed output. This directory is what turns
that into natural index order, and the choice of *how* is measured, not assumed.

🔴 **The mechanism flips with K**, so PURE / PSWAP / SCR are raced per cell
(`natorder_calibrate.h`) rather than selected by a rule. Natural-order
calibration is expensive — that cost is known and accepted.

| file | role |
|---|---|
| `natorder_perm.h` | the permutation, orientation detection, and the cycle tape |
| `natorder_exec.h` | the cycle and pair reorder passes |
| `natorder_scatter.h` | the SCR scatter terminator |
| `natorder_calibrate.h` | the PURE-vs-PSWAP-vs-SCR race |
| `natorder_mt.h` | threaded reorder and SCR passes |
