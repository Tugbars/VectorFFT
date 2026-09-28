# Extended regression run, 2026-09-28

Head build (the separation branch with the four-step and odd-row stream fixes), each rank
recalibrated from a fresh copy of the shipped store, benched against MKL, forward checked
against a long-double reference. Break = a refused or errored create, a forward error
above 1e-11, or a cell missing from the bench or the check.

## 2D: 66 cells (regr2_2d_head_2026-09-28)

**Breaks: 0** (none)

| group | cells | ratio vs MKL, median | below 0.9x |
|---|---|---|---|
| large-x-mid-odd | 24 | 1.25 | 2 |
| large-x-mid-pow2 | 8 | 1.14 | 0 |
| tall-odd-rows | 12 | 1.22 | 0 |
| tall-pow2 | 8 | 1.33 | 0 |
| wide-odd-cols | 8 | 1.39 | 0 |
| wide-pow2 | 6 | 1.37 | 0 |
| ALL | 66 | 1.25 | 2 |

Slowest cells: `61x4096` 0.57x (tpc), `1009x1024` 0.78x (tpc)

Largest forward errors: `4096x256` 1.4e-15, `2048x512` 1.3e-15, `64x16384` 1.3e-15, `256x4096` 1.3e-15, `512x2048` 1.3e-15

Control cell: 4 readings, 1.08..1.36x

## 3D: 50 cells (regr2_3d_head_2026-09-28)

**Breaks: 0** (none)

| group | cells | ratio vs MKL, median | below 0.9x |
|---|---|---|---|
| earlier-38 | 38 | 1.39 | 3 |
| odd-innermost | 12 | 1.43 | 0 |
| ALL | 50 | 1.39 | 3 |

Slowest cells: `15x15x15` 0.67x (-), `4x4x4` 0.86x (-), `12x12x12` 0.87x (-)

Largest forward errors: `2x256x4096` 5.1e-15, `4x64x8192` 2.8e-15, `8x128x2048` 2.2e-15, `8x8192x32` 2.0e-15, `128x128x128` 1.6e-15

Control cell: 4 readings, 1.25..1.29x

