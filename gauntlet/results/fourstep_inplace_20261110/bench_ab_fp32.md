# The canonical FP32 bench, before and after, in one run (5 rounds, arm order and flip alternating)

One thread. `ip` = in place, `oop` = out of place (the control: its code path and rows did not change). Before = the
build and the store before this change; after = commit f6e4a970 on the re-raced store. Our time relative to before
(median, min..max over the rounds; below 1 = faster), each arm's median time and its median MKL / ours.

| cell | after / before, our time (med, min..max) | before ms | after ms | MKL / ours, before | MKL / ours, after | route before | route after |
|---|---|---:|---:|---:|---:|---|---|
| ip:2097152 | 0.833 (0.797..0.845) | 4.392 | 3.690 | 0.829 | 0.996 | fs | fs |
| oop:2097152 | 1.003 (0.931..1.034) | 3.397 | 3.429 | 1.327 | 1.346 | fs | fs |
| ip:4194304 | 0.853 (0.841..0.860) | 9.945 | 8.472 | 1.020 | 1.199 | fs | fs |
| oop:4194304 | 1.003 (0.991..1.011) | 8.487 | 8.455 | 1.366 | 1.372 | fs | fs |
| ip:8388608 | 0.905 (0.904..0.958) | 21.755 | 19.790 | 1.113 | 1.220 | fs | fs |
| oop:8388608 | 1.000 (0.910..1.022) | 19.809 | 19.719 | 1.493 | 1.494 | fs | fs |
| ip:16777216 | 0.958 (0.943..0.960) | 44.026 | 42.186 | 1.165 | 1.224 | fs | fs |
| oop:16777216 | 1.004 (0.991..1.011) | 42.150 | 42.161 | 1.481 | 1.471 | fs | fs |
