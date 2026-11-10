# The canonical FP32 bench, before and after, in one run (7 rounds, arm order and flip alternating)

Cells: oop / ip = one thread, out of place / in place; mt = 8 threads, out of place. Our time relative to the build
before the change (median, min..max over the rounds; below 1 = faster), and each build's median MKL / ours as the
in-process control. "First form" = w = C[a] F[b], then x w (not committed); "committed form" = (x F[b]) C[a].

| cell | route | first form / before, our time (med, min..max) | committed form / before, our time (med, min..max) | MKL / ours, before | MKL / ours, first form | MKL / ours, committed form |
|---|---|---|---|---|---|---|
| oop:2097152 | fs | 0.714 (0.665..0.840) | 0.712 (0.663..0.833) | 0.908 | 1.274 | 1.276 |
| ip:2097152 | fs | 0.769 (0.757..0.791) | 0.764 (0.732..0.780) | 0.667 | 0.830 | 0.855 |
| oop:4194304 | fs | 0.758 (0.708..0.766) | 0.737 (0.702..0.766) | 1.021 | 1.343 | 1.374 |
| ip:4194304 | fs | 0.819 (0.776..0.836) | 0.788 (0.767..0.817) | 0.825 | 1.049 | 1.087 |
| oop:8388608 | fs | 0.785 (0.779..0.842) | 0.780 (0.778..0.839) | 1.179 | 1.503 | 1.512 |
| ip:8388608 | fs | 0.801 (0.798..0.827) | 0.798 (0.792..0.818) | 0.897 | 1.122 | 1.126 |
| oop:16777216 | fs | 0.798 (0.770..0.814) | 0.802 (0.760..0.818) | 1.209 | 1.511 | 1.527 |
| ip:16777216 | fs | 0.803 (0.797..0.834) | 0.799 (0.793..0.825) | 0.979 | 1.188 | 1.206 |
| mt:2097152 | fs | 0.776 (0.750..0.786) | 0.774 (0.703..0.826) | 1.366 | 1.703 | 1.744 |
| mt:4194304 | fs | 0.837 (0.775..0.893) | 0.834 (0.769..0.920) | 1.472 | 1.859 | 1.849 |
| mt:8388608 | fs | 0.890 (0.831..1.099) | 0.872 (0.705..0.955) | 1.012 | 1.121 | 1.195 |
| mt:16777216 | fs | 0.920 (0.851..1.062) | 0.916 (0.844..1.036) | 1.000 | 1.054 | 1.052 |
