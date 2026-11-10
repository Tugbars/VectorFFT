# The canonical FP64 bench, before and after, in one run (9 rounds, arm order and flip alternating)

One thread, forward. Our time relative to before (median, min..max over the rounds; below 1 = faster), each arm's
median time and its median MKL / ours.

| cell | after / before, our time (med, min..max) | before ms | after ms | MKL / ours, before | MKL / ours, after | route before | route after |
|---|---|---:|---:|---:|---:|---|---|
| oop:1048576 | 1.003 (0.984..1.030) | 3.187 | 3.193 | 1.313 | 1.294 | fs | fs |
| ip:1048576 | 0.993 (0.973..1.006) | 3.410 | 3.372 | 0.949 | 0.969 | fs | fs |
| oop:4194304 | 0.989 (0.983..0.995) | 18.735 | 18.548 | 1.314 | 1.325 | fs | fs |
| ip:4194304 | 0.994 (0.985..1.003) | 18.809 | 18.661 | 1.163 | 1.164 | fs | fs |
