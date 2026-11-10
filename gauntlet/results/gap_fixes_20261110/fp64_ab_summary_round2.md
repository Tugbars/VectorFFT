# FP64 A/B: 2026-11-10 13:50
old   <scratch>\fp64base\bench_1d_vs_mkl.exe (32044984 bytes, sha1 477b30cd3a55)
new   <scratch>\gapfix\r2\f64\bench_1d_vs_mkl.exe (32046113 bytes, sha1 89ad4ffa0d1d)
store <scratch>\ab_store
15 rounds, 27 cells per round per arm, pace 200 ms, threaded cells at T=8

| cell | arm | rounds | new/old median | min..max | MKL/ours old | MKL/ours new | route |
|---|---|---|---|---|---|---|---|
| ctl oop:4096 | - | 15 | 1.000 | 0.995..1.004 | 1.080 | 1.079 | matched (ztt) |
| oop:64 | - | 15 | 1.000 | 1.000..1.033 | 1.033 | 1.033 | matched (2p) |
| oop:128 | - | 15 | 1.000 | 1.000..1.000 | 1.062 | 1.078 | matched (2p) |
| oop:256 | - | 15 | 1.000 | 1.000..1.169 | 1.007 | 1.007 | matched (2p) |
| oop:512 | - | 15 | 1.000 | 1.000..1.091 | 0.987 | 0.987 | matched (2p) |
| oop:1024 | - | 15 | 1.001 | 0.986..1.011 | 1.176 | 1.155 | matched (ztt) |
| oop:2048 | - | 15 | 1.000 | 0.996..1.004 | 1.284 | 1.294 | matched (ztt) |
| oop:4096 | - | 15 | 0.999 | 0.994..1.004 | 1.078 | 1.079 | matched (ztt) |
| oop:8192 | - | 15 | 1.000 | 0.998..1.003 | 1.070 | 1.067 | matched (ztt) |
| oop:16384 | - | 15 | 0.999 | 0.994..1.005 | 1.076 | 1.077 | matched (ztt) |
| oop:32768 | - | 15 | 1.000 | 0.995..1.005 | 0.950 | 0.950 | matched (ztt) |
| oop:65536 | - | 15 | 1.003 | 0.992..1.012 | 1.246 | 1.244 | matched (ztt) |
| oop:131072 | - | 15 | 1.001 | 0.981..1.035 | 1.173 | 1.179 | matched (ztt) |
| oop:262144 | - | 15 | 1.000 | 0.991..1.020 | 1.312 | 1.322 | matched (ztt) |
| oop:524288 | - | 15 | 0.995 | 0.962..1.022 | 1.054 | 1.059 | matched (fs) |
| oop:1048576 | - | 15 | 1.001 | 0.985..1.062 | 1.295 | 1.298 | matched (fs) |
| ip:256 | - | 15 | 1.000 | 0.819..1.000 | 1.007 | 1.007 | matched (2p) |
| ip:512 | - | 15 | 1.000 | 0.997..1.020 | 0.980 | 0.980 | matched (2p) |
| ip:1024 | - | 15 | 0.997 | 0.992..1.008 | 1.058 | 1.054 | matched (ztt) |
| ip:2048 | - | 15 | 1.001 | 0.988..1.010 | 1.137 | 1.133 | matched (ztt) |
| ip:4096 | - | 15 | 1.000 | 0.994..1.004 | 1.069 | 1.074 | matched (ztt) |
| ip:65536 | - | 15 | 1.005 | 0.969..1.028 | 1.141 | 1.139 | matched (ztt) |
| ip:131072 | - | 15 | 1.000 | 0.987..1.025 | 1.157 | 1.158 | matched (ztt) |
| ip:262144 | - | 15 | 1.001 | 0.981..1.006 | 1.187 | 1.194 | matched (ztt) |
| tc:4096:4 (noisier) | st | 15 | 1.000 | 0.993..1.260 | 1.058 | 1.061 | matched (n/a) |
| tc:4096:4 (noisier) | mt | 15 | 0.995 | 0.973..1.097 | 7.352 | 7.239 | matched (n/a) |
| lm:1024:8 | - | 15 | 0.986 | 0.957..1.030 | 1.308 | 1.320 | matched (lm) |
| mt:65536 (noisier) | - | 15 | 1.009 | 0.977..1.028 | 1.931 | 1.880 | matched (ztt) |

new/old = our ns, new arm over old arm, paired within a round (< 1: the new build is faster); MKL/ours = the in-process control of each arm (MKL is the same library in both).

999 s; rounds.csv beside this file
