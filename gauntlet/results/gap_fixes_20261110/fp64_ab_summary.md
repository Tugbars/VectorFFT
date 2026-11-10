# FP64 A/B: 2026-11-10 12:51
old   <scratch>\fp64base\bench_1d_vs_mkl.exe (32044984 bytes, sha1 477b30cd3a55)
new   <scratch>\gapfix\f64\bench_1d_vs_mkl.exe (32046625 bytes, sha1 4bf8f3b9740e)
store <scratch>\ab_store
15 rounds, 27 cells per round per arm, pace 200 ms, threaded cells at T=8

| cell | arm | rounds | new/old median | min..max | MKL/ours old | MKL/ours new | route |
|---|---|---|---|---|---|---|---|
| ctl oop:4096 | - | 15 | 0.999 | 0.524..1.048 | 1.081 | 1.081 | matched (ztt) |
| oop:64 | - | 15 | 1.000 | 1.000..1.267 | 1.033 | 1.033 | matched (2p) |
| oop:128 | - | 15 | 1.000 | 1.000..1.000 | 1.062 | 1.062 | matched (2p) |
| oop:256 | - | 15 | 1.000 | 1.000..1.007 | 1.007 | 1.007 | matched (2p) |
| oop:512 | - | 15 | 1.000 | 0.971..1.054 | 0.987 | 1.003 | matched (2p) |
| oop:1024 | - | 15 | 1.001 | 0.939..1.073 | 1.212 | 1.194 | matched (ztt) |
| oop:2048 | - | 15 | 1.000 | 0.950..1.061 | 1.306 | 1.287 | matched (ztt) |
| oop:4096 | - | 15 | 1.000 | 0.800..1.460 | 1.079 | 1.081 | matched (ztt) |
| oop:8192 | - | 15 | 1.001 | 0.940..1.075 | 1.070 | 1.068 | matched (ztt) |
| oop:16384 | - | 15 | 0.998 | 0.753..1.065 | 1.072 | 1.078 | matched (ztt) |
| oop:32768 | - | 15 | 0.998 | 0.799..1.098 | 0.962 | 0.950 | matched (ztt) |
| oop:65536 | - | 15 | 0.998 | 0.878..1.031 | 1.245 | 1.243 | matched (ztt) |
| oop:131072 | - | 15 | 1.000 | 0.910..1.055 | 1.178 | 1.170 | matched (ztt) |
| oop:262144 | - | 15 | 0.999 | 0.970..1.099 | 1.307 | 1.288 | matched (ztt) |
| oop:524288 | - | 15 | 1.004 | 0.985..1.037 | 1.065 | 1.062 | matched (fs) |
| oop:1048576 | - | 15 | 0.998 | 0.888..1.023 | 1.293 | 1.297 | matched (fs) |
| ip:256 | - | 15 | 1.000 | 0.993..1.000 | 1.007 | 1.007 | matched (2p) |
| ip:512 | - | 15 | 1.000 | 0.965..1.051 | 0.980 | 0.980 | matched (2p) |
| ip:1024 | - | 15 | 0.999 | 0.969..1.012 | 1.050 | 1.058 | matched (ztt) |
| ip:2048 | - | 15 | 1.003 | 0.993..1.029 | 1.147 | 1.158 | matched (ztt) |
| ip:4096 | - | 15 | 0.999 | 0.993..1.030 | 1.073 | 1.069 | matched (ztt) |
| ip:65536 | - | 15 | 1.000 | 0.946..1.047 | 1.132 | 1.128 | matched (ztt) |
| ip:131072 | - | 15 | 1.002 | 0.958..1.046 | 1.149 | 1.150 | matched (ztt) |
| ip:262144 | - | 15 | 0.999 | 0.869..1.024 | 1.198 | 1.214 | matched (ztt) |
| tc:4096:4 (noisier) | st | 15 | 1.000 | 0.856..1.563 | 1.061 | 1.061 | matched (n/a) |
| tc:4096:4 (noisier) | mt | 15 | 0.999 | 0.963..1.246 | 7.530 | 7.383 | matched (n/a) |
| lm:1024:8 | - | 15 | 0.999 | 0.508..1.031 | 1.305 | 1.291 | matched (lm) |
| mt:65536 (noisier) | - | 15 | 1.011 | 0.965..1.076 | 1.738 | 1.779 | matched (ztt) |

new/old = our ns, new arm over old arm, paired within a round (< 1: the new build is faster); MKL/ours = the in-process control of each arm (MKL is the same library in both).

1006 s; rounds.csv beside this file
