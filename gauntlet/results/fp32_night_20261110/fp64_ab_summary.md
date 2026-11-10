# FP64 A/B: 2026-11-10 04:28
old   C:\Users\Tugbars\AppData\Local\Temp\claude\9663899961c0c8e5\c4ec08a0-4e66-47a3-be03-242935e811b2\scratchpad\fp64base\bench_1d_vs_mkl.exe (32044984 bytes, sha1 477b30cd3a55)
new   C:\Users\Tugbars\AppData\Local\Temp\claude\9663899961c0c8e5\c4ec08a0-4e66-47a3-be03-242935e811b2\scratchpad\fp64new\bench_1d_vs_mkl.exe (32045037 bytes, sha1 082bca9434d6)
store C:\Users\Tugbars\AppData\Local\Temp\claude\9663899961c0c8e5\c4ec08a0-4e66-47a3-be03-242935e811b2\scratchpad\ab_store
15 rounds, 27 cells per round per arm, pace 200 ms, threaded cells at T=8

| cell | arm | rounds | new/old median | min..max | MKL/ours old | MKL/ours new | route |
|---|---|---|---|---|---|---|---|
| ctl oop:4096 | - | 15 | 1.001 | 0.998..1.006 | 1.079 | 1.080 | matched (ztt) |
| oop:64 | - | 15 | 1.000 | 1.000..1.233 | 1.033 | 1.033 | matched (2p) |
| oop:128 | - | 15 | 1.000 | 1.000..1.000 | 1.062 | 1.062 | matched (2p) |
| oop:256 | - | 15 | 1.000 | 1.000..1.037 | 1.007 | 1.007 | matched (2p) |
| oop:512 | - | 15 | 1.000 | 0.993..1.003 | 0.987 | 0.983 | matched (2p) |
| oop:1024 | - | 15 | 0.999 | 0.977..1.028 | 1.159 | 1.178 | matched (ztt) |
| oop:2048 | - | 15 | 0.999 | 0.990..1.010 | 1.285 | 1.294 | matched (ztt) |
| oop:4096 | - | 15 | 1.001 | 0.998..1.004 | 1.080 | 1.080 | matched (ztt) |
| oop:8192 | - | 15 | 1.000 | 0.997..1.003 | 1.068 | 1.066 | matched (ztt) |
| oop:16384 | - | 15 | 1.001 | 0.990..1.009 | 1.074 | 1.072 | matched (ztt) |
| oop:32768 | - | 15 | 1.000 | 0.997..1.002 | 0.949 | 0.949 | matched (ztt) |
| oop:65536 | - | 15 | 1.000 | 0.994..1.012 | 1.245 | 1.246 | matched (ztt) |
| oop:131072 | - | 15 | 1.000 | 0.987..1.042 | 1.178 | 1.181 | matched (ztt) |
| oop:262144 | - | 15 | 1.001 | 0.995..1.016 | 1.314 | 1.314 | matched (ztt) |
| oop:524288 | - | 15 | 1.003 | 0.986..1.014 | 1.063 | 1.070 | matched (fs) |
| oop:1048576 | - | 15 | 1.003 | 0.983..1.023 | 1.311 | 1.308 | matched (fs) |
| ip:256 | - | 15 | 1.000 | 0.965..1.022 | 1.000 | 1.007 | matched (2p) |
| ip:512 | - | 15 | 1.000 | 0.980..1.020 | 0.980 | 0.980 | matched (2p) |
| ip:1024 | - | 15 | 1.000 | 0.994..1.006 | 1.050 | 1.049 | matched (ztt) |
| ip:2048 | - | 15 | 0.998 | 0.987..1.003 | 1.140 | 1.168 | matched (ztt) |
| ip:4096 | - | 15 | 0.999 | 0.996..1.005 | 1.069 | 1.070 | matched (ztt) |
| ip:65536 | - | 15 | 1.000 | 0.959..1.010 | 1.136 | 1.127 | matched (ztt) |
| ip:131072 | - | 15 | 1.001 | 0.993..1.018 | 1.158 | 1.156 | matched (ztt) |
| ip:262144 | - | 15 | 1.001 | 0.995..1.015 | 1.226 | 1.221 | matched (ztt) |
| tc:4096:4 (noisier) | st | 15 | 1.000 | 0.989..1.011 | 1.059 | 1.060 | matched (n/a) |
| tc:4096:4 (noisier) | mt | 15 | 1.001 | 0.981..1.028 | 7.174 | 7.262 | matched (n/a) |
| lm:1024:8 | - | 15 | 0.996 | 0.951..1.046 | 1.318 | 1.322 | matched (lm) |
| mt:65536 (noisier) | - | 15 | 0.997 | 0.966..1.078 | 1.694 | 1.797 | matched (ztt) |

new/old = our ns, new arm over old arm, paired within a round (< 1: the new build is faster); MKL/ours = the in-process control of each arm (MKL is the same library in both).

1000 s; rounds.csv beside this file
