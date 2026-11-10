# vfftf_gate after the FP32 ZTURN-T ceiling moved to 2^20: cold scratch store, 2..2^20 (2026-11-10, morning)

```
Tue Nov 10 08:36:53 RST 2026
=== FP32 gate (vfftf.h): 1D C2C pow2 N = 2..1048576, K = 1 and 3 (contiguous), T = 1 and 4, oop + ip, fwd + bwd vs a double FFT, limit 1e-05, oop and ip each; store <scratch>/gate_store3 ===
refusals:
  ok  N = 3                    refused: vfftf_create: n[0] = 3: the FP32 library serves a power-of-two n[0] of 2 or more only
  ok  N = 12                   refused: vfftf_create: n[0] = 12: the FP32 library serves a power-of-two n[0] of 2 or more only
  ok  rank 2, 16 x 16          refused: vfftf_create: n[0] = 16: the FP32 library serves rank 1 only
  ok  real (R2C), N = 64       refused: vfftf_create: n[0] = 64: the FP32 library serves VFFT_C2C only
  ok  split layout, N = 64     refused: vfftf_create: n[0] = 64: the FP32 library serves VFFT_LAYOUT_INTERLEAVED only
served:
 N        K  T  | route oop      route ip       | fwd err   bwd err   | ip fwd    ip bwd    | ip vs oop
[wisdom2] <scratch>/gate_store3: 0 record(s) loaded, writable
 2        1  1  | mono           mono           | 2.83e-08  2.83e-08 | 2.83e-08  2.83e-08 | bitwise
 2        3  1  | -              -              | 3.00e-08  3.00e-08 | 3.00e-08  3.00e-08 | bitwise
 2        1  4  | mono           mono           | 2.83e-08  2.83e-08 | 2.83e-08  2.83e-08 | bitwise
 2        3  4  | -              -              | 3.00e-08  3.00e-08 | 3.00e-08  3.00e-08 | bitwise
 4        1  1  | mono           mono           | 7.26e-08  7.26e-08 | 7.26e-08  7.26e-08 | bitwise
 4        3  1  | -              -              | 6.11e-08  6.11e-08 | 6.11e-08  6.11e-08 | bitwise
 4        1  4  | mono           mono           | 7.26e-08  7.26e-08 | 7.26e-08  7.26e-08 | bitwise
 4        3  4  | -              -              | 6.11e-08  6.11e-08 | 6.11e-08  6.11e-08 | bitwise
 8        1  1  | mono           mono           | 4.27e-08  4.27e-08 | 4.27e-08  4.27e-08 | bitwise
 8        3  1  | -              -              | 4.69e-08  4.69e-08 | 4.69e-08  4.69e-08 | bitwise
 8        1  4  | mono           mono           | 4.27e-08  4.27e-08 | 4.27e-08  4.27e-08 | bitwise
 8        3  4  | -              -              | 4.69e-08  4.69e-08 | 4.69e-08  4.69e-08 | bitwise
 16       1  1  | 2p             2p             | 5.38e-08  5.38e-08 | 5.38e-08  5.38e-08 | bitwise
 16       3  1  | -              -              | 6.48e-08  6.48e-08 | 6.48e-08  6.48e-08 | bitwise
 16       1  4  | 2p             2p             | 5.38e-08  5.38e-08 | 5.38e-08  5.38e-08 | bitwise
 16       3  4  | -              -              | 6.48e-08  6.48e-08 | 6.48e-08  6.48e-08 | bitwise
 32       1  1  | 2p             2p             | 9.47e-08  1.08e-07 | 9.47e-08  1.08e-07 | bitwise
 32       3  1  | -              -              | 1.18e-07  6.79e-08 | 1.18e-07  6.79e-08 | bitwise
 32       1  4  | 2p             2p             | 9.47e-08  1.08e-07 | 9.47e-08  1.08e-07 | bitwise
 32       3  4  | -              -              | 1.18e-07  6.79e-08 | 1.18e-07  6.79e-08 | bitwise
 64       1  1  | 2p             2p             | 8.99e-08  8.10e-08 | 8.99e-08  8.10e-08 | bitwise
 64       3  1  | -              -              | 1.03e-07  1.03e-07 | 1.03e-07  1.03e-07 | bitwise
 64       1  4  | 2p             2p             | 8.99e-08  8.10e-08 | 8.99e-08  8.10e-08 | bitwise
 64       3  4  | -              -              | 1.03e-07  1.03e-07 | 1.03e-07  1.03e-07 | bitwise
 128      1  1  | 2p             2p             | 7.68e-08  8.37e-08 | 7.68e-08  8.37e-08 | bitwise
 128      3  1  | -              -              | 1.15e-07  1.08e-07 | 1.15e-07  1.08e-07 | bitwise
 128      1  4  | 2p             2p             | 7.68e-08  8.37e-08 | 7.68e-08  8.37e-08 | bitwise
 128      3  4  | -              -              | 1.15e-07  1.08e-07 | 1.15e-07  1.08e-07 | bitwise
 256      1  1  | 2p             2p             | 1.04e-07  9.17e-08 | 1.04e-07  9.17e-08 | bitwise
 256      3  1  | -              -              | 1.17e-07  8.62e-08 | 1.17e-07  8.62e-08 | bitwise
 256      1  4  | 2p             2p             | 1.04e-07  9.17e-08 | 1.04e-07  9.17e-08 | bitwise
 256      3  4  | -              -              | 1.17e-07  8.62e-08 | 1.17e-07  8.62e-08 | bitwise
 512      1  1  | 2p             2p             | 9.63e-08  1.40e-07 | 9.63e-08  1.40e-07 | bitwise
 512      3  1  | -              -              | 1.28e-07  1.40e-07 | 1.28e-07  1.40e-07 | bitwise
 512      1  4  | 2p             2p             | 9.63e-08  1.40e-07 | 9.63e-08  1.40e-07 | bitwise
 512      3  4  | -              -              | 1.28e-07  1.40e-07 | 1.28e-07  1.40e-07 | bitwise
 1024     1  1  | 2p             2p             | 1.16e-07  1.23e-07 | 1.16e-07  1.23e-07 | bitwise
 1024     3  1  | -              -              | 1.34e-07  1.24e-07 | 1.34e-07  1.24e-07 | bitwise
 1024     1  4  | 2p             2p             | 1.16e-07  1.23e-07 | 1.16e-07  1.23e-07 | bitwise
 1024     3  4  | -              -              | 1.34e-07  1.24e-07 | 1.34e-07  1.24e-07 | bitwise
 2048     1  1  | ztt            ztt            | 1.27e-07  1.20e-07 | 1.27e-07  1.20e-07 | bitwise
 2048     3  1  | -              -              | 1.27e-07  1.28e-07 | 1.27e-07  1.28e-07 | bitwise
 2048     1  4  | ztt            ztt            | 1.27e-07  1.20e-07 | 1.27e-07  1.20e-07 | bitwise
 2048     3  4  | -              -              | 1.27e-07  1.28e-07 | 1.27e-07  1.28e-07 | bitwise
 4096     1  1  | ztt            ztt            | 1.40e-07  1.22e-07 | 1.40e-07  1.22e-07 | bitwise
 4096     3  1  | -              -              | 1.35e-07  1.29e-07 | 1.35e-07  1.29e-07 | bitwise
 4096     1  4  | ztt            ztt            | 1.40e-07  1.22e-07 | 1.40e-07  1.22e-07 | bitwise
 4096     3  4  | -              -              | 1.35e-07  1.29e-07 | 1.35e-07  1.29e-07 | bitwise
 8192     1  1  | ztt            ztt            | 1.61e-07  1.53e-07 | 1.54e-07  1.56e-07 | own plan
 8192     3  1  | -              -              | 1.78e-07  1.78e-07 | 1.71e-07  1.67e-07 | own plan
 8192     1  4  | ztt            ztt            | 1.61e-07  1.53e-07 | 1.54e-07  1.56e-07 | own plan
 8192     3  4  | -              -              | 1.78e-07  1.78e-07 | 1.71e-07  1.67e-07 | own plan
 16384    1  1  | ztt            ztt            | 1.48e-07  1.52e-07 | 1.48e-07  1.52e-07 | bitwise
 16384    3  1  | -              -              | 1.49e-07  1.52e-07 | 1.49e-07  1.52e-07 | bitwise
 16384    1  4  | ztt            ztt            | 1.48e-07  1.52e-07 | 1.48e-07  1.52e-07 | bitwise
 16384    3  4  | -              -              | 1.49e-07  1.52e-07 | 1.49e-07  1.52e-07 | bitwise
 32768    1  1  | ztt            ztt            | 1.24e-07  1.39e-07 | 1.24e-07  1.39e-07 | bitwise
 32768    3  1  | -              -              | 1.28e-07  1.39e-07 | 1.28e-07  1.39e-07 | bitwise
 32768    1  4  | ztt            ztt            | 1.24e-07  1.39e-07 | 1.24e-07  1.39e-07 | bitwise
 32768    3  4  | -              -              | 1.28e-07  1.39e-07 | 1.28e-07  1.39e-07 | bitwise
 65536    1  1  | ztt            ztt            | 1.86e-07  1.68e-07 | 1.86e-07  1.68e-07 | bitwise
 65536    3  1  | -              -              | 1.86e-07  1.87e-07 | 1.86e-07  1.87e-07 | bitwise
 65536    1  4  | ztt            ztt            | 1.86e-07  1.68e-07 | 1.86e-07  1.68e-07 | bitwise
 65536    3  4  | -              -              | 1.86e-07  1.87e-07 | 1.86e-07  1.87e-07 | bitwise
 131072   1  1  | ztt            ztt            | 1.63e-07  1.74e-07 | 1.63e-07  1.74e-07 | bitwise
 131072   3  1  | -              -              | 1.76e-07  1.61e-07 | 1.76e-07  1.61e-07 | bitwise
 131072   1  4  | ztt            ztt            | 1.63e-07  1.74e-07 | 1.63e-07  1.74e-07 | bitwise
 131072   3  4  | -              -              | 1.76e-07  1.61e-07 | 1.76e-07  1.61e-07 | bitwise
 262144   1  1  | ztt            ztt            | 1.68e-07  1.66e-07 | 1.70e-07  1.67e-07 | own plan
 262144   3  1  | -              -              | 1.82e-07  1.66e-07 | 1.81e-07  1.67e-07 | own plan
 262144   1  4  | ztt            ztt            | 1.68e-07  1.66e-07 | 1.70e-07  1.67e-07 | own plan
 262144   3  4  | -              -              | 1.82e-07  1.66e-07 | 1.81e-07  1.67e-07 | own plan
 524288   1  1  | ztt            ztt            | 1.85e-07  2.02e-07 | 1.85e-07  2.02e-07 | bitwise
 524288   3  1  | -              -              | 1.82e-07  1.99e-07 | 1.82e-07  1.99e-07 | bitwise
 524288   1  4  | ztt            ztt            | 1.85e-07  2.02e-07 | 1.85e-07  2.02e-07 | bitwise
 524288   3  4  | -              -              | 1.82e-07  1.99e-07 | 1.82e-07  1.99e-07 | bitwise
 1048576  1  1  | ztt            ztt            | 1.77e-07  1.69e-07 | 1.77e-07  1.69e-07 | bitwise
 1048576  3  1  | -              -              | 1.83e-07  1.80e-07 | 1.83e-07  1.80e-07 | bitwise
 1048576  1  4  | ztt            ztt            | 1.77e-07  1.69e-07 | 1.77e-07  1.69e-07 | bitwise
 1048576  3  4  | -              -              | 1.83e-07  1.80e-07 | 1.83e-07  1.80e-07 | bitwise
largest error: forward 1.86e-07 (N = 65536), backward 2.02e-07 (N = 524288); limit 1e-05
=== ALL PASS === (80 served cells, 5 refusals)
gate exit 0
Tue Nov 10 08:42:21 RST 2026
```
