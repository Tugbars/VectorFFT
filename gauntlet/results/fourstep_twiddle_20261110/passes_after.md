# The FP32 four-step pass by pass after the change (the banked 14900KF plans)

One run on a warmer machine than the bench runs: compare inside this table, not against other runs.

```
Tue Nov 10 15:41:21 RST 2026
N=2097152 FP32  plane 16 MB  reps 20
  oop plan: 512x4096 form 0    ip plan: 512x4096 form 0
  OOP shipped  child    3.406  transpose    0.846                      =    4.252 ms   (err 0.0e+00)
  OOP regular  child    3.458  transpose    1.656                      =    5.115 ms   (err 0.0e+00)
  OOP whole                                                             4.182 ms
  IP  shipped  child    2.915  transpose    0.819  copy back    0.965  =    4.699 ms
  IP  check    child    2.928  transpose    0.821  copy back    0.938  =    4.686 ms   (this file's sweep, streaming)
  IP  A/B first run err 0.0e+00
  IP  A        child    2.981  transpose    1.493  copy back    0.690  =    5.163 ms   (transpose: regular stores; memcpy)
  IP  B        child    2.489  transpose    1.393  copy back    0.892  =    4.774 ms   (transpose and copy: regular stores)
  IP  whole                                                             4.702 ms
  IP child's pieces on one hot row x 512: inter-pass twiddle   0.224 ms (0.11 ns/point)   row transforms   1.005 ms (0.48 ns/point)
N=4194304 FP32  plane 32 MB  reps 14
  oop plan: 2048x2048 form 0    ip plan: 1024x4096 form 0
  OOP shipped  child    7.570  transpose    1.731                      =    9.301 ms   (err 0.0e+00)
  OOP regular  child    7.969  transpose    3.956                      =   11.926 ms   (err 0.0e+00)
  OOP whole                                                             9.312 ms
  IP  shipped  child    6.789  transpose    1.684  copy back    1.946  =   10.419 ms
  IP  check    child    6.832  transpose    1.665  copy back    1.954  =   10.452 ms   (this file's sweep, streaming)
  IP  A/B first run err 5.0e-07
  IP  A        child    6.865  transpose    3.833  copy back    1.580  =   12.278 ms   (transpose: regular stores; memcpy)
  IP  B        child    6.768  transpose    3.918  copy back    2.918  =   13.604 ms   (transpose and copy: regular stores)
  IP  whole                                                            10.468 ms
  IP child's pieces on one hot row x 1024: inter-pass twiddle   0.456 ms (0.11 ns/point)   row transforms   1.988 ms (0.47 ns/point)
N=8388608 FP32  plane 64 MB  reps 10
  oop plan: 4096x2048 form 0    ip plan: 2048x4096 form 0
  OOP shipped  child   17.356  transpose    3.793                      =   21.149 ms   (err 0.0e+00)
  OOP regular  child   17.289  transpose    8.865                      =   26.154 ms   (err 0.0e+00)
  OOP whole                                                            22.071 ms
  IP  shipped  child   16.514  transpose    3.769  copy back    3.733  =   24.016 ms
  IP  check    child   16.451  transpose    3.745  copy back    3.765  =   23.961 ms   (this file's sweep, streaming)
  IP  A/B first run err 1.2e-06
  IP  A        child   15.922  transpose    8.429  copy back    3.495  =   27.847 ms   (transpose: regular stores; memcpy)
  IP  B        child   15.886  transpose    8.411  copy back    6.336  =   30.632 ms   (transpose and copy: regular stores)
  IP  whole                                                            23.073 ms
  IP child's pieces on one hot row x 2048: inter-pass twiddle   0.917 ms (0.11 ns/point)   row transforms   4.019 ms (0.48 ns/point)
N=16777216 FP32  plane 128 MB  reps 8
  oop plan: 4096x4096 form 0    ip plan: 4096x4096 form 0
  OOP shipped  child   37.324  transpose    8.167                      =   45.490 ms   (err 0.0e+00)
  OOP regular  child   37.188  transpose   19.681                      =   56.869 ms   (err 0.0e+00)
  OOP whole                                                            44.827 ms
  IP  shipped  child   31.234  transpose    7.577  copy back    7.319  =   46.130 ms
  IP  check    child   31.386  transpose    7.480  copy back    7.346  =   46.212 ms   (this file's sweep, streaming)
  IP  A/B first run err 0.0e+00
  IP  A        child   31.567  transpose   16.866  copy back    7.202  =   55.635 ms   (transpose: regular stores; memcpy)
  IP  B        child   31.783  transpose   16.851  copy back   12.504  =   61.138 ms   (transpose and copy: regular stores)
  IP  whole                                                            46.300 ms
  IP child's pieces on one hot row x 4096: inter-pass twiddle   1.842 ms (0.11 ns/point)   row transforms   8.057 ms (0.48 ns/point)
```
