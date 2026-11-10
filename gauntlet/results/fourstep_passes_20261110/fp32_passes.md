# The FP32 four-step pass by pass, 2^21..2^24 (the banked 14900KF plans)

```
Tue Nov 10 14:26:21 RST 2026
N=2097152 FP32  plane 16 MB  reps 24
  oop plan: 512x4096 form 0    ip plan: 512x4096 form 0
  OOP shipped  child    4.436  transpose    0.724                      =    5.160 ms   (err 0.0e+00)
  OOP regular  child    4.682  transpose    1.282                      =    5.965 ms   (err 0.0e+00)
  OOP whole                                                             5.168 ms
  IP  shipped  child    4.274  transpose    0.755  copy back    1.065  =    6.094 ms
  IP  check    child    4.238  transpose    0.781  copy back    0.943  =    5.962 ms   (this file's sweep, streaming)
  IP  A/B first run err 0.0e+00
  IP  A        child    4.276  transpose    1.330  copy back    0.675  =    6.282 ms   (transpose: regular stores; memcpy)
  IP  B        child    3.827  transpose    1.250  copy back    0.758  =    5.835 ms   (transpose and copy: regular stores)
  IP  whole                                                             6.027 ms
  IP child's pieces on one hot row x 512: inter-pass twiddle   1.674 ms (0.80 ns/point)   row transforms   1.041 ms (0.50 ns/point)
  EXPERIMENT the twiddle by hand in the element type (B = 64):   0.286 ms (0.14 ns/point), against the library's on the same rows 1.2e-07
  IP  no-copy form 512x4096 chain 8.8.8                                     5.956 ms   (err 0.0e+00)
  IP  no-copy form 512x4096 chain 8.4.16                                    6.937 ms   (err 1.9e-07)
N=4194304 FP32  plane 32 MB  reps 16
  oop plan: 2048x2048 form 0    ip plan: 1024x4096 form 0
  OOP shipped  child   10.107  transpose    1.659                      =   11.766 ms   (err 0.0e+00)
  OOP regular  child   10.400  transpose    3.582                      =   13.982 ms   (err 0.0e+00)
  OOP whole                                                            11.751 ms
  IP  shipped  child    9.412  transpose    1.624  copy back    1.994  =   13.030 ms
  IP  check    child    9.374  transpose    1.630  copy back    1.951  =   12.955 ms   (this file's sweep, streaming)
  IP  A/B first run err 5.5e-07
  IP  A        child    9.419  transpose    3.613  copy back    1.698  =   14.730 ms   (transpose: regular stores; memcpy)
  IP  B        child    9.356  transpose    3.673  copy back    2.537  =   15.567 ms   (transpose and copy: regular stores)
  IP  whole                                                            12.949 ms
  IP child's pieces on one hot row x 1024: inter-pass twiddle   3.352 ms (0.80 ns/point)   row transforms   2.082 ms (0.50 ns/point)
  EXPERIMENT the twiddle by hand in the element type (B = 64):   0.571 ms (0.14 ns/point), against the library's on the same rows 1.2e-07
  IP  no-copy form 1024x4096 chain 8.8.16                                   14.773 ms   (err 3.2e-07)
  IP  no-copy form 1024x4096 chain 8.16.8                                   13.688 ms   (err 3.2e-07)
N=8388608 FP32  plane 64 MB  reps 10
  oop plan: 4096x2048 form 0    ip plan: 2048x4096 form 0
  OOP shipped  child   22.340  transpose    3.583                      =   25.922 ms   (err 0.0e+00)
  OOP regular  child   22.437  transpose    8.630                      =   31.067 ms   (err 0.0e+00)
  OOP whole                                                            25.876 ms
  IP  shipped  child   20.626  transpose    3.525  copy back    3.661  =   27.812 ms
  IP  check    child   20.662  transpose    3.496  copy back    3.705  =   27.864 ms   (this file's sweep, streaming)
  IP  A/B first run err 1.2e-06
  IP  A        child   20.661  transpose    8.101  copy back    3.495  =   32.257 ms   (transpose: regular stores; memcpy)
  IP  B        child   20.405  transpose    8.172  copy back    5.798  =   34.375 ms   (transpose and copy: regular stores)
  IP  whole                                                            27.941 ms
  IP child's pieces on one hot row x 2048: inter-pass twiddle   6.738 ms (0.80 ns/point)   row transforms   4.161 ms (0.50 ns/point)
  EXPERIMENT the twiddle by hand in the element type (B = 64):   1.144 ms (0.14 ns/point), against the library's on the same rows 1.2e-07
  IP  no-copy form 2048x4096 chain 8.16.16                                  32.416 ms   (err 1.1e-06)
  IP  no-copy form 2048x4096 chain 8.32.8                                   32.046 ms   (err 1.0e-06)
N=16777216 FP32  plane 128 MB  reps 8
  oop plan: 4096x4096 form 0    ip plan: 4096x4096 form 0
  OOP shipped  child   46.605  transpose    7.497                      =   54.102 ms   (err 0.0e+00)
  OOP regular  child   46.634  transpose   19.203                      =   65.837 ms   (err 0.0e+00)
  OOP whole                                                            54.283 ms
  IP  shipped  child   41.317  transpose    7.487  copy back    7.299  =   56.103 ms
  IP  check    child   41.101  transpose    7.310  copy back    7.186  =   55.598 ms   (this file's sweep, streaming)
  IP  A/B first run err 0.0e+00
  IP  A        child   41.257  transpose   15.168  copy back    6.871  =   63.296 ms   (transpose: regular stores; memcpy)
  IP  B        child   41.076  transpose   15.214  copy back   12.221  =   68.511 ms   (transpose and copy: regular stores)
  IP  whole                                                            55.690 ms
  IP child's pieces on one hot row x 4096: inter-pass twiddle  13.474 ms (0.80 ns/point)   row transforms   8.001 ms (0.48 ns/point)
  EXPERIMENT the twiddle by hand in the element type (B = 64):   2.295 ms (0.14 ns/point), against the library's on the same rows 1.2e-07
  IP  no-copy form 4096x4096 chain 8.32.16                                  71.727 ms   (err 2.2e-07)
  IP  no-copy form 4096x4096 chain 8.64.8                                   69.841 ms   (err 2.2e-07)
Tue Nov 10 14:26:36 RST 2026
```
