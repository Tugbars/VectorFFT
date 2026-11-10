# The FP64 four-step pass by pass, 2^20..2^23 (the in-place plans raced on the scratch store: their banked rows carry no child)

```
Tue Nov 10 14:15:02 RST 2026
N=1048576 FP64  plane 16 MB  reps 16
  oop plan: 4096x256 form 0    ip plan: 4096x256 form 0
  OOP shipped  child    2.651  transpose    0.591                      =    3.242 ms   (err 0.0e+00)
  OOP regular  child    3.027  transpose    1.197                      =    4.224 ms   (err 0.0e+00)
  OOP whole                                                             3.235 ms
  IP  shipped  child    2.610  transpose    0.605  copy back    1.048  =    4.263 ms
  IP  check    child    2.619  transpose    0.623  copy back    0.994  =    4.236 ms   (this file's sweep, streaming)
  IP  A/B first run err 0.0e+00
  IP  A        child    2.600  transpose    0.820  copy back    0.801  =    4.221 ms   (transpose: regular stores; memcpy)
  IP  B        child    2.169  transpose    0.783  copy back    0.679  =    3.631 ms   (transpose and copy: regular stores)
  IP  whole                                                             4.240 ms
  IP child's pieces on one hot row x 4096: inter-pass twiddle   0.311 ms (0.30 ns/point)   row transforms   0.556 ms (0.53 ns/point)
N=2097152 FP64  plane 32 MB  reps 12
  oop plan: 4096x512 form 0    ip plan: 512x4096 form 0
  OOP shipped  child    6.792  transpose    1.434                      =    8.226 ms   (err 0.0e+00)
  OOP regular  child    7.147  transpose    3.112                      =   10.259 ms   (err 0.0e+00)
  OOP whole                                                             8.252 ms
  IP  shipped  child    5.885  transpose    1.373  copy back    1.989  =    9.247 ms
  IP  check    child    5.862  transpose    1.362  copy back    1.955  =    9.179 ms   (this file's sweep, streaming)
  IP  A/B first run err 3.9e-16
  IP  A        child    6.099  transpose    2.629  copy back    1.421  =   10.148 ms   (transpose: regular stores; memcpy)
  IP  B        child    5.677  transpose    2.858  copy back    2.617  =   11.152 ms   (transpose and copy: regular stores)
  IP  whole                                                             9.318 ms
  IP child's pieces on one hot row x 512: inter-pass twiddle   0.571 ms (0.27 ns/point)   row transforms   1.996 ms (0.95 ns/point)
N=4194304 FP64  plane 64 MB  reps 8
  oop plan: 4096x1024 form 0    ip plan: 1024x4096 form 1
  not the form-0 natural plans
N=8388608 FP64  plane 128 MB  reps 6
  oop plan: 4096x2048 form 0    ip plan: 4096x2048 form 0
  OOP shipped  child   34.519  transpose    7.755                      =   42.274 ms   (err 0.0e+00)
  OOP regular  child   34.713  transpose   16.952                      =   51.665 ms   (err 0.0e+00)
  OOP whole                                                            42.017 ms
  IP  shipped  child   31.583  transpose    7.372  copy back    7.127  =   46.082 ms
  IP  check    child   31.667  transpose    7.338  copy back    7.092  =   46.097 ms   (this file's sweep, streaming)
  IP  A/B first run err 0.0e+00
  IP  A        child   31.711  transpose   12.282  copy back    6.783  =   50.777 ms   (transpose: regular stores; memcpy)
  IP  B        child   31.944  transpose   13.889  copy back   12.108  =   57.941 ms   (transpose and copy: regular stores)
  IP  whole                                                            46.161 ms
  IP child's pieces on one hot row x 4096: inter-pass twiddle   2.379 ms (0.28 ns/point)   row transforms   7.303 ms (0.87 ns/point)
Tue Nov 10 14:23:36 RST 2026
```
