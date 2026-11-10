# The canonical FP32 bench after the change (3 processes per cell, flips 0, 1, 0; the median ratio)

```
oop:2097152:1    ratio 1.320 (1.250..1.330)  ours    3881375 ns  MKL    5066375 ns  err 2.7e-06  fs
ip:2097152:1     ratio 0.860 (0.840..0.880)  ours    4648738 ns  MKL    3987412 ns  err 2.7e-06  fs
oop:4194304:1    ratio 1.360 (1.360..1.380)  ours    9149075 ns  MKL   12463950 ns  err 7.6e-07  fs
ip:4194304:1     ratio 1.080 (1.070..1.080)  ours   10343650 ns  MKL   11114525 ns  err 1.0e-06  fs
oop:8388608:1    ratio 1.540 (1.500..1.550)  ours   21299812 ns  MKL   32437400 ns  err 1.0e-06  fs
ip:8388608:1     ratio 1.150 (1.140..1.160)  ours   22715300 ns  MKL   26326200 ns  err 7.9e-07  fs
oop:16777216:1   ratio 1.520 (1.510..1.520)  ours   44033875 ns  MKL   66891963 ns  err 1.2e-06  fs
ip:16777216:1    ratio 1.210 (1.200..1.240)  ours   45483238 ns  MKL   55679962 ns  err 1.2e-06  fs
```
