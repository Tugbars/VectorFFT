# The FP64 twiddle against a long-double reference: the AVX2 build

`lib` = the library's `_il2d_fs_twiddle`; `v1` = by hand, w = C[a] F[b] then x w; `v2` = the library's form written again in the probe. Error of each point against the product of the record's own double values computed in long double, in units of 2^-53: max and rms over 24 plane positions. The row starts 8 bytes off a vector boundary and has 16 guard doubles behind it.

```
the four-step's inter-pass twiddle, FP64, build ISA avx2: error against the long-double reference, units of 2^-53
    B     row  dir     lib max   rms      v1 max   rms      v2 max   rms
   64    4096  fwd       2.879 0.708       2.202 0.655       2.879 0.708
   64    4096  bwd       3.014 0.705       2.307 0.655       3.014 0.705
   32    2048  fwd       2.605 0.701       2.172 0.652       2.605 0.701
   32    2048  bwd       2.668 0.703       2.273 0.655       2.668 0.703
   32    1024  fwd       2.728 0.700       2.110 0.649       2.728 0.700
   32    1024  bwd       2.461 0.698       2.501 0.653       2.461 0.698
   16     512  fwd       2.502 0.693       2.294 0.649       2.502 0.693
   16     512  bwd       2.757 0.695       2.171 0.648       2.757 0.695
   16     256  fwd       2.223 0.696       2.177 0.648       2.223 0.696
   16     256  bwd       2.795 0.698       1.995 0.645       2.795 0.698
    8      64  fwd       2.313 0.668       1.642 0.622       2.313 0.668
    8      64  bwd       2.166 0.670       1.934 0.623       2.166 0.670
    2       4  fwd       1.507 0.503       1.625 0.469       1.507 0.503
    2       4  bwd       1.265 0.461       1.703 0.483       1.265 0.461
    1       1  fwd       0.000 0.000           -     -           -     -
    1       1  bwd       0.000 0.000           -     -           -     -
    1       3  fwd       1.403 0.427           -     -           -     -
    1       3  bwd       1.000 0.444           -     -           -     -
    2       7  fwd       1.605 0.501           -     -           -     -
    2       7  bwd       2.336 0.538           -     -           -     -
    4      10  fwd       1.797 0.560           -     -           -     -
    4      10  bwd       1.577 0.580           -     -           -     -
    4      23  fwd       2.008 0.656           -     -           -     -
    4      23  bwd       2.080 0.645           -     -           -     -
    8     100  fwd       2.630 0.681           -     -           -     -
    8     100  bwd       2.069 0.665           -     -           -     -
    8      61  fwd       2.019 0.671           -     -           -     -
    8      61  bwd       1.819 0.656           -     -           -     -
    8     243  fwd       2.244 0.691           -     -           -     -
    8     243  bwd       2.257 0.684           -     -           -     -
   16     500  fwd       2.432 0.688           -     -           -     -
   16     500  bwd       2.501 0.688           -     -           -     -
   16      87  fwd       1.980 0.653           -     -           -     -
   16      87  bwd       2.166 0.647           -     -           -     -
   16     191  fwd       2.578 0.687           -     -           -     -
   16     191  bwd       2.696 0.682           -     -           -     -
   32    1000  fwd       2.542 0.700           -     -           -     -
   32    1000  bwd       2.543 0.693           -     -           -     -
   64    4076  fwd       2.978 0.704           -     -           -     -
   64    4076  bwd       2.855 0.706           -     -           -     -
   64    4099  fwd       2.966 0.704           -     -           -     -
   64    4099  bwd       2.882 0.704           -     -           -     -
  128   16384  fwd       2.961 0.709           -     -           -     -
  128   16384  bwd       2.908 0.709           -     -           -     -
  256   65536  fwd       3.173 0.710           -     -           -     -
  256   65536  bwd       3.160 0.709           -     -           -     -
  128     300  fwd       2.184 0.641           -     -           -     -
  128     300  bwd       2.525 0.627           -     -           -     -
  256    1287  fwd       2.598 0.671           -     -           -     -
  256    1287  bwd       2.643 0.674           -     -           -     -
  300    1000  fwd       2.328 0.656           -     -           -     -
  300    1000  bwd       2.574 0.659           -     -           -     -
   67     500  fwd       2.425 0.687           -     -           -     -
   67     500  bwd       2.726 0.688           -     -           -     -
CHECK PASSED
```
