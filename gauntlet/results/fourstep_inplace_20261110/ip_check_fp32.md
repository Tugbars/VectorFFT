# The FP32 four-step in place against out of place, through the public door (the re-raced rows)

`fwd ip` = the in-place forward against the out-of-place plan's forward of the same input; `rt ip` = in-place
forward then backward against N times the input; `1buf` = the out-of-place plan called with one buffer against its
two-buffer result; `bwd oop` = the out-of-place roundtrip. Errors are max |a - b| / max |b|. The times alternate
the two placements inside one process: a rough comparison, the bench is the measurement.

```
FP32, T = 1, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop
   2097152  fs         fs         |  2.15e-07  7.15e-07   bitwise  7.75e-07 |     4.199     4.351  1.036
   4194304  fs         fs         |  3.03e-07  8.34e-07   bitwise  7.75e-07 |     9.477     9.569  1.010
   8388608  fs         fs         |  0.00e+00  8.34e-07   bitwise  8.34e-07 |    20.597    20.581  0.999
  16777216  fs         fs         |  0.00e+00  8.94e-07   bitwise  8.94e-07 |    43.583    43.837  1.006
CHECK PASSED
```
