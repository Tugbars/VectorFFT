# The FP64 four-step through the public door, before and after, alternated (two runs each)

`fwd ip` = the in-place forward against the out-of-place plan's; `rt ip` = the in-place roundtrip; `1buf` = the
out-of-place plan on one buffer; `bwd oop` = the out-of-place roundtrip (errors, max |a - b| / max |b|). `oop ms`,
`ip ms` = forward, `bwd ms` = the out-of-place backward: the median of 9 runs, the three alternating inside one
process.

```
=== before
FP64, T = 1, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop    bwd ms
    524288  fs         fs         |  4.27e-16  1.33e-15   bitwise  1.22e-15 |     1.819     1.874  1.030     2.571
   1000000  fs         fs         |  0.00e+00  1.86e-15   bitwise  1.86e-15 |     4.483     4.446  0.992     5.669
   1048576  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |     4.322     4.144  0.959     6.298
   2097152  fs         fs         |  3.00e-16  1.67e-15   bitwise  1.44e-15 |     9.258     9.095  0.982    12.378
   4194304  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |    19.870    19.949  1.004    25.464
CHECK PASSED
=== after
FP64, T = 1, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop    bwd ms
    524288  fs         fs         |  4.27e-16  1.33e-15   bitwise  1.11e-15 |     1.826     1.827  1.001     2.400
   1000000  fs         fs         |  0.00e+00  1.86e-15   bitwise  1.86e-15 |     4.461     4.381  0.982     5.446
   1048576  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |     4.275     4.043  0.946     5.945
   2097152  fs         fs         |  4.01e-16  1.67e-15   bitwise  1.44e-15 |     9.260     9.010  0.973    12.016
   4194304  fs         fs         |  0.00e+00  1.55e-15   bitwise  1.55e-15 |    19.921    19.973  1.003    25.179
CHECK PASSED
=== before
FP64, T = 1, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop    bwd ms
    524288  fs         fs         |  4.27e-16  1.33e-15   bitwise  1.22e-15 |     1.867     1.875  1.004     2.442
   1000000  fs         fs         |  0.00e+00  1.86e-15   bitwise  1.86e-15 |     4.344     4.414  1.016     5.501
   1048576  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |     4.248     4.161  0.979     6.219
   2097152  fs         fs         |  3.00e-16  1.67e-15   bitwise  1.44e-15 |     9.340     9.203  0.985    12.286
   4194304  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |    20.007    19.821  0.991    25.497
CHECK PASSED
=== after
FP64, T = 1, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop    bwd ms
    524288  fs         fs         |  4.27e-16  1.33e-15   bitwise  1.11e-15 |     1.861     1.902  1.022     2.585
   1000000  fs         fs         |  0.00e+00  1.86e-15   bitwise  1.86e-15 |     4.315     4.398  1.019     5.438
   1048576  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |     4.168     4.026  0.966     6.039
   2097152  fs         fs         |  4.01e-16  1.67e-15   bitwise  1.44e-15 |     9.284     9.021  0.972    12.036
   4194304  fs         fs         |  0.00e+00  1.55e-15   bitwise  1.55e-15 |    19.900    19.669  0.988    25.155
CHECK PASSED
```
