# The FP64 four-step in place against out of place, through the public door (the re-raced rows)

The columns of `ip_check_fp32.md`. One thread, then eight.

```
FP64, T = 1, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop
    524288  fs         fs         |  4.27e-16  1.33e-15   bitwise  1.22e-15 |     1.585     1.835  1.158
   1000000  fs         fs         |  0.00e+00  1.86e-15   bitwise  1.86e-15 |     4.134     4.262  1.031
   1048576  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |     4.058     4.074  1.004
   2097152  fs         fs         |  3.00e-16  1.67e-15   bitwise  1.44e-15 |     9.279     9.228  0.994
   4194304  fs         fs         |  0.00e+00  1.44e-15   bitwise  1.44e-15 |    19.860    19.842  0.999
CHECK PASSED

FP64, T = 8, interleaved natural order: in place against out of place
         N  route oop  route ip   |    fwd ip     rt ip      1buf   bwd oop |    oop ms     ip ms ip/oop
    524288  fs         fs         |  3.20e-16  1.33e-15   bitwise  1.33e-15 |     0.295     0.399  1.353
   1000000  fs         fs         |  5.94e-16  2.10e-15   bitwise  2.10e-15 |     1.231     1.042  0.846
CHECK PASSED
```
