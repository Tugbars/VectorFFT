# An out-of-place contiguous batch of solo-length transforms (N = 8) called with one buffer, under gdb

Breakpoints on the radix-8 solo kernels; a `HIT` line is the kernel the call reached, `(xn)` its count. The serial
batch is K = 3 at one thread; the threaded one K = 8 at four threads with the whole-transform slabs pinned on
(`VFFT_TCMT=1`; a debugger loses some of the hits that fire in several threads at once, so its counts run short).

## The branch before the batch fix (6f54a488)

```
FP64 batch N=8 K=3 T=1
CALL d batch two buffers
   HIT radix8_z_n1_fwd_avx2      (restrict)   (x3)
CALL d batch ONE buffer
   HIT radix8_z_n1_fwd_avx2      (restrict)   (x3)
     one buffer bitwise the two-buffer result; threaded dispatches in the two calls: 0
FP32 batch N=8 K=3 T=1
CALL s batch two buffers
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)   (x3)
CALL s batch ONE buffer
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)   (x3)
     one buffer bitwise the two-buffer result
FP64 batch N=8 K=8 T=4
CALL d batch two buffers
   HIT radix8_z_n1_fwd_avx2      (restrict)   (x5)
CALL d batch ONE buffer
   HIT radix8_z_n1_fwd_avx2      (restrict)   (x5)
     one buffer bitwise the two-buffer result; threaded dispatches in the two calls: 6
FP32 batch N=8 K=8 T=4
CALL s batch two buffers
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)   (x6)
CALL s batch ONE buffer
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)   (x5)
     one buffer bitwise the two-buffer result
PROBE OK
```

## With the batch fix

```
FP64 batch N=8 K=3 T=1
CALL d batch two buffers
   HIT radix8_z_n1_fwd_avx2      (restrict)   (x3)
CALL d batch ONE buffer
   HIT radix8_z_n1c_fwd_avx2     (alias-tolerant)   (x3)
     one buffer bitwise the two-buffer result; threaded dispatches in the two calls: 0
FP32 batch N=8 K=3 T=1
CALL s batch two buffers
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)   (x3)
CALL s batch ONE buffer
   HIT radix8_z_n1c_fwd_avx2_f32 (alias-tolerant)   (x3)
     one buffer bitwise the two-buffer result
FP64 batch N=8 K=8 T=4
CALL d batch two buffers
   HIT radix8_z_n1_fwd_avx2      (restrict)   (x6)
CALL d batch ONE buffer
   HIT radix8_z_n1c_fwd_avx2     (alias-tolerant)   (x6)
     one buffer bitwise the two-buffer result; threaded dispatches in the two calls: 6
FP32 batch N=8 K=8 T=4
CALL s batch two buffers
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)   (x7)
CALL s batch ONE buffer
   HIT radix8_z_n1c_fwd_avx2_f32 (alias-tolerant)   (x5)
     one buffer bitwise the two-buffer result
PROBE OK
```
