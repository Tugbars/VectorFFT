# The door probe (alias_probe.c, N = 8, both precisions) under gdb, breakpoints on the radix-8 solo kernels

`oop` = the out-of-place plan with two buffers, `alias` = the same plan with ONE buffer, `ip` = the in-place plan,
`ip2buf` = the in-place plan with TWO buffers. A `HIT` line is the kernel the call reached.

## The tree before the fixes (5901946f)

```
FP64 N=8: route oop mono, ip mono
CALL d oop fwd
   HIT radix8_z_n1_fwd_avx2      (restrict)
CALL d oop bwd
   HIT radix8_z_n1_bwd_avx2      (restrict)
CALL d alias fwd
   HIT radix8_z_n1_fwd_avx2      (restrict)
     alias fwd bitwise the oop result
CALL d alias bwd
   HIT radix8_z_n1_bwd_avx2      (restrict)
     alias bwd bitwise the oop result
CALL d ip fwd
   HIT radix8_z_n1c_fwd_avx2     (alias-tolerant)
     ip fwd bitwise the oop result
CALL d ip2buf fwd
   HIT radix8_z_n1c_fwd_avx2     (alias-tolerant)
     ip2buf: output WRITTEN, vfft_error: 
FP32 N=8: route oop mono, ip mono
CALL s oop fwd
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)
CALL s oop bwd
   HIT radix8_z_n1_bwd_avx2_f32  (restrict)
CALL s alias fwd
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)
     alias fwd bitwise the oop result
CALL s alias bwd
   HIT radix8_z_n1_bwd_avx2_f32  (restrict)
     alias bwd bitwise the oop result
CALL s ip fwd
   HIT radix8_z_n1c_fwd_avx2_f32 (alias-tolerant)
     ip fwd bitwise the oop result
CALL s ip2buf fwd
   HIT radix8_z_n1c_fwd_avx2_f32 (alias-tolerant)
     ip2buf: output WRITTEN, vfft_error: 
*** FAIL ***
```

## The fixed tree

```
FP64 N=8: route oop mono, ip mono
CALL d oop fwd
   HIT radix8_z_n1_fwd_avx2      (restrict)
CALL d oop bwd
   HIT radix8_z_n1_bwd_avx2      (restrict)
CALL d alias fwd
   HIT radix8_z_n1c_fwd_avx2     (alias-tolerant)
     alias fwd bitwise the oop result
CALL d alias bwd
   HIT radix8_z_n1c_bwd_avx2     (alias-tolerant)
     alias bwd bitwise the oop result
CALL d ip fwd
   HIT radix8_z_n1c_fwd_avx2     (alias-tolerant)
     ip fwd bitwise the oop result
CALL d ip2buf fwd
     ip2buf: output untouched, vfft_error: vfft_execute: this C2C plan is IN-PLACE and must be called with out == in; got distinct pointers; for separate buffers c
FP32 N=8: route oop mono, ip mono
CALL s oop fwd
   HIT radix8_z_n1_fwd_avx2_f32  (restrict)
CALL s oop bwd
   HIT radix8_z_n1_bwd_avx2_f32  (restrict)
CALL s alias fwd
   HIT radix8_z_n1c_fwd_avx2_f32 (alias-tolerant)
     alias fwd bitwise the oop result
CALL s alias bwd
   HIT radix8_z_n1c_bwd_avx2_f32 (alias-tolerant)
     alias bwd bitwise the oop result
CALL s ip fwd
   HIT radix8_z_n1c_fwd_avx2_f32 (alias-tolerant)
     ip fwd bitwise the oop result
CALL s ip2buf fwd
     ip2buf: output untouched, vfft_error: vfft_execute: this C2C plan is IN-PLACE and must be called with out == in; got distinct pointers; for separate buffers c
PROBE OK
```
