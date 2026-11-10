# vfft_tcbatch_gate (FP64 contiguous batches) after the batch fix, seeded scratch store

```
Tue Nov 10 13:41:09 RST 2026

=== transform-contiguous batch geometry, 1D C2C interleaved ===
N      K      correctness                        vs K=1    lane-major
256    K=2   fwd=1.1e-15 bwd=1.6e-15 ip=1.1e-15/5.8e-16  BITID     lane-refused
256    K=3   fwd=7.9e-16 bwd=1.6e-15 ip=7.9e-16/6.1e-16  BITID     lane-refused
256    K=4   fwd=9.1e-16 bwd=1.8e-15 ip=9.1e-16/5.6e-16  BITID     lane-refused
512    K=2   fwd=1.3e-15 bwd=1.7e-15 ip=1.3e-15/6.7e-16  BITID     lane-refused
512    K=3   fwd=1.4e-15 bwd=2.1e-15 ip=1.4e-15/7.8e-16  BITID     lane-refused
512    K=4   fwd=1.1e-15 bwd=1.7e-15 ip=1.1e-15/7.8e-16  BITID     lane-refused
1024   K=2   fwd=2.3e-15 bwd=2.1e-15 ip=2.3e-15/6.1e-16  BITID     lane-refused
1024   K=3   fwd=1.7e-15 bwd=2.4e-15 ip=1.7e-15/5.6e-16  BITID     lane-refused
1024   K=4   fwd=2.6e-15 bwd=2.6e-15 ip=2.6e-15/6.7e-16  BITID     lane-refused
2048   K=2   fwd=2.7e-15 bwd=3.6e-15 ip=2.7e-15/6.7e-16  BITID     lane-refused
2048   K=3   fwd=4.3e-15 bwd=3.5e-15 ip=4.1e-15/6.7e-16  BITID     lane-refused
2048   K=4   fwd=3.4e-15 bwd=3.9e-15 ip=3.4e-15/6.7e-16  BITID     lane-refused
4096   K=2   fwd=4.8e-15 bwd=4.8e-15 ip=4.9e-15/7.8e-16  BITID     lane-refused
4096   K=3   fwd=4.2e-15 bwd=5.4e-15 ip=4.4e-15/7.8e-16  BITID     lane-refused
4096   K=4   fwd=5.0e-15 bwd=5.0e-15 ip=5.0e-15/7.8e-16  BITID     lane-refused
96     K=3   fwd=5.1e-16 bwd=1.0e-15 ip=5.1e-16/5.9e-16  BITID     lane-refused
--- the DEFAULT geometry is transform-contiguous ---
1024   K=4   zeroed config == TRANSFORM_CONTIGUOUS: EXACT
512    K=3   zeroed config == TRANSFORM_CONTIGUOUS: EXACT
--- MT (nthreads=8, per-worker clones) == ST, bitwise ---
64     K=64  fwd=BITID bwd=BITID ip=BITID
256    K=16  fwd=BITID bwd=BITID ip=BITID
1024   K=8   fwd=BITID bwd=BITID ip=BITID
2048   K=8   fwd=BITID bwd=BITID ip=BITID
4096   K=4   fwd=BITID bwd=BITID ip=BITID
96     K=43  fwd=BITID bwd=BITID ip=BITID
512    K=5   fwd=BITID bwd=BITID ip=BITID
256    K=2   fwd=BITID bwd=BITID ip=BITID
--- the threading verdict's row per thread count ---
N=256   K=16  row at 8 threads: eng=tcb tcmt=1  at 4: eng=tcb tcmt=1  second creates replayed
N=512   K=5   row at 8 threads: eng=tcb tcmt=1  at 4: eng=tcb tcmt=1  second creates replayed
split   K=4   default=built  contiguous=built  lane=built
--- K=1 must not wrap ---
1024   K=1    geometry-identity: EXACT
256    K=1    geometry-identity: EXACT

=== ALL PASS ===
gate exit 0
Tue Nov 10 13:41:35 RST 2026
```
