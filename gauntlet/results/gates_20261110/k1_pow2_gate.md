# k1_pow2_gate: the FP64 K=1 interleaved tier at 128..4096 through the front door (a cold scratch store)

```
=== K=1 IL c2c at POW2 N through the front door: OOP + IP, both directions, both order classes, replay bit-identical, T=8 bitwise ===
N     | route (store)          | oop fwd  oop rt   | ip fwd   ip rt    | race ms replay  |
128   | 2p 4.32                | 3.1e-16  4.5e-16 | 0.0e+00  4.5e-16 |   21075       0 | replay bitwise
   mt | T=8 BITWISE rt 4.5e-16 engaged=0 (serial verdict)
  scr | ztt 4.4.8              | comb served     rt 3.9e-16 | ip 1.5e+00  rt 3.3e-16 |       0       0 | replay bitwise
   mt | T=8 BITWISE rt 3.9e-16 engaged=0 (serial verdict)
256   | 2p 16.16               | 4.5e-16  5.6e-16 | 0.0e+00  5.6e-16 |    6017       0 | replay bitwise
   mt | T=8 BITWISE rt 5.6e-16 engaged=0 (serial verdict)
  scr | ztt 4.8.8              | comb served     rt 4.4e-16 | ip 0.0e+00  rt 4.4e-16 |       0       0 | replay bitwise
   mt | T=8 BITWISE rt 4.4e-16 engaged=0 (serial verdict)
512   | 2p 16.32               | 6.3e-16  5.6e-16 | 0.0e+00  5.6e-16 |    6821       0 | replay bitwise
   mt | T=8 BITWISE rt 5.6e-16 engaged=0 (serial verdict)
  scr | ztt 4.4.4.8            | comb served     rt 6.1e-16 | ip 1.3e+00  rt 6.7e-16 |       0       0 | replay bitwise
   mt | T=8 BITWISE rt 6.1e-16 engaged=0 (serial verdict)
1024  | ztt 4.8.8.4            | 6.3e-16  5.6e-16 | 2.2e-16  6.7e-16 |    9002       0 | replay bitwise
   mt | T=8 BITWISE rt 5.6e-16 engaged=0 (serial verdict)
  scr | ztt 4.8.4.8            | comb served     rt 5.6e-16 | ip 0.0e+00  rt 5.6e-16 |       0       0 | replay bitwise
   mt | T=8 BITWISE rt 5.6e-16 engaged=0 (serial verdict)
2048  | ztt 8.8.8.4            | 6.5e-16  7.2e-16 | 2.9e-16  7.8e-16 |   10292       0 | replay bitwise
   mt | T=8 BITWISE rt 7.2e-16 engaged=0 (serial verdict)
  scr | ztt 8.8.4.8            | comb served     rt 6.7e-16 | ip 0.0e+00  rt 6.7e-16 |       0       0 | replay bitwise
   mt | T=8 BITWISE rt 6.7e-16 engaged=0 (serial verdict)
4096  | ztt 8.8.8.8            | 8.4e-16  8.9e-16 | 2.9e-16  6.7e-16 |   21139       0 | replay bitwise
   mt | T=8 BITWISE rt 8.9e-16 engaged=0 (serial verdict)
  scr | ztt 8.8.8.8            | comb served     rt 7.8e-16 | ip 0.0e+00  rt 7.8e-16 |       0       0 | replay bitwise
   mt | T=8 BITWISE rt 7.8e-16 engaged=2 (threaded)

race clock: short samples = 0

=== ALL PASS ===
```
