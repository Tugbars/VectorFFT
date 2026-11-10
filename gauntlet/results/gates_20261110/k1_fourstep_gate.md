# k1_fourstep_gate: the FP64 four-step through the front door (a scratch copy of the 14900KF store)

```
K=1 INTERLEAVED FOUR-STEP gate (front door, scratch store <scratch store>), T=8
N=524288
  natural oop: spot bins 9.0e-17 rel, replayed
  natural oop: roundtrip 1.3e-15
  natural ip: spot bins 5.4e-17 rel, roundtrip ok, own plan
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.3e-15
  scrambled ip: multiset vs natural 3.8e-16 rel, roundtrip ok, own plan
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=1048576
  natural oop: spot bins 8.7e-17 rel, replayed
  natural oop: roundtrip 1.4e-15
  natural ip: spot bins 8.7e-17 rel, roundtrip ok, bitwise oop
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.4e-15
  scrambled ip: multiset vs natural 0.0e+00 rel, roundtrip ok, bitwise oop
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=2097152
  natural oop: spot bins 7.5e-17 rel, replayed
  natural oop: roundtrip 1.6e-15
  natural ip: spot bins 7.5e-17 rel, roundtrip ok, own plan
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.6e-15
  scrambled ip: multiset vs natural 2.9e-16 rel, roundtrip ok, own plan
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
ALL PASS
```
