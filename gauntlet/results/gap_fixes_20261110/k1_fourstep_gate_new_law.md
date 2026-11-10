# k1_fourstep_gate (FP64) with its in-place law against the reference, seeded scratch store

```
Tue Nov 10 13:35:08 RST 2026
K=1 INTERLEAVED FOUR-STEP gate (front door, scratch store <scratch>/gapfix/r2/st_fs), T=8
N=524288
  natural oop: spot bins 9.0e-17 rel, replayed
  natural oop: roundtrip 1.3e-15
  natural ip: spot bins 9.0e-17 rel, roundtrip ok, own plan
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.3e-15
  scrambled ip: multiset vs natural 2.9e-16 rel, roundtrip ok, own plan
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=1048576
  natural oop: spot bins 1.1e-16 rel, replayed
  natural oop: roundtrip 1.6e-15
  natural ip: spot bins 1.0e-16 rel, roundtrip ok, own plan
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.6e-15
  scrambled ip: multiset vs natural 3.7e-16 rel, roundtrip ok, own plan
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=2097152
  natural oop: spot bins 7.1e-17 rel, replayed
  natural oop: roundtrip 1.4e-15
  natural ip: spot bins 1.2e-16 rel, roundtrip ok, own plan
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.4e-15
  scrambled ip: multiset vs natural 3.8e-16 rel, roundtrip ok, own plan
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
ALL PASS
gate exit 0
Tue Nov 10 13:41:09 RST 2026
```
