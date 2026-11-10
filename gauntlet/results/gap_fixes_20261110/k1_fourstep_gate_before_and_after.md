# k1_fourstep_gate (FP64) on a seeded scratch copy of the 14900KF store: the tree BEFORE the fixes (5901946f) and the fixed tree

Both fail the same four checks (in place bitwise out of place, at 524288 and 2097152, both order classes); apart from the
store's path and the dates the two outputs are identical line for line. The fixes do not cause the failures.

## Before the fixes

```
Tue Nov 10 12:35:59 RST 2026
K=1 INTERLEAVED FOUR-STEP gate (front door, scratch store <scratch>/gapfix/fs_store_pre), T=8
N=524288
  natural oop: spot bins 9.0e-17 rel, replayed
  natural oop: roundtrip 1.3e-15
  *** FAIL: N=524288 natural: in place != out of place (bitwise)
  natural ip: NOT BITWISE, roundtrip ok
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.3e-15
  *** FAIL: N=524288 scrambled: in place != out of place (bitwise)
  scrambled ip: NOT BITWISE, roundtrip ok
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=1048576
  natural oop: spot bins 1.1e-16 rel, replayed
  natural oop: roundtrip 1.6e-15
  natural ip: bitwise oop, roundtrip ok
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.6e-15
  scrambled ip: bitwise oop, roundtrip ok
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=2097152
  natural oop: spot bins 7.1e-17 rel, replayed
  natural oop: roundtrip 1.4e-15
  *** FAIL: N=2097152 natural: in place != out of place (bitwise)
  natural ip: NOT BITWISE, roundtrip ok
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.4e-15
  *** FAIL: N=2097152 scrambled: in place != out of place (bitwise)
  scrambled ip: NOT BITWISE, roundtrip ok
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
*** GATE FAILED ***
gate exit 1
Tue Nov 10 12:42:08 RST 2026
```

## The fixed tree

```
Tue Nov 10 12:25:00 RST 2026
K=1 INTERLEAVED FOUR-STEP gate (front door, scratch store <scratch>/gapfix/fs_store), T=8
N=524288
  natural oop: spot bins 9.0e-17 rel, replayed
  natural oop: roundtrip 1.3e-15
  *** FAIL: N=524288 natural: in place != out of place (bitwise)
  natural ip: NOT BITWISE, roundtrip ok
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.3e-15
  *** FAIL: N=524288 scrambled: in place != out of place (bitwise)
  scrambled ip: NOT BITWISE, roundtrip ok
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=1048576
  natural oop: spot bins 1.1e-16 rel, replayed
  natural oop: roundtrip 1.6e-15
  natural ip: bitwise oop, roundtrip ok
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.6e-15
  scrambled ip: bitwise oop, roundtrip ok
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
N=2097152
  natural oop: spot bins 7.1e-17 rel, replayed
  natural oop: roundtrip 1.4e-15
  *** FAIL: N=2097152 natural: in place != out of place (bitwise)
  natural ip: NOT BITWISE, roundtrip ok
  natural T=8: bitwise own serial
  natural replay: no race, bitwise
  scrambled oop: multiset vs natural 0.0e+00 rel, replayed
  scrambled oop: roundtrip 1.4e-15
  *** FAIL: N=2097152 scrambled: in place != out of place (bitwise)
  scrambled ip: NOT BITWISE, roundtrip ok
  scrambled T=8: bitwise own serial
  scrambled replay: no race, bitwise
*** GATE FAILED ***
gate exit 1
Tue Nov 10 12:31:07 RST 2026
```
