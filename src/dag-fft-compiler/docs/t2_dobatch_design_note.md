# T2 design note (NOT BUILT): dobatch copy-to-contiguous

FFTW precedent (hc2hc-direct.c / ct-hc2c-direct.c): copy a batch of
columns into a contiguous buffer, run the codelet at unit stride,
copy back; batch count deliberately non-pow2 ("should not be 2^k").

NOT a transliteration for us: FFTW's codelet walks columns at buffer
stride 1 and slots at brs = batchsize — the non-pow2 trick lives on
the COLUMN axis their codelet iterates. Our codelets iterate the
LANE axis (vl); columns are the call/range axis. A faithful analog
must decide which buffer axis carries the non-pow2 pitch:
(a) slot axis: buf[slot][lane], slot pitch = vl + pad (pad chosen so
    pitch is non-pow2, e.g. vl + 8): kills the pow2 row-stride
    conflicts INSIDE one column call; gather cost 2r rows in,
    2r rows out per column. Composes with RANGED (cs_in/cs_out are
    the buffer's column steps).
(b) column axis (FFTW-faithful): batch B = round4(r)+2 columns,
    buf[slot][column][lane] — only pays with ranged codelets, and
    the slot pitch is then B*vl (pow2 again unless padded) — so (a)
    is needed anyway.
Conclusion: implement (a) first, measure on metal, then (a)+(b).
Gather/scatter cost model: 4r * vl * 8 bytes moved per column at
streaming bandwidth vs the measured 2-4x context penalty per call —
breaks even when the penalty exceeds ~2x the copy cost; container
numbers say it would, but the container also said that about
blocking. Metal decides. Correctness gating is trivial once built
(bit-identical to the unbuffered path).
