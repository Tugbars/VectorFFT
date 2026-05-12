# R2C codelet verification

Smoke tests for the OCaml r2c math layer (doc 49).

## Quick verify (R=16 forward at K=8)

```bash
# From repo root:
./_build/default/bin/gen_radix.exe 16 --r2c --emit-c > /tmp/r16_r2c.c
gcc-11 -O3 -mavx512f -mavx512dq -mfma -march=icelake-server \
    /tmp/r16_r2c.c test/r2c/verify.c -o /tmp/r16_r2c_verify -lm
/tmp/r16_r2c_verify
```

Expected: `max abs error: ~1e-14` (FP noise floor), exit 0.

## K-sweep + timing (R=16 forward across K)

```bash
./_build/default/bin/gen_radix.exe 16 --r2c --emit-c > /tmp/r16_r2c.c
gcc-11 -O3 -mavx512f -mavx512dq -mfma -march=icelake-server \
    /tmp/r16_r2c.c test/r2c/sweep.c -o /tmp/r16_r2c_sweep -lm
/tmp/r16_r2c_sweep
```

Reports max error and ns/call for K in {8, 16, 32, 64, 128, 256, 512, 1024}.

## Ground truth

Reference is a direct DFT computed by `ref_r2c_fwd` in both harnesses:
  X[k] = sum_{n=0..N-1} x[n] * exp(-2*pi*i*n*k/N)
for k = 0..N/2. Iterates over batches in the (re, im) split-complex
batched layout that matches the codelet's calling convention.
