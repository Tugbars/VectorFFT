# R2C / C2R codelet verification

Smoke tests for the OCaml r2c math layer (docs 49, 50).

## Quick verify (R=16 r2c forward at K=8)

```bash
# From repo root:
./_build/default/bin/gen_radix.exe 16 --r2c --emit-c > /tmp/r16_r2c.c
gcc-11 -O3 -mavx512f -mavx512dq -mfma -march=icelake-server \
    /tmp/r16_r2c.c test/r2c/verify.c -o /tmp/r16_r2c_verify -lm
/tmp/r16_r2c_verify
```

## Parametric forward verify (any N, K-sweep, ns/call timing)

```bash
./_build/default/bin/gen_radix.exe 16 --r2c --emit-c > /tmp/r16_r2c.c
gcc-11 -O3 -mavx512f -mavx512dq -mfma -march=icelake-server \
    -DR2C_N=16 -DR2C_FN=radix16_r2c_fwd_avx512_gen \
    /tmp/r16_r2c.c test/r2c/verify_r2c.c -o /tmp/v -lm
/tmp/v
```

For other sizes, replace `16` with `{32, 64, 128, 256, 512}` and adjust
the function name accordingly (e.g., `radix64_r2c_fwd_avx512_gen`).

## Round-trip c2r(r2c(x)) == N*x (R=16/32/64)

```bash
GEN=./_build/default/bin/gen_radix.exe
for R in 16 32 64; do
  $GEN $R --r2c --emit-c > /tmp/r${R}_r2c.c
  $GEN $R --c2r --emit-c > /tmp/r${R}_c2r.c
  gcc-11 -O3 -mavx512f -mavx512dq -mfma -march=icelake-server \
      -DRT_N=$R \
      -DRT_R2C_FN=radix${R}_r2c_fwd_avx512_gen \
      -DRT_C2R_FN=radix${R}_c2r_avx512_gen \
      /tmp/r${R}_r2c.c /tmp/r${R}_c2r.c test/r2c/round_trip.c -o /tmp/rt_$R -lm
  /tmp/rt_$R
done
```

Expected: `Round-trip vs N*x: max err ~1e-13` for each R, exit 0.

## Files

| File | Purpose |
|------|---------|
| `verify.c` | Hardcoded R=16, K=8 single-shot verify (legacy from doc 49) |
| `verify_r2c.c` | Parametric (any N) forward verify with K-sweep + timing |
| `round_trip.c` | r2c→c2r round-trip property test for any R |

## Ground truth

Reference DFT computed by direct evaluation:
  X[k] = sum_{n=0..N-1} x[n] * exp(-2*pi*i*n*k/N) for k = 0..N/2.
