# FP64 2D interleaved cells, before and after the FP64 twiddle change (the canonical bench, --2dilnat)

The 2D row pass calls the four-step's twiddle (for four-step plans only), and the compiler re-assigned registers in it
when the twiddle changed. One thread, natural order, out of place. Our time relative to before (median, min..max over
the rounds), each arm's median time and its median MKL / ours.

## Six cells, 9 rounds

| cell | after / before, our time (med, min..max) | before ns | after ns | MKL / ours, before | MKL / ours, after | route |
|---|---|---:|---:|---:|---:|---|
| 32x32 | 0.991 (0.962..1.020) | 652 | 646 | 1.106 | 1.116 | chain+rb2 |
| 64x64 | 0.997 (0.949..1.006) | 4770 | 4742 | 1.493 | 1.495 | csk+rb |
| 256x32 | 0.999 (0.982..1.043) | 11050 | 11036 | 1.353 | 1.343 | chain+rb |
| 1024x64 | 1.013 (0.973..1.034) | 110487 | 111280 | 1.492 | 1.473 | chain+rb |
| 128x128 | 1.001 (0.981..1.006) | 23575 | 23552 | 1.273 | 1.277 | chain |
| 256x256 | 1.016 (0.991..1.035) | 108813 | 111247 | 1.202 | 1.201 | chain |

## The two cells that leaned above 1, again, 16 rounds

| cell | after / before, our time (med, min..max) | before ns | after ns | MKL / ours, before | MKL / ours, after | route |
|---|---|---:|---:|---:|---:|---|
| 1024x64 | 1.004 (0.952..1.033) | 110670 | 111382 | 1.491 | 1.479 | chain+rb |
| 256x256 | 1.008 (0.984..1.031) | 110490 | 110834 | 1.190 | 1.196 | chain |
