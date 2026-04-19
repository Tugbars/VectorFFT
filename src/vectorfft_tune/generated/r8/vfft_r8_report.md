# VectorFFT R=8 tuning report

Total measurements: **108**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 90 | — | — | — | flat |
| avx2 | 64 | 72 | 92 | — | — | — | flat |
| avx2 | 64 | 512 | 233 | — | — | — | flat |
| avx2 | 128 | 128 | 172 | — | — | — | flat |
| avx2 | 128 | 136 | 177 | — | — | — | flat |
| avx2 | 128 | 1024 | 422 | — | — | — | flat |
| avx2 | 256 | 256 | 460 | — | — | — | flat |
| avx2 | 256 | 264 | 419 | — | — | — | flat |
| avx2 | 256 | 2048 | 831 | — | — | — | flat |
| avx2 | 512 | 512 | 1811 | — | — | — | flat |
| avx2 | 512 | 520 | 1028 | — | — | — | flat |
| avx2 | 512 | 4096 | 1745 | — | — | — | flat |
| avx2 | 1024 | 1024 | 3676 | — | — | — | flat |
| avx2 | 1024 | 1032 | 2120 | — | — | — | flat |
| avx2 | 1024 | 8192 | 3452 | — | — | — | flat |
| avx2 | 2048 | 2048 | 7285 | — | — | — | flat |
| avx2 | 2048 | 2056 | 4460 | — | — | — | flat |
| avx2 | 2048 | 16384 | 14874 | — | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 168 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 168 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 415 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 333 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 234 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 711 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 779 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 628 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1525 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 2501 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1649 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 3104 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 5469 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3354 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 3452 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 12708 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 6675 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 16845 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 90 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 92 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 233 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 172 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 177 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_prefetch` | 422 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_prefetch` | 460 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 419 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_prefetch` | 831 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 1811 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_prefetch` | 1028 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_prefetch` | 1745 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit_prefetch` | 3676 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 2120 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_prefetch` | 3830 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_prefetch` | 7285 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 4460 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit_prefetch` | 14874 |