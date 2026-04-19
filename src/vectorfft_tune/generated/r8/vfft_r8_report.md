# VectorFFT R=8 tuning report

Total measurements: **216**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 230 | — | — | — | flat |
| avx2 | 64 | 72 | 233 | — | — | — | flat |
| avx2 | 64 | 512 | 504 | — | — | — | flat |
| avx2 | 128 | 128 | 485 | — | — | — | flat |
| avx2 | 128 | 136 | 492 | — | — | — | flat |
| avx2 | 128 | 1024 | 1080 | — | — | — | flat |
| avx2 | 256 | 256 | 1042 | — | — | — | flat |
| avx2 | 256 | 264 | 1040 | — | — | — | flat |
| avx2 | 256 | 2048 | 4081 | — | — | — | flat |
| avx2 | 512 | 512 | 4223 | — | — | — | flat |
| avx2 | 512 | 520 | 2119 | — | — | — | flat |
| avx2 | 512 | 4096 | 8569 | — | — | — | flat |
| avx2 | 1024 | 1024 | 9231 | — | — | — | flat |
| avx2 | 1024 | 1032 | 3982 | — | — | — | flat |
| avx2 | 1024 | 8192 | 16880 | — | — | — | flat |
| avx2 | 2048 | 2048 | 34116 | — | — | — | flat |
| avx2 | 2048 | 2056 | 14894 | — | — | — | flat |
| avx2 | 2048 | 16384 | 35213 | — | — | — | flat |
| avx512 | 64 | 64 | 118 | — | — | — | flat |
| avx512 | 64 | 72 | 118 | — | — | — | flat |
| avx512 | 64 | 512 | 405 | — | — | — | flat |
| avx512 | 128 | 128 | 244 | — | — | — | flat |
| avx512 | 128 | 136 | 243 | — | — | — | flat |
| avx512 | 128 | 1024 | 747 | — | — | — | flat |
| avx512 | 256 | 256 | 738 | — | — | — | flat |
| avx512 | 256 | 264 | 714 | — | — | — | flat |
| avx512 | 256 | 2048 | 1752 | — | — | — | flat |
| avx512 | 512 | 512 | 3284 | — | — | — | flat |
| avx512 | 512 | 520 | 1720 | — | — | — | flat |
| avx512 | 512 | 4096 | 3653 | — | — | — | flat |
| avx512 | 1024 | 1024 | 6564 | — | — | — | flat |
| avx512 | 1024 | 1032 | 3439 | — | — | — | flat |
| avx512 | 1024 | 8192 | 7224 | — | — | — | flat |
| avx512 | 2048 | 2048 | 14298 | — | — | — | flat |
| avx512 | 2048 | 2056 | 12473 | — | — | — | flat |
| avx512 | 2048 | 16384 | 14374 | — | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 230 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 233 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 587 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 485 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 487 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 1163 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 1042 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 1040 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 5098 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 5603 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 2119 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 10170 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 10203 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3982 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 20210 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 40552 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 14845 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 40124 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 246 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 247 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 504 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 498 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 492 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_prefetch` | 1080 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 1066 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 1111 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_prefetch` | 4081 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 4223 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 2233 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit_prefetch` | 8569 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 9231 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 4457 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_prefetch` | 16880 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_prefetch` | 34116 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 14894 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 35213 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 118 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 118 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 512 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 244 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 243 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 977 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 802 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 714 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 2231 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 3995 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1713 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 4375 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 8207 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 3439 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 8813 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 17721 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 14676 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 17746 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 131 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 127 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 405 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 262 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 257 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit_prefetch` | 747 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit_prefetch` | 738 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit_prefetch` | 842 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit_prefetch` | 1752 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 3284 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 1720 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 3653 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 6564 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 3567 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 7224 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 14298 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 12473 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 14374 |