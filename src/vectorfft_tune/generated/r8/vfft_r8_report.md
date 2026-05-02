# VectorFFT R=8 tuning report

Total measurements: **414**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 93 | 112 | 91 | — | t1s |
| avx2 | 64 | 72 | 95 | 109 | 87 | — | t1s |
| avx2 | 64 | 512 | 114 | 250 | 238 | — | flat |
| avx2 | 96 | 96 | 137 | 158 | 132 | — | t1s |
| avx2 | 96 | 104 | 134 | 164 | 132 | — | t1s |
| avx2 | 96 | 768 | 134 | 160 | 137 | — | flat |
| avx2 | 128 | 128 | 181 | 211 | 169 | — | t1s |
| avx2 | 128 | 136 | 189 | 227 | 175 | — | t1s |
| avx2 | 128 | 1024 | 439 | 480 | 462 | — | flat |
| avx2 | 192 | 192 | 277 | 336 | — | — | flat |
| avx2 | 192 | 200 | 277 | 328 | — | — | flat |
| avx2 | 192 | 1536 | 668 | 724 | — | — | flat |
| avx2 | 256 | 256 | 402 | 449 | — | — | flat |
| avx2 | 256 | 264 | 421 | 443 | — | — | flat |
| avx2 | 256 | 2048 | 876 | 932 | — | — | flat |
| avx2 | 384 | 384 | 743 | 739 | — | — | log3 |
| avx2 | 384 | 392 | 717 | 761 | — | — | flat |
| avx2 | 384 | 3072 | 1226 | 1348 | — | — | flat |
| avx2 | 512 | 512 | 1113 | 1803 | — | — | flat |
| avx2 | 512 | 520 | 1018 | 1049 | — | — | flat |
| avx2 | 512 | 4096 | 1727 | 1905 | — | — | flat |
| avx2 | 768 | 768 | 1570 | 1459 | — | — | log3 |
| avx2 | 768 | 776 | 1440 | 1517 | — | — | flat |
| avx2 | 768 | 6144 | 2721 | 3008 | — | — | flat |
| avx2 | 1024 | 1024 | 3548 | 3608 | — | — | flat |
| avx2 | 1024 | 1032 | 1984 | 2055 | — | — | flat |
| avx2 | 1024 | 8192 | 4026 | 4046 | — | — | flat |
| avx2 | 1536 | 1536 | 5176 | 5289 | — | — | flat |
| avx2 | 1536 | 1544 | 2901 | 2818 | — | — | log3 |
| avx2 | 1536 | 12288 | 5992 | 9947 | — | — | flat |
| avx2 | 2048 | 2048 | 6492 | 7844 | — | — | flat |
| avx2 | 2048 | 2056 | 4071 | 3976 | — | — | log3 |
| avx2 | 2048 | 16384 | 15331 | 15644 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 171 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 124 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif_prefetch` | 183 |
| avx2 | `t1_dif` | 96 | 96 | `ct_t1_dif` | 191 |
| avx2 | `t1_dif` | 96 | 104 | `ct_t1_dif_prefetch` | 199 |
| avx2 | `t1_dif` | 96 | 768 | `ct_t1_dif_prefetch` | 207 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif_prefetch` | 341 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif_prefetch` | 341 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 758 |
| avx2 | `t1_dif` | 192 | 192 | `ct_t1_dif` | 386 |
| avx2 | `t1_dif` | 192 | 200 | `ct_t1_dif` | 361 |
| avx2 | `t1_dif` | 192 | 1536 | `ct_t1_dif` | 1125 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 573 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif_prefetch` | 663 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 1229 |
| avx2 | `t1_dif` | 384 | 384 | `ct_t1_dif` | 960 |
| avx2 | `t1_dif` | 384 | 392 | `ct_t1_dif` | 1232 |
| avx2 | `t1_dif` | 384 | 3072 | `ct_t1_dif` | 2362 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif_prefetch` | 1470 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 1280 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif_prefetch` | 3064 |
| avx2 | `t1_dif` | 768 | 768 | `ct_t1_dif_prefetch` | 2739 |
| avx2 | `t1_dif` | 768 | 776 | `ct_t1_dif` | 1927 |
| avx2 | `t1_dif` | 768 | 6144 | `ct_t1_dif` | 3813 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif_prefetch` | 2812 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 2477 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 2895 |
| avx2 | `t1_dif` | 1536 | 1536 | `ct_t1_dif_prefetch` | 5452 |
| avx2 | `t1_dif` | 1536 | 1544 | `ct_t1_dif` | 3775 |
| avx2 | `t1_dif` | 1536 | 12288 | `ct_t1_dif` | 5157 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif_prefetch` | 5917 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif_prefetch` | 5640 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif_prefetch` | 16873 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 93 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 95 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit_prefetch` | 114 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit` | 137 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit` | 134 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit` | 134 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 181 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 189 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit_log1` | 439 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit` | 277 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit` | 277 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit_log1` | 668 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit_log1` | 402 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 421 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit_log1` | 876 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit` | 743 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit_log1` | 717 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit_prefetch` | 1226 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit_prefetch` | 1113 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit_prefetch` | 1018 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 1727 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit` | 1570 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit` | 1440 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit` | 2721 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 3548 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 1984 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit_prefetch` | 4026 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_log1` | 5176 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_log1` | 2901 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_prefetch` | 5992 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit_prefetch` | 6492 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit_log1` | 4071 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 15331 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 112 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 109 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 250 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3` | 158 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3` | 164 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3` | 160 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 211 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 227 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 480 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3` | 336 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3` | 328 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3` | 724 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 449 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 443 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 932 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3` | 739 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3` | 761 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3` | 1348 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 1803 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 1049 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 1905 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3` | 1459 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3` | 1517 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3` | 3008 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 3608 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 2055 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 4046 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3` | 5289 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3` | 2818 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3` | 9947 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 7844 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 3976 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 15644 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 91 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 87 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 238 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit` | 132 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit` | 132 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit` | 137 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 169 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 175 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 462 |