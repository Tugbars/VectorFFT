# VectorFFT R=32 tuning report

Total measurements: **222**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 876 | 825 | 632 | — | t1s |
| avx2 | 64 | 72 | 1044 | 809 | 791 | — | t1s |
| avx2 | 64 | 512 | 1919 | 2093 | 1897 | — | t1s |
| avx2 | 128 | 128 | 2681 | 3308 | 3075 | — | flat |
| avx2 | 128 | 136 | 2247 | 2259 | 1558 | — | t1s |
| avx2 | 128 | 1024 | 3954 | 3925 | 3932 | — | log3 |
| avx2 | 256 | 256 | 8087 | 6392 | — | — | log3 |
| avx2 | 256 | 264 | 4212 | 3659 | — | — | log3 |
| avx2 | 256 | 2048 | 8259 | 8957 | — | — | flat |
| avx2 | 512 | 512 | 17546 | 15295 | — | — | log3 |
| avx2 | 512 | 520 | 9465 | 8066 | — | — | log3 |
| avx2 | 512 | 4096 | 19423 | 22160 | — | — | flat |
| avx2 | 1024 | 1024 | 30953 | 38650 | — | — | flat |
| avx2 | 1024 | 1032 | 19984 | 16074 | — | — | log3 |
| avx2 | 1024 | 8192 | 41358 | 41223 | — | — | log3 |
| avx2 | 2048 | 2048 | 65818 | 62680 | — | — | log3 |
| avx2 | 2048 | 2056 | 51921 | 32975 | — | — | log3 |
| avx2 | 2048 | 16384 | 117387 | 130745 | — | — | flat |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 2040 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 2023 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2517 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 5060 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 3830 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 4777 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 9617 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 8174 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 9759 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 17546 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 17944 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 19423 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 36809 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 34864 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 47164 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 74445 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 66734 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 129205 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1143 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1760 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 3085 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 3903 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 2530 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 4932 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 9303 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 5806 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 9903 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 24210 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 16561 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 28635 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 39829 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 23482 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 58662 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 79727 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 69496 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 148436 |
| avx2 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 1650 |
| avx2 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 2490 |
| avx2 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 3000 |
| avx2 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 4181 |
| avx2 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 3432 |
| avx2 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 5248 |
| avx2 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 9192 |
| avx2 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 6718 |
| avx2 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 12590 |
| avx2 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 22954 |
| avx2 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 11051 |
| avx2 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 27561 |
| avx2 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 41474 |
| avx2 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 27665 |
| avx2 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 54619 |
| avx2 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 82813 |
| avx2 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 48936 |
| avx2 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 142810 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 876 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1044 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 1919 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2681 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2247 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 3954 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 8087 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 4212 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 8259 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 18396 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 9465 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 22780 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 30953 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 19984 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 41358 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 65818 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 51921 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 117387 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 825 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 809 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 2093 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3308 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 2259 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 3925 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6392 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3659 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 8957 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 15295 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 8066 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 22160 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 38650 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 16074 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 41223 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 62680 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 32975 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 130745 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 632 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 791 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1897 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3075 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1558 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 3932 |