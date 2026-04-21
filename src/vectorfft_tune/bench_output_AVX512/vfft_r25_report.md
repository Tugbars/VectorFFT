# VectorFFT R=25 tuning report

Total measurements: **512**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 8 | 8 | 232 | 208 | 201 | — | t1s |
| avx2 | 8 | 16 | 224 | 204 | 190 | — | t1s |
| avx2 | 8 | 200 | 222 | 204 | 197 | — | t1s |
| avx2 | 8 | 256 | 328 | 285 | 289 | — | log3 |
| avx2 | 16 | 16 | 421 | 400 | 368 | — | t1s |
| avx2 | 16 | 24 | 432 | 405 | 378 | — | t1s |
| avx2 | 16 | 400 | 434 | 410 | 385 | — | t1s |
| avx2 | 16 | 512 | 739 | 764 | 772 | — | flat |
| avx2 | 32 | 32 | 848 | 791 | 735 | — | t1s |
| avx2 | 32 | 40 | 840 | 800 | 736 | — | t1s |
| avx2 | 32 | 800 | 867 | 801 | 737 | — | t1s |
| avx2 | 32 | 1024 | 1851 | 2022 | 1914 | — | flat |
| avx2 | 64 | 64 | 1708 | 1578 | 1540 | — | t1s |
| avx2 | 64 | 72 | 1783 | 1568 | 1525 | — | t1s |
| avx2 | 64 | 1600 | 1762 | 1623 | 1548 | — | t1s |
| avx2 | 64 | 2048 | 3519 | 4189 | 3771 | — | flat |
| avx2 | 128 | 128 | 3698 | 3523 | 2979 | — | t1s |
| avx2 | 128 | 136 | 3627 | 3337 | 3081 | — | t1s |
| avx2 | 128 | 3200 | 5463 | 4264 | 3837 | — | t1s |
| avx2 | 128 | 4096 | 7219 | 7895 | 7519 | — | flat |
| avx2 | 256 | 256 | 11651 | 9937 | 9875 | — | t1s |
| avx2 | 256 | 264 | 10557 | 6670 | 6173 | — | t1s |
| avx2 | 256 | 6400 | 12276 | 12759 | 11978 | — | t1s |
| avx2 | 256 | 8192 | 14725 | 16077 | 14908 | — | flat |
| avx512 | 8 | 8 | 136 | 128 | 128 | — | t1s |
| avx512 | 8 | 16 | 133 | 125 | 128 | — | log3 |
| avx512 | 8 | 200 | 134 | 123 | 130 | — | log3 |
| avx512 | 8 | 256 | 200 | 163 | 182 | — | log3 |
| avx512 | 16 | 16 | 253 | 243 | 231 | — | t1s |
| avx512 | 16 | 24 | 249 | 251 | 243 | — | t1s |
| avx512 | 16 | 400 | 259 | 248 | 236 | — | t1s |
| avx512 | 16 | 512 | 475 | 373 | 413 | — | log3 |
| avx512 | 32 | 32 | 482 | 493 | 439 | — | t1s |
| avx512 | 32 | 40 | 638 | 498 | 441 | — | t1s |
| avx512 | 32 | 800 | 487 | 504 | 445 | — | t1s |
| avx512 | 32 | 1024 | 1052 | 891 | 949 | — | log3 |
| avx512 | 64 | 64 | 1041 | 976 | 867 | — | t1s |
| avx512 | 64 | 72 | 1033 | 985 | 851 | — | t1s |
| avx512 | 64 | 1600 | 1052 | 973 | 861 | — | t1s |
| avx512 | 64 | 2048 | 2099 | 1751 | 1827 | — | log3 |
| ... | ... (8 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 232 |
| avx2 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 224 |
| avx2 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 222 |
| avx2 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 328 |
| avx2 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 421 |
| avx2 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 432 |
| avx2 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 434 |
| avx2 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 739 |
| avx2 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 848 |
| avx2 | `t1_dit` | 32 | 40 | `ct_t1_dit` | 840 |
| avx2 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 867 |
| avx2 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1851 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1708 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1783 |
| avx2 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 1762 |
| avx2 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 3519 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3698 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 3627 |
| avx2 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 5463 |
| avx2 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 7219 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 11651 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 10557 |
| avx2 | `t1_dit` | 256 | 6400 | `ct_t1_dit` | 12276 |
| avx2 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 14725 |
| avx2 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 208 |
| avx2 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 204 |
| avx2 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 204 |
| avx2 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 285 |
| avx2 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 400 |
| avx2 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 405 |
| avx2 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 410 |
| avx2 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 764 |
| avx2 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 791 |
| avx2 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 800 |
| avx2 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 801 |
| avx2 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 2022 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 1578 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 1568 |
| avx2 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 1623 |
| avx2 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 4189 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 3523 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 3337 |
| avx2 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 4264 |
| avx2 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 7895 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 9937 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 6670 |
| avx2 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 12759 |
| avx2 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 16077 |
| avx2 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 201 |
| avx2 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 190 |
| avx2 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 197 |
| avx2 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 289 |
| avx2 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 368 |
| avx2 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 378 |
| avx2 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 385 |
| avx2 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 772 |
| avx2 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 735 |
| avx2 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 736 |
| avx2 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 737 |
| avx2 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 1914 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1540 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1525 |
| avx2 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 1548 |
| avx2 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 3771 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2979 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 3081 |
| avx2 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 3837 |
| avx2 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 7519 |
| avx2 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 9875 |
| avx2 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 6173 |
| avx2 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 11978 |
| avx2 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 14908 |
| avx512 | `t1_dit` | 8 | 8 | `ct_t1_dit` | 136 |
| avx512 | `t1_dit` | 8 | 16 | `ct_t1_dit` | 133 |
| avx512 | `t1_dit` | 8 | 200 | `ct_t1_dit` | 134 |
| avx512 | `t1_dit` | 8 | 256 | `ct_t1_dit` | 200 |
| avx512 | `t1_dit` | 16 | 16 | `ct_t1_dit` | 253 |
| avx512 | `t1_dit` | 16 | 24 | `ct_t1_dit` | 249 |
| avx512 | `t1_dit` | 16 | 400 | `ct_t1_dit` | 259 |
| avx512 | `t1_dit` | 16 | 512 | `ct_t1_dit` | 475 |
| avx512 | `t1_dit` | 32 | 32 | `ct_t1_dit` | 482 |
| avx512 | `t1_dit` | 32 | 40 | `ct_t1_buf_dit_tile32_temporal` | 638 |
| avx512 | `t1_dit` | 32 | 800 | `ct_t1_dit` | 487 |
| avx512 | `t1_dit` | 32 | 1024 | `ct_t1_dit` | 1052 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1041 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1033 |
| avx512 | `t1_dit` | 64 | 1600 | `ct_t1_dit` | 1052 |
| avx512 | `t1_dit` | 64 | 2048 | `ct_t1_dit` | 2099 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2361 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2314 |
| avx512 | `t1_dit` | 128 | 3200 | `ct_t1_dit` | 2850 |
| avx512 | `t1_dit` | 128 | 4096 | `ct_t1_dit` | 4196 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6737 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5036 |
| avx512 | `t1_dit` | 256 | 6400 | `ct_t1_dit` | 6840 |
| avx512 | `t1_dit` | 256 | 8192 | `ct_t1_dit` | 8817 |
| avx512 | `t1_dit_log3` | 8 | 8 | `ct_t1_dit_log3` | 128 |
| avx512 | `t1_dit_log3` | 8 | 16 | `ct_t1_dit_log3` | 125 |
| avx512 | `t1_dit_log3` | 8 | 200 | `ct_t1_dit_log3` | 123 |
| avx512 | `t1_dit_log3` | 8 | 256 | `ct_t1_dit_log3` | 163 |
| avx512 | `t1_dit_log3` | 16 | 16 | `ct_t1_dit_log3` | 243 |
| avx512 | `t1_dit_log3` | 16 | 24 | `ct_t1_dit_log3` | 251 |
| avx512 | `t1_dit_log3` | 16 | 400 | `ct_t1_dit_log3` | 248 |
| avx512 | `t1_dit_log3` | 16 | 512 | `ct_t1_dit_log3` | 373 |
| avx512 | `t1_dit_log3` | 32 | 32 | `ct_t1_dit_log3` | 493 |
| avx512 | `t1_dit_log3` | 32 | 40 | `ct_t1_dit_log3` | 498 |
| avx512 | `t1_dit_log3` | 32 | 800 | `ct_t1_dit_log3` | 504 |
| avx512 | `t1_dit_log3` | 32 | 1024 | `ct_t1_dit_log3` | 891 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 976 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 985 |
| avx512 | `t1_dit_log3` | 64 | 1600 | `ct_t1_dit_log3` | 973 |
| avx512 | `t1_dit_log3` | 64 | 2048 | `ct_t1_dit_log3` | 1751 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 1938 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1980 |
| avx512 | `t1_dit_log3` | 128 | 3200 | `ct_t1_dit_log3` | 2184 |
| avx512 | `t1_dit_log3` | 128 | 4096 | `ct_t1_dit_log3` | 3522 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 5045 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3962 |
| avx512 | `t1_dit_log3` | 256 | 6400 | `ct_t1_dit_log3` | 5460 |
| avx512 | `t1_dit_log3` | 256 | 8192 | `ct_t1_dit_log3` | 6957 |
| avx512 | `t1s_dit` | 8 | 8 | `ct_t1s_dit` | 128 |
| avx512 | `t1s_dit` | 8 | 16 | `ct_t1s_dit` | 128 |
| avx512 | `t1s_dit` | 8 | 200 | `ct_t1s_dit` | 130 |
| avx512 | `t1s_dit` | 8 | 256 | `ct_t1s_dit` | 182 |
| avx512 | `t1s_dit` | 16 | 16 | `ct_t1s_dit` | 231 |
| avx512 | `t1s_dit` | 16 | 24 | `ct_t1s_dit` | 243 |
| avx512 | `t1s_dit` | 16 | 400 | `ct_t1s_dit` | 236 |
| avx512 | `t1s_dit` | 16 | 512 | `ct_t1s_dit` | 413 |
| avx512 | `t1s_dit` | 32 | 32 | `ct_t1s_dit` | 439 |
| avx512 | `t1s_dit` | 32 | 40 | `ct_t1s_dit` | 441 |
| avx512 | `t1s_dit` | 32 | 800 | `ct_t1s_dit` | 445 |
| avx512 | `t1s_dit` | 32 | 1024 | `ct_t1s_dit` | 949 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 867 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 851 |
| avx512 | `t1s_dit` | 64 | 1600 | `ct_t1s_dit` | 861 |
| avx512 | `t1s_dit` | 64 | 2048 | `ct_t1s_dit` | 1827 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 1852 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1775 |
| avx512 | `t1s_dit` | 128 | 3200 | `ct_t1s_dit` | 2571 |
| avx512 | `t1s_dit` | 128 | 4096 | `ct_t1s_dit` | 3552 |
| avx512 | `t1s_dit` | 256 | 256 | `ct_t1s_dit` | 4780 |
| avx512 | `t1s_dit` | 256 | 264 | `ct_t1s_dit` | 3806 |
| avx512 | `t1s_dit` | 256 | 6400 | `ct_t1s_dit` | 5410 |
| avx512 | `t1s_dit` | 256 | 8192 | `ct_t1s_dit` | 7142 |