# VectorFFT R=64 tuning report

Total measurements: **600**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 5160 | 4934 | 4683 | — | t1s |
| avx2 | 64 | 72 | 5204 | 4772 | 4360 | — | t1s |
| avx2 | 64 | 512 | 11078 | 11834 | 11220 | — | flat |
| avx2 | 128 | 128 | 18116 | 16922 | 15505 | — | t1s |
| avx2 | 128 | 136 | 11142 | 9926 | 8891 | — | t1s |
| avx2 | 128 | 1024 | 22449 | 23422 | 22735 | — | flat |
| avx2 | 256 | 256 | 40080 | 42093 | — | — | flat |
| avx2 | 256 | 264 | 23163 | 19506 | — | — | log3 |
| avx2 | 256 | 2048 | 44560 | 47992 | — | — | flat |
| avx2 | 512 | 512 | 97782 | 93756 | — | — | log3 |
| avx2 | 512 | 520 | 52378 | 42605 | — | — | log3 |
| avx2 | 512 | 4096 | 92150 | 93714 | — | — | flat |
| avx2 | 1024 | 1024 | 217284 | 184003 | — | — | log3 |
| avx2 | 1024 | 1032 | 116422 | 84148 | — | — | log3 |
| avx2 | 1024 | 8192 | 211622 | 185783 | — | — | log3 |
| avx2 | 2048 | 2048 | 544114 | 395413 | — | — | log3 |
| avx2 | 2048 | 2056 | 337006 | 178482 | — | — | log3 |
| avx2 | 2048 | 16384 | 543953 | 398118 | — | — | log3 |
| avx512 | 64 | 64 | 3401 | 3117 | 3696 | — | log3 |
| avx512 | 64 | 72 | 3342 | 2722 | 2609 | — | t1s |
| avx512 | 64 | 512 | 5991 | 6178 | 5916 | — | t1s |
| avx512 | 128 | 128 | 9589 | 9320 | 9138 | — | t1s |
| avx512 | 128 | 136 | 6300 | 5513 | 5506 | — | t1s |
| avx512 | 128 | 1024 | 12075 | 12472 | 11759 | — | t1s |
| avx512 | 256 | 256 | 22053 | 21808 | — | — | log3 |
| avx512 | 256 | 264 | 13468 | 12898 | — | — | log3 |
| avx512 | 256 | 2048 | 24175 | 24801 | — | — | flat |
| avx512 | 512 | 512 | 51000 | 50420 | — | — | log3 |
| avx512 | 512 | 520 | 29106 | 24038 | — | — | log3 |
| avx512 | 512 | 4096 | 50059 | 49037 | — | — | log3 |
| avx512 | 1024 | 1024 | 117893 | 99190 | — | — | log3 |
| avx512 | 1024 | 1032 | 74403 | 48845 | — | — | log3 |
| avx512 | 1024 | 8192 | 117586 | 99957 | — | — | log3 |
| avx512 | 2048 | 2048 | 291087 | 232308 | — | — | log3 |
| avx512 | 2048 | 2056 | 204886 | 139960 | — | — | log3 |
| avx512 | 2048 | 16384 | 295003 | 229765 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 11930 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 11913 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 13554 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 24084 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile128_temporal` | 23260 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 24478 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile256_temporal` | 47916 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile256_temporal` | 45340 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 47103 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile128_temporal` | 100664 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile256_temporal` | 96038 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile256_temporal` | 99890 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile256_temporal` | 241668 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile128_temporal` | 215043 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 238926 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 575844 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 498180 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile256_temporal` | 563847 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 5917 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 5842 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 11078 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 18116 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 11995 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 22449 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 40080 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 25853 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 44560 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 97782 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 53529 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 92150 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 217284 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 128275 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 211622 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 552017 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 333929 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 543953 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 5160 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 5204 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 11699 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 18935 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 11142 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 24168 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 44490 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 23163 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 48825 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 101136 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 52378 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 99923 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 239033 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 116422 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 235167 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 544114 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 337006 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 564126 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 4934 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 4772 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 11834 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 16922 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 9926 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 23422 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 42093 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 19506 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 47992 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 93756 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 42605 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 93714 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 184003 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 84148 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 185783 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 395413 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 178482 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 398118 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 4683 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 4360 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 11220 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 15505 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 8891 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 22735 |
| avx512 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 6299 |
| avx512 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 6424 |
| avx512 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 6763 |
| avx512 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile128_temporal` | 13311 |
| avx512 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 13231 |
| avx512 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 13864 |
| avx512 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile128_temporal` | 25852 |
| avx512 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 25982 |
| avx512 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile256_temporal` | 27087 |
| avx512 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile256_temporal` | 56925 |
| avx512 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 56536 |
| avx512 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 59233 |
| avx512 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 130450 |
| avx512 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile256_temporal` | 121573 |
| avx512 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 139141 |
| avx512 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 347168 |
| avx512 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile256_temporal` | 286647 |
| avx512 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 331631 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 3969 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 3647 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 6000 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 9589 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 7407 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 12075 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 22765 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 14635 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 24175 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 51000 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 29771 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 50059 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 117893 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 80614 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 117586 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 291087 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 210682 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 310358 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 3401 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 3342 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 5991 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 10348 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 6300 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 13040 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 22053 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 13468 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 25264 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 56502 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 29106 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 51807 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 126682 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 74403 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 123683 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 297251 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 204886 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 295003 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 3117 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 2722 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 6178 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 9320 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 5513 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 12472 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 21808 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 12898 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 24801 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 50420 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 24038 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 49037 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 99190 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 48845 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 99957 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 232308 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 139960 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 229765 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 3696 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 2609 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 5916 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 9138 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 5506 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 11759 |