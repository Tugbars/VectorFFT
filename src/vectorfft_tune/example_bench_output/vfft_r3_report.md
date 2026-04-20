# VectorFFT R=3 tuning report

Total measurements: **768**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 24 | 24 | 26 | 25 | 25 | — | t1s |
| avx2 | 24 | 32 | 26 | 25 | 25 | — | t1s |
| avx2 | 24 | 72 | 25 | 26 | 27 | — | flat |
| avx2 | 24 | 192 | 25 | 26 | 25 | — | t1s |
| avx2 | 48 | 48 | 47 | 48 | 48 | — | flat |
| avx2 | 48 | 56 | 48 | 48 | 46 | — | t1s |
| avx2 | 48 | 144 | 49 | 48 | 48 | — | t1s |
| avx2 | 48 | 384 | 46 | 49 | 48 | — | flat |
| avx2 | 96 | 96 | 94 | 93 | 93 | — | t1s |
| avx2 | 96 | 104 | 93 | 93 | 94 | — | flat |
| avx2 | 96 | 288 | 90 | 93 | 93 | — | flat |
| avx2 | 96 | 768 | 93 | 93 | 94 | — | flat |
| avx2 | 192 | 192 | 186 | 183 | 181 | — | t1s |
| avx2 | 192 | 200 | 186 | 183 | 184 | — | log3 |
| avx2 | 192 | 576 | 184 | 188 | 185 | — | flat |
| avx2 | 192 | 1536 | 189 | 195 | 180 | — | t1s |
| avx2 | 384 | 384 | 364 | 375 | 382 | — | flat |
| avx2 | 384 | 392 | 375 | 364 | 377 | — | log3 |
| avx2 | 384 | 1152 | 369 | 373 | 366 | — | t1s |
| avx2 | 384 | 3072 | 368 | 363 | 369 | — | log3 |
| avx2 | 768 | 768 | 904 | 755 | 732 | — | t1s |
| avx2 | 768 | 776 | 925 | 794 | 731 | — | t1s |
| avx2 | 768 | 2304 | 912 | 758 | 734 | — | t1s |
| avx2 | 768 | 6144 | 862 | 830 | 730 | — | t1s |
| avx2 | 1536 | 1536 | 1920 | 1859 | 1650 | — | t1s |
| avx2 | 1536 | 1544 | 1917 | 1849 | 1695 | — | t1s |
| avx2 | 1536 | 4608 | 1888 | 1926 | 1657 | — | t1s |
| avx2 | 1536 | 12288 | 1948 | 1897 | 1585 | — | t1s |
| avx2 | 3072 | 3072 | 3899 | 3748 | 3816 | — | log3 |
| avx2 | 3072 | 3080 | 3779 | 3612 | 3402 | — | t1s |
| avx2 | 3072 | 9216 | 3944 | 5139 | 3394 | — | t1s |
| avx2 | 3072 | 24576 | 3964 | 3812 | 3447 | — | t1s |
| avx512 | 24 | 24 | 15 | 16 | 15 | — | flat |
| avx512 | 24 | 32 | 15 | 16 | 15 | — | flat |
| avx512 | 24 | 72 | 15 | 17 | 15 | — | flat |
| avx512 | 24 | 192 | 15 | 16 | 15 | — | flat |
| avx512 | 48 | 48 | 28 | 29 | 24 | — | t1s |
| avx512 | 48 | 56 | 28 | 29 | 23 | — | t1s |
| avx512 | 48 | 144 | 28 | 30 | 24 | — | t1s |
| avx512 | 48 | 384 | 28 | 29 | 25 | — | t1s |
| ... | ... (24 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 26 |
| avx2 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 26 |
| avx2 | `t1_dit` | 24 | 72 | `ct_t1_dit_u1` | 25 |
| avx2 | `t1_dit` | 24 | 192 | `ct_t1_dit_u1` | 25 |
| avx2 | `t1_dit` | 48 | 48 | `ct_t1_dit_u1` | 47 |
| avx2 | `t1_dit` | 48 | 56 | `ct_t1_dit_u1` | 48 |
| avx2 | `t1_dit` | 48 | 144 | `ct_t1_dit_u1` | 49 |
| avx2 | `t1_dit` | 48 | 384 | `ct_t1_dit_u1` | 46 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit_u1` | 94 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit_u1` | 93 |
| avx2 | `t1_dit` | 96 | 288 | `ct_t1_dit_u1` | 90 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit_u1` | 93 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit_u1` | 186 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit_u1` | 186 |
| avx2 | `t1_dit` | 192 | 576 | `ct_t1_dit_u1` | 184 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u1` | 189 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit_u1` | 364 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit_u1` | 375 |
| avx2 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u1` | 369 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u1` | 368 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit_u1` | 904 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit_u1` | 925 |
| avx2 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u1` | 912 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u1` | 862 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u1` | 1920 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u1` | 1917 |
| avx2 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u1` | 1888 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u1` | 1948 |
| avx2 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u1` | 3899 |
| avx2 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u1` | 3779 |
| avx2 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u1` | 3944 |
| avx2 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u1` | 3964 |
| avx2 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 25 |
| avx2 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 25 |
| avx2 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u1` | 26 |
| avx2 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u1` | 26 |
| avx2 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u1` | 48 |
| avx2 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u1` | 48 |
| avx2 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u1` | 48 |
| avx2 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u1` | 49 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u1` | 93 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u1` | 93 |
| avx2 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u1` | 93 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u1` | 93 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u1` | 183 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u1` | 183 |
| avx2 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u1` | 188 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u1` | 195 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u1` | 375 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u1` | 364 |
| avx2 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u1` | 373 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u1` | 363 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u1` | 755 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u1` | 794 |
| avx2 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u1` | 758 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u1` | 830 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 1859 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u1` | 1849 |
| avx2 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u1` | 1926 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u1` | 1897 |
| avx2 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u1` | 3748 |
| avx2 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 3612 |
| avx2 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u1` | 5139 |
| avx2 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u1` | 3812 |
| avx2 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u1` | 25 |
| avx2 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 25 |
| avx2 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u1` | 27 |
| avx2 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 25 |
| avx2 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u1` | 48 |
| avx2 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u1` | 46 |
| avx2 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u1` | 48 |
| avx2 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u1` | 48 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u1` | 93 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u1` | 94 |
| avx2 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u1` | 93 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u1` | 94 |
| avx2 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u1` | 181 |
| avx2 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u1` | 184 |
| avx2 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u1` | 185 |
| avx2 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u1` | 180 |
| avx2 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u1` | 382 |
| avx2 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u1` | 377 |
| avx2 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u1` | 366 |
| avx2 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u1` | 369 |
| avx2 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u1` | 732 |
| avx2 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u1` | 731 |
| avx2 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u1` | 734 |
| avx2 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u1` | 730 |
| avx2 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u1` | 1650 |
| avx2 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u1` | 1695 |
| avx2 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u1` | 1657 |
| avx2 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 1585 |
| avx2 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 3816 |
| avx2 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u1` | 3402 |
| avx2 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 3394 |
| avx2 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 3447 |
| avx512 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 15 |
| avx512 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 15 |
| avx512 | `t1_dit` | 24 | 72 | `ct_t1_dit_u1` | 15 |
| avx512 | `t1_dit` | 24 | 192 | `ct_t1_dit_u1` | 15 |
| avx512 | `t1_dit` | 48 | 48 | `ct_t1_dit_u2` | 28 |
| avx512 | `t1_dit` | 48 | 56 | `ct_t1_dit_u1` | 28 |
| avx512 | `t1_dit` | 48 | 144 | `ct_t1_dit_u2` | 28 |
| avx512 | `t1_dit` | 48 | 384 | `ct_t1_dit_u2` | 28 |
| avx512 | `t1_dit` | 96 | 96 | `ct_t1_dit_u2` | 55 |
| avx512 | `t1_dit` | 96 | 104 | `ct_t1_dit_u1` | 54 |
| avx512 | `t1_dit` | 96 | 288 | `ct_t1_dit_u2` | 55 |
| avx512 | `t1_dit` | 96 | 768 | `ct_t1_dit_u1` | 54 |
| avx512 | `t1_dit` | 192 | 192 | `ct_t1_dit_u2` | 100 |
| avx512 | `t1_dit` | 192 | 200 | `ct_t1_dit_u2` | 104 |
| avx512 | `t1_dit` | 192 | 576 | `ct_t1_dit_u3` | 103 |
| avx512 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u3` | 105 |
| avx512 | `t1_dit` | 384 | 384 | `ct_t1_dit_u2` | 210 |
| avx512 | `t1_dit` | 384 | 392 | `ct_t1_dit_u2` | 212 |
| avx512 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u2` | 202 |
| avx512 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u3` | 204 |
| avx512 | `t1_dit` | 768 | 768 | `ct_t1_dit_u3` | 721 |
| avx512 | `t1_dit` | 768 | 776 | `ct_t1_dit_u2` | 905 |
| avx512 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u2` | 757 |
| avx512 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u2` | 593 |
| avx512 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u2` | 1919 |
| avx512 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u2` | 1906 |
| avx512 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u2` | 1852 |
| avx512 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u3` | 1859 |
| avx512 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u3` | 3714 |
| avx512 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u2` | 3638 |
| avx512 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u2` | 3848 |
| avx512 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u2` | 3851 |
| avx512 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 16 |
| avx512 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 16 |
| avx512 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u3` | 17 |
| avx512 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u1` | 16 |
| avx512 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u2` | 29 |
| avx512 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u2` | 29 |
| avx512 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u3` | 30 |
| avx512 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u2` | 29 |
| avx512 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u2` | 57 |
| avx512 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u2` | 58 |
| avx512 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u2` | 56 |
| avx512 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u2` | 58 |
| avx512 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u2` | 114 |
| avx512 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u3` | 114 |
| avx512 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u2` | 113 |
| avx512 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u2` | 115 |
| avx512 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u2` | 232 |
| avx512 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u2` | 223 |
| avx512 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u2` | 218 |
| avx512 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u2` | 232 |
| avx512 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u2` | 485 |
| avx512 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u2` | 529 |
| avx512 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u2` | 469 |
| avx512 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u2` | 525 |
| avx512 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 1764 |
| avx512 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u2` | 1808 |
| avx512 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u2` | 1793 |
| avx512 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u2` | 1812 |
| avx512 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u3` | 3586 |
| avx512 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 3622 |
| avx512 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u1` | 3593 |
| avx512 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u3` | 3634 |
| avx512 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u3` | 15 |
| avx512 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 15 |
| avx512 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u3` | 15 |
| avx512 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 15 |
| avx512 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u2` | 24 |
| avx512 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u2` | 23 |
| avx512 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u2` | 24 |
| avx512 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u3` | 25 |
| avx512 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u3` | 46 |
| avx512 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u3` | 47 |
| avx512 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u3` | 47 |
| avx512 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u2` | 48 |
| avx512 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u3` | 92 |
| avx512 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u3` | 93 |
| avx512 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u3` | 91 |
| avx512 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u2` | 97 |
| avx512 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u3` | 184 |
| avx512 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u2` | 183 |
| avx512 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u3` | 183 |
| avx512 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u1` | 191 |
| avx512 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u2` | 372 |
| avx512 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u3` | 372 |
| avx512 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u2` | 383 |
| avx512 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u3` | 371 |
| avx512 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u3` | 1576 |
| avx512 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u3` | 1667 |
| avx512 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u2` | 1602 |
| avx512 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 1590 |
| avx512 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 3432 |
| avx512 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u1` | 3329 |
| avx512 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 3406 |
| avx512 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 3425 |