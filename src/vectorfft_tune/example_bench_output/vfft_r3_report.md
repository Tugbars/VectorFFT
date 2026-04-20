# VectorFFT R=3 tuning report

Total measurements: **768**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 24 | 24 | 25 | 25 | 25 | — | t1s |
| avx2 | 24 | 32 | 26 | 25 | 25 | — | t1s |
| avx2 | 24 | 72 | 25 | 26 | 25 | — | t1s |
| avx2 | 24 | 192 | 25 | 25 | 24 | — | t1s |
| avx2 | 48 | 48 | 49 | 48 | 47 | — | t1s |
| avx2 | 48 | 56 | 49 | 47 | 47 | — | t1s |
| avx2 | 48 | 144 | 48 | 49 | 47 | — | t1s |
| avx2 | 48 | 384 | 48 | 48 | 48 | — | flat |
| avx2 | 96 | 96 | 94 | 94 | 94 | — | flat |
| avx2 | 96 | 104 | 93 | 95 | 93 | — | flat |
| avx2 | 96 | 288 | 97 | 95 | 94 | — | t1s |
| avx2 | 96 | 768 | 93 | 95 | 93 | — | t1s |
| avx2 | 192 | 192 | 186 | 186 | 188 | — | log3 |
| avx2 | 192 | 200 | 185 | 192 | 185 | — | flat |
| avx2 | 192 | 576 | 184 | 183 | 183 | — | t1s |
| avx2 | 192 | 1536 | 185 | 183 | 193 | — | log3 |
| avx2 | 384 | 384 | 368 | 365 | 364 | — | t1s |
| avx2 | 384 | 392 | 373 | 367 | 366 | — | t1s |
| avx2 | 384 | 1152 | 367 | 369 | 369 | — | flat |
| avx2 | 384 | 3072 | 386 | 371 | 364 | — | t1s |
| avx2 | 768 | 768 | 917 | 766 | 810 | — | log3 |
| avx2 | 768 | 776 | 931 | 805 | 726 | — | t1s |
| avx2 | 768 | 2304 | 893 | 750 | 737 | — | t1s |
| avx2 | 768 | 6144 | 849 | 831 | 737 | — | t1s |
| avx2 | 1536 | 1536 | 1898 | 1932 | 1611 | — | t1s |
| avx2 | 1536 | 1544 | 1929 | 1861 | 1731 | — | t1s |
| avx2 | 1536 | 4608 | 1918 | 1866 | 1835 | — | t1s |
| avx2 | 1536 | 12288 | 1972 | 1882 | 1590 | — | t1s |
| avx2 | 3072 | 3072 | 3892 | 3733 | 3402 | — | t1s |
| avx2 | 3072 | 3080 | 3855 | 3745 | 3393 | — | t1s |
| avx2 | 3072 | 9216 | 3912 | 3828 | 3402 | — | t1s |
| avx2 | 3072 | 24576 | 3924 | 3825 | 3376 | — | t1s |
| avx512 | 24 | 24 | 15 | 16 | 15 | — | t1s |
| avx512 | 24 | 32 | 15 | 16 | 15 | — | t1s |
| avx512 | 24 | 72 | 16 | 17 | 15 | — | t1s |
| avx512 | 24 | 192 | 15 | 17 | 15 | — | t1s |
| avx512 | 48 | 48 | 29 | 30 | 24 | — | t1s |
| avx512 | 48 | 56 | 29 | 29 | 23 | — | t1s |
| avx512 | 48 | 144 | 28 | 30 | 24 | — | t1s |
| avx512 | 48 | 384 | 29 | 29 | 24 | — | t1s |
| ... | ... (24 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 25 |
| avx2 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 26 |
| avx2 | `t1_dit` | 24 | 72 | `ct_t1_dit_u1` | 25 |
| avx2 | `t1_dit` | 24 | 192 | `ct_t1_dit_u1` | 25 |
| avx2 | `t1_dit` | 48 | 48 | `ct_t1_dit_u1` | 49 |
| avx2 | `t1_dit` | 48 | 56 | `ct_t1_dit_u1` | 49 |
| avx2 | `t1_dit` | 48 | 144 | `ct_t1_dit_u1` | 48 |
| avx2 | `t1_dit` | 48 | 384 | `ct_t1_dit_u1` | 48 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit_u1` | 94 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit_u1` | 93 |
| avx2 | `t1_dit` | 96 | 288 | `ct_t1_dit_u1` | 97 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit_u1` | 93 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit_u1` | 186 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit_u1` | 185 |
| avx2 | `t1_dit` | 192 | 576 | `ct_t1_dit_u1` | 184 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u1` | 185 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit_u1` | 368 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit_u1` | 373 |
| avx2 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u1` | 367 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u1` | 386 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit_u1` | 917 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit_u1` | 931 |
| avx2 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u1` | 893 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u1` | 849 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u1` | 1898 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u1` | 1929 |
| avx2 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u1` | 1918 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u1` | 1972 |
| avx2 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u1` | 3892 |
| avx2 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u1` | 3855 |
| avx2 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u1` | 3912 |
| avx2 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u1` | 3924 |
| avx2 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 25 |
| avx2 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 25 |
| avx2 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u1` | 26 |
| avx2 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u1` | 25 |
| avx2 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u1` | 48 |
| avx2 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u1` | 47 |
| avx2 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u1` | 49 |
| avx2 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u1` | 48 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u1` | 94 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u1` | 95 |
| avx2 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u1` | 95 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u1` | 95 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u1` | 186 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u1` | 192 |
| avx2 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u1` | 183 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u1` | 183 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u1` | 365 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u1` | 367 |
| avx2 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u1` | 369 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u1` | 371 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u1` | 766 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u1` | 805 |
| avx2 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u1` | 750 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u1` | 831 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 1932 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u1` | 1861 |
| avx2 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u1` | 1866 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u1` | 1882 |
| avx2 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u1` | 3733 |
| avx2 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 3745 |
| avx2 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u1` | 3828 |
| avx2 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u1` | 3825 |
| avx2 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u1` | 25 |
| avx2 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 25 |
| avx2 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u1` | 25 |
| avx2 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 24 |
| avx2 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u1` | 47 |
| avx2 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u1` | 47 |
| avx2 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u1` | 47 |
| avx2 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u1` | 48 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u1` | 94 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u1` | 93 |
| avx2 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u1` | 94 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u1` | 93 |
| avx2 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u1` | 188 |
| avx2 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u1` | 185 |
| avx2 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u1` | 183 |
| avx2 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u1` | 193 |
| avx2 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u1` | 364 |
| avx2 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u1` | 366 |
| avx2 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u1` | 369 |
| avx2 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u1` | 364 |
| avx2 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u1` | 810 |
| avx2 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u1` | 726 |
| avx2 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u1` | 737 |
| avx2 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u1` | 737 |
| avx2 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u1` | 1611 |
| avx2 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u1` | 1731 |
| avx2 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u1` | 1835 |
| avx2 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 1590 |
| avx2 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 3402 |
| avx2 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u1` | 3393 |
| avx2 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 3402 |
| avx2 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 3376 |
| avx512 | `t1_dit` | 24 | 24 | `ct_t1_dit_u1` | 15 |
| avx512 | `t1_dit` | 24 | 32 | `ct_t1_dit_u1` | 15 |
| avx512 | `t1_dit` | 24 | 72 | `ct_t1_dit_u2` | 16 |
| avx512 | `t1_dit` | 24 | 192 | `ct_t1_dit_u2` | 15 |
| avx512 | `t1_dit` | 48 | 48 | `ct_t1_dit_u2` | 29 |
| avx512 | `t1_dit` | 48 | 56 | `ct_t1_dit_u2` | 29 |
| avx512 | `t1_dit` | 48 | 144 | `ct_t1_dit_u1` | 28 |
| avx512 | `t1_dit` | 48 | 384 | `ct_t1_dit_u2` | 29 |
| avx512 | `t1_dit` | 96 | 96 | `ct_t1_dit_u1` | 54 |
| avx512 | `t1_dit` | 96 | 104 | `ct_t1_dit_u2` | 54 |
| avx512 | `t1_dit` | 96 | 288 | `ct_t1_dit_u2` | 55 |
| avx512 | `t1_dit` | 96 | 768 | `ct_t1_dit_u2` | 54 |
| avx512 | `t1_dit` | 192 | 192 | `ct_t1_dit_u3` | 105 |
| avx512 | `t1_dit` | 192 | 200 | `ct_t1_dit_u2` | 103 |
| avx512 | `t1_dit` | 192 | 576 | `ct_t1_dit_u2` | 107 |
| avx512 | `t1_dit` | 192 | 1536 | `ct_t1_dit_u3` | 105 |
| avx512 | `t1_dit` | 384 | 384 | `ct_t1_dit_u2` | 215 |
| avx512 | `t1_dit` | 384 | 392 | `ct_t1_dit_u2` | 208 |
| avx512 | `t1_dit` | 384 | 1152 | `ct_t1_dit_u2` | 209 |
| avx512 | `t1_dit` | 384 | 3072 | `ct_t1_dit_u3` | 208 |
| avx512 | `t1_dit` | 768 | 768 | `ct_t1_dit_u2` | 799 |
| avx512 | `t1_dit` | 768 | 776 | `ct_t1_dit_u2` | 950 |
| avx512 | `t1_dit` | 768 | 2304 | `ct_t1_dit_u3` | 800 |
| avx512 | `t1_dit` | 768 | 6144 | `ct_t1_dit_u1` | 602 |
| avx512 | `t1_dit` | 1536 | 1536 | `ct_t1_dit_u2` | 1916 |
| avx512 | `t1_dit` | 1536 | 1544 | `ct_t1_dit_u2` | 1890 |
| avx512 | `t1_dit` | 1536 | 4608 | `ct_t1_dit_u3` | 1916 |
| avx512 | `t1_dit` | 1536 | 12288 | `ct_t1_dit_u2` | 1928 |
| avx512 | `t1_dit` | 3072 | 3072 | `ct_t1_dit_u2` | 3835 |
| avx512 | `t1_dit` | 3072 | 3080 | `ct_t1_dit_u2` | 3779 |
| avx512 | `t1_dit` | 3072 | 9216 | `ct_t1_dit_u2` | 3823 |
| avx512 | `t1_dit` | 3072 | 24576 | `ct_t1_dit_u2` | 3867 |
| avx512 | `t1_dit_log3` | 24 | 24 | `ct_t1_dit_log3_u1` | 16 |
| avx512 | `t1_dit_log3` | 24 | 32 | `ct_t1_dit_log3_u1` | 16 |
| avx512 | `t1_dit_log3` | 24 | 72 | `ct_t1_dit_log3_u1` | 17 |
| avx512 | `t1_dit_log3` | 24 | 192 | `ct_t1_dit_log3_u3` | 17 |
| avx512 | `t1_dit_log3` | 48 | 48 | `ct_t1_dit_log3_u3` | 30 |
| avx512 | `t1_dit_log3` | 48 | 56 | `ct_t1_dit_log3_u2` | 29 |
| avx512 | `t1_dit_log3` | 48 | 144 | `ct_t1_dit_log3_u2` | 30 |
| avx512 | `t1_dit_log3` | 48 | 384 | `ct_t1_dit_log3_u2` | 29 |
| avx512 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_u3` | 58 |
| avx512 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3_u3` | 58 |
| avx512 | `t1_dit_log3` | 96 | 288 | `ct_t1_dit_log3_u3` | 58 |
| avx512 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_u3` | 59 |
| avx512 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3_u2` | 117 |
| avx512 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_u3` | 117 |
| avx512 | `t1_dit_log3` | 192 | 576 | `ct_t1_dit_log3_u2` | 115 |
| avx512 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_u2` | 115 |
| avx512 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_u3` | 234 |
| avx512 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3_u2` | 226 |
| avx512 | `t1_dit_log3` | 384 | 1152 | `ct_t1_dit_log3_u2` | 232 |
| avx512 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_u3` | 232 |
| avx512 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3_u3` | 474 |
| avx512 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3_u3` | 477 |
| avx512 | `t1_dit_log3` | 768 | 2304 | `ct_t1_dit_log3_u2` | 482 |
| avx512 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_u1` | 541 |
| avx512 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_u1` | 1814 |
| avx512 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_u1` | 1822 |
| avx512 | `t1_dit_log3` | 1536 | 4608 | `ct_t1_dit_log3_u2` | 1824 |
| avx512 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_u3` | 1821 |
| avx512 | `t1_dit_log3` | 3072 | 3072 | `ct_t1_dit_log3_u2` | 3649 |
| avx512 | `t1_dit_log3` | 3072 | 3080 | `ct_t1_dit_log3_u1` | 3632 |
| avx512 | `t1_dit_log3` | 3072 | 9216 | `ct_t1_dit_log3_u2` | 3652 |
| avx512 | `t1_dit_log3` | 3072 | 24576 | `ct_t1_dit_log3_u2` | 3656 |
| avx512 | `t1s_dit` | 24 | 24 | `ct_t1s_dit_u1` | 15 |
| avx512 | `t1s_dit` | 24 | 32 | `ct_t1s_dit_u1` | 15 |
| avx512 | `t1s_dit` | 24 | 72 | `ct_t1s_dit_u3` | 15 |
| avx512 | `t1s_dit` | 24 | 192 | `ct_t1s_dit_u1` | 15 |
| avx512 | `t1s_dit` | 48 | 48 | `ct_t1s_dit_u2` | 24 |
| avx512 | `t1s_dit` | 48 | 56 | `ct_t1s_dit_u2` | 23 |
| avx512 | `t1s_dit` | 48 | 144 | `ct_t1s_dit_u2` | 24 |
| avx512 | `t1s_dit` | 48 | 384 | `ct_t1s_dit_u2` | 24 |
| avx512 | `t1s_dit` | 96 | 96 | `ct_t1s_dit_u2` | 46 |
| avx512 | `t1s_dit` | 96 | 104 | `ct_t1s_dit_u3` | 46 |
| avx512 | `t1s_dit` | 96 | 288 | `ct_t1s_dit_u2` | 46 |
| avx512 | `t1s_dit` | 96 | 768 | `ct_t1s_dit_u2` | 49 |
| avx512 | `t1s_dit` | 192 | 192 | `ct_t1s_dit_u3` | 92 |
| avx512 | `t1s_dit` | 192 | 200 | `ct_t1s_dit_u2` | 92 |
| avx512 | `t1s_dit` | 192 | 576 | `ct_t1s_dit_u3` | 93 |
| avx512 | `t1s_dit` | 192 | 1536 | `ct_t1s_dit_u2` | 95 |
| avx512 | `t1s_dit` | 384 | 384 | `ct_t1s_dit_u3` | 183 |
| avx512 | `t1s_dit` | 384 | 392 | `ct_t1s_dit_u2` | 183 |
| avx512 | `t1s_dit` | 384 | 1152 | `ct_t1s_dit_u3` | 185 |
| avx512 | `t1s_dit` | 384 | 3072 | `ct_t1s_dit_u2` | 191 |
| avx512 | `t1s_dit` | 768 | 768 | `ct_t1s_dit_u2` | 369 |
| avx512 | `t1s_dit` | 768 | 776 | `ct_t1s_dit_u2` | 373 |
| avx512 | `t1s_dit` | 768 | 2304 | `ct_t1s_dit_u2` | 375 |
| avx512 | `t1s_dit` | 768 | 6144 | `ct_t1s_dit_u3` | 373 |
| avx512 | `t1s_dit` | 1536 | 1536 | `ct_t1s_dit_u2` | 1602 |
| avx512 | `t1s_dit` | 1536 | 1544 | `ct_t1s_dit_u3` | 1676 |
| avx512 | `t1s_dit` | 1536 | 4608 | `ct_t1s_dit_u3` | 1607 |
| avx512 | `t1s_dit` | 1536 | 12288 | `ct_t1s_dit_u1` | 1608 |
| avx512 | `t1s_dit` | 3072 | 3072 | `ct_t1s_dit_u1` | 3439 |
| avx512 | `t1s_dit` | 3072 | 3080 | `ct_t1s_dit_u3` | 3448 |
| avx512 | `t1s_dit` | 3072 | 9216 | `ct_t1s_dit_u1` | 3431 |
| avx512 | `t1s_dit` | 3072 | 24576 | `ct_t1s_dit_u1` | 3443 |