# VectorFFT R=32 tuning report

Total measurements: **402**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 1014 | 860 | 717 | — | t1s |
| avx2 | 64 | 72 | 1073 | 892 | 763 | — | t1s |
| avx2 | 64 | 512 | 2363 | 1979 | 1900 | — | t1s |
| avx2 | 96 | 96 | 1622 | 1388 | 1367 | — | t1s |
| avx2 | 96 | 104 | 1546 | 1402 | 1289 | — | t1s |
| avx2 | 96 | 768 | 2834 | 3046 | 2272 | — | t1s |
| avx2 | 128 | 128 | 3495 | 2034 | 2532 | — | log3 |
| avx2 | 128 | 136 | 2081 | 1811 | 1818 | — | log3 |
| avx2 | 128 | 1024 | 4685 | 4902 | 4122 | — | t1s |
| avx2 | 192 | 192 | 2880 | 3317 | — | — | flat |
| avx2 | 192 | 200 | 3851 | 3050 | — | — | log3 |
| avx2 | 192 | 1536 | 6268 | 6175 | — | — | log3 |
| avx2 | 256 | 256 | 8121 | 6957 | — | — | log3 |
| avx2 | 256 | 264 | 5619 | 4717 | — | — | log3 |
| avx2 | 256 | 2048 | 8961 | 8711 | — | — | log3 |
| avx2 | 384 | 384 | 9858 | 6740 | — | — | log3 |
| avx2 | 384 | 392 | 7397 | 8358 | — | — | flat |
| avx2 | 384 | 3072 | 11675 | 15499 | — | — | flat |
| avx2 | 512 | 512 | 16642 | 15970 | — | — | log3 |
| avx2 | 512 | 520 | 9794 | 8439 | — | — | log3 |
| avx2 | 512 | 4096 | 18813 | 19822 | — | — | flat |
| avx2 | 768 | 768 | 22386 | 20543 | — | — | log3 |
| avx2 | 768 | 776 | 13800 | 13653 | — | — | log3 |
| avx2 | 768 | 6144 | 29055 | 30353 | — | — | flat |
| avx2 | 1024 | 1024 | 38454 | 32619 | — | — | log3 |
| avx2 | 1024 | 1032 | 18786 | 16151 | — | — | log3 |
| avx2 | 1024 | 8192 | 41256 | 46204 | — | — | flat |
| avx2 | 1536 | 1536 | 50349 | 49650 | — | — | log3 |
| avx2 | 1536 | 1544 | 35511 | 25108 | — | — | log3 |
| avx2 | 1536 | 12288 | 64150 | 67555 | — | — | flat |
| avx2 | 2048 | 2048 | 71112 | 72612 | — | — | flat |
| avx2 | 2048 | 2056 | 43132 | 33280 | — | — | log3 |
| avx2 | 2048 | 16384 | 136770 | 129676 | — | — | log3 |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 2281 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 2313 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 3029 |
| avx2 | `t1_buf_dit` | 96 | 96 | `ct_t1_buf_dit_tile64_temporal` | 2750 |
| avx2 | `t1_buf_dit` | 96 | 104 | `ct_t1_buf_dit_tile64_temporal` | 2548 |
| avx2 | `t1_buf_dit` | 96 | 768 | `ct_t1_buf_dit_tile64_temporal` | 3476 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 4189 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 4803 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 5231 |
| avx2 | `t1_buf_dit` | 192 | 192 | `ct_t1_buf_dit_tile128_temporal` | 6610 |
| avx2 | `t1_buf_dit` | 192 | 200 | `ct_t1_buf_dit_tile128_temporal` | 6330 |
| avx2 | `t1_buf_dit` | 192 | 1536 | `ct_t1_buf_dit_tile128_temporal` | 7212 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 8121 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 9377 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 8961 |
| avx2 | `t1_buf_dit` | 384 | 384 | `ct_t1_buf_dit_tile64_temporal` | 13577 |
| avx2 | `t1_buf_dit` | 384 | 392 | `ct_t1_buf_dit_tile64_temporal` | 12838 |
| avx2 | `t1_buf_dit` | 384 | 3072 | `ct_t1_buf_dit_tile64_temporal` | 15156 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 19270 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 19028 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 18813 |
| avx2 | `t1_buf_dit` | 768 | 768 | `ct_t1_buf_dit_tile64_temporal` | 28718 |
| avx2 | `t1_buf_dit` | 768 | 776 | `ct_t1_buf_dit_tile64_temporal` | 29451 |
| avx2 | `t1_buf_dit` | 768 | 6144 | `ct_t1_buf_dit_tile64_temporal` | 29055 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 39104 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 35290 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 44988 |
| avx2 | `t1_buf_dit` | 1536 | 1536 | `ct_t1_buf_dit_tile64_temporal` | 60992 |
| avx2 | `t1_buf_dit` | 1536 | 1544 | `ct_t1_buf_dit_tile64_temporal` | 55316 |
| avx2 | `t1_buf_dit` | 1536 | 12288 | `ct_t1_buf_dit_tile128_temporal` | 64150 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 83223 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 70982 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile128_temporal` | 140139 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1783 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1175 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2665 |
| avx2 | `t1_dif` | 96 | 96 | `ct_t1_dif` | 2898 |
| avx2 | `t1_dif` | 96 | 104 | `ct_t1_dif` | 1830 |
| avx2 | `t1_dif` | 96 | 768 | `ct_t1_dif` | 3516 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 4073 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 3498 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 6056 |
| avx2 | `t1_dif` | 192 | 192 | `ct_t1_dif` | 5839 |
| avx2 | `t1_dif` | 192 | 200 | `ct_t1_dif` | 5687 |
| avx2 | `t1_dif` | 192 | 1536 | `ct_t1_dif` | 9101 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 11323 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 5646 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 13104 |
| avx2 | `t1_dif` | 384 | 384 | `ct_t1_dif` | 11508 |
| avx2 | `t1_dif` | 384 | 392 | `ct_t1_dif` | 12067 |
| avx2 | `t1_dif` | 384 | 3072 | `ct_t1_dif` | 19124 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 24644 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 11248 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 30266 |
| avx2 | `t1_dif` | 768 | 768 | `ct_t1_dif` | 28495 |
| avx2 | `t1_dif` | 768 | 776 | `ct_t1_dif` | 25381 |
| avx2 | `t1_dif` | 768 | 6144 | `ct_t1_dif` | 43372 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 48617 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 33650 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 48886 |
| avx2 | `t1_dif` | 1536 | 1536 | `ct_t1_dif` | 63573 |
| avx2 | `t1_dif` | 1536 | 1544 | `ct_t1_dif` | 33886 |
| avx2 | `t1_dif` | 1536 | 12288 | `ct_t1_dif` | 100416 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 104910 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 50707 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 154295 |
| avx2 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 1224 |
| avx2 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 1322 |
| avx2 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 3257 |
| avx2 | `t1_dif_log3` | 96 | 96 | `ct_t1_dif_log3` | 2124 |
| avx2 | `t1_dif_log3` | 96 | 104 | `ct_t1_dif_log3` | 2896 |
| avx2 | `t1_dif_log3` | 96 | 768 | `ct_t1_dif_log3` | 4350 |
| avx2 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 3119 |
| avx2 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 2656 |
| avx2 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 6260 |
| avx2 | `t1_dif_log3` | 192 | 192 | `ct_t1_dif_log3` | 4257 |
| avx2 | `t1_dif_log3` | 192 | 200 | `ct_t1_dif_log3` | 4170 |
| avx2 | `t1_dif_log3` | 192 | 1536 | `ct_t1_dif_log3` | 8300 |
| avx2 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 11978 |
| avx2 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 6263 |
| avx2 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 10876 |
| avx2 | `t1_dif_log3` | 384 | 384 | `ct_t1_dif_log3` | 14693 |
| avx2 | `t1_dif_log3` | 384 | 392 | `ct_t1_dif_log3` | 8918 |
| avx2 | `t1_dif_log3` | 384 | 3072 | `ct_t1_dif_log3` | 18282 |
| avx2 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 23811 |
| avx2 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 13156 |
| avx2 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 25580 |
| avx2 | `t1_dif_log3` | 768 | 768 | `ct_t1_dif_log3` | 35797 |
| avx2 | `t1_dif_log3` | 768 | 776 | `ct_t1_dif_log3` | 23704 |
| avx2 | `t1_dif_log3` | 768 | 6144 | `ct_t1_dif_log3` | 36319 |
| avx2 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 51145 |
| avx2 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 32909 |
| avx2 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 50830 |
| avx2 | `t1_dif_log3` | 1536 | 1536 | `ct_t1_dif_log3` | 61139 |
| avx2 | `t1_dif_log3` | 1536 | 1544 | `ct_t1_dif_log3` | 45655 |
| avx2 | `t1_dif_log3` | 1536 | 12288 | `ct_t1_dif_log3` | 75446 |
| avx2 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 85679 |
| avx2 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 52203 |
| avx2 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 149544 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1014 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1073 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2363 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit` | 1622 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit` | 1546 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit` | 2834 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 3495 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2081 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4685 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit` | 2880 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit` | 3851 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit` | 6268 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 8830 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5619 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 9774 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit` | 9858 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit` | 7397 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit` | 11675 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 16642 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 9794 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 20728 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit` | 22386 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit` | 13800 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit` | 36617 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 38454 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 18786 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 41256 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit` | 50349 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit` | 35511 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit` | 67754 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 71112 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 43132 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 136770 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 860 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 892 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1979 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3` | 1388 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3` | 1402 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3` | 3046 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 2034 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1811 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 4902 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3` | 3317 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3` | 3050 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3` | 6175 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 6957 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 4717 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 8711 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3` | 6740 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3` | 8358 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3` | 15499 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 15970 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 8439 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 19822 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3` | 20543 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3` | 13653 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3` | 30353 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 32619 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 16151 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 46204 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3` | 49650 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3` | 25108 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3` | 67555 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 72612 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 33280 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 129676 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 717 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 763 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1900 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit` | 1367 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit` | 1289 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit` | 2272 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2532 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 1818 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 4122 |