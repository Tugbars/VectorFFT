# VectorFFT R=16 tuning report

Total measurements: **936**

## Cross-protocol winners (fwd direction)

Best-of-protocol ns/call at each sweep point (informs plan-level
choice of twiddle-table layout).

| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |
|---|---|---|---|---|---|---|---|
| avx2 | 64 | 64 | 1254 | 992 | 2365 | — | log3 |
| avx2 | 64 | 72 | 1272 | 946 | 2363 | — | log3 |
| avx2 | 64 | 512 | 2176 | 1960 | 2559 | — | log3 |
| avx2 | 96 | 96 | 1994 | 1395 | 2935 | — | log3 |
| avx2 | 96 | 104 | 1977 | 1417 | 3029 | — | log3 |
| avx2 | 96 | 768 | 2712 | 1825 | 3084 | — | log3 |
| avx2 | 128 | 128 | 2781 | 2015 | 3844 | — | log3 |
| avx2 | 128 | 136 | 2686 | 1942 | 4674 | — | log3 |
| avx2 | 128 | 1024 | 4902 | 4535 | 4796 | — | log3 |
| avx2 | 192 | 192 | 4135 | 2945 | — | — | log3 |
| avx2 | 192 | 200 | 3999 | 2964 | — | — | log3 |
| avx2 | 192 | 1536 | 7476 | 7145 | — | — | log3 |
| avx2 | 256 | 256 | 6793 | 5221 | — | — | log3 |
| avx2 | 256 | 264 | 5422 | 3978 | — | — | log3 |
| avx2 | 256 | 2048 | 10472 | 9349 | — | — | log3 |
| avx2 | 384 | 384 | 8959 | 5919 | — | — | log3 |
| avx2 | 384 | 392 | 8201 | 5998 | — | — | log3 |
| avx2 | 384 | 3072 | 14564 | 14133 | — | — | log3 |
| avx2 | 512 | 512 | 18419 | 16498 | — | — | log3 |
| avx2 | 512 | 520 | 11904 | 7950 | — | — | log3 |
| avx2 | 512 | 4096 | 22518 | 18173 | — | — | log3 |
| avx2 | 768 | 768 | 20617 | 15364 | — | — | log3 |
| avx2 | 768 | 776 | 17125 | 11720 | — | — | log3 |
| avx2 | 768 | 6144 | 31663 | 28529 | — | — | log3 |
| avx2 | 1024 | 1024 | 44200 | 38447 | — | — | log3 |
| avx2 | 1024 | 1032 | 25763 | 16180 | — | — | log3 |
| avx2 | 1024 | 8192 | 44402 | 37949 | — | — | log3 |
| avx2 | 1536 | 1536 | 66498 | 60695 | — | — | log3 |
| avx2 | 1536 | 1544 | 40433 | 25996 | — | — | log3 |
| avx2 | 1536 | 12288 | 73302 | 56007 | — | — | log3 |
| avx2 | 2048 | 2048 | 103418 | 78832 | — | — | log3 |
| avx2 | 2048 | 2056 | 57939 | 35535 | — | — | log3 |
| avx2 | 2048 | 16384 | 99732 | 77618 | — | — | log3 |
| avx512 | 64 | 64 | 727 | 536 | 1302 | — | log3 |
| avx512 | 64 | 72 | 744 | 534 | 1292 | — | log3 |
| avx512 | 64 | 512 | 1334 | 1056 | 1572 | — | log3 |
| avx512 | 96 | 96 | 1273 | 822 | 1732 | — | log3 |
| avx512 | 96 | 104 | 1349 | 888 | 1653 | — | log3 |
| avx512 | 96 | 768 | 1900 | 1316 | 2268 | — | log3 |
| avx512 | 128 | 128 | 1759 | 1176 | 2192 | — | log3 |
| ... | ... (26 rows omitted) | | | | | | |

## Per-dispatcher winners (fwd)

Within each dispatcher slot (variants that compute the same
mathematical function), which variant wins each point?

| isa | dispatcher | me | ios | winner | ns |
|---|---|---|---|---|---|
| avx2 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 2130 |
| avx2 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 2056 |
| avx2 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 2168 |
| avx2 | `t1_buf_dit` | 96 | 96 | `ct_t1_buf_dit_tile64_temporal` | 2806 |
| avx2 | `t1_buf_dit` | 96 | 104 | `ct_t1_buf_dit_tile64_temporal` | 2765 |
| avx2 | `t1_buf_dit` | 96 | 768 | `ct_t1_buf_dit_tile64_temporal` | 3246 |
| avx2 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 4302 |
| avx2 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 3974 |
| avx2 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 4858 |
| avx2 | `t1_buf_dit` | 192 | 192 | `ct_t1_buf_dit_tile64_temporal` | 6438 |
| avx2 | `t1_buf_dit` | 192 | 200 | `ct_t1_buf_dit_tile64_temporal` | 6420 |
| avx2 | `t1_buf_dit` | 192 | 1536 | `ct_t1_buf_dit_tile64_temporal` | 7476 |
| avx2 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 8626 |
| avx2 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 8342 |
| avx2 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 10472 |
| avx2 | `t1_buf_dit` | 384 | 384 | `ct_t1_buf_dit_tile64_temporal` | 13675 |
| avx2 | `t1_buf_dit` | 384 | 392 | `ct_t1_buf_dit_tile64_temporal` | 12125 |
| avx2 | `t1_buf_dit` | 384 | 3072 | `ct_t1_buf_dit_tile64_temporal` | 14564 |
| avx2 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 19744 |
| avx2 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 16941 |
| avx2 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 22284 |
| avx2 | `t1_buf_dit` | 768 | 768 | `ct_t1_buf_dit_tile64_temporal` | 28023 |
| avx2 | `t1_buf_dit` | 768 | 776 | `ct_t1_buf_dit_tile64_temporal` | 25021 |
| avx2 | `t1_buf_dit` | 768 | 6144 | `ct_t1_buf_dit_tile64_temporal` | 31663 |
| avx2 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile128_temporal` | 44443 |
| avx2 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile64_temporal` | 39453 |
| avx2 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile128_temporal` | 44402 |
| avx2 | `t1_buf_dit` | 1536 | 1536 | `ct_t1_buf_dit_tile64_temporal` | 71360 |
| avx2 | `t1_buf_dit` | 1536 | 1544 | `ct_t1_buf_dit_tile64_temporal` | 62496 |
| avx2 | `t1_buf_dit` | 1536 | 12288 | `ct_t1_buf_dit_tile64_temporal` | 75290 |
| avx2 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile128_temporal` | 103418 |
| avx2 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 86029 |
| avx2 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 99732 |
| avx2 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 1034 |
| avx2 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 1003 |
| avx2 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 2328 |
| avx2 | `t1_dif` | 96 | 96 | `ct_t1_dif` | 1583 |
| avx2 | `t1_dif` | 96 | 104 | `ct_t1_dif` | 1586 |
| avx2 | `t1_dif` | 96 | 768 | `ct_t1_dif` | 2163 |
| avx2 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 2077 |
| avx2 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 11529 |
| avx2 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 5606 |
| avx2 | `t1_dif` | 192 | 192 | `ct_t1_dif` | 3228 |
| avx2 | `t1_dif` | 192 | 200 | `ct_t1_dif` | 3151 |
| avx2 | `t1_dif` | 192 | 1536 | `ct_t1_dif` | 8526 |
| avx2 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 5754 |
| avx2 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 4468 |
| avx2 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 12053 |
| avx2 | `t1_dif` | 384 | 384 | `ct_t1_dif` | 6400 |
| avx2 | `t1_dif` | 384 | 392 | `ct_t1_dif` | 6256 |
| avx2 | `t1_dif` | 384 | 3072 | `ct_t1_dif` | 17215 |
| avx2 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 19101 |
| avx2 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 10098 |
| avx2 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 25118 |
| avx2 | `t1_dif` | 768 | 768 | `ct_t1_dif` | 17987 |
| avx2 | `t1_dif` | 768 | 776 | `ct_t1_dif` | 13728 |
| avx2 | `t1_dif` | 768 | 6144 | `ct_t1_dif` | 182688 |
| avx2 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 52077 |
| avx2 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 20988 |
| avx2 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 49955 |
| avx2 | `t1_dif` | 1536 | 1536 | `ct_t1_dif` | 79969 |
| avx2 | `t1_dif` | 1536 | 1544 | `ct_t1_dif` | 35658 |
| avx2 | `t1_dif` | 1536 | 12288 | `ct_t1_dif` | 76974 |
| avx2 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 109584 |
| avx2 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 55033 |
| avx2 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 109612 |
| avx2 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 961 |
| avx2 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 980 |
| avx2 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 2061 |
| avx2 | `t1_dif_log3` | 96 | 96 | `ct_t1_dif_log3` | 1500 |
| avx2 | `t1_dif_log3` | 96 | 104 | `ct_t1_dif_log3` | 1486 |
| avx2 | `t1_dif_log3` | 96 | 768 | `ct_t1_dif_log3` | 1834 |
| avx2 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 2111 |
| avx2 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 2133 |
| avx2 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 4982 |
| avx2 | `t1_dif_log3` | 192 | 192 | `ct_t1_dif_log3` | 4890 |
| avx2 | `t1_dif_log3` | 192 | 200 | `ct_t1_dif_log3` | 4933 |
| avx2 | `t1_dif_log3` | 192 | 1536 | `ct_t1_dif_log3` | 7709 |
| avx2 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 5651 |
| avx2 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 4299 |
| avx2 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 10255 |
| avx2 | `t1_dif_log3` | 384 | 384 | `ct_t1_dif_log3` | 6798 |
| avx2 | `t1_dif_log3` | 384 | 392 | `ct_t1_dif_log3` | 6285 |
| avx2 | `t1_dif_log3` | 384 | 3072 | `ct_t1_dif_log3` | 17572 |
| avx2 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 17104 |
| avx2 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 8767 |
| avx2 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 20208 |
| avx2 | `t1_dif_log3` | 768 | 768 | `ct_t1_dif_log3` | 16373 |
| avx2 | `t1_dif_log3` | 768 | 776 | `ct_t1_dif_log3` | 12616 |
| avx2 | `t1_dif_log3` | 768 | 6144 | `ct_t1_dif_log3` | 32112 |
| avx2 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 42666 |
| avx2 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 17305 |
| avx2 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 41414 |
| avx2 | `t1_dif_log3` | 1536 | 1536 | `ct_t1_dif_log3` | 64891 |
| avx2 | `t1_dif_log3` | 1536 | 1544 | `ct_t1_dif_log3` | 28364 |
| avx2 | `t1_dif_log3` | 1536 | 12288 | `ct_t1_dif_log3` | 62928 |
| avx2 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 83610 |
| avx2 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 38904 |
| avx2 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 81938 |
| avx2 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 1254 |
| avx2 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 1272 |
| avx2 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 2176 |
| avx2 | `t1_dit` | 96 | 96 | `ct_t1_dit` | 1994 |
| avx2 | `t1_dit` | 96 | 104 | `ct_t1_dit` | 1977 |
| avx2 | `t1_dit` | 96 | 768 | `ct_t1_dit` | 2712 |
| avx2 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 2781 |
| avx2 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 2686 |
| avx2 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 4902 |
| avx2 | `t1_dit` | 192 | 192 | `ct_t1_dit` | 4135 |
| avx2 | `t1_dit` | 192 | 200 | `ct_t1_dit` | 3999 |
| avx2 | `t1_dit` | 192 | 1536 | `ct_t1_dit` | 7677 |
| avx2 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 6793 |
| avx2 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 5422 |
| avx2 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 11898 |
| avx2 | `t1_dit` | 384 | 384 | `ct_t1_dit` | 8959 |
| avx2 | `t1_dit` | 384 | 392 | `ct_t1_dit` | 8201 |
| avx2 | `t1_dit` | 384 | 3072 | `ct_t1_dit` | 15210 |
| avx2 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 18419 |
| avx2 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 11904 |
| avx2 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 22518 |
| avx2 | `t1_dit` | 768 | 768 | `ct_t1_dit` | 20617 |
| avx2 | `t1_dit` | 768 | 776 | `ct_t1_dit` | 17125 |
| avx2 | `t1_dit` | 768 | 6144 | `ct_t1_dit` | 33138 |
| avx2 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 44200 |
| avx2 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 25763 |
| avx2 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 48014 |
| avx2 | `t1_dit` | 1536 | 1536 | `ct_t1_dit` | 66498 |
| avx2 | `t1_dit` | 1536 | 1544 | `ct_t1_dit` | 40433 |
| avx2 | `t1_dit` | 1536 | 12288 | `ct_t1_dit` | 73302 |
| avx2 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 111649 |
| avx2 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 57939 |
| avx2 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 107076 |
| avx2 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3` | 992 |
| avx2 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3` | 946 |
| avx2 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log3` | 1960 |
| avx2 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3` | 1395 |
| avx2 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3` | 1417 |
| avx2 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3` | 1825 |
| avx2 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3` | 2015 |
| avx2 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1942 |
| avx2 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3` | 4535 |
| avx2 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3` | 2945 |
| avx2 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3` | 2964 |
| avx2 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3` | 7145 |
| avx2 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 5221 |
| avx2 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3` | 3978 |
| avx2 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3` | 9349 |
| avx2 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3` | 5919 |
| avx2 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3` | 5998 |
| avx2 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3` | 14133 |
| avx2 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 16498 |
| avx2 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3` | 7950 |
| avx2 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3` | 18173 |
| avx2 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3` | 15364 |
| avx2 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3` | 11720 |
| avx2 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3` | 28529 |
| avx2 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3` | 38447 |
| avx2 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3` | 16180 |
| avx2 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3` | 37949 |
| avx2 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3` | 60695 |
| avx2 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3` | 25996 |
| avx2 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3` | 56007 |
| avx2 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3` | 78832 |
| avx2 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 35535 |
| avx2 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3` | 77618 |
| avx2 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 2365 |
| avx2 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 2363 |
| avx2 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 2559 |
| avx2 | `t1s_dit` | 96 | 96 | `ct_t1s_dit` | 2935 |
| avx2 | `t1s_dit` | 96 | 104 | `ct_t1s_dit` | 3029 |
| avx2 | `t1s_dit` | 96 | 768 | `ct_t1s_dit` | 3084 |
| avx2 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 3844 |
| avx2 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 4674 |
| avx2 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 4796 |
| avx512 | `t1_buf_dit` | 64 | 64 | `ct_t1_buf_dit_tile64_temporal` | 1526 |
| avx512 | `t1_buf_dit` | 64 | 72 | `ct_t1_buf_dit_tile64_temporal` | 1495 |
| avx512 | `t1_buf_dit` | 64 | 512 | `ct_t1_buf_dit_tile64_temporal` | 1425 |
| avx512 | `t1_buf_dit` | 96 | 96 | `ct_t1_buf_dit_tile64_temporal` | 2074 |
| avx512 | `t1_buf_dit` | 96 | 104 | `ct_t1_buf_dit_tile64_temporal` | 2067 |
| avx512 | `t1_buf_dit` | 96 | 768 | `ct_t1_buf_dit_tile64_temporal` | 2113 |
| avx512 | `t1_buf_dit` | 128 | 128 | `ct_t1_buf_dit_tile64_temporal` | 3111 |
| avx512 | `t1_buf_dit` | 128 | 136 | `ct_t1_buf_dit_tile64_temporal` | 2959 |
| avx512 | `t1_buf_dit` | 128 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 2985 |
| avx512 | `t1_buf_dit` | 192 | 192 | `ct_t1_buf_dit_tile128_temporal` | 4152 |
| avx512 | `t1_buf_dit` | 192 | 200 | `ct_t1_buf_dit_tile128_temporal` | 4206 |
| avx512 | `t1_buf_dit` | 192 | 1536 | `ct_t1_buf_dit_tile64_temporal` | 4504 |
| avx512 | `t1_buf_dit` | 256 | 256 | `ct_t1_buf_dit_tile64_temporal` | 6300 |
| avx512 | `t1_buf_dit` | 256 | 264 | `ct_t1_buf_dit_tile64_temporal` | 6117 |
| avx512 | `t1_buf_dit` | 256 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 6641 |
| avx512 | `t1_buf_dit` | 384 | 384 | `ct_t1_buf_dit_tile64_temporal` | 9250 |
| avx512 | `t1_buf_dit` | 384 | 392 | `ct_t1_buf_dit_tile64_temporal` | 8711 |
| avx512 | `t1_buf_dit` | 384 | 3072 | `ct_t1_buf_dit_tile64_temporal` | 9595 |
| avx512 | `t1_buf_dit` | 512 | 512 | `ct_t1_buf_dit_tile64_temporal` | 12070 |
| avx512 | `t1_buf_dit` | 512 | 520 | `ct_t1_buf_dit_tile64_temporal` | 12050 |
| avx512 | `t1_buf_dit` | 512 | 4096 | `ct_t1_buf_dit_tile64_temporal` | 13416 |
| avx512 | `t1_buf_dit` | 768 | 768 | `ct_t1_buf_dit_tile64_temporal` | 18298 |
| avx512 | `t1_buf_dit` | 768 | 776 | `ct_t1_buf_dit_tile64_temporal` | 18286 |
| avx512 | `t1_buf_dit` | 768 | 6144 | `ct_t1_buf_dit_tile64_temporal` | 19622 |
| avx512 | `t1_buf_dit` | 1024 | 1024 | `ct_t1_buf_dit_tile64_temporal` | 26172 |
| avx512 | `t1_buf_dit` | 1024 | 1032 | `ct_t1_buf_dit_tile128_temporal` | 25673 |
| avx512 | `t1_buf_dit` | 1024 | 8192 | `ct_t1_buf_dit_tile64_temporal` | 27972 |
| avx512 | `t1_buf_dit` | 1536 | 1536 | `ct_t1_buf_dit_tile64_temporal` | 44167 |
| avx512 | `t1_buf_dit` | 1536 | 1544 | `ct_t1_buf_dit_tile64_temporal` | 40919 |
| avx512 | `t1_buf_dit` | 1536 | 12288 | `ct_t1_buf_dit_tile64_temporal` | 45290 |
| avx512 | `t1_buf_dit` | 2048 | 2048 | `ct_t1_buf_dit_tile64_temporal` | 68029 |
| avx512 | `t1_buf_dit` | 2048 | 2056 | `ct_t1_buf_dit_tile64_temporal` | 58964 |
| avx512 | `t1_buf_dit` | 2048 | 16384 | `ct_t1_buf_dit_tile64_temporal` | 62966 |
| avx512 | `t1_dif` | 64 | 64 | `ct_t1_dif` | 591 |
| avx512 | `t1_dif` | 64 | 72 | `ct_t1_dif` | 647 |
| avx512 | `t1_dif` | 64 | 512 | `ct_t1_dif` | 1478 |
| avx512 | `t1_dif` | 96 | 96 | `ct_t1_dif` | 1009 |
| avx512 | `t1_dif` | 96 | 104 | `ct_t1_dif` | 942 |
| avx512 | `t1_dif` | 96 | 768 | `ct_t1_dif` | 1860 |
| avx512 | `t1_dif` | 128 | 128 | `ct_t1_dif` | 1397 |
| avx512 | `t1_dif` | 128 | 136 | `ct_t1_dif` | 1357 |
| avx512 | `t1_dif` | 128 | 1024 | `ct_t1_dif` | 3122 |
| avx512 | `t1_dif` | 192 | 192 | `ct_t1_dif` | 2032 |
| avx512 | `t1_dif` | 192 | 200 | `ct_t1_dif` | 1944 |
| avx512 | `t1_dif` | 192 | 1536 | `ct_t1_dif` | 4773 |
| avx512 | `t1_dif` | 256 | 256 | `ct_t1_dif` | 5037 |
| avx512 | `t1_dif` | 256 | 264 | `ct_t1_dif` | 2776 |
| avx512 | `t1_dif` | 256 | 2048 | `ct_t1_dif` | 6791 |
| avx512 | `t1_dif` | 384 | 384 | `ct_t1_dif` | 4259 |
| avx512 | `t1_dif` | 384 | 392 | `ct_t1_dif` | 4179 |
| avx512 | `t1_dif` | 384 | 3072 | `ct_t1_dif` | 9489 |
| avx512 | `t1_dif` | 512 | 512 | `ct_t1_dif` | 12716 |
| avx512 | `t1_dif` | 512 | 520 | `ct_t1_dif` | 6071 |
| avx512 | `t1_dif` | 512 | 4096 | `ct_t1_dif` | 13339 |
| avx512 | `t1_dif` | 768 | 768 | `ct_t1_dif` | 14966 |
| avx512 | `t1_dif` | 768 | 776 | `ct_t1_dif` | 8904 |
| avx512 | `t1_dif` | 768 | 6144 | `ct_t1_dif` | 19110 |
| avx512 | `t1_dif` | 1024 | 1024 | `ct_t1_dif` | 27574 |
| avx512 | `t1_dif` | 1024 | 1032 | `ct_t1_dif` | 15278 |
| avx512 | `t1_dif` | 1024 | 8192 | `ct_t1_dif` | 27145 |
| avx512 | `t1_dif` | 1536 | 1536 | `ct_t1_dif` | 43238 |
| avx512 | `t1_dif` | 1536 | 1544 | `ct_t1_dif` | 21118 |
| avx512 | `t1_dif` | 1536 | 12288 | `ct_t1_dif` | 45268 |
| avx512 | `t1_dif` | 2048 | 2048 | `ct_t1_dif` | 66349 |
| avx512 | `t1_dif` | 2048 | 2056 | `ct_t1_dif` | 37773 |
| avx512 | `t1_dif` | 2048 | 16384 | `ct_t1_dif` | 60320 |
| avx512 | `t1_dif_log3` | 64 | 64 | `ct_t1_dif_log3` | 601 |
| avx512 | `t1_dif_log3` | 64 | 72 | `ct_t1_dif_log3` | 587 |
| avx512 | `t1_dif_log3` | 64 | 512 | `ct_t1_dif_log3` | 1224 |
| avx512 | `t1_dif_log3` | 96 | 96 | `ct_t1_dif_log3` | 892 |
| avx512 | `t1_dif_log3` | 96 | 104 | `ct_t1_dif_log3` | 850 |
| avx512 | `t1_dif_log3` | 96 | 768 | `ct_t1_dif_log3` | 1475 |
| avx512 | `t1_dif_log3` | 128 | 128 | `ct_t1_dif_log3` | 1227 |
| avx512 | `t1_dif_log3` | 128 | 136 | `ct_t1_dif_log3` | 1318 |
| avx512 | `t1_dif_log3` | 128 | 1024 | `ct_t1_dif_log3` | 2242 |
| avx512 | `t1_dif_log3` | 192 | 192 | `ct_t1_dif_log3` | 2074 |
| avx512 | `t1_dif_log3` | 192 | 200 | `ct_t1_dif_log3` | 2076 |
| avx512 | `t1_dif_log3` | 192 | 1536 | `ct_t1_dif_log3` | 3510 |
| avx512 | `t1_dif_log3` | 256 | 256 | `ct_t1_dif_log3` | 3859 |
| avx512 | `t1_dif_log3` | 256 | 264 | `ct_t1_dif_log3` | 2744 |
| avx512 | `t1_dif_log3` | 256 | 2048 | `ct_t1_dif_log3` | 4597 |
| avx512 | `t1_dif_log3` | 384 | 384 | `ct_t1_dif_log3` | 4226 |
| avx512 | `t1_dif_log3` | 384 | 392 | `ct_t1_dif_log3` | 5052 |
| avx512 | `t1_dif_log3` | 384 | 3072 | `ct_t1_dif_log3` | 6872 |
| avx512 | `t1_dif_log3` | 512 | 512 | `ct_t1_dif_log3` | 9727 |
| avx512 | `t1_dif_log3` | 512 | 520 | `ct_t1_dif_log3` | 5285 |
| avx512 | `t1_dif_log3` | 512 | 4096 | `ct_t1_dif_log3` | 9338 |
| avx512 | `t1_dif_log3` | 768 | 768 | `ct_t1_dif_log3` | 11842 |
| avx512 | `t1_dif_log3` | 768 | 776 | `ct_t1_dif_log3` | 8519 |
| avx512 | `t1_dif_log3` | 768 | 6144 | `ct_t1_dif_log3` | 14172 |
| avx512 | `t1_dif_log3` | 1024 | 1024 | `ct_t1_dif_log3` | 19385 |
| avx512 | `t1_dif_log3` | 1024 | 1032 | `ct_t1_dif_log3` | 11000 |
| avx512 | `t1_dif_log3` | 1024 | 8192 | `ct_t1_dif_log3` | 18199 |
| avx512 | `t1_dif_log3` | 1536 | 1536 | `ct_t1_dif_log3` | 28892 |
| avx512 | `t1_dif_log3` | 1536 | 1544 | `ct_t1_dif_log3` | 17243 |
| avx512 | `t1_dif_log3` | 1536 | 12288 | `ct_t1_dif_log3` | 29058 |
| avx512 | `t1_dif_log3` | 2048 | 2048 | `ct_t1_dif_log3` | 39430 |
| avx512 | `t1_dif_log3` | 2048 | 2056 | `ct_t1_dif_log3` | 24138 |
| avx512 | `t1_dif_log3` | 2048 | 16384 | `ct_t1_dif_log3` | 39058 |
| avx512 | `t1_dit` | 64 | 64 | `ct_t1_dit` | 727 |
| avx512 | `t1_dit` | 64 | 72 | `ct_t1_dit` | 744 |
| avx512 | `t1_dit` | 64 | 512 | `ct_t1_dit` | 1334 |
| avx512 | `t1_dit` | 96 | 96 | `ct_t1_dit` | 1273 |
| avx512 | `t1_dit` | 96 | 104 | `ct_t1_dit` | 1349 |
| avx512 | `t1_dit` | 96 | 768 | `ct_t1_dit` | 1900 |
| avx512 | `t1_dit` | 128 | 128 | `ct_t1_dit` | 1759 |
| avx512 | `t1_dit` | 128 | 136 | `ct_t1_dit` | 1664 |
| avx512 | `t1_dit` | 128 | 1024 | `ct_t1_dit` | 2948 |
| avx512 | `t1_dit` | 192 | 192 | `ct_t1_dit` | 2479 |
| avx512 | `t1_dit` | 192 | 200 | `ct_t1_dit` | 2615 |
| avx512 | `t1_dit` | 192 | 1536 | `ct_t1_dit` | 4376 |
| avx512 | `t1_dit` | 256 | 256 | `ct_t1_dit` | 4932 |
| avx512 | `t1_dit` | 256 | 264 | `ct_t1_dit` | 3469 |
| avx512 | `t1_dit` | 256 | 2048 | `ct_t1_dit` | 6071 |
| avx512 | `t1_dit` | 384 | 384 | `ct_t1_dit` | 5228 |
| avx512 | `t1_dit` | 384 | 392 | `ct_t1_dit` | 5164 |
| avx512 | `t1_dit` | 384 | 3072 | `ct_t1_dit` | 8942 |
| avx512 | `t1_dit` | 512 | 512 | `ct_t1_dit` | 12305 |
| avx512 | `t1_dit` | 512 | 520 | `ct_t1_dit` | 7416 |
| avx512 | `t1_dit` | 512 | 4096 | `ct_t1_dit` | 12690 |
| avx512 | `t1_dit` | 768 | 768 | `ct_t1_dit` | 15115 |
| avx512 | `t1_dit` | 768 | 776 | `ct_t1_dit` | 10949 |
| avx512 | `t1_dit` | 768 | 6144 | `ct_t1_dit` | 18438 |
| avx512 | `t1_dit` | 1024 | 1024 | `ct_t1_dit` | 23951 |
| avx512 | `t1_dit` | 1024 | 1032 | `ct_t1_dit` | 16097 |
| avx512 | `t1_dit` | 1024 | 8192 | `ct_t1_dit` | 23445 |
| avx512 | `t1_dit` | 1536 | 1536 | `ct_t1_dit` | 37198 |
| avx512 | `t1_dit` | 1536 | 1544 | `ct_t1_dit` | 24546 |
| avx512 | `t1_dit` | 1536 | 12288 | `ct_t1_dit` | 38487 |
| avx512 | `t1_dit` | 2048 | 2048 | `ct_t1_dit` | 59238 |
| avx512 | `t1_dit` | 2048 | 2056 | `ct_t1_dit` | 39909 |
| avx512 | `t1_dit` | 2048 | 16384 | `ct_t1_dit` | 55644 |
| avx512 | `t1_dit_log3` | 64 | 64 | `ct_t1_dit_log3_isub2` | 536 |
| avx512 | `t1_dit_log3` | 64 | 72 | `ct_t1_dit_log3_isub2` | 534 |
| avx512 | `t1_dit_log3` | 64 | 512 | `ct_t1_dit_log_half` | 1056 |
| avx512 | `t1_dit_log3` | 96 | 96 | `ct_t1_dit_log3_isub2` | 822 |
| avx512 | `t1_dit_log3` | 96 | 104 | `ct_t1_dit_log3` | 888 |
| avx512 | `t1_dit_log3` | 96 | 768 | `ct_t1_dit_log3_isub2` | 1316 |
| avx512 | `t1_dit_log3` | 128 | 128 | `ct_t1_dit_log3_isub2` | 1176 |
| avx512 | `t1_dit_log3` | 128 | 136 | `ct_t1_dit_log3` | 1152 |
| avx512 | `t1_dit_log3` | 128 | 1024 | `ct_t1_dit_log3_isub2` | 2057 |
| avx512 | `t1_dit_log3` | 192 | 192 | `ct_t1_dit_log3` | 1830 |
| avx512 | `t1_dit_log3` | 192 | 200 | `ct_t1_dit_log3_isub2` | 1757 |
| avx512 | `t1_dit_log3` | 192 | 1536 | `ct_t1_dit_log3_isub2` | 3289 |
| avx512 | `t1_dit_log3` | 256 | 256 | `ct_t1_dit_log3` | 3628 |
| avx512 | `t1_dit_log3` | 256 | 264 | `ct_t1_dit_log3_isub2` | 2379 |
| avx512 | `t1_dit_log3` | 256 | 2048 | `ct_t1_dit_log3_isub2` | 4147 |
| avx512 | `t1_dit_log3` | 384 | 384 | `ct_t1_dit_log3_isub2` | 3523 |
| avx512 | `t1_dit_log3` | 384 | 392 | `ct_t1_dit_log3` | 3721 |
| avx512 | `t1_dit_log3` | 384 | 3072 | `ct_t1_dit_log3_isub2` | 6246 |
| avx512 | `t1_dit_log3` | 512 | 512 | `ct_t1_dit_log3` | 8897 |
| avx512 | `t1_dit_log3` | 512 | 520 | `ct_t1_dit_log3_isub2` | 4511 |
| avx512 | `t1_dit_log3` | 512 | 4096 | `ct_t1_dit_log3_isub2` | 8222 |
| avx512 | `t1_dit_log3` | 768 | 768 | `ct_t1_dit_log3` | 11129 |
| avx512 | `t1_dit_log3` | 768 | 776 | `ct_t1_dit_log3` | 7582 |
| avx512 | `t1_dit_log3` | 768 | 6144 | `ct_t1_dit_log3_isub2` | 12952 |
| avx512 | `t1_dit_log3` | 1024 | 1024 | `ct_t1_dit_log3_isub2` | 17185 |
| avx512 | `t1_dit_log3` | 1024 | 1032 | `ct_t1_dit_log3_isub2` | 9811 |
| avx512 | `t1_dit_log3` | 1024 | 8192 | `ct_t1_dit_log3_isub2` | 16614 |
| avx512 | `t1_dit_log3` | 1536 | 1536 | `ct_t1_dit_log3_isub2` | 27049 |
| avx512 | `t1_dit_log3` | 1536 | 1544 | `ct_t1_dit_log3_isub2` | 15560 |
| avx512 | `t1_dit_log3` | 1536 | 12288 | `ct_t1_dit_log3_isub2` | 26038 |
| avx512 | `t1_dit_log3` | 2048 | 2048 | `ct_t1_dit_log3_isub2` | 35181 |
| avx512 | `t1_dit_log3` | 2048 | 2056 | `ct_t1_dit_log3` | 23170 |
| avx512 | `t1_dit_log3` | 2048 | 16384 | `ct_t1_dit_log3_isub2` | 35665 |
| avx512 | `t1s_dit` | 64 | 64 | `ct_t1s_dit` | 1302 |
| avx512 | `t1s_dit` | 64 | 72 | `ct_t1s_dit` | 1292 |
| avx512 | `t1s_dit` | 64 | 512 | `ct_t1s_dit` | 1572 |
| avx512 | `t1s_dit` | 96 | 96 | `ct_t1s_dit` | 1732 |
| avx512 | `t1s_dit` | 96 | 104 | `ct_t1s_dit` | 1653 |
| avx512 | `t1s_dit` | 96 | 768 | `ct_t1s_dit` | 2268 |
| avx512 | `t1s_dit` | 128 | 128 | `ct_t1s_dit` | 2192 |
| avx512 | `t1s_dit` | 128 | 136 | `ct_t1s_dit` | 2107 |
| avx512 | `t1s_dit` | 128 | 1024 | `ct_t1s_dit` | 2557 |