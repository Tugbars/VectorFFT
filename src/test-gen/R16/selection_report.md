# Codelet Selection Report (R=16)

Tie threshold: **2.0%**
Total decisions: **72**
Distinct winners: **6**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1_dit_log3__avx2` | 22 |
| `ct_t1_dit_log3__avx512` | 22 |
| `ct_t1s_dit__avx2` | 8 |
| `ct_t1_dit__avx512` | 8 |
| `ct_t1s_dit__avx512` | 6 |
| `ct_t1_buf_dit__avx2__tile128_draintemporal` | 6 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx2` | 580 |
| 72 | 64 | `ct_t1s_dit__avx2` | 623 |
| 128 | 64 | `ct_t1_dit_log3__avx2` | 385 |
| 128 | 128 | `ct_t1s_dit__avx2` | 963 |
| 136 | 128 | `ct_t1s_dit__avx2` | 1058 |
| 192 | 128 | `ct_t1_dit_log3__avx2` | 1150 |
| 256 | 256 | `ct_t1_dit_log3__avx2` | 5024 |
| 264 | 256 | `ct_t1_dit_log3__avx2` | 2595 |
| 320 | 256 | `ct_t1_dit_log3__avx2` | 2744 |
| 512 | 512 | `ct_t1_buf_dit__avx2__tile128_draintemporal` | 10990 |
| 520 | 512 | `ct_t1_dit_log3__avx2` | 5152 |
| 576 | 512 | `ct_t1_dit_log3__avx2` | 5480 |
| 1024 | 1024 | `ct_t1_buf_dit__avx2__tile128_draintemporal` | 21592 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2` | 10663 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 11143 |
| 2048 | 2048 | `ct_t1_buf_dit__avx2__tile128_draintemporal` | 44764 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2` | 23721 |
| 2112 | 2048 | `ct_t1s_dit__avx2` | 23677 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx2` | 593 |
| 72 | 64 | `ct_t1s_dit__avx2` | 597 |
| 128 | 64 | `ct_t1_dit_log3__avx2` | 394 |
| 128 | 128 | `ct_t1_dit_log3__avx2` | 926 |
| 136 | 128 | `ct_t1s_dit__avx2` | 1021 |
| 192 | 128 | `ct_t1_dit_log3__avx2` | 1129 |
| 256 | 256 | `ct_t1_dit_log3__avx2` | 5045 |
| 264 | 256 | `ct_t1_dit_log3__avx2` | 2674 |
| 320 | 256 | `ct_t1_dit_log3__avx2` | 2671 |
| 512 | 512 | `ct_t1_buf_dit__avx2__tile128_draintemporal` | 11076 |
| 520 | 512 | `ct_t1_dit_log3__avx2` | 5126 |
| 576 | 512 | `ct_t1_dit_log3__avx2` | 5539 |
| 1024 | 1024 | `ct_t1_buf_dit__avx2__tile128_draintemporal` | 21463 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2` | 10737 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 10864 |
| 2048 | 2048 | `ct_t1_buf_dit__avx2__tile128_draintemporal` | 46765 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2` | 24445 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2` | 22148 |

### avx512 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx512` | 368 |
| 72 | 64 | `ct_t1s_dit__avx512` | 415 |
| 128 | 64 | `ct_t1_dit_log3__avx512` | 235 |
| 128 | 128 | `ct_t1_dit_log3__avx512` | 539 |
| 136 | 128 | `ct_t1s_dit__avx512` | 637 |
| 192 | 128 | `ct_t1_dit_log3__avx512` | 798 |
| 256 | 256 | `ct_t1_dit__avx512` | 2951 |
| 264 | 256 | `ct_t1_dit_log3__avx512` | 1855 |
| 320 | 256 | `ct_t1s_dit__avx512` | 2021 |
| 512 | 512 | `ct_t1_dit_log3__avx512` | 6035 |
| 520 | 512 | `ct_t1_dit_log3__avx512` | 3660 |
| 576 | 512 | `ct_t1_dit_log3__avx512` | 4100 |
| 1024 | 1024 | `ct_t1_dit__avx512` | 12480 |
| 1032 | 1024 | `ct_t1_dit_log3__avx512` | 7708 |
| 1088 | 1024 | `ct_t1_dit_log3__avx512` | 7889 |
| 2048 | 2048 | `ct_t1_dit_log3__avx512` | 24477 |
| 2056 | 2048 | `ct_t1_dit_log3__avx512` | 16222 |
| 2112 | 2048 | `ct_t1_dit_log3__avx512` | 18036 |

### avx512 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dit__avx512` | 375 |
| 72 | 64 | `ct_t1s_dit__avx512` | 435 |
| 128 | 64 | `ct_t1_dit__avx512` | 262 |
| 128 | 128 | `ct_t1_dit_log3__avx512` | 527 |
| 136 | 128 | `ct_t1s_dit__avx512` | 652 |
| 192 | 128 | `ct_t1_dit_log3__avx512` | 819 |
| 256 | 256 | `ct_t1_dit__avx512` | 2933 |
| 264 | 256 | `ct_t1_dit_log3__avx512` | 1842 |
| 320 | 256 | `ct_t1_dit__avx512` | 1970 |
| 512 | 512 | `ct_t1_dit__avx512` | 6335 |
| 520 | 512 | `ct_t1_dit_log3__avx512` | 3905 |
| 576 | 512 | `ct_t1_dit_log3__avx512` | 4181 |
| 1024 | 1024 | `ct_t1_dit__avx512` | 12602 |
| 1032 | 1024 | `ct_t1_dit_log3__avx512` | 7403 |
| 1088 | 1024 | `ct_t1_dit_log3__avx512` | 8147 |
| 2048 | 2048 | `ct_t1_dit_log3__avx512` | 27861 |
| 2056 | 2048 | `ct_t1_dit_log3__avx512` | 16110 |
| 2112 | 2048 | `ct_t1_dit_log3__avx512` | 16456 |
