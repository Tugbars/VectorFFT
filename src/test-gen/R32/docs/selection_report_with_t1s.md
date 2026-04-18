# Codelet Selection Report (R=32)

Tie threshold: **2.0%**
Total decisions: **72**
Distinct winners: **22**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1s_dit__avx512` | 18 |
| `ct_t1s_dit__avx2` | 11 |
| `ct_t1_dit_log3__avx2` | 5 |
| `ct_t1_dit__avx2__tpf8r2` | 4 |
| `ct_t1_dit__avx512__tpf32r2` | 4 |
| `ct_t1_dit__avx2__tpf16r2` | 3 |
| `ct_t1_dit__avx512__tpf32r1` | 3 |
| `ct_t1_dit__avx512__tpf8r1` | 3 |
| `ct_t1_dit_log3__avx2__tpf16r1` | 3 |
| `ct_t1_dit__avx2__tpf16r1` | 2 |
| `ct_t1_dit__avx512` | 2 |
| `ct_t1_dit__avx2` | 2 |
| `ct_t1_dit_log3__avx2__tpf32r1` | 2 |
| `ct_t1_dit_log3__avx512__tpf4r1` | 2 |
| `ct_t1_dit_log3__avx512__tpf16r1` | 1 |
| `ct_t1_dit_log3__avx512__tpf8r2` | 1 |
| `ct_t1_dit__avx2__tpf32r2` | 1 |
| `ct_t1_dit__avx2__tpf4r1` | 1 |
| `ct_t1_dit__avx2__tpf32r1` | 1 |
| `ct_t1_dit_log3__avx512__tpf16r2` | 1 |
| `ct_t1_dit_log3__avx512__tpf32r2` | 1 |
| `ct_t1_dit_log3__avx2__tpf32r2` | 1 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx2` | 1484 |
| 72 | 64 | `ct_t1_dit__avx2__tpf16r1` | 1585 |
| 128 | 64 | `ct_t1_dit__avx2__tpf32r2` | 1855 |
| 128 | 128 | `ct_t1_dit__avx2__tpf16r2` | 3673 |
| 136 | 128 | `ct_t1_dit__avx2__tpf16r1` | 3527 |
| 192 | 128 | `ct_t1_dit__avx2__tpf4r1` | 3590 |
| 256 | 256 | `ct_t1_dit__avx2__tpf8r2` | 12168 |
| 264 | 256 | `ct_t1_dit__avx2` | 7634 |
| 320 | 256 | `ct_t1_dit__avx2__tpf32r1` | 7527 |
| 512 | 512 | `ct_t1_dit_log3__avx2__tpf16r1` | 25700 |
| 520 | 512 | `ct_t1_dit_log3__avx2__tpf32r1` | 15626 |
| 576 | 512 | `ct_t1_dit_log3__avx2` | 15634 |
| 1024 | 1024 | `ct_t1_dit_log3__avx2__tpf16r1` | 51519 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2` | 31868 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 31269 |
| 2048 | 2048 | `ct_t1_dit_log3__avx2` | 100893 |
| 2056 | 2048 | `ct_t1s_dit__avx2` | 58019 |
| 2112 | 2048 | `ct_t1s_dit__avx2` | 53417 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx2` | 1495 |
| 72 | 64 | `ct_t1_dit__avx2__tpf8r2` | 1568 |
| 128 | 64 | `ct_t1_dit__avx2__tpf16r2` | 1830 |
| 128 | 128 | `ct_t1_dit__avx2__tpf16r2` | 3636 |
| 136 | 128 | `ct_t1_dit__avx2__tpf8r2` | 3532 |
| 192 | 128 | `ct_t1_dit__avx2` | 3565 |
| 256 | 256 | `ct_t1_dit__avx2__tpf8r2` | 11944 |
| 264 | 256 | `ct_t1s_dit__avx2` | 7282 |
| 320 | 256 | `ct_t1s_dit__avx2` | 7333 |
| 512 | 512 | `ct_t1_dit_log3__avx2__tpf16r1` | 25280 |
| 520 | 512 | `ct_t1s_dit__avx2` | 15339 |
| 576 | 512 | `ct_t1s_dit__avx2` | 14550 |
| 1024 | 1024 | `ct_t1_dit_log3__avx2` | 50099 |
| 1032 | 1024 | `ct_t1s_dit__avx2` | 28792 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2__tpf32r1` | 31849 |
| 2048 | 2048 | `ct_t1_dit_log3__avx2__tpf32r2` | 102851 |
| 2056 | 2048 | `ct_t1s_dit__avx2` | 61094 |
| 2112 | 2048 | `ct_t1s_dit__avx2` | 58097 |

### avx512 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dit_log3__avx512__tpf8r2` | 993 |
| 72 | 64 | `ct_t1_dit__avx512` | 1064 |
| 128 | 64 | `ct_t1_dit__avx512__tpf32r2` | 1320 |
| 128 | 128 | `ct_t1_dit__avx512__tpf32r2` | 2609 |
| 136 | 128 | `ct_t1_dit__avx512__tpf32r1` | 2379 |
| 192 | 128 | `ct_t1_dit__avx512__tpf32r1` | 2421 |
| 256 | 256 | `ct_t1s_dit__avx512` | 6248 |
| 264 | 256 | `ct_t1s_dit__avx512` | 4965 |
| 320 | 256 | `ct_t1_dit__avx512__tpf8r1` | 5115 |
| 512 | 512 | `ct_t1_dit_log3__avx512__tpf32r2` | 14406 |
| 520 | 512 | `ct_t1s_dit__avx512` | 9922 |
| 576 | 512 | `ct_t1s_dit__avx512` | 10140 |
| 1024 | 1024 | `ct_t1s_dit__avx512` | 28773 |
| 1032 | 1024 | `ct_t1s_dit__avx512` | 20221 |
| 1088 | 1024 | `ct_t1s_dit__avx512` | 20333 |
| 2048 | 2048 | `ct_t1s_dit__avx512` | 52308 |
| 2056 | 2048 | `ct_t1s_dit__avx512` | 40382 |
| 2112 | 2048 | `ct_t1s_dit__avx512` | 31618 |

### avx512 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dit_log3__avx512__tpf16r1` | 1003 |
| 72 | 64 | `ct_t1_dit__avx512` | 1071 |
| 128 | 64 | `ct_t1_dit__avx512__tpf32r2` | 1309 |
| 128 | 128 | `ct_t1_dit__avx512__tpf32r2` | 2602 |
| 136 | 128 | `ct_t1_dit__avx512__tpf32r1` | 2398 |
| 192 | 128 | `ct_t1_dit__avx512__tpf8r1` | 2368 |
| 256 | 256 | `ct_t1s_dit__avx512` | 6263 |
| 264 | 256 | `ct_t1s_dit__avx512` | 4965 |
| 320 | 256 | `ct_t1_dit__avx512__tpf8r1` | 5161 |
| 512 | 512 | `ct_t1_dit_log3__avx512__tpf16r2` | 14597 |
| 520 | 512 | `ct_t1s_dit__avx512` | 10055 |
| 576 | 512 | `ct_t1s_dit__avx512` | 10137 |
| 1024 | 1024 | `ct_t1_dit_log3__avx512__tpf4r1` | 28753 |
| 1032 | 1024 | `ct_t1s_dit__avx512` | 20238 |
| 1088 | 1024 | `ct_t1s_dit__avx512` | 20533 |
| 2048 | 2048 | `ct_t1s_dit__avx512` | 56731 |
| 2056 | 2048 | `ct_t1_dit_log3__avx512__tpf4r1` | 33232 |
| 2112 | 2048 | `ct_t1s_dit__avx512` | 32335 |
