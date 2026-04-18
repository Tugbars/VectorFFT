# Codelet Selection Report (R=8)

Tie threshold: **2.0%**
Total decisions: **72**
Distinct winners: **4**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1_dif__avx2` | 34 |
| `ct_t1_dif__avx512` | 25 |
| `ct_t1_dit__avx512` | 11 |
| `ct_t1_dit__avx2` | 2 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dif__avx2` | 221 |
| 72 | 64 | `ct_t1_dif__avx2` | 217 |
| 128 | 64 | `ct_t1_dif__avx2` | 214 |
| 128 | 128 | `ct_t1_dif__avx2` | 448 |
| 136 | 128 | `ct_t1_dif__avx2` | 437 |
| 192 | 128 | `ct_t1_dif__avx2` | 593 |
| 256 | 256 | `ct_t1_dif__avx2` | 765 |
| 264 | 256 | `ct_t1_dif__avx2` | 729 |
| 320 | 256 | `ct_t1_dif__avx2` | 754 |
| 512 | 512 | `ct_t1_dif__avx2` | 3689 |
| 520 | 512 | `ct_t1_dif__avx2` | 1918 |
| 576 | 512 | `ct_t1_dif__avx2` | 1896 |
| 1024 | 1024 | `ct_t1_dif__avx2` | 6736 |
| 1032 | 1024 | `ct_t1_dif__avx2` | 4159 |
| 1088 | 1024 | `ct_t1_dif__avx2` | 3987 |
| 2048 | 2048 | `ct_t1_dif__avx2` | 13383 |
| 2056 | 2048 | `ct_t1_dif__avx2` | 8304 |
| 2112 | 2048 | `ct_t1_dif__avx2` | 7830 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dif__avx2` | 216 |
| 72 | 64 | `ct_t1_dif__avx2` | 218 |
| 128 | 64 | `ct_t1_dif__avx2` | 212 |
| 128 | 128 | `ct_t1_dif__avx2` | 457 |
| 136 | 128 | `ct_t1_dif__avx2` | 445 |
| 192 | 128 | `ct_t1_dif__avx2` | 601 |
| 256 | 256 | `ct_t1_dif__avx2` | 778 |
| 264 | 256 | `ct_t1_dif__avx2` | 693 |
| 320 | 256 | `ct_t1_dif__avx2` | 751 |
| 512 | 512 | `ct_t1_dif__avx2` | 3681 |
| 520 | 512 | `ct_t1_dif__avx2` | 1925 |
| 576 | 512 | `ct_t1_dif__avx2` | 1816 |
| 1024 | 1024 | `ct_t1_dif__avx2` | 6746 |
| 1032 | 1024 | `ct_t1_dit__avx2` | 4075 |
| 1088 | 1024 | `ct_t1_dif__avx2` | 4003 |
| 2048 | 2048 | `ct_t1_dit__avx2` | 9171 |
| 2056 | 2048 | `ct_t1_dif__avx2` | 8371 |
| 2112 | 2048 | `ct_t1_dif__avx2` | 8355 |

### avx512 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dif__avx512` | 124 |
| 72 | 64 | `ct_t1_dif__avx512` | 131 |
| 128 | 64 | `ct_t1_dif__avx512` | 131 |
| 128 | 128 | `ct_t1_dif__avx512` | 255 |
| 136 | 128 | `ct_t1_dit__avx512` | 279 |
| 192 | 128 | `ct_t1_dif__avx512` | 361 |
| 256 | 256 | `ct_t1_dif__avx512` | 439 |
| 264 | 256 | `ct_t1_dif__avx512` | 434 |
| 320 | 256 | `ct_t1_dit__avx512` | 545 |
| 512 | 512 | `ct_t1_dit__avx512` | 2964 |
| 520 | 512 | `ct_t1_dit__avx512` | 1523 |
| 576 | 512 | `ct_t1_dif__avx512` | 1484 |
| 1024 | 1024 | `ct_t1_dit__avx512` | 6287 |
| 1032 | 1024 | `ct_t1_dif__avx512` | 3236 |
| 1088 | 1024 | `ct_t1_dif__avx512` | 3277 |
| 2048 | 2048 | `ct_t1_dit__avx512` | 12245 |
| 2056 | 2048 | `ct_t1_dif__avx512` | 6680 |
| 2112 | 2048 | `ct_t1_dif__avx512` | 6437 |

### avx512 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1_dit__avx512` | 127 |
| 72 | 64 | `ct_t1_dif__avx512` | 130 |
| 128 | 64 | `ct_t1_dif__avx512` | 129 |
| 128 | 128 | `ct_t1_dif__avx512` | 256 |
| 136 | 128 | `ct_t1_dif__avx512` | 255 |
| 192 | 128 | `ct_t1_dif__avx512` | 369 |
| 256 | 256 | `ct_t1_dif__avx512` | 442 |
| 264 | 256 | `ct_t1_dif__avx512` | 434 |
| 320 | 256 | `ct_t1_dit__avx512` | 559 |
| 512 | 512 | `ct_t1_dit__avx512` | 2993 |
| 520 | 512 | `ct_t1_dif__avx512` | 1472 |
| 576 | 512 | `ct_t1_dif__avx512` | 1437 |
| 1024 | 1024 | `ct_t1_dit__avx512` | 6238 |
| 1032 | 1024 | `ct_t1_dif__avx512` | 3110 |
| 1088 | 1024 | `ct_t1_dif__avx512` | 3329 |
| 2048 | 2048 | `ct_t1_dit__avx512` | 12435 |
| 2056 | 2048 | `ct_t1_dif__avx512` | 6822 |
| 2112 | 2048 | `ct_t1_dif__avx512` | 6623 |
