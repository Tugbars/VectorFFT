# Codelet Selection Report (R=16)

Tie threshold: **2.0%**
Total decisions: **36**
Distinct winners: **3**

## Win counts

| Candidate | Regions won |
|-----------|-------------|
| `ct_t1s_dit__avx2` | 25 |
| `ct_t1_dit_log3__avx2` | 9 |
| `ct_t1_buf_dit__avx2__tile16_draintemporal` | 2 |

## Decisions by (isa, direction)

### avx2 / bwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx2` | 327 |
| 72 | 64 | `ct_t1s_dit__avx2` | 332 |
| 128 | 64 | `ct_t1s_dit__avx2` | 639 |
| 128 | 128 | `ct_t1s_dit__avx2` | 920 |
| 136 | 128 | `ct_t1s_dit__avx2` | 982 |
| 192 | 128 | `ct_t1s_dit__avx2` | 1135 |
| 256 | 256 | `ct_t1_buf_dit__avx2__tile16_draintemporal` | 3703 |
| 264 | 256 | `ct_t1s_dit__avx2` | 2064 |
| 320 | 256 | `ct_t1s_dit__avx2` | 2250 |
| 512 | 512 | `ct_t1s_dit__avx2` | 8260 |
| 520 | 512 | `ct_t1_dit_log3__avx2` | 4479 |
| 576 | 512 | `ct_t1s_dit__avx2` | 4736 |
| 1024 | 1024 | `ct_t1s_dit__avx2` | 16195 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2` | 8652 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 8779 |
| 2048 | 2048 | `ct_t1s_dit__avx2` | 30850 |
| 2056 | 2048 | `ct_t1s_dit__avx2` | 17287 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2` | 17217 |

### avx2 / fwd

| ios | me | winner | ns |
|-----|-----|--------|-----|
| 64 | 64 | `ct_t1s_dit__avx2` | 334 |
| 72 | 64 | `ct_t1s_dit__avx2` | 333 |
| 128 | 64 | `ct_t1s_dit__avx2` | 648 |
| 128 | 128 | `ct_t1s_dit__avx2` | 898 |
| 136 | 128 | `ct_t1s_dit__avx2` | 940 |
| 192 | 128 | `ct_t1s_dit__avx2` | 1144 |
| 256 | 256 | `ct_t1_buf_dit__avx2__tile16_draintemporal` | 3650 |
| 264 | 256 | `ct_t1s_dit__avx2` | 2078 |
| 320 | 256 | `ct_t1_dit_log3__avx2` | 2273 |
| 512 | 512 | `ct_t1s_dit__avx2` | 8434 |
| 520 | 512 | `ct_t1s_dit__avx2` | 4346 |
| 576 | 512 | `ct_t1s_dit__avx2` | 4555 |
| 1024 | 1024 | `ct_t1s_dit__avx2` | 16403 |
| 1032 | 1024 | `ct_t1_dit_log3__avx2` | 8538 |
| 1088 | 1024 | `ct_t1_dit_log3__avx2` | 8680 |
| 2048 | 2048 | `ct_t1s_dit__avx2` | 32103 |
| 2056 | 2048 | `ct_t1_dit_log3__avx2` | 17073 |
| 2112 | 2048 | `ct_t1_dit_log3__avx2` | 16570 |
