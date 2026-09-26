# AVX-512 tail handling for the IL codelets

**Status: DECIDED and in the tree (2026-09-26).** The law is L10 in
`src/core/planning/policy.h` (`vfft_policy_il_tail_arm`); its one implementation is
`tail_policy` in `src/dag-fft-compiler/generator/lib/gen/c2c_il.ml`; every codelet under
`codelets/zil/avx512/` is emitted under it. Measured on an Emerald Rapids Xeon (4-vCPU
KVM guest, gcc 13), in a Claude Code cloud session.

---

## 1. The problem

An interleaved-complex (IL) codelet walks its columns in whole vectors. At AVX2 a ymm
holds 2 complex, so a column count that is not a multiple of 2 leaves one column over;
that column runs a narrow copy of the same DAG at 128 bits (the shipped "odd-count
tail"). At AVX-512 a zmm holds 4 complex, so `count % 4` leaves **1, 2 or 3** columns.

When does the tail run? Whenever a kernel's column count is not a multiple of 4:

- **batched transforms with K not a multiple of 4** — the column dimension is the batch,
  so every call's count is K (the common case, and the one that matters most);
- **odd-length transforms** — the stage counts are odd factors;
- **the K=1 solo kernels** (N ≤ 64): count = 1, the tail *is* the whole transform.

It never runs at power-of-two N with K = 1 through ZTURN-T (its geometry is whole
vectors) or through pair plans (their counts are powers of two).

Two correctness traps had to be fixed before any timing meant anything (both silent,
both found on this host, both now fixed in the generator):

- the narrow tail read **lane 0** of the twiddle record for every leftover column —
  right at AVX2, where the leftover is always lane 0; wrong at AVX-512 for columns 4g+1
  and 4g+2;
- the column-stride kinds had no width-8 gather at all.

## 2. What we tried

Every arm renders the *same* scheduled DAG, only at a different width; all are
bit-identical to each other and to the AVX2 codelet (checked in every run below).

| arm | what runs on the leftover | name in the generator |
|---|---|---|
| per-column xmm | one 128-bit pass per column (1–3 passes) | `narrowfix` |
| masked zmm | one full-width pass with a k-mask (`_mm512_maskz_loadu_pd` / `mask_storeu`) | `masked` |
| ladder | a 256-bit pass for 2 columns, then a 128-bit pass for 1 | `ladder` |
| hybrid | a k-masked ymm pass for 1–2, a k-masked zmm pass for 3 | `hyb2` |
| **ladder, masked at 3** | xmm for 1, ymm for 2, one k-masked zmm for 3 | **`ladder_m3`** |

`VFFT_TAIL512=<name>` selects an arm at generation time (recorded in the codelet's
provenance Env line).

## 3. What we found

Protocol: one pinned core, each arm timed as the minimum of 5 batches, 21 rounds with the
arm order alternated, median reported; the AVX2 codelets built as real AVX2
(`-mno-avx512f`, no EVEX). Run-to-run drift on this VM is 5–15%; the rankings below held
in every run. Raw logs (`*_results.log`) and sources: `docs/roadmap/zil_avx512_prototypes/harness/tail_policy/`.

### 3.1 Batched N = 1024, K interleaved transforms

32 × `n1c` radix-32 (count K) then one `t2c` radix-32 over 32 digits (count K). ns per
transform:

| K | leftover | per-column xmm | ladder | masked | hybrid | AVX2 |
|---|---|---|---|---|---|---|
| 1 | 1 | 5529 | **4570** | 8158 | 5693 | 5211 |
| 3 | 3 | 3468 | **2472** | 2473 | 2666 | 3462 |
| 5 | 1 | 2200 | **1990** | 2466 | 2271 | 2580 |
| 6 | 2 | 2491 | **1810** | 2266 | 1939 | 2245 |
| 7 | 3 | 2568 | 2200 | **2112** | 2165 | 2676 |
| 8 | 0 | 1915 | 1891 | 1916 | 1931 | 3172 |
| 9 | 1 | 2022 | **1984** | 2270 | 2100 | 2590 |
| 10 | 2 | 1951 | **1591** | 1774 | 1626 | 1985 |
| 13 | 1 | **1633** | 1738 | 1963 | 1869 | 2455 |
| 16 | 0 | 1815 | 1835 | 1799 | 1796 | 3388 |
| 33 | 1 | 1735 | **1682** | 1876 | 1872 | 2527 |
| 63 | 3 | 1740 | 1696 | **1600** | 1625 | 2482 |

K = 8 and 16 are the controls: no tail, and every arm times the same.

### 3.2 Odd lengths near 1024 and the K = 1 solos

Two-stage transforms (radix-R2 `n1` with count R1, then radix-R1 `t2` with count R2),
and the solo kernels at count 1. ns:

| case | leftover | per-column xmm | ladder | masked | hybrid | AVX2 |
|---|---|---|---|---|---|---|
| N=989 (43×23) | 3 / 3 | 3920 | 4780 | **3569** | 3697 | 5061 |
| N=999 (37×27) | 3 / 1 | 3546 | 3513 | **3486** | 3655 | 4747 |
| N=1025 (41×25) | 1 / 1 | **3103** | 3226 | 3376 | 3333 | 4744 |
| N=1073 (37×29) | 1 / 1 | **3109** | 3171 | 3301 | 3281 | 4734 |
| solo N=8 | 1 | **6.3** | 7.1 | 10.9 | 9.4 | 6.8 |
| solo N=16 | 1 | **13.9** | 15.2 | 24.9 | 21.5 | 14.1 |
| solo N=32 | 1 | 51.7 | **38.9** | 63.2 | 51.7 | 52.5 |
| solo N=64 | 1 | 114.6 | **99.1** | 168.4 | 131.7 | 127.1 |

### 3.3 What the numbers say

- **1 leftover:** a single narrow pass wins. Masked zmm pays a whole vector for one
  column: 1.3–1.9× slower on the solos, 1.8× at K = 1. The ladder's 1-column rung *is*
  a narrow pass, so it lands with the per-column loop (sometimes ahead on layout —
  solo N = 32/64 — sometimes 8–10% behind, N = 1025/1073).
- **2 leftovers:** the ymm pass wins clearly (K = 6: 1810 vs 2266 masked, 2491
  per-column).
- **3 leftovers:** masked zmm wins or ties. The ladder ties it at radix 32 (K = 3) but
  loses **25%** at the large odd radices (N = 989, radices 43/23) — the ymm+xmm pair
  costs two passes of a large DAG.
- **Why masked is not the free lunch it looks like:** a zmm pass costs about 1.5–1.8
  narrow passes on this core (512-bit ops issue on 2 ports, 128/256-bit on 3; measured
  7.7 vs 11.0 ops/ns on the butterfly mix); masked-off bytes still pay cache-line splits;
  and a masked access whose inactive bytes reach an untouched or unmapped page pays a
  fault-suppression assist (~120 ns per load, ~75 ns per store) on every call.
- **Code size** (the avx512 zil set): masked 6.5 MB, per-column 7.0 MB, ladder 10.5 MB;
  `ladder_m3` carries all three arms and is the largest. The tail runs once per call,
  so this is i-cache footprint, not loop cost.

## 4. What we decided

**`ladder_m3`**: 1 leftover → one xmm pass; 2 → one ymm pass; 3 → one k-masked zmm pass.
It takes the winning arm in every leftover class above, at the price of carrying three
arms per codelet. The branch is on `count - k`, known at run time, taken once per call.

Re-measured after landing (same benches, `ladder_m3` in place of the hybrid, which never
won):

| case | per-column xmm | ladder | masked | **ladder_m3** | AVX2 |
|---|---|---|---|---|---|
| N=989 (3/3) | 3257 | 4004 | 2988 | **3086** | 4258 |
| K=7 (3) | 2748 | 2131 | 2226 | **1933** | 2749 |
| K=6 (2) | 2466 | 1846 | 2330 | **1858** | 2300 |
| K=1 (1) | 5537 | 4150 | 8324 | **4310** | 6019 |
| solo N=64 (1) | 137.2 | 111.5 | 199.5 | **112.4** | 143.2 |

**Exception — corner-turned kinds** (n1t, t2t, t2tg and their row/tangent twins) keep
the per-column xmm arm at every leftover: their store writes one column per vector
address, so a two-column ymm rung cannot store through it. They were verified that way
(bitwise equal to AVX2 over counts 1–13, four buffer offsets, a guard page after the
output).

**AVX2 is unchanged**: at 2 complex per vector the leftover is at most one column, and
the shipped narrow tail stays byte-identical.

## 5. What this says about AVX-512 vs AVX2 in batched use

With the decided policy, batched N = 1024, ns per transform:

| K | 2 | 6 | 10 | 14 | 18 | 30 | 62 |
|---|---|---|---|---|---|---|---|
| AVX-512 | 2233 | 1878 | 1796 | 1396 | 1623 | 1526 | 1527 |
| AVX2 | 3150 | 2413 | 2305 | 2008 | 2383 | 2411 | 2353 |
| ratio | 1.41× | 1.28× | 1.28× | 1.44× | 1.47× | 1.58× | 1.54× |

- AVX-512 is ahead at every even K, **including K = 2**, where it never fills a zmm: its
  ymm pass is compiled with EVEX encodings and 32 registers (inference: fewer spills).
- At K = 4, 8, 16, 32, 64 the ratio reads 1.7–2.1×, but that is inflated: the AVX2 build
  runs about 40% slower there than at the neighbouring K (power-of-two leg strides of
  2–32 KB — cache-set and 4K-aliasing conflicts), while AVX-512 loses about 5%. The fair
  figure is the K % 4 = 2 row: **1.3–1.6×**. The stride effect itself is a batch-layout
  finding (pad the batch distance; `planning/pad_calibrate.h`), not a tail one.
- At odd K the advantage is 1.25–1.5×; at K = 1 AVX2 is about 10% faster than every
  AVX-512 arm.

## 6. Open items

- Every number is from one Intel VM. Zen 4 (512-bit ops as two 256-bit halves) and Zen 5
  may rank the arms differently; if they do, the arm table becomes per-uarch data (the
  generator's `tail_policy` is already a single switch).
- The static helper bodies of the group-loop and ZTT/msz kinds carry no target attribute
  of their own (116 avx512 files, and the same gap in the AVX2 tree), so they compile only
  with the build's `-m` flags. Harmless under both build systems; worth fixing with the
  helper-body attribute hook.
- The ZTT odd-mid `msz` kernels keep their own narrow arms (sse2, then scalar) and are
  not under this law yet; the flat DIT races msz against the other forms per stage.
