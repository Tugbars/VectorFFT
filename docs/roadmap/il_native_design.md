# IL-NATIVE CODELET FAMILY (tier-2) — design from the FFTW 3.3.10 ground truth

**Status**: design phase (2026-07-24). User directives: derived-IL population DELETED
(305 files; il_execute.h = connection-point shim); K=1 boundary-lattice twins live only
until this family replaces them; tier-1 (z-scratch on split compute) SKIPPED — build the
true interleaved-native family directly.

**Why**: every codelet we emit computes on split re/im registers; z-layout exists only as
boundary lattices. Measured cost: IL axis 0.48–0.71× MKL-IL everywhere, while MKL's own
IL kernels beat their zero-shuffle split kernels 20–38% (one memory stream vs two) and
their mid-N W-kernels stream duplicated twiddle pairs. The winning architecture on this
hardware is interleaved-native compute; FFTW's `dft/simd` family is the readable reference
implementation of exactly that, and this doc is its distillation against our emitter.

## 1. The FFTW mechanics (read from `simd-support/simd-avx2.h` + `dft/simd/common/*`)

### 1.1 Vector model
- `V = __m256d` holds **VL = 2 complex**, one `(re,im)` pair per 128-bit lane.
- `LD(x, ivs)`: lane 0 = complex at `x`, lane 1 = complex at `x + ivs` — the vector packs
  **two iterations of the codelet's own loop axis** (their `m`, our columns/me). Stride is
  a free parameter per call site. `ST` mirrors (two 128-bit stores).
- `LDA/STA`: contiguous 2-complex load/store (one `vmovupd`) when the loop axis is
  unit-stride in memory — the hot case for our four-step column geometry.
- `STM2/STN2/STN4`: strided/transposing store forms — the interleaved successors of our
  UL / il_out store lattices (FFTW's column-kernel sectioned stores).

### 1.2 The op algebra (what replaces our split add/sub/fma emission)
| primitive | AVX2 form | cost (2 complex) |
|---|---|---|
| add/sub | `vaddpd`/`vsubpd` directly on V | 1 op, ports 0/1/5 |
| × real const | `VMUL` with splat constant (`DVK`) | 1 mul |
| × i (rotation) | `VFMAI(b,c) = addsub(c, FLIP_RI(b))` | 1 shuffle + 1 addsub |
| ± conj forms | `VFMACONJ/VFNMSCONJ` = single `addsub` | ~1 op |
| × twiddle (computed) | `VZMUL(t,s) = fmaddsub(s, dupL(t), mul(flip(s), dupH(t)))` | 3 shuf + 1 mul + 1 fmaddsub |
| × twiddle (table, **VTW2**) | `BYTW2: fma(tr, s, mul(ti, flip(s)))` | **1 shuf + 1 mul + 1 fma** |
| conj-twiddle (DIT) | `BYTWJ2: fnms(ti, flip(s), mul(tr, s))` | same |

Key facts: (a) the `addsub` family makes ±i rotations and conjugate-adds 1–2 ops — the
interleaved format's compensation for half vector density; (b) with VTW2 tables the
per-twiddle shuffle count drops 3→1 because duplication AND the sign pattern are baked at
plan time; (c) FMA-port pressure per complex ≈ split's (shuffles ride port 5 under FMA
latency). Split retains a small edge on pure ±i butterflies (free by operand swap); the
bet — proven by MKL-IL and FFTW numbers — is stream shape > density above L1 scale.

### 1.3 VTW2 twiddle storage (double precision, per twiddle, per 2-lane vector)
```
twp[0] = [ cos_m,  cos_m,  cos_m',  cos_m' ]     (m, m' = the two lanes)
twp[1] = [ -sin_m, +sin_m, -sin_m', +sin_m' ]    (sign FOLDED into the table)
```
2 V per twiddle (TWVL2 = 2·VL), consumption-ordered with ONE advancing cursor — our twl
linear-table work generalized. This is the storage class MKL's mid-N W-kernels stream
(§12.1: `0x40 B/complex` splat-duplicated pairs). VTW3 = VTW1 storage with the t3
few-twiddle derivation (log3-class: load {w¹,w³,w⁹}, derive the rest via VZMUL) — the
IL analog of our leg-axis LOG3, a per-cell calibrated variant later, not M1.

### 1.4 Codelet shapes (`n1fv_8`, `t2fv_8` read end-to-end)
- `t2fv_8(ri, ii, W, rs, mb, me, ms)`: ONE array (`x = ri`, im implicit at +1); loop
  `m += VL`, `x += VL·ms`, `W += TWVL·(2(R−1))`; loads leg k at `x[2·k·rs_c]` with the
  second lane from `x + 2·ms_c`; `BYTWJ` immediately on load; pure V-algebra butterfly;
  in-place stores. This IS our t1 concept: vectorized over the m/columns axis, interleaved,
  tables streamed.
- `n1fv_R`: same, no twiddles — our leaf concept. `n2fv` adds the strided/sectioned store
  forms.
- Available radices cover our needs: n1fv up to 64 (+128), t2fv up to 64.

## 2. Mapping to our emitter (`codelet_oop.ml`) — the tier-2 deltas

1. **New backend value class `zvec`** (2 complex per ymm). The butterfly DAG
   (`prepare_butterfly`) is unchanged — only the scalar→SIMD lowering changes: emit the
   §1.2 algebra instead of split re/im pairs. Twiddle mults lower to BYTW2 against a
   VTW2 table; ±i and conj patterns lower to the addsub forms (new primitives in the
   render layer: `FLIP_RI`, `fmaddsub`, `addsub`, `movedup`, lane-dup).
2. **Twiddle emission**: plan-time VTW2 fill (cos-dup + sign-folded sin-dup, consumption
   order, one cursor). Reuses the twl linear-layout machinery conceptually.
3. **ABI**: z-pointer native — `fn(const double *z_in, double *z_out, const double *Wz,
   size_t Lz, size_t Gz, size_t OLz, size_t OGz, size_t count)` (strides in COMPLEX
   units; in==out allowed where the DAG permits). No re/im pointer pairs anywhere.
4. **K=1 plan shapes unchanged**: keep our two-pass four-step / ccol structures (they beat
   FFTW's recursion on the split axis); swap the kernel family. Column identity holds:
   lane = column, now 2 columns per vector instead of 4.
5. **VL=2 consequence**: iteration counts double vs split codelets. Expect the family to
   WIN ≥~512 (stream-bound) and possibly LOSE at 64/128 (L1-resident, density-bound —
   split mono-64 already ties MKL-IL). Both families stay in the calibrator ladder;
   per-cell wisdom decides, as always.
6. **Re-wire targets when the family lands**: il_execute.h sockets (classic z→z folds),
   K=1 IL routes (then DELETE the 32 boundary-lattice twins), ND-r2c `il2sp/sp2il` sweeps,
   r2c `_rfft_zrow` packing, JIT wrapper table rows for IL routes.

## 3. Milestones (mono-64 discipline: oracle first, then generalize)

- **M1**: `zvec` lowering for ONE codelet — `radix8_z_n1` (n1fv_8 analog) emitted by
  `gen_radix.exe 8 --z-native`; det+rnd gate vs naive; race vs (a) split leaf + lattices,
  (b) MKL-IL, at a K=8-column microbench. GO/NO-GO on codegen quality (spill count,
  port mix vs FFTW's own objdump).
- **M2**: `radix8_z_t2` (t2fv analog, VTW2 tables) + the two-pass z→z K=1 route at one
  cell (512 = 64×8 or 8×64); gate bit-vs-tolerance, race vs MKL-IL 291 ns.
- **M3**: radix set {4,8,16,32,64} × {n1,t2}, both directions (bwd = conj tables +
  swapped addsub forms — NOT text derivation); calibrator arms; wisdom.
- **M4**: re-wire the §2.6 targets; retire population 2; extend JIT wrapper table.

## 4. Open questions (resolve by measurement, not argument)
- VZMUL vs BYTW2 crossover (table bytes vs shuffles) — per-cell, likely BYTW2 everywhere
  like MKL/FFTW mid-N.
- Whether the leaf (no twiddles) prefers z-native or split-with-lattices at small R —
  the M1 race answers this.
- AVX-512: VL=4, `vpermilpd`-class dups scale; defer until AVX2 family is calibrated
  (same policy as everything else on this i9).
