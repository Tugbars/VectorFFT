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
  cell. **⚠ split choice is a CALIBRATOR AXIS, not fixed**: MKL's disassembled N=512 uses
  **16×32** (small first radix, larger co-size — small p5-bound reshape leaf, FMA density in
  pass 2), NOT the 64×8/8×64 this doc first proposed. Race 16×32 alongside 64×8/8×64 vs
  MKL-IL 291 ns; treat column-radix and W-length as independent axes. See
  [../research/mkl_il_512_anatomy.md](../research/mkl_il_512_anatomy.md) §8 item 5.
- **M3**: radix set {4,8,16,32,64} × {n1,t2}, both directions (bwd = conj tables +
  swapped addsub forms — NOT text derivation); calibrator arms; wisdom. Ship
  **direction-specialized fwd/bwd codelets** (MKL does: fwd/bwd differ only by swapped
  `vfmadd`↔`vfnmadd` + ±i sign, no runtime branch). Amortize digit-reversal into a
  **precomputed input-offset stack table** (MKL: `mov 0xNN(%rsp),%r13; vmovupd (%rcx,%r13,1)`)
  so the permutation costs address-gen, not vector ports.
- **M4**: re-wire the §2.6 targets; retire population 2; extend JIT wrapper table.

### 3a. MKL-confirmed emitter directives (from docs/research/mkl_il_512_anatomy.md, verified)
- **Cap the M1 leaf radix at 8, not 16.** MKL's radix-8 column sibling runs spill-free in ≤16
  ymm (26 FP-arith + 8 transpose + 6 ±i ops); its radix-16 needs a 0x218 stack frame + mid-body
  spills. Spill-free radix-8 is the AVX2 sweet spot; reserve radix-16/32 for M3.
- **Two-regime scheduler keyed to arithmetic intensity.** Twiddle-free/low-AI → compact,
  register-resident, LOOPED (matches our lazy-load sink-first). Twiddle-dense → SPILL-TO-WIDEN:
  raise the live cap above 16, park completed sub-DAGs in an L1-resident stack pool, unroll
  fully straight-line (MKL parks ~62 ymm-slots / ~2.4 KB to expose ILP). Keep the streamed
  VTW2 twiddle-apply a SEPARATE stage feeding scratch — do NOT fuse BYTW2 into the FMA-dense combine.
- **Port 5 is a first-class scheduler resource.** The twiddle-free pass is p5-BOUND (8 mandatory
  `vinsertf128`/`vperm2f128` + 3 `vshufpd` vs 4 FMA); a cp_dist scheduler over one generic
  "vector ALU" will starve it. Add a p5 pressure term; minimize interleave→strided transpose
  pairs/output. GO/NO-GO metric = match MKL's arith (~26 flop/output, FMA:add≈1:1) while cutting
  its movement half (MKL is only ~40% arithmetic, ~23% spill, ~12% `vmovapd` copies).
- **VTW2 record is COS-FIRST** `[c,c,c',c'][-s,+s,-s',+s']` (64 B, two lanes = two adjacent
  rows, exp r·k), single forward cursor `add $0x40`. Verified byte-level against MKL's live table.
- **Gate VTW2 (32 B/pt) vs compact (16 B/pt) by twiddle-stream residency, per cell** — VTW2 wins
  where the stream is L1/L2-resident (≤mid-N, our IL-hot zone); compact wins where it spills to DRAM.

## 3c. M1 EMITTER — DONE (2026-07-24)

`codelet_zil.ml` = the new complex-vector backend (the existing emitter's `expr` IR is
real-valued split arithmetic — a genfft-scalar model — so native IL required a separate
backend, not a renderer swap). `gen_radix.exe 8 --z-native --isa avx2 --emit-c` emits
`codelets/zil/avx2/radix8_z_n1_avx2.c` (z-pointer oop11-shaped ABI, complex-unit strides,
2 complex/ymm, ±i via `permute 0x5 + xor` sign mask). Family dir `codelets/zil/` wired into
build.py. **GATE: emitted == hand oracle BIT-IDENTICAL (0.0); oracle == naive 2.94e-15.**
NEXT (M2): radices 4/16/32/64 n1 + the t2 twiddle kernel (VTW2 cos-first + BYTW2), then the
two-pass z→z K=1 route raced vs MKL-IL per §3a.

## 3d. INTERIM PURE-IL SCOREBOARD (2026-07-24, `zil_ladder_race.c`) — the baseline

z-native two-pass (ONE uncalibrated shape/cell, strided-load t2s, no scheduling work) vs
LIVE MKL-IL, same process, in-place, all cells gated ~1e-15 first:

| N | shape | z (ns) | MKL-IL (ns) | Z/MKL |
|--:|---|--:|--:|--:|
| 64 | 8×8 | 29.6 | 30.2 | **1.02 WIN** |
| 128 | 8×16 | 67.2 | 68.3 | **1.02 WIN** |
| 256 | 8×32 | 174.4 | 135.9 | 0.78 |
| 512 | 8×64 | 398.7 | 290.5 | 0.73 (0.82 in the earlier run — band) |
| 1024 | 16×64 | 1320.9 | 799.9 | 0.61 |
| 2048 | 32×64 | 3107.0 | 2633.2 | 0.85 |
| 4096 | 64×64 | 11162.9 | 4208.2 | 0.38 |

Regime read: **≤128 = WON on first attempt** (vs MKL's hand-polished mono kernels).
Mid-N 0.61–0.85 with the top levers unbuilt (item 4 corner-turn stores kills the strided
2×128 load tax; per-cell shape calibration; scheduling). **4096 = the predicted VTW2
residency cliff** (~126 KB twiddle stream → the 6b compact-table knob) + monolithic
spilling r64 t2s + single two-pass vs MKL's deeper recursion. This table is the baseline
every subsequent lever is measured against.

## 3b. M1 GO/NO-GO — MEASURED (2026-07-24): GO, regime-dependent

Hand-written z-native radix-8 leaf vs split-radix-8 + il_in/il_out boundary (our current
arch), both bit-exact (2.94e-15), on the same interleaved buffer. Bench
`build_tuned/benches/il_r8_m1_race.c`; full numbers `../research/m1_znative_r8_race.txt`.

| K (batch) | z-native | split+il | verdict |
|--:|--:|--:|--|
| 8 | 1.32 | 3.24 | **z-native 2.5× WIN** |
| 64 | 1.30 | 3.18 | **z-native 2.4× WIN** |
| 256 | 4.73 | 4.09 | split 1.15× |
| 1024 | 4.47 | 3.92 | split 1.14× |
| 4096 | 3.79 | 4.00 | z-native ~tie |

**Verdict: GO.** z-native **dominates 2.5× at small K** (≤64) — the leaf's common regime (the
four-step leaf runs at K=R1; the mono tier is exactly here). It loses ~15% at mid-K (256–1024)
where its VL=2 penalty (2× butterfly instrs vs split's 4-wide) is exposed once L1-resident
latency stops hiding it. **This twiddle-free leaf is z-native's WORST case** (no VTW2/BYTW2
advantage); the M2 t2 kernel should favor it more. Two consequences: (a) build M1; (b) the
regime-dependence means **keep both families in the calibrator** — per-cell choice, as the
whole engine already does.

**Bonus — the memory-bound thesis, measured again:** split on *two* streams (`re[]`+`im[]`) is
**49% SLOWER** than split-with-il-conversion on *one* z stream at K=1024. The il boundary buys
single-stream access and pays for itself mid-K — the two-memory-stream tax is real and large.

## 5. FAMILY BUILD CHECKLIST (numbered; each item gated before the next depends on it)

Twiddle-strategy note first, because it shapes items 5–8: the split engine ships THREE
calibrated twiddle strategies — **t1/FLAT** (per-lane vector loads from parallel per-leg
rows, `loadu(&tw_re[l*me+b])`), **t1s/BROADCAST** (`set1(tw_re[l])`, one W shared by all
lanes — admissible only when W is constant across the SIMD axis), **LOG3** (sparse base
rows + in-register cmul derivation chains) — plus twl (consumption-order linear layout of
FLAT). The three-twiddle-methods thesis (per-cell measured mixing) is a core contribution
and CARRIES OVER: the IL family gets the analog of each, raced per cell, not one collapsed
method. The twiddle geometry law carries too: in z-vectors the two 128-bit lanes are two
adjacent columns/rows, so K=1 four-step twiddles VARY across the vector → broadcast-class
inadmissible there, per-lane-class (VTW2) and leg-axis LOG3 admissible — same law, new clothes.

1. **[DONE] z-n1 radix-8 leaf** — `codelet_zil.ml`, `--z-native`, emitted
   `codelets/zil/avx2/radix8_z_n1_avx2.c`; GATE: bit-identical to the hand oracle (0.0),
   oracle vs naive 2.94e-15. IL op-selection rules 1–5 codified in the module header.
2. **z-n1 radices 4/16/32/64** — same rules; rule 3 (`FLIP+vaddsubpd` lone rotations)
   activates at R≥16; keep each body spill-free ≤16 ymm or split per the two-regime rule.
   GATE: per-radix vs naive; bit vs hand oracle where one exists.
3. **Count/tail contract** — the z loop is 2 complex/iter: state and assert `count % 2 == 0`
   (the UL twins' `me%4` precedent). All K=1 call sites use multiples of 4; the contract
   must still be explicit.
4. **z store lattice (corner-turn-in-stores) — RACED (2026-07-24): NEGATIVE, banked.**
   Built as `n1t` (`--z-n1t`, vperm2f128 pair-repack, full-width sectioned stores) + plain
   t2 pass 2; C-shapes (grid scratch + t2s) WON every cell (256: 164 vs 178/256 · 512: 359
   vs 443/572 · 1024: 972 vs ~1830 · 2048: 3796 vs 4346/5145; `zil_item4_race.c`).
   **LESSON: transposing the scratch MOVES striding to the LEG axis (stride R2, the
   degraded pattern) — it doesn't remove it. The grid layout + t2s (contiguous legs +
   cheap 2×128 columns) is already pass-2 consumption order on the axis that matters;
   MKL's corner-turn wins only into a fully consumption-ordered scratch. The 2×128 load
   "tax" is nearly free.** Same run: champion mid-N ratios 0.81–0.83 (256/512/1024) —
   the ladder's 0.61–0.78 was band. n1t kernels remain in-tree for future 3-pass shapes.
5. **z load lattice (transpose-in-loads)** — the UL analog, same lane ops on the load side.
6. **Twiddle strategy family — the IL analog of each split strategy, each a separate twin:**
   - **6a. z-T1/VTW2 (dense stream, DEFAULT)** — table emitted at generation: cos-first
     `[c,c,c',c'][-s,+s,-s',+s']`, 64 B/record, consumption-order single cursor (this IS
     the twl idea unified with dup+sign-folding); apply = BYTW2 (2 loads + FLIP + mul + fma,
     zero table-side shuffles). Per-128-bit-lane twiddles → K=1-admissible dense form.
   - **6b. z-T1 compact (residency-gated variant)** — plain `[c,s]` 16 B/pt + VZMUL
     (3 shuffles): half the table bytes, more port-5; wins only where the VTW2 stream
     spills past L2 (large N). Calibrator decides per cell (§3a residency gate).
   - **6c. z-T1S (broadcast)** — `set1(c)` + sign-alternated sin vector (set1+xor or
     emit-time VLIT), then the same FLIP+mul+fma core. **Admissible ONLY where both lanes
     share W** (batch contexts; NOT K=1 columns — geometry law). MKL's baked-immediate
     combine roots are this strategy's constant-folded extreme.
   - **6d. z-LOG3 (VTW3/t3-class)** — sparse base twiddles + VZMUL leg-axis derivation
     chains (FFTW t3fv proves transfer; ~7.75× table-byte cut at R=32). K=1-admissible.
   GATE: all four bit-or-tolerance vs a common reference; VTW2 fill verified against the
   MKL live-table dump (docs/research, cos-first Pythagorean check).
7. **z-t2 kernel, radix-8 first** — 6a apply + combine with `addsub` ±i triads + rule-4
   FMA folding; two-regime scheduling (unroll + spill-to-widen per §3a). GATE vs naive.
8. **Backward twins** — direction-specialized codelets: conjugate table fill (`+sin` fold)
   + swapped `vfmadd↔vfnmadd` + ±i sign — never a runtime branch, never text derivation.
9. **Registry + build wiring** — `vfft_z_n1_fn(R)` / `vfft_z_t2_fn(R,strategy)` resolvers;
   `codelets/zil/` already in build.py.
10. **Two-pass z→z K=1 composition at 512** — race the splits (16×32 per MKL, 64×8/8×64
    per our engine) × twiddle strategies vs MKL-IL 291 ns. GATE vs naive first.
11. **Calibrator + wisdom** — z-native routes as new IL-axis arms in calibrate_k1; kind-3
    il_route codes extended; median discipline as always.
12. **M4 retirement** — wire the K=1 engine's IL dispatch to the z family, DELETE the
    32 boundary-lattice twins (population 2), re-wire the il_execute.h sockets for the
    classic engine's z→z paths.
13. **Standing gates at every step** — public-API ladder green; master K=1 gate green;
    no step lands without its numbered gate above.
14. **Blocked (Tier-B) zil lowering — BUILT AND RACED (2026-07-24), verdict per-radix.**
    Blocked twins emitted (`radix{16,32,64}_z_n1b`, `radix{16,32}_z_t2b`): PASS 1a/1b
    half-DFTs with per-half loads parking to a function-scope packed `zspill[]` (z bonus:
    ONE array, not split's re/im pair), PASS 2 reload-on-demand + store-on-compute —
    the split family's structure carried detail-for-detail. Race (`zil_blocked_race.c`,
    best-of-7 rotated, K ∈ 16..4096, cross-checked + gated vs naive first):
    **r16 → BLOCKED (n1 −14…−19%, t2 −1…−11%) · r32 → BLOCKED (n1 −5…−22%, t2 −14…−33%
    across two runs) · r64 single-level → WORSE (+12…+40%; radix-32 halves too fat) ·
    r64 8×8 TWO-LEVEL (`--z-blocked2`, user-directed) → WINS −17…−31% at every K.**
    The 8×8 = eight spill-free radix-8 sub-DFTs parking to zspill[8i+j] + eight twiddled
    (const W64^(i·j), full class selection) radix-8 combines over strided slots — the
    inter-stage corner-turn is FREE in slot indexing (vs mono-64's 62 lane ops).
    **CONSTRUCTION LAW (measured): sub-bodies must be SPILL-FREE — cut every radix to
    radix-8 pieces.** ADOPTED: r4/r8 monolithic; r16/r32 blocked; r64 blocked2 (8×8).
    Follow-up: r32 as 4×8/8×4 CT (its current radix-16 halves spill ~9% — the law predicts
    a further win). Verdict radix-determined, not K-determined — static, no calibrator
    axis. Spill census (monolithic): r8 = 0 (129 insns), r16 ≈ 9%, r32 ≈ 14-15%,
    r64 ≈ 17%; doc-58's ~50% catastrophe was bodies-of-bodies (mono-256), not these.

15. **Blocked z executor v1 — RACED (2026-07-24): NEGATIVE RESULT, banked.** Built as
    designed (`t2d` post-twiddle outer + `t2ss` strided-store inner, `zil_4096_blocked.c`);
    gate 3.46e-15 PASS but **12190 ns vs flat 64×64's 8849 vs MKL 4087 (0.34× vs 0.46×)**.
    Post-mortem, three compounding causes the projection missed: (a) the shape is THREE
    full-array sweeps vs the flat's two — the extra 64 KB round-trip was unpriced; (b) the
    inner final stores are a SCATTER (stride-16-complex half-width stores across the whole
    output; MKL's recursion survives because its stores are sectioned-contiguous — store
    locality is the whole game at this size); (c) the outer pass's stride-4KB legs are the
    degraded strided-leaf pattern from the residency probe. LESSON: split-point recursion
    only pays when the final stores stay SECTIONED; this decomposition can't provide that
    without an extra reorder pass or scrambled output. 4096 effort REDIRECTED to: item 4
    corner-turn stores on the FLAT two-pass (remove its strided-load tax — the winning
    shape) + 6b compact twiddle table (its 126 KB stream). Blocked-v2 (sectioned-store
    variant) only if those two leave a measured structural gap.

16. **z staged cascade for N ≥ 2048 — the high-N answer (research 2026-07-24:
    docs/research/high_n_loss_analysis.md).** Root cause of the high-N loss: the flat
    two-pass FORCES ~N-wide strides in both passes (our pass 1 alone = 4148 ns at 4096 ≈
    MKL's entire 4087 ns transform); MKL switches families at exactly this boundary to a
    multi-stage contiguous-streaming cascade (census: ping-pong scratch, ~5 passes, reused
    small looped stage kernels, user memory touched once each way). Build: 3–4 radix-8/16
    z stages through ping-pong scratch, per-stage small VTW2 streams; v1 = a C driver
    composing the EXISTING gated kernels via their (Ls,Gs,OLs,OGs,count) params. Crossover
    two-pass↔cascade ≈ 1024/2048, per-cell calibrated. Explains both banked negatives
    (blocked-v1, n1t) under one principle: every pass must stream contiguously; the
    exchange amortizes across log-many stages.

## 6. ADOPTED CONSTRUCTION TABLE (measured 2026-07-24; the authority for every zil emission)

**Construction law: sub-bodies must be SPILL-FREE — cut every radix down to radix-8
pieces.** Radix-8 is the largest interleaved body that runs entirely in 16 ymm (0 spills,
129 insns); any sub-body that spills internally makes the pass structure pay parking on
top of its own spills (the single-level r64 failure). The inter-stage corner-turn in a
blocked body is FREE — absorbed into the zspill slot indexing, zero lane ops (contrast:
mono-64's in-register transpose = 62 `vinsertf128`/`vperm2f128`).

| radix | construction | flag | evidence (`zil_blocked_race.c`, best-of-7 rotated, K 16–4096) |
|---|---|---|---|
| r4, r8 | **monolithic** | `--z-native` | spill-free by nature; r8 = the gold bit-gated body |
| r16 | **blocked** (2× radix-8) | `--z-blocked` | n1 −14…−19%, t2 −1…−21% |
| r32 | **blocked** (2× radix-16) | `--z-blocked` | n1 −5…−22%, t2 −14…−33% (two runs) |
| r64 | **blocked2** (8×8 CT) | `--z-blocked2` | **−17…−31% vs monolithic at every K**; single-level halving was +12…+40% WORSE |

The 8×8 shape (mono-64's factorization at codelet scope): PASS 1 = eight spill-free
radix-8 sub-DFTs over the residue classes parking to `zspill[8i+j]`; PASS 2 = eight
groups reloading strided slots `{8i+j}`, constant `W64^{i·j}` twiddles (full class
selection), radix-8 combine, natural-order stores. Verdicts are radix-determined, not
K-determined → static per-radix defaults, no calibrator axis.

Follow-up predicted by the law (queued, unbuilt): r32 as a 4×8 CT should beat its current
radix-16 halves (which spill ~9% internally).

## 4. Open questions (resolve by measurement, not argument)
- VZMUL vs BYTW2 crossover (table bytes vs shuffles) — per-cell, likely BYTW2 everywhere
  like MKL/FFTW mid-N.
- Whether the leaf (no twiddles) prefers z-native or split-with-lattices at small R —
  the M1 race answers this.
- AVX-512: VL=4, `vpermilpd`-class dups scale; defer until AVX2 family is calibrated
  (same policy as everything else on this i9).
