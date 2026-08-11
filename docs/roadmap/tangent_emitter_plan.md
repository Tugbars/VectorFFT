# Tangent-interior emission through cx_math — implementation plan

Goal: emit the tangent-scaled butterfly construction (see
`docs/performance/tangent_scaled_butterflies.md`) through the real cx
pipeline (SR scheduler + render), replacing the hand-generated proof
kernels (`w16tg*/w32tg*`, scratchpad) with production codelets, then let
the dp re-race place them and sunset the classic interiors per slot
(pool-sunset policy).

Measured motivation (2026-08-11, all 5/5 paired, controls ≤0.2%): R16 mid
−25%, R16 leaf −17%, pure-tangent N=256 −20…−24%, pure-tangent N=512
−14…−17% (route (32,16): tg-leaf32 + tg-mid16, floors ~306 ns).

## Where the change lives

`generator/lib/cx_math.ml : butterfly_pair` is the ONE shared twiddle-class
selector for the pow2 DIT recursion (`dft_cx`). Its arms today:

- `k=0` — plain butterfly (keep)
- `4k=n` — ±i via `crot/crotp` (keep — already multiply-free)
- `8k=n`, `8k=3n` — √½ classes: `x = o ± rot(o)` then `cfma/cfnma √½ x e`
  — **already the tangent form** (shear + normalization fused into the
  butterfly). Keep unchanged.
- `else` — **the classic arm that loses**: `ctw c s ok` (full complex
  rotation) then naked `cadd/csub`. This is the only arm to change.

## The tangent general arm (v1)

For angle θ = 2πk/n, sign-aware (`rot` is already ±i by direction):

```ocaml
(* w = e^{sgn·iθ} = c·(1 + sgn·i·t), c = cos θ, t = tan θ.
   shear = ok + t·rot(ok)   (one FMA; |shear| = |ok|/|c| — no cancellation)
   out   = ek ± c·shear     (normalization fused into the butterfly FMAs)
   Signs of t and c ride in the OPCODE (cfma vs cfnma), same convention as
   cscale_chain — magnitudes only in constants. *)
let t = tan θ and c = cos θ in
let sh = if t >= 0. then cfma t (rot ok) ok else cfnma (-.t) (rot ok) ok in
if c >= 0. then cfma c sh ek, cfnma c sh ek
else            cfnma (-.c) sh ek, cfma (-.c) sh ek
```

Per site vs classic: {shuf, mul, fma, add, sub} (2×p01 + 3×p15) becomes
{rot, fma, fma, fma} (3×p01 + rot) — the naked adds are gone. This is
exactly the raced construction: the hand kernels used sign-folded constant
*vectors* where the IR uses sign-folded *opcodes*; same arithmetic.

Numerics: |t| ≤ tan(7π/16) ≈ 5.03 at R32 — gated at 3.4e-13 (mid) /
1.7e-13 (leaf) on the hand kernels. Add a loud `failwith` if |t| > 8
(first fires at R64's 15π/32 site; revisit with quarter-turn composition
`w = rot(w')`, `ek ± c'·rot(sh')` — `rot` commutes with the scalar so the
fusion survives at the cost of one extra rot node).

## Plumbing

1. An optional `variant` parameter (polymorphic variant, default Classic)
   on `butterfly_pair`/`dft_cx`/`dft_small`/`dft_chain` — default Classic
   ⇒ the 183-case byte-identity matrix is untouched by construction.
2. `gen_radix --cil-tangent` flag → codelet_cil kind wrappers pass the
   variant through. Orthogonal to kind (n1t/t2/mono/t2t) — ingest and I/O
   shells unchanged; the mid consumes the SAME VTW2 record stream.
3. Symbol/file tag `tg` (e.g. `radix16_z_n1tg_...`, `t2tg`) per the
   sunset-policy naming rule — grep must never conflate interiors.

## Radix coverage policy

- **R4: skip** — no general/√½ sites exist; tangent ≡ classic (vacuous).
- **R8: free** — π/4 sites only, already tangent-form in the shared arms;
  emitting the variant is a no-op diff at R8 except naming. Race anyway.
- **R16, R32: the targets** — leaf + mid, fwd + bwd, both ISAs.
- **R64: defer, re-priced by measurement** — the 4096-Bailey-vs-cascade
  hypothesis was probed at N=1024 (32×32, the first L1-exceeding size,
  buildable from existing kernels): the hand tangent route **inverts to
  +16% slower** (899.8 → 1003.9 ns, 5/35, ctrl 0.00%). See below — the
  loss is attributed to kernel SHAPE, not tangent arithmetic. R64 waits
  until the emitted tangent-BLOCKED forms re-race 1024; if the edge
  returns there, the 4096 question reopens (and the |t|>8 guard needs the
  quarter-turn composition first). Note the cascade's own stage kernels
  inherit the tangent arm for free through `dft_cx`, so at ≥2048 tangent
  likely lifts BOTH routes — the tier boundary re-races, never assumed.

## Blocking — the load-bearing lesson from the 1024 probe

The hand proof kernels bundle TWO properties: the tangent arithmetic AND a
monolithic, bit-rev-grouped-load structure co-designed for L1 residency.
At ≤512 (working set inside L1d) the bundle wins big; at 1024 (~80KB
working set) the non-monotone access order and unblocked structure pay
memory latency that the classic 4.4/4.8 blocked forms were raced to avoid
— and the route inverts to +16% slower. This is the same L1-conditionality
the construction's origin had (its source kernel was tier-scoped ≤512 for
the same reason).

**The emitter's job is to unbundle them**: tangent interior arithmetic
(the `butterfly_pair` arm) composed with the existing blocked structures
(4.4/4.8 splits, sequential-friendly access) — a combination neither hand
kernel tested. Concretely:

- R16: SR-scheduled monolithic tangent is proven register-fit; also emit
  the 4.4-blocked tangent variant and race per slot.
- R32: monolithic = 32 live legs. Try SR-monolithic first (check emitted
  spill counts vs the hand kernel's), but expect the blocked tangent
  (4.8-split with tangent interiors per block) to be the production form —
  it is also the form that can survive L1 exit.
- Tier scope until the blocked-tangent race: tangent kernels are
  ≤512-class. The 1024 cell and the Bailey/cascade boundary re-race only
  with blocked-tangent members in the pool.

## Gates and races (in order)

1. Flag OFF: 183-case emission matrix byte-identical (existing gate).
2. Flag ON: tolerance gate vs golden per direction (roundtrip forbidden as
   a gate per house rule), R16+R32 × leaf/mid × fwd/bwd.
3. Speed gate (gate_new_kernels_on_speed_too): emitted-tangent vs the
   hand-generated proof kernels (`w16tg`, `w32tgL`...) — does SR beat gcc's
   schedule of the mechanical SSA? Then vs classic per slot, paired
   protocol (ONE arena + 64B skews + ≥200 ms pace + control arm).
4. dp re-race across the wisdom grid (plans are pool-relative — the 512
   matrix proved factorization verdicts flip with the pool). `calibrate_*`
   front doors only; never hand-set plan slots.
5. Sunset per slot on stable verdicts: banner first, delete in the
   promoting commit (pool-sunset policy).

## Deliverables checklist

- [x] `cx_math.ml` variant arm + guard (2026-08-11: `tangent` ref, env
      `VFFT_CX_TANGENT` + `--cil-tangent` in gen_main; OFF byte-safe by
      construction, OFF-determinism verified byte-identical)
- [x] R16 MONO × t2/n1t fwd emitted + tolerance-gated (t2tg 7.8e-14,
      n1ttg 5.1e-14 — first build through the full pipeline)
- [ ] **BLOCKED forms**: the blocked builder makes its cross-rotations via
      its own `ctw` calls in codelet_cil.ml, NOT butterfly_pair — tangent
      does not fire there yet. Second patch site (found, not patched).
- [x] **v2 render folds (a)+(b)** (2026-08-11): CTwC c=1 peephole in
      cx_render (`fmadd(_s, cflip x, x)`, 2 uops, fires only on tangent
      shears — classic never builds c=1) + shear switched to
      `ctw 1.0 (sgn·t)` + √½ arms flag-gated to unit-cosine shears
      (conjugate shear + swapped opcodes for the 3n/8 class). Census:
      **143 FP / bound 47.7 / CP 79 / spills 4/4** — LOGIC 17→7 (= hand),
      now statically AHEAD of classic (151/50.3). Gates unchanged
      (7.8e-14 / 5.1e-14); OFF-determinism re-verified byte-identical.
- [ ] **v3 (optional, chases hand's 131/43.7)**: multi-level deferral —
      the hand kernel rides √½ scales through MORE than one butterfly
      level before absorbing (12-leg-footprint FMAs in the struct dump),
      merging normalizations; ours defers exactly one level. +addsub use.
      Price after the v2-vs-hand race says whether the last ~9% static
      gap matters on hardware.
- [ ] `tg` naming in emitter (currently sed-rename per house practice)
- [ ] bwd variants gated
- [ ] speed gates vs hand kernels + vs classic (DEFERRED: machine noisy
      2026-08-11 night — correctness/static only until it calms)
- [ ] R32 mono-vs-halves race (only if mono spills)
- [ ] dp re-race + wisdom rebank
- [ ] sunset banners on out-raced classic files
