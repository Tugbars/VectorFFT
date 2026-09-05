# The odd-N engine: the flat mixed-radix DIT

`src/core/oop/il_flatdit.h` · route `VFFT_K1_IL_FLAT` · wisdom `il_route=flat il_flat=… il_forms=…`

This is the reference for the K=1 interleaved c2c engine that serves N with
odd factors (any N whose factors lie in the radix pool below, to 2¹⁸). It
describes the transform, the memory layout, every kernel the engine calls
and its contract, every twiddle table and its exact contents, the backward
transform, the per-stage kernel-form race, the planner and wisdom
integration, the front-door dispatch, the reasons behind each choice, and
how the engine differs from the power-of-two cascade. It is written so
that the engine can be understood, extended and debugged from this file
alone.

---

## 1. What the engine computes and where it sits

The engine computes the unnormalized complex DFT of one interleaved
vector of N points (re, im adjacent), forward and backward, out of place
or in place, natural order in; natural order out, or — the SCRAMBLED
class — the mixed-radix digit reversal (§2.3):

    X[k] = Σ_n x[n] · e^(−2πi·nk/N)          (forward)
    x[n] = Σ_k X[k] · e^(+2πi·nk/N)          (backward, N·x on a roundtrip)

It is one of the K=1 IL tier's engines. The tier's planner races every
engine that can express a cell — the solo kernels, the two-stage pair,
the three-stage chain, and this one — and banks the winner on the cell's
kind-3 wisdom row. The flat DIT is enumerated for any N that is not a
power of two and that is either below 2048 or has no factor of 4 (an N
with a factor of 4 at or above 2048 belongs to the cascade). Bluestein
is used only for factors no chain can express.

The name: "flat" because every stage is a plain sweep over the whole
vector with one kernel form; "mixed-radix" because the stage radices are
any mix drawn from the pool; "DIT" because each stage twiddles its
inputs before its butterflies.

---

## 2. The transform, stage by stage

### 2.1 Factorization, runs, blocks, digits

Let N = R₀·R₁·…·R_{K−1}, K ≥ 2, every Rₛ from the pool

    { 9, 7, 5, 3, 25, 27, 21, 15, 13, 11, 8, 4, 16 }

(this is also the seed order the planner enumerates in; §9.1). Define

    Dₛ  = N / (R₀·R₁·…·Rₛ)      the RUN at stage s (D_{K−1} = 1)
    Lₛ  = Rₛ · Dₛ               the BLOCK SPAN at stage s (= D_{s−1})
    nblkₛ = N / Lₛ = R₀·…·R_{s−1}   the number of blocks at stage s
    Mₛ  = N / Dₛ = R₀·…·Rₛ      the TWIDDLE MODULUS at stage s

Stage 0 is the leaf. It views the input as R₀ legs of D₀ columns each,
leg l at offset l·D₀: input index n = l·D₀ + c. The leaf computes, for
every column c, the R₀-point DFT over the legs and writes digit q₀ where
leg l was. No twiddle is involved.

Every later stage s ≥ 1 views the staging plane as nblkₛ blocks of span
Lₛ, each block being Rₛ legs of Dₛ columns (leg l at offset l·Dₛ inside
the block). The block index b encodes the digits already produced,
q₀ … q_{s−1}, with q₀ most significant:

    b = q₀·(R₁·…·R_{s−1}) + q₁·(R₂·…·R_{s−1}) + … + q_{s−1}

The stage multiplies leg l of every column of block b by the twiddle
w_{Mₛ}^(l·Q_b), then runs the Rₛ-point DFT over the legs, and writes the
new digit qₛ where the leg was. Q_b is the block's natural-order value:

    Q_b = q₀ + q₁·R₀ + q₂·R₀R₁ + … + q_{s−1}·(R₀·…·R_{s−2})

i.e. the digits weighted by W_i = R₀·…·R_{i−1}. The helper
`_ilfd_block_Q(p, s, b)` decomposes b into its digits and returns Q_b.

### 2.2 Why those twiddles, derived once

Split n = l·D₀ + c and k = q₀ + R₀·k′:

    X[q₀ + R₀k′] = Σ_c w_{D₀}^(c·k′) · [ w_N^(c·q₀) · Y_c[q₀] ],
    Y_c[q₀] = Σ_l x[l·D₀ + c] · w_{R₀}^(l·q₀)

Y_c is the leaf. The factor w_N^(c·q₀) is the twiddle the next stages
owe; the D₀-point DFT over c for a fixed q₀ is the next level, applied
to block q₀. At that level c = l·D₁ + c′ and the owed factor splits as

    w_N^(c·q₀) = w_N^((l·D₁ + c′)·q₀) = w_{N/D₁}^(l·q₀) · w_N^(c′·q₀)

The first factor depends only on the leg l and the block (q₀): that is
the stage-1 pre-twiddle with modulus M₁ = N/D₁. The second is deferred
again. Repeating the argument, the pre-twiddle at stage s for leg l of
block b is w_{Mₛ}^(l·Q_b) with Q_b the accumulated digits, and nothing
is left owed after the last stage because D_{K−1} = 1.

### 2.3 Output order and the last stage

The output index is k = q₀ + R₀q₁ + R₀R₁q₂ + … . After the last stage a
block b holds, at leg position l, the bin with q_{K−1} = l, i.e. bin
Q_b + l·N/R_{K−1}. The engine never digit-reverses: the last stage
writes its results straight to their natural positions,

    zout[ natbase[b] + l·nstride ],   natbase[b] = Q_b,   nstride = N/R_{K−1}

Only the last stage writes zout. Every earlier stage works in place on
the staging plane.

**The scrambled class** (`p->scr = 1`). The same plan with the last
stage's redirection removed: it writes zout in the plane's own block
order, position b·R_{K−1} + l holding bin natbase[b] + l·nstride — the
mixed-radix digit reversal of the natural spectrum. No scatter and no
order table (the identity base table `ipb[K−1]` stands in for
`natbase`), so the forward is the natural plan minus its one strided
write pass; its inverse is the transposed pipeline (§6.2). Natural and
scrambled are two ORDER CLASSES of the same chain: the front door serves
each from its own wisdom cell (§9.3) and never compares them.

### 2.4 A worked example, N = 60 = 3·4·5

    R = (3, 4, 5);   D = (20, 5, 1);   L = (60, 20, 5);   nblk = (1, 3, 12);   M = (3, 12, 60)

- Stage 0 (leaf, radix 3): 3 legs at stride 20. Column c holds
  x[c], x[20+c], x[40+c]. The 3-point DFT over them lands in the plane at
  positions q₀·20 + c.
- Stage 1 (radix 4): 3 blocks (b = q₀) of span 20; each block is 4 legs
  at stride 5. Pre-twiddle for leg l of block q₀: w₁₂^(l·q₀). Result at
  block q₀, position q₁·5 + c′.
- Stage 2 (radix 5, last): 12 blocks (b = q₀·4 + q₁) of span 5; each
  block is 5 legs at stride 1, one column. Pre-twiddle for leg l:
  w₆₀^(l·(q₀ + 3q₁)). Output bin for leg l: q₀ + 3q₁ + 12·l, so
  natbase[b] = q₀ + 3q₁ and nstride = 12.

At stage 2 the 4 consecutive blocks with the same q₀ (q₁ = 0..3) have
natural bases q₀, q₀+3, q₀+6, q₀+9: bases step by W = 3, the weight of
q₁. That constant step is what the tail kernels exploit (§4.4).

---

## 3. Memory layout

- Input and output are interleaved z: `z[2n]` = re, `z[2n+1]` = im.
- The staging plane `p->stg` is 2N doubles, also interleaved, in the
  un-turned block layout of §2: block b of stage s starts at complex
  index b·Lₛ, its leg l at +l·Dₛ, its column c at +c. Because
  Lₛ = D_{s−1}, a block of stage s is exactly one leg of stage s−1.
- In place (zin == zout) is legal by construction: the leaf reads all of
  zin into the plane before any stage writes zout, and only the last
  stage writes zout.
- Per plan: the plane (16N bytes), the twiddle tables of §5, the natural
  base table `natbase` (nblk_{K−1} entries), the group order `gorder`
  (§4.5), and for in-place tail stages a base table `ipb[s]`. The
  planner's scratch (four planes of 2¹⁸ complex) grows on demand and is
  separate from any plan.

---

## 4. The kernels

The engine calls two codelet families. Both read and write interleaved
z. They differ in what a vector register holds inside the kernel.

**Pure-IL (packed) family** — `codelets/zil/avx2/pure_il/`, emitted by
`generator/lib/gen/c2c_il.ml`. A 256-bit register holds two complex
numbers as [re, im, re, im]. The complex multiply needs a lane swap per
product. Every kernel processes 2 columns per wide iteration and carries
the inline VEX-128 odd-count arm, so `count` may be any value ≥ 1.

**Boundary-split family** — `codelets/zil/avx2/boundary_split/`, emitted
by `generator/lib/gen/cascade_z.ml`. The kernel deinterleaves on load
into a re register and an im register ([re×4] and [im×4]: four columns
per iteration), runs the whole butterfly on split planes in registers —
a complex multiply is four real FMAs, no lane operation — and
reinterleaves on store. The split layout exists only inside the kernel;
memory stays interleaved. This is "IL at the boundary, split in the
body".

All kernels share the eleven-argument z ABI:

    f(zin, zin2, zout, zout2, tw_re, tw_im, Ls, Gs, OLs, OGs, count)

with per-kind meanings for the size and pointer slots given below.

### 4.1 Leaf: `radix{R₀}_z_n1c_{fwd,bwd}_avx2` (pure-IL)

Natural in, natural out, twiddle-free, alias-tolerant. Called once:

    lf(zin, 0, stg, 0, 0, 0, D₀, 0, D₀, 0, D₀)

Ls = OLs = leg stride D₀, count = D₀ columns. It reads the input legs at
stride D₀ and writes the digits at the same stride into the plane.

### 4.2 Mid stages, form t: `radix{Rₛ}_z_t2cp_fwd_avx2` / `t2c_bwd` (pure-IL)

The 2D tier's column-stage kernel with the twiddle applied BEFORE the
butterfly at forward (`--cil-t2cp`; the backward `t2c` kernel is
pre-twiddle by construction). One digit per call, called once per
block:

    f(blk, 0, out, 0, tw, 0, Dₛ, 0, Dₛ, 1, Dₛ)      blk = stg + 2·b·Lₛ

Ls = OLs = Dₛ (leg stride), OGs = 1 (one digit), count = Dₛ. `tw` is the
block's record set of §5.1. For s < K−1, out = blk (in place); for the
last stage the driver instead passes out = zout + 2·natbase[b] with
OLs = nstride, which is the natural scatter of §2.3 — but the last stage
is normally on a tail form (§4.4), where the run is 1.

### 4.3 Mid stages, form m: `radix{Rₛ}_z_msz_{fwd,bwd}_avx2` (boundary-split)

Emitted for radices 3, 5, 7, 9, 15 (`--zp-msz`, `--zp-mszb`). One call
per stage; the kernel loops over the blocks itself:

    fz(0, 0, stg, 0, tz, 0, Dₛ, nblkₛ, 0, 0, Dₛ)

Ls = count = Dₛ, Gs = the number of blocks. Inside, `bp` starts at zout
and advances by 2·Rₛ·Ls doubles per block, and the record cursor `twg`
advances by (Rₛ−1)·8 doubles per block. In place always (zin is
ignored).

Its body is the cascade's `msg` mid: per column group, the legs are
loaded as two z vectors each, deinterleaved with `unpacklo/unpackhi`
into [re×4] and [im×4], the split butterfly runs with the block's
broadcast records, and the results are reinterleaved with the same two
unpacks and stored. The lanes are left in the order the unpacks produce
([0,2,1,3]); no `permute4x64` is spent putting them in order, which is
legal because every record is broadcast (all four lanes equal), so lane
order never enters the arithmetic. The loop is

    size_t k = 0;
    for (; k + 4 <= count; k += 4) { … four columns, __m256d … }
    for (; k + 2 <= count; k += 2) { … two columns, __m128d  … }
    for (; k < count; ++k)         { … one column, scalar    … }

The two trailing arms are the SAME scheduled DAG rendered at the
narrower ISA widths (`Isa.sse2`, `Isa.scalar`), so the kernel takes any
count ≥ 1. At two lanes the unpack of two one-complex loads is already
the full deinterleave; at one lane re and im are loaded directly. The
twiddle records keep the wide width (`Emit_render.Cfg.tw_vw` pins
[c×4][s×4] even when a narrow arm reads two or one of the four equal
values).

### 4.4 Tail stages: `t2csg` and `t2csgn` (pure-IL)

A stage whose run Dₛ ≤ 4 (`VFFT_ILFD_TAIL_D`) is a tail stage. There the
column form of §4.2 would process one or two columns per block and
spend its time in prologues, so the tail kinds turn the geometry
sideways: the COLUMNS of a call are consecutive BLOCKS.

A **group** is G = R_{s−1} consecutive blocks — all values of the digit
q_{s−1} with every higher digit fixed. Their natural bases step by the
constant W = R₀·…·R_{s−2} (§2.4), and their twiddles factor as

    w_{Mₛ}^(l·Q_b) = w_{Mₛ}^(l·Q_hi) · w_{Mₛ}^(l·c·W),   b = (hi, c), c = q_{s−1}

so one broadcast record per group (T2 = w^(Q_hi)) and one small table
per column pair (T1[c] = w^(c·W)) let the kernel form W¹ = T1·T2 for a
column pair and derive the higher legs by squaring in registers. That
is the generated twiddle stream (`--cil-t2csg`, "gen2"): the driver
hands the kernel two tables of a few hundred doubles instead of a
per-block stream.

`radix{Rₛ}_z_t2csg_fwd_avx2` (and `_bwd`) processes one group per call:

    fcs(in, 0, out, 0, T1, T2_g, Dₛ, Lₛ, OLs, OGs, G)

Ls = Dₛ (leg stride), Gs = Lₛ (the stride between the columns, i.e.
between consecutive blocks), count = G. Column c of the call is block
(g·G + c). For an in-place tail stage out = in, OLs = Dₛ, OGs = Lₛ. For
the last stage out = zout + 2·natbase[g·G], OLs = nstride, OGs = W: the
natural scatter, with the G columns landing W apart.

`radix{Rₛ}_z_t2csgn_{fwd,bwd}_avx2` is the same kernel body emitted as a
`static always_inline` function plus an exported wrapper that runs the
GROUP LOOP in-kernel (`--cil-t2csgn`), one call per stage:

    fgl(stg, (const double*)base, out, (double*)order, T1, T2, Dₛ, ngrp, OLs, OGs, G)

- `zin_unused` carries the base table: `base[g·G]` is the output base of
  group g in complex units (for the last stage `natbase`; for an
  in-place tail stage the identity table `ipb[s]`, base[g·G] = g·G·Lₛ).
- `zout_unused` carries an optional group ORDER (size_t per group): when
  non-null the wrapper visits the groups in that order (§4.5).
- Gs is the group count; the wrapper derives the body's column stride
  as Rₛ·Ls.
- T2 for group g is `tw_im + 8·g`.

The wrapper is

    for (gq = 0; gq < Gs; gq++) {
        g = order ? order[gq] : gq;
        for (c = 0; c < Ls; c++)
            body(zin + 2·(g·G·R·Ls + c), 0, zout + 2·(base[g·G] + c), 0,
                 tw_re, tw_im + 8·g, Ls, R·Ls, OLs, OGs, G);
    }

### 4.5 The last stage's group order

The last stage is a scatter: every complex it writes lands in a different
cache line, because consecutive blocks differ in a high digit. Written in
block order, every output line is write-allocated once per complex it
receives. The driver therefore builds `gorder`: the groups sorted by
ascending natural base (a counting sort over natbase[g·G], which are
distinct and below N). Walking the groups in that order, consecutive
groups write adjacent output positions, so each output line is
allocated once and filled while it is hot; the input side becomes the
scattered one, which costs little because a group's input is contiguous
(G·Rₛ complex) and reads carry no ownership traffic. This is the form
letter `o` (§7); the same wrapper in block order is `n`.

### 4.6 Where each kernel form applies

    stage        run Dₛ     forms available            default
    leaf (s=0)   D₀         n1c                        —
    mid          Dₛ > 4     t2cp (t) | msz (m)         t2cp
    tail, s<K−1  Dₛ ≤ 4     t2csg (t) | t2csgn (n) | msz (m)   t2csgn
    last         D=1        t2csg (t) | t2csgn (n) | t2csgn ordered (o)   o

"Default" is what a plan is built with before the forms are raced; the
planner replaces it with the measured pick (§7).

---

## 5. Twiddle tables, exactly

All angles are a = −2π·(e mod Mₛ)/Mₛ with e the exponent named below;
c = cos a, s = sin a. Every table has a backward twin with the sign of
s reversed (§6). Modulus L in the code is Mₛ = N/Dₛ.

### 5.1 t2cp / t2c records (per block, sign-folded)

For block b and leg l ≥ 1, one record of 8 doubles at
`tf[s] + (b·(Rₛ−1) + (l−1))·8`, exponent e = l·Q_b:

    [ c, c, c, c, −s, +s, −s, +s ]

The sign folding matches the packed complex multiply
(re·c − im·s, re·s + im·c) done with an add-sub on a lane-swapped copy.
Backward: `[c,c,c,c, +s,−s,+s,−s]`.

### 5.2 msz records (per block, splat-pair)

Same indexing into `tz[s]`, plain sine:

    [ c, c, c, c,  s,  s,  s,  s ]

The split body multiplies re·c − im·s and re·s + im·c directly.
Backward `tzb[s]`: `[c,c,c,c, −s,−s,−s,−s]`.

### 5.3 t2csg / t2csgn tables (per stage)

T1 (`tf[s]`), one record per column PAIR pp, lanes j = 0, 1 for columns
2pp + j, exponent e = W·(2pp + j):

    [ c₀, c₀, c₁, c₁, −s₀, +s₀, −s₁, +s₁ ]

T2 (`t2g[s]`), one broadcast record per group g, exponent e = Q_hi of
the group's first block:

    [ c, c, c, c, −s, +s, −s, +s ]

The kernel forms the per-pair W¹ = T1·T2 in registers and the higher
legs by squaring. Backward twins `tfb[s]`, `t2gb[s]` with the sines
negated.

### 5.4 Sizes

- t2cp / msz: nblkₛ·(Rₛ−1)·8 doubles per direction — largest at small
  runs (a run-3 stage of a 10⁵ point transform carries a table of the
  order of the data).
- t2csg / t2csgn: ceil(G/2)·8 + ngrp·8 doubles per direction.
- natbase: N/R_{K−1} entries; gorder: N/(R_{K−1}·R_{K−2}) entries.

---

## 6. The backward transform: two pipelines

### 6.1 The natural class: the conjugate pipeline

The natural class's inverse is NOT the transpose of the forward pipeline
run in reverse (the natural scatter would have to be un-scattered
first). It is the same pipeline with every stage conjugated:

    IDFT(x) = conj( DFT( conj(x) ) )

and since each stage is (pre-twiddle T, then DFT block B), its conjugate
is (conj-twiddle, then IDFT block). So the backward runs the SAME stage
order — leaf, mids, tails, the same natural scatter with the same
`natbase`, `gorder`, the same forms — with backward kernels and the
conjugated tables:

    leaf      n1c_bwd                     (IDFT block)
    t         t2c_bwd                     (pre-twiddle conj + IDFT — the 2D tier's backward column kernel)
    m         msz_bwd  ("mszb")           (cascade_z kind {msz with bwd; dif=true}: (DIF, Bwd) places the twiddle PRE)
    n, o      t2csg_bwd, t2csgn_bwd       (the T2 backward with the pre-twiddle placement forced)

No conjugation happens in any kernel; the tables carry it. The
direction picks the kernel set and the tables when the stage's call
record is bound (§8); everything else is shared. A plan whose tail stage
has no backward twin (the plain `t2cs` kind) sets `bwd_ok = 0` and is
refused by the front door, which serves both directions from one
handle.

### 6.2 The scrambled class: the transposed pipeline

A scrambled forward F = S_{K−1} ··· S_1 · L (leaf L, stages S_s, no
final scatter) has the inverse N·F⁻¹ = F^H = L^H · S_1^H ··· S_{K−1}^H:
the stages run in REVERSE order, each replaced by its Hermitian
transpose, and since a stage is (pre-twiddle T_s, then DFT block B_s),
its transpose is (IDFT block, then conj-twiddle POST). The last stage's
transpose consumes the comb first, reading zin in block order into the
plane; the mids and tails follow in reverse order in place; the leaf's
transpose writes natural zout last. The kernels are the transposed
twins, tagged "t" where the natural backward already had the name:

    t         t2cp_bwd                     (IDFT block, POST-twiddle conj: t2cp's placement toggle at bwd)
    m         mszt_bwd  ("mszt")           (cascade_z kind {msz with bwd}: (DIT, Bwd) places the twiddle POST)
    n         t2csgt_bwd, t2csgnt_bwd      (the tail kinds transposed: no forced pre-twiddle placement)
    leaf      n1c_bwd                      (the leaf's transpose is its conjugate)

The conjugated tables are shared with the conjugate pipeline
(post-multiplying by conj(w) reads the same records). `scr_ok` = every
stage has its transposed twin; a scrambled plan without it is refused.
The letter `o` (natural-base group order) has no meaning in this class:
`vfft_ilfd_create_scr_of` builds the class from a natural token reading
`o` as `n`, and a `t` last stage writes block order as well.

---

## 7. Kernel forms: measured per stage, never ruled

Every stage keeps ALL its forms' tables and kernel pointers in the plan;
two integers name the form: `msz[s]` (1 = form m) and `gl[s]` (1 = the
in-kernel group loop), plus `gord` for the last stage's order and `scr`
for the class. The plan never rebuilds to change form: the fields are
re-bound into the stage's call record (§8).

`vfft_ilfd_race_forms(p, zin, zout, now_ns)` races them on real data in
pipeline order: it executes the leaf, then for each stage times the
candidate forms on that stage's actual input (the previous stages'
output), keeps the fastest, executes the stage on the chosen form, and
moves on. The tail forms are raced first (7 rounds, t2csg vs t2csgn vs
ordered t2csgn on the last stage) with msz off, then msz against the
best of them (3 rounds). Real inputs matter: the run-3 and run-4 stages
that decide between the forms behave differently on random data and on
a transformed vector.

The verdict is a token, one letter per stage s ≥ 1, joined by dots:

    t   the default column form (t2cp, or t2csg on a tail stage)
    m   msz
    n   t2csgn, block order
    o   t2csgn, natural-base order (last stage only)

`vfft_ilfd_forms_str` writes it, `vfft_ilfd_apply_forms` applies it and
refuses the whole token if any letter names a form the stage cannot
serve (msz needs its records, n/o need the group-loop kernel, o needs
the last stage). The token is what wisdom stores.

Observed on this box, for orientation only: msz wins the run-4 stages by
about two to one (a full four-column iteration against the half-width
column form), sits at parity with t2cp on long runs of radix 5 and 7,
and loses at radix 9 and on runs of 3; the ordered group loop takes the
count-1 last stage from roughly 1.7 to 0.9 ns per point at 10⁵ points.
None of this is a rule in the code — the race decides every cell.

---

## 8. Execution: the bound call lists

Execution does no planning work. At bind time (`vfft_ilfd_bind` — run
by create, by the forms token, by the form race and by the scrambled
builder: every writer of a form field) the plan derives, from the form
fields and the stage geometry, ONE call record per stage for each of
its three pipelines:

    cf[s]   the forward                       (leaf, stages 1..K−1)
    cb[s]   the conjugate backward            (natural class, same order)
    ct[i]   the transposed backward           (scrambled class: stages K−1..1, then the leaf)

A record (`vfft_ilfd_call_t`) holds the kernel pointer, which buffer
each of the two z arguments is (the plane, zin, zout, or none), the
table pointers and the five strides/counts of the z ABI — everything the
call needs except the caller's zin/zout addresses. The executor
(`_ilfd_run`) walks the list and calls; per stage that is two indexed
loads and one indirect call. No division, no digit-weight product, no
form or direction branch and no resolver runs at execute time; the front
door's dispatch (`vfft_execute` → route `VFFT_K1_IL_FLAT`) is the only
code between the API and the records.

Three record shapes exist:

- **ONE** — one kernel call for the whole stage. The leaf; msz (its own
  block loop, Gs = blocks); t2csgn (its own group loop, Gs = groups);
  and t2cp in place or on the scrambled last stage: the t2cp kernel
  carries an outer loop over OGs blocks with Gs as the block pitch and
  its record pointer advancing R−1 records per block, so a mid stage of
  nb blocks is ONE call with OGs = nb, Gs = R·D. The per-block driver
  loop this replaced paid a call and a prologue per block; the same-run
  A/B (`benches/bind_ab.c`, spectra bitwise identical) put it at
  1.01–1.05× the bound list (4095 1.036, 6561 t.t.t 1.035–1.042, 98415
  1.049, 19683/59049 1.011), while the per-stage arithmetic itself was
  within noise (0.99–1.06).
- **COL** — the t2csg per-column tail form (letter t on a tail stage):
  ngrp × D calls, the group's T2 record advancing per group, the natural
  last stage redirected through `natbase`.
- **BLK** — t2cp per block on a natural last stage without a tail form
  (no pool radix lacks one; kept for completeness).

`vfft_ilfd_stage(p, s, …)` binds stage s from the CURRENT fields into a
local record and runs it: the form race and the probes flip a field and
time a stage without rebinding the plan, and rebind when done. The
served path never rebinds.

---

## 9. Planner, wisdom, front door

### 9.1 Candidates

`_il_dp_enumerate_flat` (dp_planner_il.h) walks the ordered
compositions of N over the pool in the pool's order (so the greedy seed
chain — 9 first, then 7, 5, 3, 25, 27, … — is the first candidate),
depth 2 to `VFFT_ILFD_MAX_K` (10), capped at 24 per cell. The cap is
LOGGED when it bites (`flat chain pool capped at 24 (n more compositions
not raced)`); it bites hard on smooth N. Each candidate is a chain only;
its forms are raced when it is benched. Kernel availability, run
contracts and the existence of the inverse are validated by
`vfft_ilfd_create_chain` at build — a chain that cannot build or cannot
invert is dropped, never patched.

### 9.2 Measurement

The planner's bench builds the candidate, runs the per-stage form race
on the planner's own data and clock, writes the resulting token into the
candidate, and times the forward transform. Every candidate is gated
before timing against an independent reference: the mixed-radix scalar
DIT in long double (`_il_dp_ref_dft_mixed`, O(N·Σp) — it shares no code,
table or plan with the engines), itself spot-checked against direct
sums. The tolerance is 10⁻¹² relative to the reference's scale.

### 9.3 Banking and replay: one cell per order

The winner banks its route, chain and token on the kind-3 IL row of the
cell's ORDER. Natural and scrambled are separate cells, keyed `ord=nat`
and `ord=scr`, each with its own race and its own verdict; the two are
never compared and neither is derived from the other:

    @cell t=c2c n=19683 q=1 ord=nat place=oop role=comp lay=il |
        eng=k1 il_route=flat il_flat=9.9.9.9.3 il_forms=t.t.n.o il_kv=0 | ran=1 ns=… src=race
    @cell t=c2c n=19683 q=1 ord=scr place=oop role=comp lay=il |
        eng=k1 il_route=flat il_flat=9.9.3.9.9 il_forms=t.t.m.n il_kv=0 | ran=1 ns=… src=race

The scrambled pool, at every cell the K=1 IL tier races (below 2048, or
any N without a factor of 4), is every natural engine (the solo kernels,
the pair, the chain — a scrambled request that only a natural-writing
engine wins is served natural, and the row says so) plus, at a
non-power-of-two, the flat chains in their scrambled class; a `chain3`
or pair verdict on an ord=scr row means a natural-writing engine won
that cell's scrambled race. The in-place mode row of a SCRAMBLED
request signposts that ord=scr row (`ref=cell(…,ord=scr,place=oop,
role=comp,lay=il)`); `vfft_ilp_front_gate` holds the power-of-two cells
to it (own verdict, replay without a race). `il_flat` is a
structural field; `il_forms` is a local one (kernel placement is
machine-tied and re-raced on a host mismatch, like every kv). A create
with a banked flat row builds the chain and applies the token —
`vfft_ilfd_create_scr_of` for an ord=scr row — there is no default flat
build anywhere: the planner is the only source of a flat plan. Replay is
bit-identical to the raced plan (the gate checks this by comparing
spectra).

### 9.4 Front door

- OOP create (`c2c_oop_create.h`): the IL axis reads the request's order
  cell — `vw2_oop_lookup_k1` (ord=nat) for DEFAULT and NATURAL,
  `vw2_oop_lookup_k1_scr` (ord=scr) for SCRAMBLED. A banked
  `il_route=flat` row replays into `hk->k1ilfd` (an ord=scr row through
  `vfft_ilfd_create_scr_of`); a miss below 2048 or at any N without a
  factor of 4 (up to `VFFT_K1_IL_PLAN_ODD_MAX_N` = 2¹⁸) runs the plan
  race for that order first.
- In-place create (`c2c_ip_create.h` → `k1_commit.h`): the same order
  cell; `_k1_il_candidate` returns the flat plan of the request's class,
  raced against the cascade where one exists and attached as the cell's
  ILP engine.
- Execute (`vfft_execute.h`): route `VFFT_K1_IL_FLAT` dispatches
  `vfft_ilfd_execute_fwd/bwd`; in place with zin == zout.
- The plan is engine-pure (own staging plane, no pool), so the
  transform-contiguous batch tier may clone it per worker.
- `benches/flatdit_gate.c` (cold wisdom dir) is the machine proof: a
  natural pass and a scrambled pass — forward against sampled bins (the
  scrambled pass through the digit-reversal map), roundtrips in both
  placements, bitwise replay, `il_route=flat` asserted above 27³ where
  no other native route exists, and the ord=scr rows read back.

---

## 10. The insights behind the design

**Un-turned, natural throughout.** The transform never transposes and
never digit-reverses. Each stage is a contiguous sweep in place on one
interleaved plane; the output order is produced by addressing at the
very last stage. There is no corner turn, no sectioned scratch and no
reorder pass, so any radix mix is legal — the odd factors the pool
carries are ordinary stages.

**One record set per block.** Because the DIT owes exactly one twiddle
factor per (leg, block) at each stage (§2.2), a stage's twiddles are
broadcast records, not per-lane streams. That is what lets the split
body work with unordered lanes and lets the tail kernels generate their
stream from two tiny tables.

**The split body pays only where the packed form is starved.** On long
runs both bodies stream at the same rate; the shuffle count of the
packed complex multiply is not the limiter. The split body wins the
run-4 stages because it always runs a full four-column iteration where
the packed tail form runs half-width. Hence forms are raced per stage;
a single body would be wrong somewhere.

**The last stage is the cost, and it is an ownership problem.** Its
work is trivial; its writes are a scatter. Moving the group loop
in-kernel removes the per-group call and prologue, but the decisive
change is the ORDER of the groups: visiting them by ascending natural
base turns one write-allocate per complex into one per line. The
engine keeps the input contiguous and lets the input side absorb the
scatter, because reads are cheap and writes are not.

**Tails process blocks as columns.** A stage whose run is 1 to 4 has
nothing to vectorize across within a block; across blocks it has G
columns with a constant base step and factorizable twiddles. Turning the
call sideways is what makes short runs vectorizable at all.

**Any count, everywhere.** The odd-count arms in both families (the
pure-IL kernels at width 1, the split kernel at widths 2 and 1) mean the
runs need no padding and no divisibility beyond N's own factorization.

**Nothing is a rule.** Chain, stage forms and order are measured per
cell by the planner and stored; the engine has no heuristic path to a
plan.

---

## 11. How this differs from the power-of-two cascade

The cascade (`oop/zturn.h`, the ZTURN-S route for N with a factor of 4
at or above 2048) and the flat DIT are both multi-stage
Cooley–Tukey with split-body mid kernels, and both keep memory
interleaved at the API. Everything else differs:

| | CT cascade (zturn) | flat mixed-radix DIT |
|---|---|---|
| radices | chain[0] = 4 fixed, mids 4/8 (odd mids 3..15 for 2ᵃ·odd), last 4 or 8 | any mix from the pool, first to last |
| scratch | a SECTIONED split plane: 64-byte [re×4][im×4] records in four sections | one interleaved plane in natural block layout |
| ingest | radix-4 leaf with the corner turn fused into its stores | plain leaf, no turn |
| mids | `msg`: split planes in memory, per-lane splat-pair tables per group | `t2cp` (packed) or `msz` (split in registers only), one broadcast record set per block, raced per stage |
| terminator | section taps, no load shuffles, reinterleaving comb stores | tail kinds over block groups with a generated twiddle stream and the in-kernel group loop |
| output order | SCRAMBLED (digit-reversed) by default; natural through the `stfn` variants | natural by the last stage's redirected stores; the SCRAMBLED class (digit-reversed) by the same stage in block order — its own wisdom cell |
| twiddle streams | per-lane tables tiled by section | per-block broadcast records; two small tables at the tails |
| backward | transposed pipeline (terminator first, un-turn last) | natural class: conjugate pipeline, same order; scrambled class: transposed pipeline, reverse order |
| where lanes are ordered | the plane holds ordered lanes; the boundary kernels pay two permutes per leg per four columns | lanes stay in unpack order inside msz; nothing else cares |

The cascade buys contiguous terminator loads by paying for the turn at
ingest; the flat DIT buys freedom of radix and natural order by paying
the scatter at the end and then arranging that scatter to be cheap.

---

## 12. Limits and extension points

- **Radix pool.** A new radix needs the pure-IL kinds n1c, t2cp/t2c
  (both directions), t2csg and t2csgn (both directions) — all emitted
  from the registry — and, to be raceable as form m, the split kinds
  msz and mszb. Add it to the pool in `vfft_ilfd_default_chain` and in
  `_il_dp_flat_rec`.
- **Prime factors outside the pool** are not the engine's business; the
  planner offers no flat chain and the cell falls to the prime engine.
- **Stage count** is bounded by `VFFT_ILFD_MAX_K` (10).
- **Batches and threads.** The plan is single-transform; K>1 rides the
  transform-contiguous tier by cloning. Intra-transform threading and
  cache banding are open.
- **Memory.** The per-block record tables of small-run mid stages scale
  like the data; a generated stream for those stages (as the tails
  already have) would remove them.
- **The candidate cap** (24 compositions) is a logged budget, not a
  search; smooth N leave many orderings unraced.
- **ISA.** Everything is AVX2. The pure-IL family emits AVX-512 from the
  same source; the split family's edges assume four columns.

---

## 13. File map

    src/core/oop/il_flatdit.h                 the engine: plan, create, tables, bind + the executor, race_forms, forms token, the scrambled builder
    src/core/oop/il2p.h                       resolvers: t2cp(_bwd), t2csg(_bwd, t_bwd), t2csgn(_bwd, t_bwd), msz(_bwd, t_bwd), n1c, t2c
    src/core/planning/dp_planner_il.h         flat candidates, bench-time form race, mixed-radix reference, banking
    src/core/oop/k1_commit.h                  plan-race admission (odd N to 2^18), the replay for in-place, the request's order cell
    src/core/oop/c2c_oop_create.h             OOP replay of a banked flat row from the request's order cell
    src/core/oop/c2c_ip_create.h              in-place candidate and attach
    src/core/vfft_execute.h                   dispatch both placements, destroy
    src/core/wisdom2/wisdom2_oop_reader.h     il_flat= / il_forms= parse and write; the ord=nat / ord=scr lookups
    codelets/zil/avx2/pure_il/*_t2cp_*, *_t2csg_*, *_t2csgn_*, *_n1c_*, *_t2c_*  (each with its _bwd, the tails with _t_bwd)
    codelets/zil/avx2/boundary_split/*_msz_*, *_msz_bwd_*, *_mszt_bwd_*
    generator/lib/gen/c2c_il.ml               t2cp / t2cs / t2csg / t2csgn emission, the group-loop wrapper
    generator/lib/gen/cascade_z.ml            msz / mszb: interleaved edges, unordered lanes, the narrow arms
    build_tuned/benches/flatdit_gate.c        the front-door gate
    build_tuned/benches/ilfd_probe.c          the standalone probe (chains raced, vs MKL)
    build_tuned/benches/msz_probe.c           the per-stage form A/B
    build_tuned/benches/bind_ab.c             the executor's identity check + same-run A/B (bound list vs per-stage vs per-block)
