# THE IL CODELET TREE — READ THIS FIRST

This is the orientation document for the interleaved-complex (IL) codelet family. If you are
about to pick a codelet out of `avx2/`, read §1 and §5 at minimum.

**The one-paragraph version.** There are **three methods** for computing a K=1 transform in IL:
**mono**, **Bailey**, and **cascade**, each owning a range of N. They are not variations on one
theme — they have different codelet vocabularies and different *data layouts*. On top of that,
the family has **two layout flavours**: **pure IL** (packed complex all the way through) and
**boundary-IL / split-interior** (interleaved at the edges, separate re/im planes inside).
Pure IL is right below N≈2048; the boundary-converting shape is right above it. Both fences are
measured, not chosen.

---

## 1. THE THREE METHODS

| method | N range | shape | layout | codelet kinds |
|---|---|---|---|---|
| **Mono** | ≤ 64 | ONE codelet, no passes, no scratch, twiddles baked into rodata at emit time | **pure IL** | `vfft_k1_mono64_*` (lives in `codelets/oop/`, not here) |
| **Bailey** (two-pass / four-step) | 128 – 4096 | two passes with the transpose fused into stage 1's stores | **pure IL** is the target | `n1t` (leaf) → `t2` (mid) |
| **Cascade** (Cooley–Tukey, staged) | ≥ 2048 | many stages through a scratch plane, ping-pong style | **boundary IL, split interior** | `s0t`/`s0s` → `msg`×k → `stf`/`sterm` |

Why the fences sit where they do — both were established by measurement and both survived
attempts to move them:

- **Mono dies at 64 for register reasons, not pass overhead.** At N=128 a mono measures 98–104 ns
  against our own two-pass at 86; at 256, mono 272 vs four-step 175. A mono has *zero* pass
  overhead and still loses — 8+ radix-16 bodies plus N-sized function-scope state on 16 ymm
  registers is a spill storm.
- **Bailey collapses at 2048** because the flat two-pass four-step forces ~N-wide strides in
  *both* passes. At 4096, pass 1 *alone* costs about as much as a whole competing transform.
  The cascade exists specifically to fix that, and it took 4096 from 0.46× to 0.81×.

---

## 2. THE CASCADE, IN DETAIL — THIS IS THE INTERESTING ONE

The cascade is not "a longer Bailey". It is a staged pipeline with **four distinct roles**, and
its scratch plane is **not interleaved**: it is 64-byte `[re ×4][im ×4]` blocks — a blocked
structure-of-arrays. That single fact is what makes the interior split.

### 2a. The roles

**① INGEST / THE TURN — `s0t`, `s0s`** (stage 0)
Reads the caller's **packed interleaved** input and writes **split planes** into the scratch.
Two variants:
- `s0s` — *"z-in → split-out leaf, twiddle-free, **DEINT loads**"*. The straightforward
  deinterleaving ingest.
- `s0t_r4` — *"ZTURN-S **fused-turn** ingest: natural z leg loads, twiddle-free radix-4, one
  64-B record per position at section `bitrev2(p mod 4)`, **4 rate-matched cursors**"*.
  This is the "turn": the corner-turn (four-step transpose) is **fused into the stores**, and
  the four output cursors are deliberately rate-matched so the store stream doesn't sweep
  addresses mod-4K and alias. Rate-matching is the design rule — sectioning is just how it is
  achieved.

**② MIDS — `msg`** (repeated, one call per stage)
*"GROUP-LOOPED split mid: one call/stage, in-kernel `bp`/`twg` bumps"*. Runs entirely on the
split planes. Its complex multiply is 4 real multiplies with **zero lane operations** — no
shuffle, no `permute_pd`. Twiddles arrive as per-group splat-pair sets, 8 doubles per leg
`[c×4][s×4]`.

**③ TERMINATOR — `stf`, `sterm`** (last stage)
Reads split planes, does the final butterfly, and **re-interleaves on the stores** so the caller
gets packed complex back. Two families:
- `sterm` — *"SPLIT-INPUT terminator: **TR4 loads**, packed w¹ squaring tree, **REINT drev-comb
  stores**"*. Pays a load-side TR4 transpose.
- `stf_r4` — *"ZTURN-S terminator: 4 section taps, 2 consecutive records = 128 B contiguous per
  tap, **NO load shuffles**, packed w¹ squaring tree, REINT drev-comb stores"*. The ZTURN
  restructure deleted the terminator's load-side TR4 by making the ingest store in the geometry
  the terminator wants to read. That is the whole point of ZTURN.

Both derive w²…w⁷ **in-register from w¹ by a squaring tree**, so the table only carries w¹ per
column: `[c(k..k+3)][s(k..k+3)]`.

**④ TWINS — `sterm2`, `stf2`**
*"2-quad unroll-and-jam terminator twin (SU-braided 2-instance DAG + baseline-shaped tail;
**bit-identical pair** with sterm, per-cell `t2q` pick)"*. Same math, different scheduling. Which
one wins turns on ±5% **placement luck**, so the pick (`t2q`) is **measured per cell at create
time and banked** — never hand-set. A create-time race times `stf` against `stf2`.

### 2b. Contract and scratch

All cascade codelets: **`count % 4 == 0`** (4 columns per iteration), scratch = 64-B
`[re ×4][im ×4]` blocks, `z` addressing `+4` for the imaginary half, one stream per leg row.

### 2c. Where the flat codelets fit

The flat family in this directory — `n1`, `n1t`, `t2` — is a *different* vocabulary:
- `n1` — solo leaf, natural order in and out, twiddle-free.
- `n1t` — Bailey stage-1 leaf, four-step **transpose fused into the stores**.
- `t2` — Bailey stage-2 mid, streamed VTW2 twiddles, BYTW2 apply.

Note `n1t` and `s0t` both "fuse a turn into the stores" — but `n1t` stays **packed** and feeds a
packed mid, while `s0t` **deinterleaves** into split planes. Same idea, opposite layout. Do not
substitute one for the other.

---

## 3. THE TWO LAYOUT FLAVOURS, AND THE FENCE

| flavour | what it means | where it is right |
|---|---|---|
| **Pure IL** | packed complex (re/im adjacent in the register) through **all** arithmetic; complex multiply = `cflip` + `mul` + `fma` | **N ≲ 2048** — mono and Bailey |
| **Boundary IL / split interior** | packed at the buffer edges; deinterleave once at ingest, compute on split planes, re-interleave once at the terminator | **N ≳ 4096** — the cascade |

**Why the split interior wins at large N.** A packed complex multiply must swap re/im within
each complex (`cflip` → a port-5 shuffle uop) on **every** multiply. A split complex multiply is
4 real multiplies with **no shuffle at all**. Packed is denser and more local; split frees the
shuffle port and drops the per-multiply op count. Below ~2048 locality wins; above it, the port
pressure does.

Our own instruction mix shows the shape plainly:

```
s0t_r4   deint=16  permute_pd=2   fmadd=0     <- ingest: deinterleaves
msg      deint=0   permute_pd=0   fmadd=6     <- mids: pure split cmul, ZERO lane ops
sterm    deint=48  permute_pd=0   fmadd=34    <- terminator: re-interleaves on store
```

**This was measured, not assumed.** A full-IL cascade interior was built and raced against the
split interior under **identical chains**, and lost: **6157 vs 5609 ns @4096 (−8.9%)** and
**−12.6% @16384**. Cause: our packed BYTW2 pays that `cflip` on every complex multiply.

🔴 **Do not propose a full-IL cascade interior again.** It has now been refuted twice by
independent measurement.

---

## 4. WHAT BENCHMARKING AGAINST MKL SHOWS

Racing the front door against MKL across the size range agrees with the internal races, and
that agreement is the strongest evidence we have that the tiering is structural rather than
tuning:

- Our **packed-IL** routes are competitive at small N — mono-64 lands at ~30 ns, at parity with
  MKL's interleaved path and ~1.2× over MKL split; Bailey at 256 wins 1.26× against MKL split.
- Our **boundary-converting cascade** is what closes the gap at large N: **2048 → 1.07×,
  4096 → 0.85×, 8192 → 0.89×, 16384 → 1.03×** (>1 means we beat MKL). We beat MKL at 2048 and
  16384; the residual 11–15% at 4096/8192 is a known architectural gap, not a missing kernel.
- **The crossover between the two flavours falls between N=2048 and N=4096** — the same place
  our own full-IL-interior race flipped sign, and the same place the mono→Bailey→cascade tier
  boundaries were independently measured to sit.

Two measurements from opposite directions landing on the same fence is why this is treated as
settled.

⚠️ The cascade chains currently banked in `oop_wisdom.txt` were re-promoted on a hot machine and
are marked **UNCONFIRMED** (a +15% drift was observed at 4096 on an identical-config repeat).
The parked predecessors are recorded in that file's comment block.

---

## 5. WHAT IS ACTUALLY IN `avx2/` — MIXED PROVENANCE

🔴 **This is ONE family in three HOSTING states — not three families.** `codelet_zil.ml` was the
original self-contained emitter used to develop the IL codelets; every idea in it was then baked
into the full DAG pipeline. `codelet_cil.ml` and `codelet_zsplit.ml` are that same family
**re-hosted on the production machinery** (`algsimp`, the SU scheduler, `regalloc`,
`emit_render`/`emit_c`, `Isa` parameterization). Do not read the provenance line as "a rival
implementation" — read it as **"ported yet?"**

Identify any file by its **first two lines**, which name the emitter.

| emitter | files | hosting state | layout |
|---|---|---|---|
| `codelet_cil.ml` | **152** | **PORTED.** Pipeline-hosted (`zil_pipeline_port.md §11`). All odd/prime radices + all pow2 **backward** | **pure IL** |
| `codelet_zil.ml` | **43** | **NOT YET PORTED.** Every pow2 **FORWARD** kernel, plus exotic kinds never productionised (`t2c`, `t2s`, `t2sp`, `t2spt`, `t2st`, `t2sq`, `t2sqt`, `t2ss`, `n1b2`) | pure IL (packed) |
| `codelet_zsplit.ml` | **25** | **PORTED** (tranche 1). The cascade: `s0t_r4`, `s0s`, `msg`, `ms`, `msz`, `stf_r4`, `stf2_r4`, `sterm`, `sterm2` | **boundary IL / split interior** |

Recognise un-ported (`codelet_zil.ml`) output on sight: `(k >> 1)` instead of `(k / 2)`, and
`in0`/`out0` naming instead of `z0`/`z1`.

**Why the port matters — and why it is NOT about i9 speed.** `zil` is a self-contained C-string
emitter with 486 literal `_mm256_` intrinsics and a hard `vec_width <> 4 -> failwith` gate. It
bypasses ~9.2K lines of shared machinery, so: **no AVX-512 / EPYC path**, every new kind is
another hand template, and pass improvements (FMA-lift, scheduler wisdom, regalloc widening)
**never reach it**. `zil_pipeline_port.md` §0 is explicit that arithmetic parity with the
pipeline is *already proven* — the port is for **reach and maintainability**.

**Why forward is un-ported but backward is not:** `zil` was **forward-only**, so the backward
kernels had to be born in `cil` (there was nothing to be bit-identical to). The pow2 forward
kernels are simply the unfinished tranche.

⇒ **The port gate is BIT-IDENTITY, not an A/B race.** Tranche 1 (cascade) established the
recipe: all 11 production kernels regenerated **bit-identical** to the legacy emission. Finishing
tranches 2 (bailey2: `n1t`/`t2`) and 3 (solo: `n1`) follows that recipe. The exotic kinds are
bench-only research residue and should be dropped with their spikes, not ported.

**The cascade files are not a mistake and must not be "fixed" into pure IL.** See §3.

---

## 6. THE RULE

- **N ≤ 2048 (mono, Bailey): PURE IL.** Use `codelet_cil.ml` kernels. Full IL beats the
  `il_in`/`il_out` hybrids **0.51–0.89×** when L1-resident (route-level 0.558@64, 0.765@256).
- **N ≥ 4096 (cascade): boundary conversion is right.** Keep the `codelet_zsplit.ml` shape.
- **NEVER build a new IL-boundary / split-interior *codelet*.** The forbidden shape is
  `il_in`/`il_out` in `codelets/oop/avx2/` — a codelet whose **ABI** mixes layouts and therefore
  converts **at every pass boundary** of a 2–3 pass route. The cascade converts **twice for the
  entire transform**, so the cost amortises across every stage. Same conversion, a fraction of
  the work to spread it over. That is why the hybrids measure slow and the cascade does not.
  **Test: a signature with `in_re` + `in_im` is HYBRID.** Every file here takes a single
  interleaved `zin`/`zout`, so none of them are.

**Three axes that keep getting conflated. They are not the same question:**
1. **layout through the arithmetic** — packed vs split planes;
2. **how often you convert** — once per transform vs once per codelet call. *This* is what
   separates the cascade (fine) from the `il_in`/`il_out` hybrids (slow);
3. **codelet ABI** — whether the signature exposes `in_re`+`in_im`. This is what "hybrid" means
   in this tree.

"The cascade goes split at high N" does **not** mean "keep the hybrids".

---

## 7. TRAPS

- 🔴 **Symbol ≠ filename.** `radix16_n1_oop_il_in_avx2.c` defines
  `radix16_n1_oop_fwd_avx2_UG_UG_il_in`. Grepping the basename returns **zero** references and
  makes a load-bearing file look deletable. **Always check by exported symbol.** (This broke the
  build on 2026-07-28.)
- 🔴 **No odd-COUNT tail exists.** Every kernel here loops `k += per` (per = 2 on AVX2; the
  cascade is `count % 4 == 0`) and silently drops trailing columns if `count` is not a multiple.
  `il2p_create` refuses odd R1/R2 up front, so nothing is wrong today — but a mixed split like
  **N=48 = 16×3** would hand `count = 3` to a pow2 leaf. Design decided (inline narrow VEX-128
  arm, mirroring the split family's existing peel in `emit_c.ml` §3995-4062), not yet built.
- **`sterm`/`sterm2` and `stf`/`stf2` are bit-identical pairs** whose winner turns on ±5%
  placement luck ⇒ the `t2q` pick is measured per cell, never hand-set.
- **Blocked kernels** (`n1b`, `t2b`, `t2b_log3`) carry their own per-pass edges and keep the
  even-count contract.
- **`log3` is T2-only** — the emitter refuses `--cil-log3` on `n1`/`n1t`.
- **Blocked tags the SYMBOL** with `b` (`radix25_z_n1b_fwd_avx2`), so blocked and flat variants
  can coexist in one binary.

---

## 8. REGENERATING

Regenerate **in place** only the `codelet_cil.ml`-provenance files, deriving the flag set
(kind / blocked / log3 / bwd / `--cil-split`) from each filename, so the file **set** never
drifts and the other two emitters' files are left alone. Never bare `dune build` — use an
absolute `--root` with `DUNE_CACHE=disabled`.
