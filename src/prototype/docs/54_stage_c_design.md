# 54. Stage C Design: Sorensen-style r2c Cascade via Hermitian-Packed Codelets

## Goal

Close the 25–34% op-count gap to the Sorensen bound (doc 53) for r2c
at N ≥ 128 by porting FFTW's `gen_hc2hc` / `gen_hc2c` architecture into
our math layer.

## Why doc 52's "wire it up" plan didn't work

Doc 52 proposed a 2-stage cascade as `r2c_first × M + t1_dit × M +
butterfly`. Working through the math (see session 2026-05-12 trace, also
the analysis below) reveals it's structurally invalid for N > 2R.

`r2c_first` reads 2R *consecutive* reals and produces a plain DFT-R of
the pair-packed complex values. For Z[m] = ∑_n z[n]·W_N^(mn), substituting
n = R·g + a gives:

```
Z[m] = ∑_g exp(−2πi·m·g/M) · [∑_a z[Rg + a] · exp(−2πi·m·a/N)]
```

The inner sum's kernel `exp(−2πi·m·a/N)` depends on the full output
index m (not just its low bits). A t1_dit second stage can't decompose
this into a per-(m1, m2) twiddle — the kernel doesn't factor.

Concretely at N=8, factoring 4 = 2×2, the cascade only correctly
computes Z[0] and Z[2] (even k). Odd-k bins require additional
first-stage evaluations with pre-twiddled inputs. **The trivial
cascade test at N=16 passed only because N/2 = R, so M = 1 — there's
no inner stage at all.**

The fix is what FFTW does: keep data in **Hermitian-packed** form
throughout the cascade and use hc2hc-style codelets that exploit the
symmetry at every stage.

## FFTW algorithm summary

Three codelet types compose into a full r2c cascade for N = N₁·N₂·…·N_k:

| Codelet | FFTW source | Role |
|---|---|---|
| `rdft` (size N_k) | gen_r2cf.ml | First stage: N_k reals → N_k/2+1 unique complex (Hermitian-compact) |
| `hc2hc` (size N_i, middle) | gen_hc2hc.ml | Middle stage: Hermitian-packed in → twiddle → DFT-N_i → symmetry transform → Hermitian-packed out, in-place |
| `hc2c` (size N_1, last) | gen_hc2c.ml | Last stage: Hermitian-packed in → twiddle → DFT-N_1 → unpack to natural-order complex |

### Key insight from trig.ml

```ocaml
(* DFT of real input *)
let rdft sign n input = Fft.dft sign n (Complex.real @@ input)

(* DFT of hermitian input *)
let hdft sign n input = Fft.dft sign n (Complex.hermitian n input)
```

**FFTW doesn't write a separate real DFT or Hermitian DFT.** It runs the
standard c2c DFT on inputs that satisfy specific constraints
(im(input) = 0 for rdft; conjugate-symmetric inputs for hdft). algsimp
folds the zero/conjugate redundancy at math-layer time. The "real
DFT" is c2c with structural simplification.

Same trick for outputs — the Hermitian-packed *storage* is what makes
the cascade efficient, not the DFT computation itself.

## Hermitian-packed storage convention

For N complex values that are Hermitian-symmetric (X[n-k] = conj(X[k])),
only N/2+1 unique complex values exist (and X[0], X[N/2] are real).

### Middle stages (hc2hc): `(cr, ci)` arrays of length N

Storage: position i (0 ≤ i ≤ N/2) stores X[i], position N-i stores
conj(X[i]). This is "redundant" storage from a memory-saving standpoint
but matches the access pattern of an in-place butterfly. Both arrays
hold N reals total (= N/2+1 complex × 2 minus DC/Nyquist doubles).

The codelet operates on positions 0..N-1 as if they were full complex,
but the values at i > N/2 are conjugates by construction.

### Last stage (hc2c): split pointers `(Rp, Ip, Rm, Im)`

- `Rp[i]`, `Ip[i]` for i ∈ [0, N/2]: real and imag of the "positive" half (X[i])
- `Rm[i]`, `Im[i]` for i ∈ [0, N/2]: real and imag at the "negative" mirror position (X[N-i])

The two ranges are pointer-arithmetic siblings. From gen_hc2c.ml:

```ocaml
let locri i = if i mod 2 == 0 then locr (i/2) else loci ((i-1)/2)
and locpm i = if i < n - i then locp i else locm (n-1-i)
```

- `locri` (load): interleaves R0/R1 → even/odd indices from two streams (the Hermitian-packed input pattern from the previous hc2hc stage).
- `locpm` (store): positions < N/2 go to `(Rp, Ip)`, positions ≥ N/2 go to `(Rm, Im)`. This is the natural-order complex output.

## sym1, sym2, sym — the symmetry transforms

From gen_hc2hc.ml:

```ocaml
let byi = Complex.times Complex.i
let byui = Complex.times (Complex.uminus Complex.i)

let sym1 n f i = Complex.plus [Complex.real (f i); byi (Complex.imag (f (n - 1 - i)))]
let sym2 n f i = if (i < n - i) then f i else byi (f i)
let sym2i n f i = if (i < n - i) then f i else byui (f i)
```

From gen_hc2c.ml:

```ocaml
let sym n f i = if (i < n - i) then f i else Complex.conj (f i)
```

### Semantics

- **`sym2 n f i`**: pre-rotates by `+i` the upper half (i ≥ n-i). DIT pre-stage symmetry adjustment.
- **`sym2i n f i`**: pre-rotates by `−i` instead. Used for DIF.
- **`sym1 n f i`**: combines `Re(f(i))` with `i·Im(f(n-1-i))`. This is the Hermitian-packing combiner that's applied AFTER the c2c DFT — turns `(complex, complex_at_mirror)` pairs into the packed half-spectrum representation.
- **`sym n f i`**: at the last stage, position i < n/2 stays, position i ≥ n/2 gets conjugated. Unpacks Hermitian.

### DIT vs DIF wiring

```ocaml
(* hc2hc DIT *)
((sym1 n) @@ (sym2 n)) (Fft.dft sign n (byw input))
(* read packed input → pre-twiddle → c2c DFT → sym2 (post-rotate upper half) → sym1 (combine pairs) → write packed *)

(* hc2hc DIF *)
byw (Fft.dft sign n (((sym2i n) @@ (sym1 n)) input))
(* read packed input → sym1 → sym2i → c2c DFT → post-twiddle → write packed *)
```

DIT applies twiddles before the DFT; DIF after. The sym1/sym2
ordering reverses correspondingly.

## Math primitives to add (our codebase)

Add to `lib/dft_r2c.ml`:

```ocaml
(* Real DFT of n reals — outputs n/2+1 unique complex values.
 * Internally just c2c with im=0; algsimp eliminates redundancy. *)
val dft_rdft : ?sign:[`Fwd | `Bwd] -> int
            -> (int -> Expr.expr)            (* input_re : int -> real expr *)
            -> Expr.expr array * Expr.expr array  (* (out_re[0..n/2], out_im[0..n/2]) *)

(* Hermitian-input DFT — for the c2r (backward) path.
 * Inputs at positions 0..n/2 are unique; positions n/2+1..n-1 are conjugates.
 * Outputs are n real values. *)
val dft_hdft : ?sign:[`Fwd | `Bwd] -> int
            -> (int -> Expr.expr * Expr.expr)  (* input(i) : (re, im) pair *)
            -> Expr.expr array                  (* out[0..n-1] real *)

(* Middle-stage cascade codelet: twiddled DFT-n on Hermitian-packed data,
 * in-place. Reads from (cr_arr, ci_arr) of length n in packed form,
 * writes back same. Applies inter-stage twiddle pre-multiply. *)
val dft_hc2hc : ?sign:[`Fwd | `Bwd]
             -> ?direction:[`Dit | `Dif]
             -> int                            (* radix *)
             -> (int -> Expr.expr * Expr.expr) (* packed input load *)
             -> (int -> Expr.expr * Expr.expr) (* per-element twiddle (W^m, sign) *)
             -> (Expr.expr * Expr.expr) array  (* packed output, length n *)

(* Last-stage cascade codelet: twiddled DFT-n on Hermitian-packed input,
 * outputs natural-order complex via split pointers.
 * - "ri" load pattern: interleaved from two streams (even/odd from prev hc2hc)
 * - "pm" store pattern: positions [0, n/2) → (Rp, Ip); positions [n/2, n) → (Rm, Im) *)
val dft_hc2c : ?sign:[`Fwd | `Bwd]
            -> ?direction:[`Dit | `Dif]
            -> int
            -> (int -> Expr.expr * Expr.expr) (* "ri" packed input load *)
            -> (int -> Expr.expr * Expr.expr) (* twiddle *)
            -> (Expr.expr * Expr.expr) array  (* output, with pm storage convention encoded by caller *)
```

### Naming for emitted codelets

| Math primitive | CLI flag | Function name pattern |
|---|---|---|
| `dft_rdft` | `--rdft` | `radix{N}_rdft_{sgn}_{isa}_gen{suffix}` |
| `dft_hdft` | `--hdft` | `radix{N}_hdft_{sgn}_{isa}_gen{suffix}` |
| `dft_hc2hc` | `--hc2hc` | `radix{N}_hc2hc_{dir}_{sgn}_{isa}_gen{suffix}` |
| `dft_hc2c` | `--hc2c` | `radix{N}_hc2c_{dir}_{sgn}_{isa}_gen{suffix}` |

`--bwd` flips `sgn_suffix` as elsewhere. `--dif` switches DIT→DIF for hc2hc/hc2c.

## Phased implementation plan

### Phase 1: foundational primitive — `dft_rdft`

- Add `Dft.dft` already takes `(input_re, input_im)`. `dft_rdft` is a
  thin wrapper that calls it with `input_im k = Const 0.0`.
- Output: array of length n/2+1 (return only the unique half; the
  caller knows the symmetry).
- Validate: brute-force real DFT reference, compare DAG output.
- CLI: add `--rdft` flag; function naming.
- **Size: ~30 lines math + ~30 lines CLI + ~100 lines validation.**

This delivers something useful by itself: `--rdft 16` emits a codelet
that computes a 16-point real DFT with Hermitian-compact output. Drop-in
replacement for our current `dft_r2c_direct` for the *monolithic* case
(it's functionally equivalent, just cleaner — no separate "first stage
+ butterfly" structure).

### Phase 2: `dft_hc2hc` middle-stage codelet

- Port `sym1`, `sym2`, `sym2i` from gen_hc2hc.ml lines 67–74 directly.
  These are 3–5 lines each in our expression language.
- `dft_hc2hc n input_load twiddle` builds:
  - For DIT: `(sym1 n) @@ (sym2 n) (Fft.dft sign n (byw input))`
  - For DIF: `byw (Fft.dft sign n (((sym2i n) @@ (sym1 n)) input))`
- Twiddle scaffolding: reuse existing `Twiddle.twiddle_policy` —
  hc2hc uses `twiddle_policy 1 false` (one twiddle per position, not log3-shared).
- CLI: `--hc2hc` flag, `--dif`/`--dit` direction.
- **Size: ~60 lines math + ~30 lines CLI.**

### Phase 3: `dft_hc2c` last-stage codelet

- Port `sym` from gen_hc2c.ml line 50.
- Implement split-pointer storage (4-array storage convention).
- Indexing helpers: `locri` (load) and `locpm` (store) port directly.
- This is structurally similar to hc2hc but with different storage convention.
- **Size: ~80 lines math + ~30 lines CLI.**

### Phase 4: cascade harness + validation

- Build a cascade at N=128 = (R=8 rdft) → (R=16 hc2c). 
  Or N=128 = (R=8 rdft) → (R=8 hc2hc) → (R=2 hc2c). Multiple
  factorizations possible; pick the simplest first.
- Validate: brute-force DFT-128 reference, compare batch-0 output.
- Bench against Paths A (monolithic) and B (3-pass) on AVX2.
- **Size: ~250 lines.**

### Phase 5: production wire-up (later)

- Planner integration: pick cascade factorization vs monolithic at
  plan time based on N and available codelets.
- Wisdom format extension for r2c plans.
- Multi-thread story (parallelize over batch K, simpler than the
  current per-pass parallelization in `r2c.h`).

## Validation strategy

Three reference levels:

1. **Brute-force DFT** in scalar C. O(N²) but trivially correct.
2. **`dft_r2c_direct` monolithic** for sizes where it exists (N ≤ 256
   or wherever the monolithic codelet works). Match cascade output
   against monolithic.
3. **MKL or FFTW** for production-realistic numbers.

At each phase, run all three. The brute-force reference is the gating
check — anything that disagrees with it is wrong.

## Op-count expectations (recall doc 53)

| N | Ours current (3-pass) | Sorensen bound | Cascade target |
|---|---|---|---|
| 64 | 806 | 518 | ~518 |
| 128 | 1882 | 1286 | ~1286 |
| 256 | 4283 | 3078 | ~3078 |
| 512 | 9580 | 7174 | ~7174 |

Cascade should land within 5–10% of Sorensen (algsimp will recover
slightly more than the theoretical count via FMA fusion and CSE).
**At production sizes, 25–32% op reduction.**

## References

| File | Lines | Concept |
|---|---|---|
| `genfft/trig.ml` | 26–27 | `rdft = c2c with im=0` |
| `genfft/trig.ml` | 30–31 | `hdft = c2c with Hermitian inputs` |
| `genfft/gen_r2cf.ml` | 71–126 | First-stage codelet emission |
| `genfft/gen_hc2hc.ml` | 67–74 | `sym1`, `sym2`, `sym2i` definitions |
| `genfft/gen_hc2hc.ml` | 92–104 | DIT/DIF wiring with byw and sym chain |
| `genfft/gen_hc2c.ml` | 50 | `sym` definition for last-stage |
| `genfft/gen_hc2c.ml` | 100–101 | `locri`, `locpm` indexing |

Our equivalents:

| Their file | Our file |
|---|---|
| `genfft/trig.ml` | (none — add to `lib/dft_r2c.ml`) |
| `genfft/gen_r2cf.ml` | `bin/gen_radix.ml` (CLI), `lib/dft_r2c.ml` (math) |
| `genfft/gen_hc2hc.ml` | same |
| `genfft/gen_hc2c.ml` | same |

## Status

- [x] Design doc (this file)
- [x] Phase 1: `dft_rdft` math primitive (lib/dft_r2c.ml)
- [x] Phase 1: CLI wiring + emission (`--rdft`, `radix{N}_rdft_{sgn}_{isa}_gen`)
- [x] Phase 1: op-count validation — R=8: 23 ops (Sorensen bound = 22, **+4.5%** vs prior **+91%**)
- [x] Phase 2: `dft_hc2hc` with DIT/DIF dispatch + sym1/sym2/sym2i helpers
- [x] Phase 2: CLI wiring (`--hc2hc`, `radix{R}_hc2hc_{dir}_{sgn}_{isa}_gen`)
- [x] Phase 3: `dft_hc2c` with DIT/DIF dispatch + sym helper
- [x] Phase 3: CLI wiring (`--hc2c`, `radix{R}_hc2c_{dir}_{sgn}_{isa}_gen`)
- [x] Phase 4a: scalar reference cascade at N=8 — max_err 3.3e-16 PASS
      (test/r2c/cascade_scalar_ref.c)
- [x] Phase 4b: codelet-based cascade at N=8 (rdft_4 × 2 + hand combine) —
      max_err 6.1e-16 PASS. Validates rdft_R + Hermitian-symmetry combine
      against brute-force DFT-8. (test/r2c/cascade_codelet_n8.c)
- [ ] Phase 4c: swap hand combine → hc2c_M codelet (validate split-pointer
      convention)
- [ ] Phase 4d: scale to N=128 = 16×8 (or per-wisdom factorization)
- [ ] Phase 4e: bench vs Path A (monolithic r2c) and Path B (3-pass)
- [ ] Phase 5: production wire-up

### Op-count snapshot (vector instructions, fwd direction, AVX2)

| R | rdft | hc2hc | hc2c | t1_dit (c2c baseline) | Sorensen bound for rdft |
|---|---|---|---|---|---|
| 8  | 23  | 88   | 88   | 84   | 22 |
| 16 | 78  | 235  | 235  | 227  | 70 |
| 32 | 249 | 598  | 598  | 582  | 198 |
| 64 | 747 | 1444 | 1444 | 1412 | 518 |

**rdft is at the bound for small R.** hc2hc/hc2c add ~3-5% over plain
t1_dit per call (the symmetry transform cost). The cascade-level win
comes from hc2hc/hc2c operating on Hermitian-packed data — only n/2+1
unique values per stage — which roughly halves total work across a
multi-stage chain. Concrete cascade op counts pending Phase 4.
