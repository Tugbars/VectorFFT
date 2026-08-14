(* dft_recurse.ml — the c2c recursive constructions.
 *
 * Middle layer of the Dft chain (Dft_select < Dft_recurse < Dft).
 * One atomic mutual-recursion group builds the expr DAG for any N:
 * dft dispatches on Dft_select.pick_algorithm into dft_direct (naive,
 * n = 2), dft_direct_conjugate_pair (odd primes), the Winograd 5 / 25
 * / 7 constructions, dft_ct (Cooley-Tukey n1 x n2), or the
 * split_radix.ml callback. The group is one `and` chain — it moves as
 * a unit or not at all. const_cmul (the constant-twiddle complex
 * multiply with the FFTW-style factored |cr| = |ci| path) lives here
 * because every construction in the chain leans on it.
 *
 * Output convention: functions take input accessors (int -> expr) and
 * return (out_re, out_im) expr arrays; no assignment lists yet — the
 * facade's dft_expand wrappers do that packaging.
 * ------------------------------------------------------------------
 * MODULE CARD (dft_recurse.ml — grep "MODULE CARD" for the full set)
 * ROLE: The DAG-building recursion for c2c DFTs (all algorithms).
 * PIPELINE: Dft facade wrappers -> this recursion -> Expr trees
 * PUBLIC SURFACE (measured): zero direct Dft_recurse.X references —
 * dft / dft_ct / const_cmul etc. are reached as Dft.X via the facade
 * (heaviest external user: dft_r2c with Dft_recurse.dft x17).
 * DEPS: Dft_select via include; Expr (open); Cnum(4); Split_radix(2).
 * ------------------------------------------------------------------
 *)

open Dft_select  (* M1: was `include` — Dft_select is no longer re-exported *)
open Expr

(* === COOLEY-TUKEY DIT DECOMPOSITION ===
 *
 * Radix-2 DIT decomposition of DFT-N (N even):
 *
 *   X[k]       = E[k] + ω_N^k · O[k]      for k in 0..N/2-1
 *   X[k+N/2]   = E[k] - ω_N^k · O[k]      for k in 0..N/2-1
 *
 * where E = DFT-(N/2) on even-indexed inputs, O = DFT-(N/2) on
 * odd-indexed inputs, and ω_N = exp(-2πi/N) (forward DFT).
 *
 * The complex multiply ω_N^k · O[k] uses constant twiddles (known at
 * code-gen time). For k=0: ω = 1, no multiplication. For k=N/4:
 * ω = -i, just a swap. For k=N/2: ω = -1, just a sign flip.
 * For other k: ω = (cos θ, -sin θ) with θ = 2πk/N — these need
 * actual multiplications by cos/sin constants.
 *
 * We use plain Mul/Add/Sub (NOT Cmul opaque atoms) for these constant
 * twiddle multiplies — algsimp's constant folding handles the trivial
 * k=0/N/4/N/2 cases, and hash-consing shares the √2/2 constant across
 * all uses. Cmul is reserved for runtime-loaded twiddles (the t1_dit
 * premultiplication stage), where we want to PROTECT the cmul structure
 * from reassoc.
 *)

(* Helper: compute the symbolic complex multiply by a constant twiddle.
 * (a + ib) * (c + id) = (ac - bd) + i(ad + bc).
 *
 * Path B optimization (FFTW-style): when |cr| = |ci| = K (the W^k twiddles
 * with k ∈ {1,3,5,7,...} for radix-{2^n}), every output reduces to one
 * of ±K*(xr+xi), ±K*(xr-xi). We emit the FACTORED form so the K-multiply
 * happens AFTER the sum/diff, not before. Two effects:
 *
 *   1. Operation count drops from 6 (4 mul + 1 add + 1 sub) to 4 (2 mul +
 *      1 add + 1 sub). Saves 2 ops per |cr|=|ci| twiddle.
 *
 *   2. The remaining Mul(K, sum) and Mul(K, diff) each have ≤2 downstream
 *      consumers (one + and one -) that look like K*S ± value. These are
 *      perfect FMA absorption targets — multi_use_fma_lift folds each
 *      Mul into its consumers. Net effect from absorbing the Mul: -1 op
 *      AND +1 FMA per absorption, closing the FMA-count gap vs FFTW.
 *
 * For the general |cr| ≠ |ci| case, the original 4-mul form is optimal
 * (no shared factor to extract; algsimp's existing fma_lift handles it).
 *
 * Sign analysis for |cr|=|ci|=K case:
 *   (+K, +K):  out_re = K*(xr-xi),  out_im = K*(xr+xi)
 *   (+K, -K):  out_re = K*(xr+xi),  out_im = K*(xi-xr) = -K*(xr-xi)
 *   (-K, +K):  out_re = -K*(xr+xi), out_im = K*(xr-xi)
 *   (-K, -K):  out_re = K*(xi-xr) = -K*(xr-xi),  out_im = -K*(xr+xi)
 *
 * In all 4 cases the building blocks are K*(xr+xi) and K*(xr-xi); only
 * the signs of the assignments differ. Hash-consing then SHARES the two
 * K-multiplied sums across all uses of this twiddle. *)
let const_cmul (xr : expr) (xi : expr) (cr : float) (ci : float) : expr * expr =
  let abs_eq = abs_float cr = abs_float ci in
  if abs_eq && cr <> 0.0
  then (
    let k = abs_float cr in
    let k_e = Const k in
    let s = Add (xr, xi) in
    (* xr + xi *)
    let d = Sub (xr, xi) in
    (* xr - xi *)
    let ks = Mul (k_e, s) in
    (* K*(xr+xi) *)
    let kd = Mul (k_e, d) in
    (* K*(xr-xi) *)
    let sr = if cr > 0.0 then 1 else -1 in
    let si = if ci > 0.0 then 1 else -1 in
    let with_sign sgn e = if sgn > 0 then e else Neg e in
    match sr, si with
    | 1, 1 -> kd, ks (* out_re=K*D, out_im=K*S *)
    | 1, -1 -> ks, with_sign (-1) kd (* out_re=K*S, out_im=-K*D *)
    | -1, 1 -> with_sign (-1) ks, kd (* out_re=-K*S, out_im=K*D *)
    | -1, -1 -> with_sign (-1) kd, with_sign (-1) ks (* out_re=-K*D, out_im=-K*S *)
    | _ -> assert false)
  else (
    (* General case |cr| ≠ |ci|: emit in TAN-FACTORED form (FFTW genfft
     * with -fma flag). Pick the larger of |cr|, |ci| as the OUTER factor
     * (the "cos"), and the ratio as the INNER factor (the "tan"):
     *
     *   if |cr| ≥ |ci|:
     *     y_re = cr · (xr − (ci/cr)·xi)
     *     y_im = cr · (xi + (ci/cr)·xr)
     *
     *   if |ci| > |cr|:
     *     swap roles; ci is the outer factor, cr/ci is the inner ratio.
     *
     * IMPORTANT — canonical inputs for cross-rotation Const sharing:
     * For symmetric angles, the inner ratio is mathematically the same
     * (e.g. tan(π/8) for both ω^4 and ω^12 at N=64) but the FP inputs
     * differ by 1 ulp because of how sin/cos round. To unify, we round
     * cr and ci to 13 significant digits BEFORE computing the ratio.
     * This is the same precision the smart constructor mk_const uses
     * downstream when the inner constant is hash-consed, but applying
     * it here matters because the rounding-to-13e happens on the
     * INPUTS to the division, not the OUTPUT — and the division
     * amplifies any sub-ulp difference into a 1-ulp difference in
     * the ratio.
     *
     * The unification only fires where there's actually a matching
     * symmetric angle. Non-symmetric rotations keep their distinct
     * (cr, ci) bit patterns up to the 13-digit rounding mk_const
     * would apply anyway. *)
    (* Unification of symmetric-angle constants now happens in Ir.mk_const,
       * which dedups on a 14-sig-digit KEY while storing the first-seen
       * full-precision value. Quantizing the inputs here (the previous
       * behavior) stored the damaged values themselves: ~22-30 ulp on the
       * cosine leaf and ~90 ulp on the tangent ratio, measured by the
       * accuracy harness. Full precision in, full precision emitted. *)
    let round_13 x = x in
    let cr_r = round_13 cr in
    let ci_r = round_13 ci in
    let acr = abs_float cr_r in
    let aci = abs_float ci_r in
    let r_abs = min acr aci /. max acr aci in
    if acr >= aci
    then (
      let tn = if ci_r >= 0.0 = (cr_r >= 0.0) then r_abs else -.r_abs in
      let cr_e = Const cr_r in
      let tn_e = Const tn in
      let inner_re = Sub (xr, Mul (tn_e, xi)) in
      (* xr - tan*xi *)
      let inner_im = Add (xi, Mul (tn_e, xr)) in
      (* xi + tan*xr *)
      let out_re = Mul (cr_e, inner_re) in
      let out_im = Mul (cr_e, inner_im) in
      out_re, out_im)
    else (
      let ct = if cr_r >= 0.0 = (ci_r >= 0.0) then r_abs else -.r_abs in
      let ci_e = Const ci_r in
      let ct_e = Const ct in
      let inner_re = Sub (Mul (ct_e, xr), xi) in
      (* ct*xr - xi *)
      let inner_im = Add (xr, Mul (ct_e, xi)) in
      (* xr + ct*xi *)
      let out_re = Mul (ci_e, inner_re) in
      let out_im = Mul (ci_e, inner_im) in
      out_re, out_im))
;;

(* The recursive DFT computation.
 *
 * Inputs:
 *   n          — transform size
 *   input_re k — Expr tree for the k-th input's real component
 *   input_im k — Expr tree for the k-th input's imag component
 *
 * Returns: (re_outputs, im_outputs) as arrays of Expr trees indexed by k.
 *
 * The algorithm dispatch and recursion happens here. Algsimp is NOT
 * called inside — it runs once at the top level, on the fully-expanded
 * tree. This way hash-consing catches sharing across CT recursion levels.
 *)
(* The recursive DFT computation.
 *
 * Inputs:
 *   n          — transform size
 *   input_re k — Expr tree for the k-th input's real component
 *   input_im k — Expr tree for the k-th input's imag component
 *   ?sign      — Fwd uses θ = -2πk/N (DFT); Bwd uses θ = +2πk/N (IDFT, no /N).
 *)
let rec dft ?(sign = `Fwd) (n : int) (input_re : int -> expr) (input_im : int -> expr)
  : expr array * expr array
  =
  (* Hand-derived Winograd-25 codelet, gated for A/B comparison.
   * See dft_winograd25 (below) for the algebra. *)
  if n = 25 && Sys.getenv_opt "VFFT_WINOGRAD25" = Some "1"
  then dft_winograd25 ~sign input_re input_im
  else (
    match pick_algorithm n with
    | Direct ->
      (* For odd N >= 3, the conjugate-pair construction produces a much
       * better-structured DAG than naive direct DFT: pair sums/diffs are
       * shared, per-pair-output intermediates (p_re_m, q_im_m, etc.) are
       * shared between X[m] and X[N-m], and inner sums use linear FMA
       * chains. For N=2 there's nothing to factor; for even N we never
       * reach Direct anyway (CT-decomposed).
       *
       * Special case for N=5: Winograd-5 (dft_winograd5) exploits algebraic
       * identities of 5th roots of unity to reduce 36 ops to 32 and matches
       * FFTW's gen_notw -fma R=5 codelet exactly. Propagates through any
       * radix that decomposes to DFT-5 (R=15, R=20, R=25, R=50, R=100, ...).
       *
       * Empirical trade-off measured for the R=25 cascade
       * (sandbox Xeon 2.80 GHz, 31 trials × 5000 reps × 10 runs):
       *   AVX2  : Winograd ~2% faster (spill traffic dominates chain depth)
       *   AVX-512: Winograd ~4% SLOWER (port parallelism > chain depth)
       *
       * The AVX-512 regression is real and reproducible, but the underlying
       * gap to FFTW (+31 ops at R=25, all butterfly-pair-shared Muls in
       * inter-pass twiddles) is structurally tied to our binary IR. The
       * n-ary `Plus` rewrite that FFTW's genfft uses would unblock both
       * the AVX-512 regression and the R=25/R=64 gap to FFTW — see doc 59
       * addendum for sizing. Until that lands, the choice is between:
       *   (a) Pure default win on AVX2, small loss on AVX-512 R=25
       *   (b) Flag-based dispatch (one more flag, more cognitive load)
       * We pick (a). Code simplicity > marginal AVX-512 R=25 perf. *)
      if n = 5
      then
        if Sys.getenv_opt "VFFT_CNUM_W5" = Some "1"
        then (
          let input = Cnum.signal_of_re_im input_re input_im in
          let out = dft_winograd5_cnum ~sign input in
          Cnum.split_re_im out)
        else dft_winograd5 ~sign input_re input_im
      else if n = 7
      then dft_winograd7 ~sign input_re input_im
      else if n >= 3 && n mod 2 = 1
      then dft_direct_conjugate_pair ~sign n input_re input_im
      else dft_direct ~sign n input_re input_im
    | Cooley_Tukey (n1, n2) -> dft_ct ~sign n1 n2 input_re input_im
    | Split_radix ->
      (* Split-radix lives in its own module (lib/split_radix.ml). Cross-
       * module mutual recursion uses the callback pattern: we pass `dft`
       * itself in as `dft_rec` so SR can recurse on its sub-DFT inputs
       * (size N/2 and N/4) which dispatch back through the picker. *)
      if newsplit_enabled ()
      then Split_radix.dft_newsplit ~sign n input_re input_im
      else
        Split_radix.dft_split_radix
          ~dft_rec:(fun ~sign:s n' f g -> dft ~sign:s n' f g)
          ~sign
          n
          input_re
          input_im)

(* Direct DFT: matrix-vector form.
 *   X[k].re = Σ_n  a[n] * cos(±2πnk/n) - b[n] * sin(±2πnk/n)
 *   X[k].im = Σ_n  a[n] * sin(±2πnk/n) + b[n] * cos(±2πnk/n)
 * The sign of θ is - for Fwd, + for Bwd. *)
and dft_direct ?(sign = `Fwd) (n : int) (input_re : int -> expr) (input_im : int -> expr)
  : expr array * expr array
  =
  let pi = 4.0 *. atan 1.0 in
  let sgn =
    match sign with
    | `Fwd -> -1.0
    | `Bwd -> 1.0
  in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k = 0 to n - 1 do
    let re_sum = ref (Const 0.0) in
    let im_sum = ref (Const 0.0) in
    for nn = 0 to n - 1 do
      let theta = sgn *. 2.0 *. pi *. float_of_int (nn * k) /. float_of_int n in
      let c = cos theta in
      let s = sin theta in
      let a_nn = input_re nn in
      let b_nn = input_im nn in
      re_sum := Add (!re_sum, Sub (Mul (a_nn, Const c), Mul (b_nn, Const s)));
      im_sum := Add (!im_sum, Add (Mul (a_nn, Const s), Mul (b_nn, Const c)))
    done;
    out_re.(k) <- !re_sum;
    out_im.(k) <- !im_sum
  done;
  out_re, out_im

(* === DIRECT DFT WITH EXPLICIT CONJUGATE-PAIR FACTORING ===
 *
 * For a real input pair (x[j], x[N-j]) with j in 1..(N-1)/2, the
 * twiddle factors satisfy:
 *
 *   cos(2π·j·m/N)     =  cos(2π·(N-j)·m/N)     [cos is even]
 *   sin(2π·j·m/N)     = -sin(2π·(N-j)·m/N)     [sin is odd]
 *
 * So for forward DFT (using exp(-2πi·j·m/N) = cos(2πjm/N) - i·sin(2πjm/N)):
 *
 *   X[m].re = x[0].re + Σ_{j=1..H} (cos(jm)·s_re_j + sin(jm)·d_im_j)
 *   X[N-m].re = x[0].re + Σ_{j=1..H} (cos(jm)·s_re_j - sin(jm)·d_im_j)
 *
 *   X[m].im = x[0].im + Σ_{j=1..H} (cos(jm)·s_im_j - sin(jm)·d_re_j)
 *   X[N-m].im = x[0].im + Σ_{j=1..H} (cos(jm)·s_im_j + sin(jm)·d_re_j)
 *
 * where H = (N-1)/2 and:
 *   s_re_j = x[j].re + x[N-j].re      d_re_j = x[j].re - x[N-j].re
 *   s_im_j = x[j].im + x[N-j].im      d_im_j = x[j].im - x[N-j].im
 *
 * Per pair (m, N-m) we compute four shared intermediates ONCE:
 *   p_re_m = Σ cos(jm)·s_re_j      (shared between X[m].re and X[N-m].re)
 *   p_im_m = Σ cos(jm)·s_im_j      (shared between X[m].im and X[N-m].im)
 *   q_re_m = Σ sin(jm)·d_re_j      (shared between im outputs, opposite signs)
 *   q_im_m = Σ sin(jm)·d_im_j      (shared between re outputs, opposite signs)
 *
 * The OCaml `expr` is value-shared for the four intermediates; Algsimp's
 * hash-cons preserves the sharing through `of_expr`. The outer outputs
 * use BINARY structure (no flattening) so the shared sub-trees remain
 * intact — caller must use `reassoc:false`.
 *
 * For backward DFT (sign=`Bwd), exp(+2πi·j·m/N), so sin signs flip:
 *
 *   X[m].re   = x[0].re + Σ (cos(jm)·s_re_j - sin(jm)·d_im_j)
 *   X[N-m].re = x[0].re + Σ (cos(jm)·s_re_j + sin(jm)·d_im_j)
 *   X[m].im   = x[0].im + Σ (cos(jm)·s_im_j + sin(jm)·d_re_j)
 *   X[N-m].im = x[0].im + Σ (cos(jm)·s_im_j - sin(jm)·d_re_j)
 *)
and dft_direct_conjugate_pair
      ?(sign = `Fwd)
      (n : int)
      (input_re : int -> expr)
      (input_im : int -> expr)
  : expr array * expr array
  =
  let pi = 4.0 *. atan 1.0 in
  let sgn =
    match sign with
    | `Fwd -> -1.0
    | `Bwd -> 1.0
  in
  let half = (n - 1) / 2 in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  (* === STAGE 1: pair sums and diffs (the s_jk and d_jk subterms) ===
   * Computed once each, shared everywhere via OCaml value reuse → hash-cons. *)
  let s_re =
    Array.init (half + 1) (fun j ->
      if j = 0 then input_re 0 else Add (input_re j, input_re (n - j)))
  in
  let s_im =
    Array.init (half + 1) (fun j ->
      if j = 0 then input_im 0 else Add (input_im j, input_im (n - j)))
  in
  let d_re =
    Array.init (half + 1) (fun j ->
      if j = 0 then Const 0.0 else Sub (input_re j, input_re (n - j)))
  in
  let d_im =
    Array.init (half + 1) (fun j ->
      if j = 0 then Const 0.0 else Sub (input_im j, input_im (n - j)))
  in
  (* === STAGE 2: linear-chain weighted sums ===
   * Two variants:
   *
   * `make_sum_with_init initial coeffs terms`:
   *   Builds initial + sign(c1)|c1|·t1 + sign(c2)|c2|·t2 + ... as a left-fold
   *   chain. For positive c, emits `Add(acc, Mul(t, |c|))`; for negative c,
   *   emits `Sub(acc, Mul(t, |c|))`. After fma_lift:
   *     - Add(acc, Mul) lifts to fmadd  →  a*b + acc
   *     - Sub(acc, Mul) lifts to fnmadd →  -a*b + acc
   *   This produces a single FMA chain with mixed +/- coefficients encoded
   *   in the FMA opcode (matching FFTW codelet style). The deepest addend
   *   is `initial`, free at the asm level (it's the FMA's `c` operand).
   *
   * `make_sum coeffs terms`:
   *   Same but no initial accumulator — first term starts as a Mul.
   *   Used for q chains (sine sums) which don't have x[0] to absorb.
   *
   * Why sign-aware: when coeffs have mixed signs (typical for prime DFTs
   * where cos(2πjm/N) > 0 for some j and < 0 for others), our pipeline's
   * factor pass otherwise splits the sum into "positive coefficient" and
   * "negative coefficient" sub-chains, costing ~4 extra ops per pair output
   * block (start-mul on each sub-chain + extra structural sub for combining). *)
  let make_sum_with_init initial coeffs terms =
    let acc = ref initial in
    for j = 1 to half do
      let c = coeffs.(j) in
      let abs_c = Float.abs c in
      let term = Mul (terms.(j), Const abs_c) in
      acc := if c < 0.0 then Sub (!acc, term) else Add (!acc, term)
    done;
    !acc
  in
  let make_sum coeffs terms =
    let acc = ref None in
    for j = 1 to half do
      let c = coeffs.(j) in
      let abs_c = Float.abs c in
      let term = Mul (terms.(j), Const abs_c) in
      acc
      := match !acc with
         | None ->
           (* First term: if positive, start with the Mul as-is.
            * If negative, start with Neg(Mul). fma_lift catches the Neg(Mul)
            * pattern when it's then Add'd to something later, producing fnmadd. *)
           Some (if c < 0.0 then Neg term else term)
         | Some a -> Some (if c < 0.0 then Sub (a, term) else Add (a, term))
    done;
    match !acc with
    | Some a -> a
    | None -> Const 0.0
  in
  (* === STAGE 3: X[0] — sum of all real/imag inputs ===
   * Using the s_re/s_im pair sums, X[0].re = x[0].re + Σ s_re_j
   * (each pair sum is reused from STAGE 1, no duplicate adds.) *)
  let x0_re = ref (input_re 0) in
  let x0_im = ref (input_im 0) in
  for j = 1 to half do
    x0_re := Add (!x0_re, s_re.(j));
    x0_im := Add (!x0_im, s_im.(j))
  done;
  out_re.(0) <- !x0_re;
  out_im.(0) <- !x0_im;
  (* === STAGE 4: pair outputs X[m] and X[N-m] for m = 1..half ===
   * Compute four intermediates once per pair, then combine. *)
  for m = 1 to half do
    let cos_arr =
      Array.init (half + 1) (fun j ->
        cos (2.0 *. pi *. float_of_int (j * m) /. float_of_int n))
    in
    (* Sin coefficient for the q_im_m = Σ sin(jm)·d_im_j intermediate.
     * The output combinations below use:
     *   out_re.(m)     = x[0].re + p_re + q_im   ← needs +sin for forward
     *   out_re.(n-m)   = x[0].re + p_re - q_im
     *
     * For forward (exp(-iθ) = cos-i·sin), X[m].re = Σcos·s_re + Σsin·d_im,
     * so coefficient inside q_im needs to be +sin → factor = +1 → -sgn.
     *
     * For backward (exp(+iθ) = cos+i·sin), X[m].re = Σcos·s_re - Σsin·d_im,
     * so coefficient inside q_im is -sin → factor = -1 → -sgn.
     *
     * Both cases: sin_arr coefficient = -sgn · sin(2πjm/N). *)
    let sin_arr =
      Array.init (half + 1) (fun j ->
        -.sgn *. sin (2.0 *. pi *. float_of_int (j * m) /. float_of_int n))
    in
    (* p_re_m / p_im_m: cosine sums WITH x[0] absorbed as the deepest addend.
     * The chain shape is:
     *   ((((x[0] +/- |c1|·s1) +/- |c2|·s2) +/- |c3|·s3) +/- |c4|·s4) +/- |c5|·s5
     * After fma_lift, this becomes 5 nested FMAs (mix of fmadd/fnmadd based on
     * coefficient signs). The deepest fma uses x[0] as its `c` addend (free).
     *
     * q_re_m / q_im_m: sine sums WITHOUT x[0] (first term starts as Mul or
     * Neg(Mul) depending on sign). After fma_lift: 1 mul + 4 fmas chain.
     * Output combines absorb sin chain via Add/Sub at top level. *)
    let p_re_m = make_sum_with_init (input_re 0) cos_arr s_re in
    let p_im_m = make_sum_with_init (input_im 0) cos_arr s_im in
    let q_re_m = make_sum sin_arr d_re in
    let q_im_m = make_sum sin_arr d_im in
    (* Output combinations. p_re_m / p_im_m already include x[0]; we just
     * add or subtract the q chain. Each output requires exactly 1 op
     * (1 add or 1 sub) at this combining level — matching hand-coded
     * FFTW-style codelet structure. *)
    out_re.(m) <- Add (p_re_m, q_im_m);
    out_re.(n - m) <- Sub (p_re_m, q_im_m);
    out_im.(m) <- Sub (p_im_m, q_re_m);
    out_im.(n - m) <- Add (p_im_m, q_re_m)
  done;
  out_re, out_im

(* General Cooley-Tukey DIT decomposition: N = N1 · N2.
 *
 * Standard DIT convention (matches user's gen_radix*.py):
 *   - Input mapping:  n = n1 + n2 · N1  (low digit n1, high digit n2)
 *   - Output mapping: k = k1 · N2 + k2  (high digit k1, low digit k2)
 *
 * Decomposition:
 *   X[k1·N2 + k2] = Σ_{n1} ω_{N1}^{n1·k1} ·
 *                   ω_N^{n1·k2} ·
 *                   Σ_{n2} x[n1 + n2·N1] · ω_{N2}^{n2·k2}
 *
 * Read as three nested operations:
 *   PASS 1: For each n1 (offset), compute DFT-N2 on the strided slice
 *           x[n1], x[n1 + N1], x[n1 + 2·N1], ..., x[n1 + (N2-1)·N1].
 *           Output: pass1[n1][k2].
 *   TWIDDLE: Multiply pass1[n1][k2] by the inter-stage twiddle ω_N^{n1·k2}.
 *   PASS 2: For each k2 (output sub-index), compute DFT-N1 on the
 *           column twiddled[0..N1-1][k2]. Output goes to X[k1·N2 + k2].
 *
 * Concrete examples:
 *   R=4 = CT(2, 2): PASS 1 splits inputs by parity (even/odd).
 *   R=8 = CT(2, 4): PASS 1 splits by parity, two DFT-4s (even/odd indices).
 *   R=16 = CT(4, 4): PASS 1 splits by mod-4 residue, four DFT-4s.
 *)
and dft_winograd5 ?(sign = `Fwd) (input_re : int -> expr) (input_im : int -> expr)
  : expr array * expr array
  =
  (* Winograd 5-point DFT — exploits algebraic identities of 5th roots of
   * unity to reduce arithmetic vs. the naive direct DFT.
   *
   * Identities used:
   *   cos(2π/5) + cos(4π/5) = -1/2
   *   cos(2π/5) - cos(4π/5) =  √5/2
   *   sin(4π/5)/sin(2π/5)   =  1/φ   (= 2cos(2π/5) = (√5-1)/2)
   *
   * Four hard-coded constants:
   *   k_quarter  = 0.25
   *   k_root5_4  = √5/4
   *   k_sin_2pi5 = sin(2π/5)        ≈ 0.951
   *   k_inv_phi  = 1/φ              ≈ 0.618
   *
   * Op count: 14 add/sub + 18 fma = 32 total (matches FFTW gen_notw -fma).
   *
   * Compared with dft_direct_conjugate_pair's 36 ops at R=5: saves 4 ops
   * via the cos(2π/5) ± cos(4π/5) identity (Winograd cos channel uses 2
   * muls instead of 4) and the sin(4π/5) = sin(2π/5)/φ identity (Winograd
   * sin channel shares one s1 factor across both sin terms).
   *
   * Scheduling for register pressure: emit in the same order as FFTW's
   * gen_notw output — real pre-adds, then output 0, then real outputs
   * (which only need real pre-adds and the imag pair-diffs), then imag
   * outputs. The two channels share no live intermediates after their
   * own outputs are emitted, so peak live ≈ 14 IR nodes (well under the
   * 16-ymm AVX2 budget; trivially fits AVX-512).
   *
   * Sign convention: forward (`Fwd) uses ω = exp(-2πi/5), backward (`Bwd)
   * uses ω = exp(+2πi/5). The cos channel is sign-agnostic; only the
   * imag-channel contribution to real outputs (and vice versa) flips.
   * We handle this by flipping the sign of `k_inv_phi` and `k_sin_2pi5`
   * for Bwd in the appropriate slots. *)
  let two_pi = 8.0 *. atan 1.0 in
  let k_quarter = Const 0.25 in
  let k_root5_4 = Const (sqrt 5.0 /. 4.0) in
  let k_inv_phi = Const ((sqrt 5.0 -. 1.0) /. 2.0) in
  (* Sign convention for the sin-channel coupling. For Fwd DFT (θ = -2πnk/N),
   * working through the algebra of (x_n.re + i x_n.im) · (cos(θ) - i sin(θ))
   * gives  Re(X_1) = ... + sin(2π/5)·(x_1.im - x_4.im) + sin(4π/5)·(x_2.im - x_3.im)
   * with POSITIVE sin coupling. For Bwd (θ = +2πnk/N) the sign flips.
   * Absorb that into the constant so the structural signs in the output
   * assignments below stay identical for both directions. *)
  let s_sign =
    match sign with
    | `Fwd -> 1.0
    | `Bwd -> -1.0
  in
  let k_sin_2pi5_s = Const (s_sign *. sin (two_pi /. 5.0)) in
  (* Real-channel pre-additions
   *   t4 = x_1.re + x_4.re      ta = t4 - t7
   *   t7 = x_2.re + x_3.re      tt = x_2.re - x_3.re
   *   t8 = t4 + t7              ts = x_1.re - x_4.re *)
  let x0r = input_re 0 in
  let x1r = input_re 1 in
  let x4r = input_re 4 in
  let t4 = Add (x1r, x4r) in
  let x2r = input_re 2 in
  let x3r = input_re 3 in
  let t7 = Add (x2r, x3r) in
  let t8 = Add (t4, t7) in
  let tt = Sub (x2r, x3r) in
  let ta = Sub (t4, t7) in
  let ts = Sub (x1r, x4r) in
  (* Imag-channel pre-additions — same structure. *)
  let x0i = input_im 0 in
  let x1i = input_im 1 in
  let x4i = input_im 4 in
  let tm = Add (x1i, x4i) in
  let x2i = input_im 2 in
  let x3i = input_im 3 in
  let tn = Add (x2i, x3i) in
  let te = Sub (x1i, x4i) in
  let tq = Sub (tm, tn) in
  let th = Sub (x2i, x3i) in
  let to_ = Add (tm, tn) in
  let out_re = Array.make 5 (Const 0.0) in
  let out_im = Array.make 5 (Const 0.0) in
  (* Output 0 — trivial sums, no multiplications. *)
  out_re.(0) <- Add (x0r, t8);
  out_im.(0) <- Add (x0i, to_);
  (* Real outputs X_1.re .. X_4.re.
   *   ti = te + (1/φ)·th       — shares sin1·(C.im + D.im·1/φ) = s1·C.im+s2·D.im
   *   tk = th - (1/φ)·te       — pairs for X_2/X_3
   *   t9 = x_0.re - 0.25·t8    — common real anchor
   *   tb = t9 + (√5/4)·ta      — cos-channel for X_1/X_4
   *   tj = t9 - (√5/4)·ta      — cos-channel for X_2/X_3
   * Then:
   *   X_1.re = tb + s1·ti      (FMA)
   *   X_4.re = tb - s1·ti      (FNMS)
   *   X_2.re = tj - s1·tk      (FNMS)
   *   X_3.re = tj + s1·tk      (FMA) *)
  let ti = Add (te, Mul (k_inv_phi, th)) in
  let tk = Sub (th, Mul (k_inv_phi, te)) in
  let t9 = Sub (x0r, Mul (k_quarter, t8)) in
  let tb = Add (t9, Mul (k_root5_4, ta)) in
  let tj = Sub (t9, Mul (k_root5_4, ta)) in
  out_re.(1) <- Add (tb, Mul (k_sin_2pi5_s, ti));
  out_re.(4) <- Sub (tb, Mul (k_sin_2pi5_s, ti));
  out_re.(2) <- Sub (tj, Mul (k_sin_2pi5_s, tk));
  out_re.(3) <- Add (tj, Mul (k_sin_2pi5_s, tk));
  (* Imag outputs X_1.im .. X_4.im. Mirror of real, with (ts, tt) playing
   * the role of (te, th), (tq, to_) playing (ta, t8), and the cross-
   * coupling sign flipped relative to the real channel — that flip is
   * absorbed into k_sin_2pi5_s above for the real channel; here the
   * direct s1·X form is used (sin contribution to imag is positive in
   * Fwd, negative in Bwd, which is what sgn captures via the same
   * k_sin_2pi5_s constant). *)
  let tu = Add (ts, Mul (k_inv_phi, tt)) in
  let tw = Sub (tt, Mul (k_inv_phi, ts)) in
  let tp = Sub (x0i, Mul (k_quarter, to_)) in
  let tr = Add (tp, Mul (k_root5_4, tq)) in
  let tv = Sub (tp, Mul (k_root5_4, tq)) in
  (* Note the sign flip relative to real channel: imag outputs use the
   * OPPOSITE sign on the s1·tu and s1·tw FMAs. For Fwd, X_1.im uses
   * `tr - s1·tu`; for Bwd, `tr + s1·tu`. The k_sin_2pi5_s constant
   * already carries `sgn`, so we keep the structural sign as +real and
   * −imag (which becomes −real, +imag for Bwd). *)
  out_im.(1) <- Sub (tr, Mul (k_sin_2pi5_s, tu));
  out_im.(4) <- Add (tr, Mul (k_sin_2pi5_s, tu));
  out_im.(2) <- Add (tv, Mul (k_sin_2pi5_s, tw));
  out_im.(3) <- Sub (tv, Mul (k_sin_2pi5_s, tw));
  out_re, out_im

(* ============================================================
 * dft_winograd5_cnum — same algorithm as dft_winograd5, written
 * via the Cnum combinator layer.
 *
 * This is a TEST CASE for the Cnum infrastructure: it should produce
 * the same (or strictly fewer) ops as dft_winograd5. If it doesn't,
 * something is wrong with Cnum or the Expr smart constructors.
 *
 * The algebra is identical to dft_winograd5. What changes is the
 * SHAPE of the constructed Expr tree — specifically, the `Mul`
 * nodes are placed at leaves of `Sub`/`Add` chains via mk_sub/mk_add,
 * letting the Expr.mk_mul rotation rule fire when downstream code
 * multiplies this by a constant. (For W5 used standalone, there is
 * no downstream constant multiplication; the rotation only matters
 * when W5 is composed in a larger codelet via cscale or cmul.)
 * ============================================================ *)
and dft_winograd5_cnum ?(sign = `Fwd) (input : int -> Cnum.cnum) : Cnum.cnum array =
  let two_pi = 8.0 *. atan 1.0 in
  let k_quarter = Const 0.25 in
  let k_root5_4 = Const (sqrt 5.0 /. 4.0) in
  let k_inv_phi = Const ((sqrt 5.0 -. 1.0) /. 2.0) in
  let s_sign =
    match sign with
    | `Fwd -> 1.0
    | `Bwd -> -1.0
  in
  let k_sin_2pi5_s = Const (s_sign *. sin (two_pi /. 5.0)) in
  let open Expr in
  let open Cnum in
  let x0 = input 0 in
  let x1 = input 1 in
  let x2 = input 2 in
  let x3 = input 3 in
  let x4 = input 4 in
  (* Real-channel pre-additions. *)
  let t4 = mk_add x1.re x4.re in
  let t7 = mk_add x2.re x3.re in
  let t8 = mk_add t4 t7 in
  let tt = mk_sub x2.re x3.re in
  let ta = mk_sub t4 t7 in
  let ts = mk_sub x1.re x4.re in
  (* Imag-channel pre-additions. *)
  let tm = mk_add x1.im x4.im in
  let tn = mk_add x2.im x3.im in
  let te = mk_sub x1.im x4.im in
  let tq = mk_sub tm tn in
  let th = mk_sub x2.im x3.im in
  let to_ = mk_add tm tn in
  let out = Array.make 5 czero in
  (* Output 0 — trivial sums. *)
  out.(0) <- cnum (mk_add x0.re t8) (mk_add x0.im to_);
  (* Real outputs X_1.re .. X_4.re *)
  let ti = mk_add te (mk_mul k_inv_phi th) in
  let tk = mk_sub th (mk_mul k_inv_phi te) in
  let t9 = mk_sub x0.re (mk_mul k_quarter t8) in
  let tb = mk_add t9 (mk_mul k_root5_4 ta) in
  let tj = mk_sub t9 (mk_mul k_root5_4 ta) in
  let x1_re = mk_add tb (mk_mul k_sin_2pi5_s ti) in
  let x4_re = mk_sub tb (mk_mul k_sin_2pi5_s ti) in
  let x2_re = mk_sub tj (mk_mul k_sin_2pi5_s tk) in
  let x3_re = mk_add tj (mk_mul k_sin_2pi5_s tk) in
  (* Imag outputs X_1.im .. X_4.im *)
  let tu = mk_add ts (mk_mul k_inv_phi tt) in
  let tw = mk_sub tt (mk_mul k_inv_phi ts) in
  let tp = mk_sub x0.im (mk_mul k_quarter to_) in
  let tr = mk_add tp (mk_mul k_root5_4 tq) in
  let tv = mk_sub tp (mk_mul k_root5_4 tq) in
  let x1_im = mk_sub tr (mk_mul k_sin_2pi5_s tu) in
  let x4_im = mk_add tr (mk_mul k_sin_2pi5_s tu) in
  let x2_im = mk_add tv (mk_mul k_sin_2pi5_s tw) in
  let x3_im = mk_sub tv (mk_mul k_sin_2pi5_s tw) in
  out.(1) <- cnum x1_re x1_im;
  out.(2) <- cnum x2_re x2_im;
  out.(3) <- cnum x3_re x3_im;
  out.(4) <- cnum x4_re x4_im;
  out

(* ============================================================
 * dft_winograd25 — hand-derived N=25 Winograd codelet.
 *
 * CT(5,5) decomposition with PLUS-OF-TIMES twiddles (instead of the
 * tan-factored form const_cmul uses).
 *
 * Plus-of-times: z = T * Y emitted as
 *   z.re = Sub(Mul(cr, Y.re), Mul(ci, Y.im))
 *   z.im = Add(Mul(cr, Y.im), Mul(ci, Y.re))
 *
 * Tan-factored (current dft_ct + const_cmul):
 *   z.re = Mul(cr, Sub(Y.re, Mul(tan, Y.im)))      [outer Mul opaque]
 *   z.im = Mul(cr, Add(Y.im, Mul(tan, Y.re)))
 *
 * The two forms compute the same value but have different IR shape.
 * Plus-of-times places all multiplications at leaf level, which lets
 * fma_lift fuse them into surrounding Add/Sub chains. Tan-factored
 * wraps the result in an outer Mul that has no FMA target.
 *
 * Trade-off: plus-of-times has 4 muls per cmul (both forms have 4
 * abstract muls, but plus-of-times keeps them leaf-level vs tan's outer).
 * Whether this nets a savings depends on how pass-2 W5 consumes the
 * twiddled values — specifically whether the W5 internal sums benefit
 * from leaf-level Mul shapes available for FMA fusion.
 *)
and dft_winograd25 ?(sign = `Fwd) (input_re : int -> expr) (input_im : int -> expr)
  : expr array * expr array
  =
  let two_pi = 8.0 *. atan 1.0 in
  let sgn =
    match sign with
    | `Fwd -> -1.0
    | `Bwd -> 1.0
  in
  let n = 25 in
  (* Pass 1: 5 W5s on input columns. *)
  let p1_re = Array.make_matrix 5 5 (Const 0.0) in
  let p1_im = Array.make_matrix 5 5 (Const 0.0) in
  for j_0 = 0 to 4 do
    let col_re k = input_re (j_0 + (5 * k)) in
    let col_im k = input_im (j_0 + (5 * k)) in
    let r, i = dft_winograd5 ~sign col_re col_im in
    for k = 0 to 4 do
      p1_re.(j_0).(k) <- r.(k);
      p1_im.(j_0).(k) <- i.(k)
    done
  done;
  (* Twiddle: plus-of-times form (NOT tan-factored).
   * For (j_0, k_0) with j_0=0 or k_0=0, twiddle is 1 — pass through. *)
  let z_re = Array.make_matrix 5 5 (Const 0.0) in
  let z_im = Array.make_matrix 5 5 (Const 0.0) in
  for j_0 = 0 to 4 do
    for k_0 = 0 to 4 do
      let theta = sgn *. two_pi *. float_of_int (j_0 * k_0) /. float_of_int n in
      let cr = cos theta in
      let ci = sin theta in
      let yr = p1_re.(j_0).(k_0) in
      let yi = p1_im.(j_0).(k_0) in
      if j_0 = 0 || k_0 = 0
      then (
        (* Identity twiddle. *)
        z_re.(j_0).(k_0) <- yr;
        z_im.(j_0).(k_0) <- yi)
      else (
        (* Plus-of-times: all 4 muls at leaf level. *)
        let cr_e = Const cr in
        let ci_e = Const ci in
        z_re.(j_0).(k_0) <- Sub (Mul (cr_e, yr), Mul (ci_e, yi));
        z_im.(j_0).(k_0) <- Add (Mul (cr_e, yi), Mul (ci_e, yr)))
    done
  done;
  (* Pass 2: 5 W5s on rows of z. *)
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k_0 = 0 to 4 do
    let row_re j_0 = z_re.(j_0).(k_0) in
    let row_im j_0 = z_im.(j_0).(k_0) in
    let r, i = dft_winograd5 ~sign row_re row_im in
    for k_1 = 0 to 4 do
      out_re.(k_0 + (5 * k_1)) <- r.(k_1);
      out_im.(k_0 + (5 * k_1)) <- i.(k_1)
    done
  done;
  out_re, out_im

and dft_winograd7 ?(sign = `Fwd) (input_re : int -> expr) (input_im : int -> expr)
  : expr array * expr array
  =
  (* Winograd 7-point DFT — Rader-style decomposition exploiting the
   * multiplicative-group structure of (Z/7Z)*. The cyclic-convolution
   * subproblem factors via Winograd's small-convolution algorithms,
   * yielding 18 add/sub + 42 fma = 60 ops total. Matches FFTW's
   * gen_notw -fma -n 7 codelet exactly (vs 66 ops from our generic
   * conjugate-pair Direct path).
   *
   * Six derived constants come out of the Winograd derivation. Of these,
   * KP_974927912 = sin(4π/7) is the only one carrying cross-channel sin
   * coupling — it flips sign for Bwd. The other five are sign-invariant
   * (rational combinations of the cos values).
   *
   * Algorithm mirrors FFTW's emitted ordering (the algebra is otherwise
   * the same): real and imag pre-additions, output 0, then three output
   * pairs (1,6), (2,5), (3,4). Within each pair, lower-indexed output is
   * FMA, higher-indexed is FNMS — the sign convention encoded structurally
   * for Fwd, flipped by KP_974927912's sign for Bwd.
   *
   * Note the imag-channel pair-diffs use LOW-HIGH (Tj = x_1.im - x_6.im,
   * etc.) while real-channel uses HIGH-LOW (TI = x_6.re - x_1.re). This
   * asymmetry handles the sign of sin in the DFT formula and is required
   * for correctness — flipping it would invert the imag outputs. *)
  let kp_356895867 = Const 0.356895867892209443894399510021300583399127187 in
  let kp_554958132 = Const 0.554958132087371191422194871006410481067288862 in
  let kp_801937735 = Const 0.801937735804838252472204639014890102331838324 in
  let kp_692021471 = Const 0.692021471630095869627814897002069140197260599 in
  let kp_900968867 = Const 0.900968867902419126236102319507445051165919162 in
  let s_sign =
    match sign with
    | `Fwd -> 1.0
    | `Bwd -> -1.0
  in
  let kp_974927912_s =
    Const (s_sign *. 0.974927912181823607018131682993931217232785801)
  in
  (* Real-channel pre-additions *)
  let x0r = input_re 0 in
  let x1r = input_re 1 in
  let x2r = input_re 2 in
  let x3r = input_re 3 in
  let x4r = input_re 4 in
  let x5r = input_re 5 in
  let x6r = input_re 6 in
  let t_4 = Add (x1r, x6r) in
  (* T4: pair sum  (x1+x6) *)
  let t_i = Sub (x6r, x1r) in
  (* TI: pair diff (x6-x1)  [HIGH-LOW] *)
  let t_7 = Add (x2r, x5r) in
  let t_h = Sub (x5r, x2r) in
  let t_a = Add (x3r, x4r) in
  let t_g = Sub (x4r, x3r) in
  let t_b = Sub (t_4, Mul (kp_356895867, t_7)) in
  (* Tb = T4 - K356·T7 *)
  let t_p = Sub (t_a, Mul (kp_356895867, t_4)) in
  let t_u = Sub (t_7, Mul (kp_356895867, t_a)) in
  let t_tt = Add (t_i, Mul (kp_554958132, t_g)) in
  (* TT = TI + K555·TG *)
  let t_o2 = Add (t_g, Mul (kp_554958132, t_h)) in
  (* TO = TG + K555·TH *)
  let t_jj = Sub (t_h, Mul (kp_554958132, t_i)) in
  (* TJ = TH - K555·TI *)
  (* Imag-channel pre-additions  (pair diffs are LOW-HIGH here) *)
  let x0i = input_im 0 in
  let x1i = input_im 1 in
  let x2i = input_im 2 in
  let x3i = input_im 3 in
  let x4i = input_im 4 in
  let x5i = input_im 5 in
  let x6i = input_im 6 in
  let t_aA = Add (x1i, x6i) in
  (* TA: pair sum *)
  let t_jI = Sub (x1i, x6i) in
  (* Tj: pair diff (LOW-HIGH) *)
  let t_bB = Add (x2i, x5i) in
  let t_gI = Sub (x2i, x5i) in
  let t_cC = Add (x3i, x4i) in
  let t_mM = Sub (x3i, x4i) in
  let t_qQ = Sub (t_aA, Mul (kp_356895867, t_bB)) in
  (* TQ = TA - K356·TB *)
  let t_lL = Sub (t_cC, Mul (kp_356895867, t_aA)) in
  let t_dD = Sub (t_bB, Mul (kp_356895867, t_cC)) in
  let t_nN = Add (t_jI, Mul (kp_554958132, t_mM)) in
  (* Tn = Tj + K555·Tm *)
  let t_sS = Add (t_mM, Mul (kp_554958132, t_gI)) in
  (* Ts = Tm + K555·Tg *)
  let t_xX = Sub (t_gI, Mul (kp_554958132, t_jI)) in
  (* Tx = Tg - K555·Tj *)
  let out_re = Array.make 7 (Const 0.0) in
  let out_im = Array.make 7 (Const 0.0) in
  (* Output 0 — chained sums, 3 adds each channel *)
  out_re.(0) <- Add (Add (Add (x0r, t_4), t_7), t_a);
  out_im.(0) <- Add (Add (Add (x0i, t_aA), t_bB), t_cC);
  (* Pair (1, 6) *)
  let to_ = Add (t_gI, Mul (kp_801937735, t_nN)) in
  (* To = Tg + K801·Tn *)
  let t_c = Sub (t_a, Mul (kp_692021471, t_b)) in
  (* Tc = Ta - K692·Tb *)
  let t_d = Sub (x0r, Mul (kp_900968867, t_c)) in
  (* Td = x_0 - K900·Tc *)
  out_re.(1) <- Add (t_d, Mul (kp_974927912_s, to_));
  out_re.(6) <- Sub (t_d, Mul (kp_974927912_s, to_));
  let t_u2 = Add (t_h, Mul (kp_801937735, t_tt)) in
  let t_r2 = Sub (t_cC, Mul (kp_692021471, t_qQ)) in
  let t_s2 = Sub (x0i, Mul (kp_900968867, t_r2)) in
  out_im.(1) <- Add (t_s2, Mul (kp_974927912_s, t_u2));
  out_im.(6) <- Sub (t_s2, Mul (kp_974927912_s, t_u2));
  (* Pair (2, 5) *)
  let t_tT = Sub (t_jI, Mul (kp_801937735, t_sS)) in
  (* Tt = Tj - K801·Ts *)
  let t_q = Sub (t_7, Mul (kp_692021471, t_p)) in
  let t_r = Sub (x0r, Mul (kp_900968867, t_q)) in
  out_re.(2) <- Add (t_r, Mul (kp_974927912_s, t_tT));
  out_re.(5) <- Sub (t_r, Mul (kp_974927912_s, t_tT));
  let t_pP = Sub (t_i, Mul (kp_801937735, t_o2)) in
  (* TP = TI - K801·TO *)
  let t_mM2 = Sub (t_bB, Mul (kp_692021471, t_lL)) in
  let t_nN2 = Sub (x0i, Mul (kp_900968867, t_mM2)) in
  out_im.(2) <- Add (t_nN2, Mul (kp_974927912_s, t_pP));
  out_im.(5) <- Sub (t_nN2, Mul (kp_974927912_s, t_pP));
  (* Pair (3, 4) *)
  let t_y = Sub (t_mM, Mul (kp_801937735, t_xX)) in
  let t_v = Sub (t_4, Mul (kp_692021471, t_u)) in
  let t_w = Sub (x0r, Mul (kp_900968867, t_v)) in
  out_re.(3) <- Add (t_w, Mul (kp_974927912_s, t_y));
  out_re.(4) <- Sub (t_w, Mul (kp_974927912_s, t_y));
  let t_kK = Sub (t_g, Mul (kp_801937735, t_jj)) in
  let t_eE = Sub (t_aA, Mul (kp_692021471, t_dD)) in
  let t_fF = Sub (x0i, Mul (kp_900968867, t_eE)) in
  out_im.(3) <- Add (t_fF, Mul (kp_974927912_s, t_kK));
  out_im.(4) <- Sub (t_fF, Mul (kp_974927912_s, t_kK));
  out_re, out_im

and dft_ct
      ?(sign = `Fwd)
      (n1 : int)
      (n2 : int)
      (input_re : int -> expr)
      (input_im : int -> expr)
  : expr array * expr array
  =
  let n = n1 * n2 in
  let pi = 4.0 *. atan 1.0 in
  let sgn =
    match sign with
    | `Fwd -> -1.0
    | `Bwd -> 1.0
  in
  (* PASS 1: N1 sub-FFTs of size N2.
   * For each n1_idx in [0, N1), compute DFT-N2 on inputs at
   *   x[n1_idx], x[n1_idx + N1], x[n1_idx + 2·N1], ...
   * pass1[n1_idx][k2] = DFT-N2 result at output bin k2. *)
  let pass1_re = Array.make_matrix n1 n2 (Const 0.0) in
  let pass1_im = Array.make_matrix n1 n2 (Const 0.0) in
  for n1_idx = 0 to n1 - 1 do
    let inner_input_re k2 = input_re (n1_idx + (k2 * n1)) in
    let inner_input_im k2 = input_im (n1_idx + (k2 * n1)) in
    let r, i = dft ~sign n2 inner_input_re inner_input_im in
    for k2 = 0 to n2 - 1 do
      pass1_re.(n1_idx).(k2) <- r.(k2);
      pass1_im.(n1_idx).(k2) <- i.(k2)
    done
  done;
  (* INTERNAL TWIDDLES: multiply pass1[n1_idx][k2] by ω_N^{n1_idx·k2}.
   * For n1_idx=0 or k2=0, the twiddle is 1 and const_cmul folds away.
   * Other (n1_idx, k2) pairs introduce non-trivial cmul nodes. *)
  let twiddled_re = Array.make_matrix n1 n2 (Const 0.0) in
  let twiddled_im = Array.make_matrix n1 n2 (Const 0.0) in
  for n1_idx = 0 to n1 - 1 do
    for k2 = 0 to n2 - 1 do
      let theta = sgn *. 2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
      let cr = cos theta in
      let ci = sin theta in
      let tr, ti = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
      twiddled_re.(n1_idx).(k2) <- tr;
      twiddled_im.(n1_idx).(k2) <- ti
    done
  done;
  (* PASS 2: N2 sub-FFTs of size N1.
   * For each k2 in [0, N2), compute DFT-N1 on the column
   *   twiddled[0..N1-1][k2]
   * Output: X[k1·N2 + k2] = pass2_result[k1]. *)
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k2 = 0 to n2 - 1 do
    let outer_input_re n1_idx = twiddled_re.(n1_idx).(k2) in
    let outer_input_im n1_idx = twiddled_im.(n1_idx).(k2) in
    let r, i = dft ~sign n1 outer_input_re outer_input_im in
    for k1 = 0 to n1 - 1 do
      out_re.((k1 * n2) + k2) <- r.(k1);
      out_im.((k1 * n2) + k2) <- i.(k1)
    done
  done;
  out_re, out_im
;;
