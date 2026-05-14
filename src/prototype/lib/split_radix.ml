(* split_radix.ml — split-radix decomposition for pow2 N ≥ 8.
 *
 * Asymmetric N → DFT(N/2) over x[2n] + 2 × DFT(N/4) over x[4n+1] / x[4n+3],
 * with ~33% fewer multiplications than radix-2 CT. Refs: Yavne 1968,
 * Sorensen-Heideman-Burrus 1986. Conjugate-pair refinement (Johnson-Frigo
 * 2007) is future work.
 *
 * Routed via VFFT_SPLIT_RADIX=1 env var; otherwise pow2 sizes use the CT
 * factorizations in dft.ml.
 *
 *
 * ALGEBRA
 * -------
 *
 * Let N = 4M. For k in [0, M):
 *
 *   T1[k]  = W_N^{k}  · O1[k]                  (twiddled O1)
 *   T3[k]  = W_N^{3k} · O3[k]                  (twiddled O3)
 *   U[k]   = T1[k] + T3[k]                     (shared sum)
 *   V[k]   = T1[k] - T3[k]                     (shared diff)
 *
 * Then the four output groups are:
 *
 *   X[k]       = E[k]    + U[k]                              (no factor)
 *   X[k+M]     = E[k+M]  + (-j) · V[k]                       (× -j for fwd)
 *   X[k+2M]    = E[k]    - U[k]                              (negate U)
 *   X[k+3M]    = E[k+M]  + (+j) · V[k]                       (× +j for fwd)
 *
 * Note that E is periodic with period N/2 (E[k+N/2] = E[k]), and similarly
 * O1, O3 are periodic with period N/4. We use only the first M=N/4 values
 * of each O array; for E, we use the first N/2 values directly indexed.
 *
 * Multiplying by -j in real/imag form:
 *   -j · (a + bi) = b - ai  →  (re, im) = (b, -a) = (V_im, -V_re)
 *
 * So for FORWARD DFT (θ has negative sign convention):
 *   X[k+M].re   = E[k+M].re + V[k].im
 *   X[k+M].im   = E[k+M].im - V[k].re
 *   X[k+3M].re  = E[k+M].re - V[k].im
 *   X[k+3M].im  = E[k+M].im + V[k].re
 *
 * For BACKWARD (θ has positive sign), the j and -j swap, flipping signs:
 *   X[k+M].re   = E[k+M].re - V[k].im
 *   X[k+M].im   = E[k+M].im + V[k].re
 *   X[k+3M].re  = E[k+M].re + V[k].im
 *   X[k+3M].im  = E[k+M].im - V[k].re
 *
 *
 * RECURSION AND BASE CASES
 * ------------------------
 *
 * SR is gated to N ≥ 8 (power-of-two). The recursion bottoms via:
 *
 *   N=8  → SR splits into E (DFT-4) + O1 (DFT-2) + O3 (DFT-2)
 *   N=4  → not SR; existing CT(2,2) path handles this
 *   N=2  → trivial; Direct
 *
 * For N=16, 32, 64, ..., SR recurses through itself (E is a smaller SR)
 * until N reaches 8, at which point the sub-DFTs land at sizes 4 and 2
 * which are not SR. The picker handles the dispatch at each level.
 *
 *
 * INTEGRATION WITH EXISTING PIPELINE
 * ----------------------------------
 *
 * The construction here builds raw expr trees using Add/Sub/Mul/Const just
 * like dft_ct. Algsimp runs once at the top level on the fully-expanded
 * tree, so all the existing smart constructors (mk_add, mk_sub, mk_mul,
 * mk_neg) apply to the SR-constructed IR exactly as they do for CT.
 *
 * The Mul-by-twiddle pattern uses const_cmul (same as dft_ct's internal
 * twiddle stage), so the resulting Sub(Mul(_,c), Mul(_,c)) and
 * Add(Mul(_,c), Mul(_,c)) patterns match what algsimp already handles.
 *
 * The lift_sub_neg_mul pass (doc 30) will fire on whatever Sub(Neg(Mul), c)
 * patterns SR's DAG produces, just as it does for the CT family.
 *
 *
 * CALLER CONTRACT
 * ---------------
 *
 * dft_split_radix takes a recursive callback `dft_rec` rather than calling
 * dft directly. This is the standard OCaml pattern for cross-module mutual
 * recursion: dft.ml provides `dft` as the callback, and SR uses it to
 * recurse on the N/2 and N/4 sub-DFTs (which may themselves dispatch back
 * to SR, or to CT, or to Direct, based on the picker).
 *
 * Signature mirrors dft_direct, dft_ct in dft.ml:
 *   Inputs are functions int → expr giving the k-th input's real / imag
 *   Output is (re_outputs, im_outputs) as arrays indexed by output bin k
 *)

open Expr

(* Complex multiplication by a constant (cr, ci):
 *   out = (xr + i·xi) · (cr + i·ci) = (xr·cr - xi·ci) + i·(xr·ci + xi·cr)
 *
 * Duplicated locally rather than imported from Dft to keep this module
 * independent of dft.ml's exports. Identical formula. *)
let const_cmul (xr : expr) (xi : expr) (cr : float) (ci : float) : expr * expr =
  let cr_e = Const cr in
  let ci_e = Const ci in
  let out_re = Sub (Mul (xr, cr_e), Mul (xi, ci_e)) in
  let out_im = Add (Mul (xr, ci_e), Mul (xi, cr_e)) in
  (out_re, out_im)

(* Type alias for the recursive callback. Matches dft.ml's `dft` signature
 * with sign as an explicit (non-optional) parameter to keep the callsite
 * unambiguous when passed across module boundaries. *)
type dft_callback =
  sign:[`Fwd | `Bwd] -> int ->
  (int -> expr) -> (int -> expr) ->
  expr array * expr array

(* dft_split_radix: top-level split-radix construction.
 *
 *   dft_rec   — recursive callback into the main dft dispatcher. SR uses
 *               this to compute E (size N/2), O1 (size N/4), O3 (size N/4),
 *               each of which may dispatch to SR, CT, or Direct depending
 *               on its size.
 *   sign      — Fwd uses θ = -2πk/N (DFT); Bwd uses θ = +2πk/N (IDFT).
 *   n         — transform size. Must be a power of 2 and ≥ 8.
 *   input_re,
 *   input_im  — input element accessors.
 *
 * Returns: (re_outputs, im_outputs) as arrays of expr trees indexed by k.
 *)
let dft_split_radix
    ~(dft_rec : dft_callback)
    ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
  : expr array * expr array =
  (* Preconditions: pow2 AND >= 8.
   * N=4 should go to CT(2,2), N=2 to Direct, smaller is invalid. *)
  assert (n >= 8);
  assert (n land (n - 1) = 0);

  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in
  let m = n / 4 in
  let n2 = n / 2 in

  (* === RECURSIVE SUB-DFTS ===
   *
   * E:  DFT of size N/2 over inputs x[0], x[2], x[4], ..., x[N-2]
   * O1: DFT of size N/4 over inputs x[1], x[5], x[9], ..., x[N-3]
   * O3: DFT of size N/4 over inputs x[3], x[7], x[11], ..., x[N-1]
   *)
  let even_re k = input_re (2 * k) in
  let even_im k = input_im (2 * k) in
  let e_re, e_im = dft_rec ~sign n2 even_re even_im in

  let o1_in_re k = input_re (4 * k + 1) in
  let o1_in_im k = input_im (4 * k + 1) in
  let o1_re, o1_im = dft_rec ~sign m o1_in_re o1_in_im in

  let o3_in_re k = input_re (4 * k + 3) in
  let o3_in_im k = input_im (4 * k + 3) in
  let o3_re, o3_im = dft_rec ~sign m o3_in_re o3_in_im in

  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in

  (* === COMBINE STAGE ===
   *
   * For each k in [0, M):
   *   Compute T1[k] = W_N^k    · O1[k]     (twiddle complex multiply)
   *           T3[k] = W_N^{3k} · O3[k]
   *           U[k]  = T1[k] + T3[k]
   *           V[k]  = T1[k] - T3[k]
   *   Then fill X[k], X[k+M], X[k+2M], X[k+3M] using U, V, and E values.
   *
   * Special cases for k=0:
   *   W_N^0  = 1, so T1[0] = O1[0] (const_cmul folds the trivial twiddle)
   *   W_N^0  = 1, so T3[0] = O3[0]
   * Both fold automatically via mk_mul's Const(1)/Const(0) peephole; no
   * special handling needed in this construction.
   *)
  for k = 0 to m - 1 do
    (* Twiddle angles. The sgn factor is - for Fwd, + for Bwd, mirroring
     * dft_ct's convention. cos/sin of these directly give the correct
     * complex twiddle for the requested direction. *)
    let theta1 = sgn *. 2.0 *. pi *. float_of_int k /. float_of_int n in
    let theta3 = sgn *. 2.0 *. pi *. float_of_int (3 * k) /. float_of_int n in
    let t1_re, t1_im = const_cmul o1_re.(k) o1_im.(k) (cos theta1) (sin theta1) in
    let t3_re, t3_im = const_cmul o3_re.(k) o3_im.(k) (cos theta3) (sin theta3) in

    (* Shared sum and difference. These are the "33% savings" — they get
     * reused across the four output groups instead of computing
     * W·O1 + W·O3 separately for each group. *)
    let u_re = Add (t1_re, t3_re) in
    let u_im = Add (t1_im, t3_im) in
    let v_re = Sub (t1_re, t3_re) in
    let v_im = Sub (t1_im, t3_im) in

    (* X[k]   = E[k] + U *)
    out_re.(k) <- Add (e_re.(k), u_re);
    out_im.(k) <- Add (e_im.(k), u_im);

    (* X[k+2M] = E[k] - U  (E is periodic mod N/2, so E[k+2M] = E[k]) *)
    out_re.(k + 2 * m) <- Sub (e_re.(k), u_re);
    out_im.(k + 2 * m) <- Sub (e_im.(k), u_im);

    (* X[k+M]  = E[k+M] + (factor_j) · V
     * X[k+3M] = E[k+M] - (factor_j) · V
     *
     * factor_j is -j for Fwd, +j for Bwd. In real/imag form:
     *   For Fwd:  -j · V = ( V_im, -V_re), so:
     *               X[k+M].re  = E[k+M].re + V_im
     *               X[k+M].im  = E[k+M].im - V_re
     *               X[k+3M].re = E[k+M].re - V_im
     *               X[k+3M].im = E[k+M].im + V_re
     *   For Bwd:  +j · V = (-V_im,  V_re), so:
     *               X[k+M].re  = E[k+M].re - V_im
     *               X[k+M].im  = E[k+M].im + V_re
     *               X[k+3M].re = E[k+M].re + V_im
     *               X[k+3M].im = E[k+M].im - V_re
     *)
    (match sign with
     | `Fwd ->
       out_re.(k + m)     <- Add (e_re.(k + m), v_im);
       out_im.(k + m)     <- Sub (e_im.(k + m), v_re);
       out_re.(k + 3 * m) <- Sub (e_re.(k + m), v_im);
       out_im.(k + 3 * m) <- Add (e_im.(k + m), v_re)
     | `Bwd ->
       out_re.(k + m)     <- Sub (e_re.(k + m), v_im);
       out_im.(k + m)     <- Add (e_im.(k + m), v_re);
       out_re.(k + 3 * m) <- Add (e_re.(k + m), v_im);
       out_im.(k + 3 * m) <- Sub (e_im.(k + m), v_re))
  done;

  (out_re, out_im)
