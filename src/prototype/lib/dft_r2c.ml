(* dft_r2c.ml — math layer for real-to-complex and complex-to-real DFTs.
 *
 * Separate from dft.ml (complex-to-complex) because r2c/c2r are
 * fundamentally different transform types with their own decomposition
 * strategies. Mirrors FFTW's separation between gen_dft.ml (c2c) and
 * gen_r2cf.ml / gen_r2cb.ml / gen_hc2c.ml / gen_hc2hc.ml.
 *
 * As this module grows it will host:
 *   - dft_r2c_direct  (this file, forward r2c via pair-pack + post-process)
 *   - dft_c2r_direct  (TODO: backward c2r via pre-process + unpack)
 *   - dft_r2c_first   (TODO: first-stage cascade codelet, v1.1 t1_r2c_first_R)
 *   - dft_r2c_last    (TODO: last-stage cascade codelet, v1.1 t1_r2c_last_R)
 *   - dft_hc2hc       (TODO: middle stages, Hermitian-packed in & out)
 *
 * The inner complex sub-DFT is delegated to Dft.dft, so all the c2c
 * machinery (CT decomposition, conjugate-pair, recursive dispatch) is
 * reused unchanged. This module only handles the r2c-specific structure:
 * pair-packing at input, Hermitian-extraction butterfly at output.
 *)

open Expr

(* === R2C FORWARD: dft_r2c_direct ===
 *
 * Compute the N-point real-to-complex DFT directly via pair-packing +
 * post-process butterfly, fused into a single straight-line DAG.
 *
 * Algorithm (matching the executor's r2c.h, but FUSED at math layer):
 *   1. Pair-pack: z[k] = x[2k] + i*x[2k+1] for k = 0..N/2-1
 *      → N/2 complex values constructed by reinterpreting N reals.
 *      No actual work; the DAG just routes reals to the c2c sub-DFT.
 *   2. (N/2)-point complex DFT of z (uses Dft.dft, which dispatches to
 *      direct/conjugate-pair/Cooley-Tukey based on the picker).
 *   3. Post-process butterfly (Hermitian symmetry extraction):
 *      X[0]    = Re(Z[0]) + Im(Z[0])             (real)
 *      X[N/2]  = Re(Z[0]) - Im(Z[0])             (real)
 *      X[k]    = E[k] + W_N^k * (-i*O[k])        for k = 1..N/2-1
 *        E[k] = (Z[k] + conj(Z[N/2-k])) / 2
 *        O[k] = (Z[k] - conj(Z[N/2-k])) / 2
 *
 * The math layer produces ONE fused DAG. Algsimp/scheduler/emit handle
 * it unchanged — they don't know it's r2c, they just see a DAG with N
 * real Input loads and N/2+1 complex Output stores. The pack and
 * post-process are inlined into the natural arithmetic.
 *
 * Output: (re_array, im_array) of length N/2+1. The DC (k=0) and Nyquist
 * (k=N/2) outputs have im=0 by construction. The imaginary stores for
 * those slots end up as `Const 0.0`, which algsimp folds away — the
 * emitter sees no work for them.
 *)
let dft_r2c_direct ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) : expr array * expr array =
  assert (n >= 2 && n mod 2 = 0);
  let half = n / 2 in
  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1.0 | `Bwd -> +1.0 in

  (* Step 1+2: feed pair-packed reals into a half-point complex DFT.
   * z[k] = x[2k] + i*x[2k+1] is just routing — the c2c DFT reads
   * input_re from offset 2k and input_im from offset 2k+1. The
   * "pack" is purely an indexing trick at the math layer.
   *)
  let z_in_re k = input_re (2 * k) in
  let z_in_im k = input_re (2 * k + 1) in
  let z_re_arr, z_im_arr = Dft.dft half z_in_re z_in_im in

  (* Step 3: post-process butterfly.
   * Output arrays size N/2+1: slots [0..N/2]. *)
  let out_re = Array.make (half + 1) (Const 0.0) in
  let out_im = Array.make (half + 1) (Const 0.0) in

  (* DC and Nyquist: purely real, derived only from Z[0]. *)
  let z0_re = z_re_arr.(0) in
  let z0_im = z_im_arr.(0) in
  out_re.(0)    <- Add (z0_re, z0_im);     (* X[0]   = Re(Z[0]) + Im(Z[0]) *)
  out_im.(0)    <- Const 0.0;
  out_re.(half) <- Sub (z0_re, z0_im);     (* X[N/2] = Re(Z[0]) - Im(Z[0]) *)
  out_im.(half) <- Const 0.0;

  (* Pair butterflies: for k in 1..N/2-1, X[k] from (Z[k], Z[N/2-k]).
   * E[k] = (Z[k] + conj(Z[m])) / 2 where m = N/2 - k
   * O[k] = (Z[k] - conj(Z[m])) / 2
   * Substituting conj(Z[m]) = (Re(Z[m]), -Im(Z[m])):
   *   E.re = (Z[k].re + Z[m].re) / 2
   *   E.im = (Z[k].im - Z[m].im) / 2
   *   O.re = (Z[k].re - Z[m].re) / 2
   *   O.im = (Z[k].im + Z[m].im) / 2
   * Then (-i * O) = (O.im, -O.re), and
   *   W_N^k * (O.im, -O.re):
   *     real = W.re * O.im - W.im * (-O.re) = W.re*O.im + W.im*O.re
   *     imag = W.re * (-O.re) + W.im * O.im = -W.re*O.re + W.im*O.im
   * X[k] = E + W*(-i*O):
   *   X.re = E.re + W.re*O.im + W.im*O.re
   *   X.im = E.im - W.re*O.re + W.im*O.im
   *
   * We use sgn for the rotation direction (forward vs backward).
   *)
  let half_const = Const 0.5 in
  for k = 1 to half - 1 do
    let m = half - k in
    let zk_re = z_re_arr.(k) in let zk_im = z_im_arr.(k) in
    let zm_re = z_re_arr.(m) in let zm_im = z_im_arr.(m) in
    let e_re = Mul (Add (zk_re, zm_re), half_const) in
    let e_im = Mul (Sub (zk_im, zm_im), half_const) in
    let o_re = Mul (Sub (zk_re, zm_re), half_const) in
    let o_im = Mul (Add (zk_im, zm_im), half_const) in
    let theta = sgn *. 2.0 *. pi *. float_of_int k /. float_of_int n in
    let wr = Const (cos theta) in
    let wi = Const (sin theta) in
    let x_re = Add (e_re, Add (Mul (wr, o_im), Mul (wi, o_re))) in
    let x_im = Add (e_im, Sub (Mul (wi, o_im), Mul (wr, o_re))) in
    out_re.(k) <- x_re;
    out_im.(k) <- x_im
  done;

  (out_re, out_im)

(* Assignment-list wrapper for r2c forward.
 * Outputs at slots [0..N/2]. Slots [N/2+1..N-1] are not written by the
 * codelet — the caller is responsible for understanding Hermitian
 * packing (those slots are conjugates of slots [1..N/2-1]).
 *
 * Input loads only reference Input(k, true) for k in 0..N-1. The codelet
 * signature still has in_im for ABI compatibility with the c2c emit
 * machinery, but no expressions in the DAG load from it (gcc will
 * optimize the parameter away or just leave it unused).
 *)
let dft_expand_r2c ?(sign = `Fwd) (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let half = n / 2 in
  let out_re, out_im = dft_r2c_direct ~sign n input_re in
  let acc = ref [] in
  for k = half downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc
