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

(* === C2R BACKWARD: dft_c2r_direct ===
 *
 * Inverse of dft_r2c_direct. Given Hermitian-packed input X[0..N/2]
 * (complex), produce N real outputs x[0..N-1].
 *
 * Algorithm (mirror of forward):
 *   1. Pre-process butterfly (reverse the post-process from forward):
 *      Z[0].re = X[0] + X[N/2]
 *      Z[0].im = X[0] - X[N/2]
 *      For k = 1..N/2-1, m = N/2-k:
 *        S = X[k] + conj(X[m])        (sum)
 *        D = X[k] - conj(X[m])        (diff)
 *        Z[k] = S + i*conj(W_N^k)*D
 *      Then Z[m] follows from Hermitian: Z[m] = conj(Z[k]_at_m_index)
 *      computed via the same formula with k↔m, so we compute both
 *      Z[k] and Z[m] in the loop for k = 1..(N/2-1)/2, then handle
 *      the self-pair (k = m, only when N/2 is even and k = N/4) once.
 *   2. N/2-point complex IFFT (backward) of Z → z (un-normalized).
 *   3. Unpack pairs: x[2n] = 2*Re(z[n]),  x[2n+1] = 2*Im(z[n])
 *
 * Normalization: c2r(r2c(x))[n] = N * x[n].
 *   Forward applies factor 1/2 in E, O (in dft_r2c_direct).
 *   c2c bwd multiplies by N/2 (un-normalized).
 *   c2r pre-process here uses NO /2 (cancels forward's 1/2).
 *   Unpack applies *2.
 *   Total: 1 * (N/2) * 2 = N. ✓
 *)
let dft_c2r_direct (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
    : expr array =
  assert (n >= 2 && n mod 2 = 0);
  let half = n / 2 in
  let pi = 4.0 *. atan 1.0 in

  (* Step 1: build Z[0..N/2-1] from X[0..N/2].
   * X[0] and X[N/2] are real (input im is ignored for those slots).
   *)
  let z_re = Array.make half (Const 0.0) in
  let z_im = Array.make half (Const 0.0) in
  let x0_re   = input_re 0 in
  let xnyq_re = input_re half in
  z_re.(0) <- Add (x0_re, xnyq_re);   (* Z[0].re = X[0] + X[N/2] *)
  z_im.(0) <- Sub (x0_re, xnyq_re);   (* Z[0].im = X[0] - X[N/2] *)

  for k = 1 to half - 1 do
    let m = half - k in
    let xk_re = input_re k    in let xk_im = input_im k    in
    let xm_re = input_re m    in let xm_im = input_im m    in
    (* conj(X[m]) = (xm_re, -xm_im)
     * S = X[k] + conj(X[m]) = (xk_re + xm_re, xk_im - xm_im)
     * D = X[k] - conj(X[m]) = (xk_re - xm_re, xk_im + xm_im)
     *)
    let s_re = Add (xk_re, xm_re) in
    let s_im = Sub (xk_im, xm_im) in
    let d_re = Sub (xk_re, xm_re) in
    let d_im = Add (xk_im, xm_im) in
    (* W = W_N^k for forward sign = (cos(2πk/N), -sin(2πk/N))
     * (using fwd sgn = -1 in forward, so W_fwd has angle -2πk/N).
     * conj(W) = (cos(2πk/N), +sin(2πk/N))
     *)
    let theta = 2.0 *. pi *. float_of_int k /. float_of_int n in
    let cwr = Const (cos theta) in        (* conj(W).re *)
    let cwi = Const (sin theta) in        (* conj(W).im *)
    (* conj(W)*D = (cwr*d_re - cwi*d_im, cwr*d_im + cwi*d_re)
     * i * conj(W) * D = (-(cwr*d_im + cwi*d_re), cwr*d_re - cwi*d_im)
     * Z[k] = S + i*conj(W)*D:
     *   Z[k].re = s_re - cwr*d_im - cwi*d_re
     *   Z[k].im = s_im + cwr*d_re - cwi*d_im
     *)
    let z_k_re =
      Sub (Sub (s_re, Mul (cwr, d_im)), Mul (cwi, d_re)) in
    let z_k_im =
      Sub (Add (s_im, Mul (cwr, d_re)), Mul (cwi, d_im)) in
    z_re.(k) <- z_k_re;
    z_im.(k) <- z_k_im
  done;

  (* Step 2: N/2-point complex BACKWARD c2c of Z → z.
   * Use the c2c dispatcher with sign=`Bwd. *)
  let z_in_re k = z_re.(k) in
  let z_in_im k = z_im.(k) in
  let z_out_re, z_out_im = Dft.dft ~sign:`Bwd half z_in_re z_in_im in

  (* Step 3: unpack. With the "no /2 in pre-process" convention, the
   * c2c bwd output is already N*z, so unpack is just Re/Im extraction
   * with no scaling. Total normalization: c2r(r2c(x)) = N*x.
   *
   * Why no /2 in pre-process: forward's E,O each carry a /2 factor.
   * Computing Z_recovered = S + i*conj(W)*D (without /2) gives 2*Z.
   * The c2c bwd multiplies by N/2 (un-normalized), yielding 2*(N/2)*z = N*z.
   * Direct unpack (no *2) gives N*x. Cleaner than the alternative
   * (/2 pre-process + *2 unpack) because it saves N multiplies.
   *)
  let out = Array.make n (Const 0.0) in
  for nn = 0 to half - 1 do
    out.(2 * nn)     <- z_out_re.(nn);
    out.(2 * nn + 1) <- z_out_im.(nn)
  done;
  out

(* Assignment-list wrapper for c2r backward.
 * Inputs: Hermitian-packed X[0..N/2] at slots Input(k, true) for
 * real parts and Input(k, false) for imag parts, k in [0..N/2].
 * Inputs at slots [N/2+1..N-1] are not referenced (Hermitian implicit).
 * X[0] and X[N/2] are mathematically real; the codelet does not load
 * their imag parts (algsimp folds them away if it did).
 *
 * Outputs: x[0..N-1] real, written to Output(n, true). The imaginary
 * slots Output(n, false) are NOT written — c2r's output is purely
 * real, so emitting `Const 0.0` stores to im slots would waste N
 * stores per call. The caller allocates an out_im buffer for ABI
 * compatibility but its contents are undefined after the call.
 *)
let dft_expand_c2r (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let out_re = dft_c2r_direct n input_re input_im in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out_re.(k)) :: !acc
  done;
  List.rev !acc

(* === R2C FIRST-STAGE: dft_r2c_first ===
 *
 * Cascade boundary codelet: pair-pack + first DIT stage of the inner
 * (N/2)-point c2c FFT, fused into one DAG.
 *
 * Use case: for large N r2c (N > 512 where monolithic codelets don't
 * fit registers), the transform is decomposed as:
 *
 *   N reals
 *      → [t1_r2c_first_R, called M times]    ← THIS codelet (no twiddles)
 *      → [twiddle multiplications]
 *      → [t1_R, called R times]              ← existing c2c twiddled codelet
 *      → [Hermitian-extraction butterfly]    ← separate pass for now
 *   N/2+1 complex
 *
 * where N/2 = R * M (e.g., N=128 → R=8, M=8; N=1024 → R=32, M=16).
 *
 * Math: the codelet does ONE R-point complex DFT on R complex inputs
 * formed by pair-packing 2R real inputs:
 *   z[k] = real_input[2k] + i * real_input[2k+1]  for k = 0..R-1
 *   Z = DFT_R(z)
 *
 * For batched execution (K parallel calls), the input layout is:
 *   real_input[(2k)*K + b]   = even-indexed reals for batch b
 *   real_input[(2k+1)*K + b] = odd-indexed reals for batch b
 * The codelet's outer K loop processes K batches in SIMD lanes.
 *
 * NOTE: this is the FIRST stage of the inner c2c (no incoming twiddles).
 * It's an n1-style codelet — no `tw_re`/`tw_im` consumed at the codelet
 * level, just like the existing n1 c2c codelets. Twiddles between this
 * stage and the next are applied externally (by the planner) between
 * codelet calls.
 *)
let dft_r2c_first ?(sign = `Fwd) (r : int)
    (input_re : int -> expr) : expr array * expr array =
  assert (r >= 2);
  (* Pair-pack: even reals → real part, odd reals → imag part. *)
  let z_in_re k = input_re (2 * k) in
  let z_in_im k = input_re (2 * k + 1) in
  Dft.dft ~sign r z_in_re z_in_im

(* Assignment-list wrapper for r2c first-stage codelet.
 * Inputs: 2R reals at slots Input(j, true) for j=0..2R-1.
 * Outputs: R complex at slots Output(k, true/false) for k=0..R-1.
 *
 * Calling convention reuses the c2c codelet ABI:
 *   void f(in_re, in_im, out_re, out_im, tw_re, tw_im, K)
 * The codelet reads 2R values from in_re per batch (in_re[0*K..2R*K]).
 * It does not load in_im, tw_re, or tw_im. The caller must allocate
 * those arrays for ABI compatibility but their contents are unused.
 *)
let dft_expand_r2c_first ?(sign = `Fwd) (r : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out_re, out_im = dft_r2c_first ~sign r input_re in
  let acc = ref [] in
  for k = r - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc
