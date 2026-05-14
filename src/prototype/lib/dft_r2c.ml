(* dft_r2c.ml — math layer for real-valued transforms.
 *
 * Each transform has its own algorithm; the inner complex sub-DFT is
 * delegated to Dft.dft so all c2c machinery (CT decomposition,
 * conjugate-pair, recursion) is reused unchanged.
 *
 *   r2c / c2r     pair-pack + Hermitian-extraction butterfly
 *   rdft          Hermitian-compact c2c via N-point r2c
 *   hc2hc / hc2c  middle-stage codelets, Hermitian-packed in & out
 *   dct2 / dct3   Makhoul reduction (N-point rdft + butterfly)
 *   dct4          Lee 1984 (N/2-point c2c IFFT + pre/post twiddle)
 *   dst2 / dst3   DCT-II/III wrappers with sign-flip + reversal
 *   dht           N-point rdft + butterfly H[k] = Re(X[k]) ± Im(X[k])
 *
 * CLI flags: --r2c / --c2r / --rdft / --hc2hc / --hc2c / --dct{2,3,4} /
 * --dst{2,3} / --dht. See doc 55 for the trig family bench results. *)

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

(* === RDFT: real-input DFT (FFTW-style, Hermitian-compact output) ===
 *
 * Port of FFTW's Trig.rdft (genfft/trig.ml:26–27):
 *     let rdft sign n input = Fft.dft sign n (Complex.real @@ input)
 *
 * Idea: a real-input DFT is just c2c with im(input) = 0. algsimp folds
 * out the zero-multiplications, leaving the streamlined real DFT for
 * free. Output is Hermitian-symmetric (X[N-k] = conj(X[k])), so only
 * the unique half [0..N/2] is returned.
 *
 * Inputs:
 *   n          — transform size (real input length)
 *   input_re k — Expr tree for the k-th real input (k = 0..n-1)
 *   ?sign      — `Fwd or `Bwd
 *
 * Returns: (out_re, out_im) arrays of length n/2+1.
 *   - out_re/out_im[0]    : DC bin (out_im[0]   = 0 by symmetry)
 *   - out_re/out_im[n/2]  : Nyquist (out_im[n/2] = 0; only when n even)
 *   - out_re/out_im[k]    : kth bin, 1 <= k <= n/2 - 1
 *
 * The caller knows the output is Hermitian-compact and must store only
 * positions [0..n/2]; positions [n/2+1..n-1] = conj(positions [n/2-1..1])
 * by construction (no work to emit them).
 *)
let dft_rdft ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) : expr array * expr array =
  assert (n >= 2);
  let zero_im _ = Const 0.0 in
  let full_re, full_im = Dft.dft ~sign n input_re zero_im in
  let half = n / 2 in
  let out_re = Array.make (half + 1) (Const 0.0) in
  let out_im = Array.make (half + 1) (Const 0.0) in
  for k = 0 to half do
    out_re.(k) <- full_re.(k);
    out_im.(k) <- full_im.(k)
  done;
  (out_re, out_im)

(* Assignment-list wrapper for rdft.
 * Inputs at Input(k, true) for k = 0..n-1.
 * Outputs at Output(k, true)/Output(k, false) for k = 0..n/2.
 * Positions k > n/2 are conjugates of the stored values; not emitted. *)
let dft_expand_rdft ?(sign = `Fwd) (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let half = n / 2 in
  let out_re, out_im = dft_rdft ~sign n input_re in
  let acc = ref [] in
  for k = half downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc

(* === DCT-II via Makhoul's reduction ===
 *
 * Port of the production algorithm from src/core/dct.h. Convention is
 * FFTW's REDFT10:
 *   Y[k] = 2 · Σ_{n=0..N-1} x[n] · cos(π·k·(2n+1)/(2N))
 *
 * Makhoul's pipeline (fused into one DAG so algsimp can collapse it):
 *   1. Permute reals into buf such that
 *        buf[0]      = x[0]
 *        buf[i]      = x[2i]      for i = 1..N/2-1
 *        buf[N-i]    = x[2i-1]
 *        buf[N/2]    = x[N-1]     (only when N even)
 *   2. N-point R2C of buf → Z[0..N/2] complex
 *   3. Post-process butterfly:
 *        Y[0]    = 2·Re(Z[0])
 *        Y[N/2]  = 2·cos(π/4)·Re(Z[N/2])           (N even)
 *        For i = 1..N/2-1, with a = 2·Re(Z[i]), b = 2·Im(Z[i]):
 *          wa = cos(π·i/(2N)),  wb = sin(π·i/(2N))
 *          Y[i]    = wa·a + wb·b
 *          Y[N-i]  = wb·a - wa·b
 *
 * In this DAG-fused form, the permutation is just index manipulation
 * (zero cost), rdft gets its inputs through the permuted indices, and
 * the post-process butterfly is regular arithmetic. algsimp folds and
 * schedules the whole thing as one straight-line codelet.
 *
 * Reference: J. Makhoul, IEEE Trans. ASSP-28 (1), 27–34 (1980).
 *)
let dft_dct2 (n : int) (input_re : int -> expr) : expr array =
  assert (n >= 2 && n mod 2 = 0);
  let half = n / 2 in
  let pi = 4.0 *. atan 1.0 in
  (* Step 1: build the permuted input as a virtual index function.
   * No actual memory permutation — algsimp sees through it. *)
  let buf k =
    if k = 0 then input_re 0
    else if k = half && n mod 2 = 0 then input_re (n - 1)
    else if k < half then input_re (2 * k)
    else (* k > half, in range [half+1, n-1] *)
      input_re (2 * (n - k) - 1)
  in
  (* Step 2: N-point real DFT of the permuted buffer. *)
  let z_re, z_im = dft_rdft ~sign:`Fwd n buf in
  (* Step 3: post-process butterfly. Output is N reals (indices 0..N-1). *)
  let out = Array.make n (Const 0.0) in
  (* DC bin Y[0] = 2·Re(Z[0]). *)
  out.(0) <- Mul (Const 2.0, z_re.(0));
  (* Nyquist bin Y[N/2] = 2·cos(π/4)·Re(Z[N/2]).
   * cos(π/4) = √2/2 ≈ 0.7071. *)
  let nyq_scale = 2.0 *. cos (pi /. 4.0) in
  out.(half) <- Mul (Const nyq_scale, z_re.(half));
  (* Pair butterflies for i = 1..N/2-1. *)
  for i = 1 to half - 1 do
    let theta = pi *. float_of_int i /. (2.0 *. float_of_int n) in
    let wa = Const (cos theta) in
    let wb = Const (sin theta) in
    let a = Mul (Const 2.0, z_re.(i)) in
    let b = Mul (Const 2.0, z_im.(i)) in
    (*  Y[i]   = wa·a + wb·b  *)
    out.(i)     <- Add (Mul (wa, a), Mul (wb, b));
    (*  Y[N-i] = wb·a - wa·b  *)
    out.(n - i) <- Sub (Mul (wb, a), Mul (wa, b))
  done;
  out

(* Assignment-list wrapper for dct2.
 * Inputs at Input(k, true) for k = 0..n-1.
 * Outputs at Output(k, true) for k = 0..n-1. Output is purely real —
 * we skip Output(k, false) entirely to avoid emitting n×K useless
 * stores per call. Caller treats out_im as undefined / scratch. *)
let dft_expand_dct2 (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dct2 n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out.(k)) :: !acc
  done;
  List.rev !acc

(* === DCT-II via FFTW's trigII embedding ===
 *
 * Alternative to Makhoul. Port of FFTW's approach from genfft/trig.ml:
 *     trigII n input = Fft.dft 1 (4n) (Complex.hermitian (4n) (interleave_zero input))
 *     dctII = make_dct Complex.one 0 trigII
 *
 * Construct a 4N-point real signal g such that
 *   g[2k+1] = x[k]   for k = 0..N-1     (odd positions in [0..2N))
 *   g[2k]   = 0      for k = 0..2N-1    (all even positions)
 *   g[4N-i] = g[i]   for i = 1..2N-1    (Hermitian mirror; real => no conj)
 *
 * Then DFT(g)[k] = sum_{m=0..4N-1} g[m] · exp(-2πi · m · k / (4N))
 *                = 2 · sum_{j=0..N-1} x[j] · cos(π · k · (2j+1) / (2N))
 *                = Y_DCT-II[k]
 *
 * The 32-point (for N=8) c2c DFT is run on a signal where most inputs
 * are zero (16 of 32 inputs are zero by construction; another 8 are
 * conjugate-symmetric mirrors). algsimp should fold the zeros and
 * exploit the symmetry, producing fewer ops than Makhoul's explicit
 * permute-then-rdft-then-butterfly DAG.
 *
 * Reference: FFTW genfft/trig.ml lines 60-64.
 *)
let dft_dct2_trigII (n : int) (input_re : int -> expr) : expr array =
  assert (n >= 1);
  let fourn = 4 * n in
  let zero = Const 0.0 in
  (* Build the 4N-point real signal g via interleave-zero + Hermitian. *)
  let in_re i =
    if i mod 2 = 0 then zero
    else if i < 2 * n then input_re ((i - 1) / 2)
    else input_re ((fourn - 1 - i) / 2)   (* Hermitian mirror: g[4N-i] = g[i] *)
  in
  let in_im _ = zero in
  let full_re, _ = Dft.dft ~sign:`Fwd fourn in_re in_im in
  (* DCT-II outputs are the first N real parts of the 4N-point DFT. *)
  let out = Array.make n zero in
  for k = 0 to n - 1 do
    out.(k) <- full_re.(k)
  done;
  out

let dft_expand_dct2_trigII (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dct2_trigII n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out.(k)) :: !acc
  done;
  List.rev !acc

(* === DCT-III via inverse-Makhoul (FFTW REDFT01 convention) ===
 *
 * Fills the production gap: production has only a specialized N=8
 * codelet (`dct3_n8_avx2.h`); the general-N dispatcher is deferred per
 * the v1.0 limitation note in `src/core/dct.h`.
 *
 *   Y[k] = X[0] + 2 · Σ_{n=1..N-1} X[n] · cos(π · n · (2k+1) / (2N))
 *
 * DCT-III is the inverse of DCT-II up to scale 2N. We invert Makhoul:
 *   1. Inverse butterfly: build complex Z[0..N/2] from input X
 *      Re(Z[0])    = X[0] / 2,                       Im(Z[0])    = 0
 *      Re(Z[i])    = (wa · X[i] + wb · X[N-i]) / 2,  Im(Z[i])    = (wb · X[i] - wa · X[N-i]) / 2
 *        where wa = cos(π·i/(2N)), wb = sin(π·i/(2N))
 *      Re(Z[N/2])  = X[N/2] · √2/2,                  Im(Z[N/2])  = 0   (N even)
 *   2. N-point inverse R2C (C2R) of Z → buf[0..N-1] real
 *   3. Un-permute (inverse of DCT-II's pre-permute):
 *        Y[0]   = buf[0]
 *        Y[2i]  = buf[i]      for i = 1..N/2-1
 *        Y[2i-1]= buf[N-i]    for i = 1..N/2-1
 *        Y[N-1] = buf[N/2]    (N even)
 *)
let dft_dct3 (n : int) (input_re : int -> expr) : expr array =
  assert (n >= 2 && n mod 2 = 0);
  let half = n / 2 in
  let pi = 4.0 *. atan 1.0 in
  (* Step 1: inverse butterfly. We intentionally produce 2·Z (twice the
   * actual Re(Z), Im(Z)) — the doubling propagates through the
   * unnormalized IFFT (which gives N · buf) to yield 2N · x at the end,
   * matching production's REDFT01 convention DCT-III(DCT-II(x)) = 2N·x. *)
  let z_re = Array.make (half + 1) (Const 0.0) in
  let z_im = Array.make (half + 1) (Const 0.0) in
  (* Forward: Y[0] = 2·Re(Z[0]). For 2·Re(Z[0]) = Y[0], use z_re[0] = Y[0]. *)
  z_re.(0) <- input_re 0;
  (* Forward: Y[N/2] = 2·cos(π/4)·Re(Z[N/2]).
   * For 2·Re(Z[N/2]) = Y[N/2]/cos(π/4), use z_re[N/2] = Y[N/2]/cos(π/4). *)
  let nyq_scale = 1.0 /. cos (pi /. 4.0) in
  z_re.(half) <- Mul (Const nyq_scale, input_re half);
  for i = 1 to half - 1 do
    let theta = pi *. float_of_int i /. (2.0 *. float_of_int n) in
    let wa = cos theta in
    let wb = sin theta in
    let y_i   = input_re i in
    let y_nmi = input_re (n - i) in
    (*  a = wa·Y[i] + wb·Y[N-i] = 2·Re(Z[i])  (no /2 — keep 2·Re)
     *  b = wb·Y[i] - wa·Y[N-i] = 2·Im(Z[i])                       *)
    z_re.(i) <- Add (Mul (Const wa, y_i), Mul (Const wb, y_nmi));
    z_im.(i) <- Sub (Mul (Const wb, y_i), Mul (Const wa, y_nmi))
  done;
  (* Step 2: N-point inverse R2C of Z → buf.
   * We reuse Dft.dft with sign=Bwd on the Hermitian-extended Z.
   * For positions i > N/2, Z is the conjugate of Z[N-i]. *)
  let z_in_re k =
    if k <= half then z_re.(k)
    else z_re.(n - k)
  in
  let z_in_im k =
    if k <= half then z_im.(k)
    else Neg (z_im.(n - k))
  in
  let buf_re, _buf_im = Dft.dft ~sign:`Bwd n z_in_re z_in_im in
  (* Step 3: inverse permutation to produce Y. *)
  let out = Array.make n (Const 0.0) in
  out.(0) <- buf_re.(0);
  out.(n - 1) <- buf_re.(half);
  for i = 1 to half - 1 do
    out.(2 * i)     <- buf_re.(i);       (* Y[2i]   = buf[i]   *)
    out.(2 * i - 1) <- buf_re.(n - i)    (* Y[2i-1] = buf[N-i] *)
  done;
  out

let dft_expand_dct3 (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dct3 n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out.(k)) :: !acc
  done;
  List.rev !acc

(* === DHT — Discrete Hartley Transform (FFTW convention) ===
 *
 * Port of production's `src/core/dht.h` algorithm — no specialized
 * codelet exists at any N in production, so we fuse the whole pipeline
 * into one DAG.
 *
 *   H[k] = sum_{n=0..N-1} x[n] · (cos(2π·k·n/N) + sin(2π·k·n/N))
 *
 * Self-inverse up to 1/N: DHT(DHT(x)) = N · x.
 *
 * Algorithm: given X = R2C(x) with X[N-k] = conj(X[k]):
 *   H[0]   = Re(X[0])
 *   H[k]   = Re(X[k]) - Im(X[k])   for k = 1..N/2-1
 *   H[N-k] = Re(X[k]) + Im(X[k])   for k = 1..N/2-1
 *   H[N/2] = Re(X[N/2])            (N even)
 *
 * One N-point rdft + an O(N) butterfly. algsimp folds the whole thing
 * into a single straight-line codelet. Constraint: N must be even.
 *)
let dft_dht (n : int) (input_re : int -> expr) : expr array =
  assert (n >= 2 && n mod 2 = 0);
  let half = n / 2 in
  let x_re, x_im = dft_rdft ~sign:`Fwd n input_re in
  let out = Array.make n (Const 0.0) in
  out.(0)    <- x_re.(0);                          (* H[0]   = Re(X[0])  *)
  out.(half) <- x_re.(half);                       (* H[N/2] = Re(X[N/2]) *)
  for k = 1 to half - 1 do
    out.(k)     <- Sub (x_re.(k), x_im.(k));       (* H[k]   = Re - Im *)
    out.(n - k) <- Add (x_re.(k), x_im.(k))        (* H[N-k] = Re + Im *)
  done;
  out

let dft_expand_dht (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dht n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out.(k)) :: !acc
  done;
  List.rev !acc

(* === DST-II (FFTW RODFT10) via DCT-II wrapper ===
 *
 *   Y[k] = 2 · sum_{n=0..N-1} x[n] · sin(π · (k+1) · (2n+1) / (2N))
 *
 * Port of production's `src/core/dst.h` identity:
 *   DST-II[k] = DCT-II[(-1)^n · x[n]][N-1-k]
 *
 * Pre: sign-flip every other element of x.
 * Core: DCT-II of the sign-flipped input.
 * Post: reverse the output indices.
 * All three steps fold into one DAG. *)
let dft_dst2 (n : int) (input_re : int -> expr) : expr array =
  let signed_input k =
    if k mod 2 = 0 then input_re k
    else Neg (input_re k)
  in
  let dct2_out = dft_dct2 n signed_input in
  let out = Array.make n (Const 0.0) in
  for k = 0 to n - 1 do
    out.(k) <- dct2_out.(n - 1 - k)
  done;
  out

let dft_expand_dst2 (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dst2 n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out.(k)) :: !acc
  done;
  List.rev !acc

(* === DST-III (FFTW RODFT01) via DCT-III wrapper ===
 *
 *   Y[k] = (-1)^k · X[N-1] + 2 · sum_{n=0..N-2} X[n] · sin(π · (n+1) · (2k+1) / (2N))
 *
 * Identity: DST-III[k] = (-1)^k · DCT-III[reversed_input][k]
 *
 * Pre: reverse the input.
 * Core: DCT-III of the reversed input.
 * Post: sign-flip every other output element. *)
let dft_dst3 (n : int) (input_re : int -> expr) : expr array =
  let reversed k = input_re (n - 1 - k) in
  let dct3_out = dft_dct3 n reversed in
  let out = Array.make n (Const 0.0) in
  for k = 0 to n - 1 do
    if k mod 2 = 0 then out.(k) <- dct3_out.(k)
    else out.(k) <- Neg (dct3_out.(k))
  done;
  out

let dft_expand_dst3 (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dst3 n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out.(k)) :: !acc
  done;
  List.rev !acc

(* === DCT-IV via Lee 1984 (FFTW REDFT11 convention) ===
 *
 *   Y[k] = 2 · sum_{n=0..N-1} x[n] · cos(π · (2k+1) · (2n+1) / (4N))
 *
 * Port of production's `src/core/dct4.h` Lee 1984 algorithm:
 *   1. z[m]   = x[2m] - i · x[N-1-2m]                     for m = 0..N/2-1
 *   2. psi[m] = z[m] · exp(i·π·m/N)                       (pre-twiddle)
 *   3. IFFT_{N/2}(psi)[k'] = Σ_m psi[m] · exp(+2πi·m·k'/(N/2))   (unnormalized)
 *   4. Z[k']  = 2 · exp(i·π·(4k'+1)/(4N)) · IFFT(psi)[k']  (post-twiddle)
 *   5. Y[2k']     = Re(Z[k'])
 *      Y[N-1-2k'] = Im(Z[k'])
 *
 * The c2c IFFT_{N/2} is the inner transform; everything else is index
 * manipulation + complex multiplies that algsimp folds.
 *
 * Constraint: N must be even (matches production).
 * Involutory: DCT-IV(DCT-IV(x)) = 2N · x.
 *)
let dft_dct4 (n : int) (input_re : int -> expr) : expr array =
  assert (n >= 2 && n mod 2 = 0);
  let half = n / 2 in
  let pi = 4.0 *. atan 1.0 in
  (* Step 2: pre-twiddle psi[m] = z[m] · exp(iπm/N) where z[m] = x[2m] - i·x[N-1-2m].
   * Compute z_re = x[2m], z_im = -x[N-1-2m] inline. *)
  let psi_re = Array.make half (Const 0.0) in
  let psi_im = Array.make half (Const 0.0) in
  for m = 0 to half - 1 do
    let z_re = input_re (2 * m) in
    let z_im = Neg (input_re (n - 1 - 2 * m)) in
    let phi = pi *. float_of_int m /. float_of_int n in
    let c = Const (cos phi) in
    let s = Const (sin phi) in
    (* psi = (z_re + i·z_im) · (c + i·s)
     *     = (z_re·c - z_im·s) + i(z_re·s + z_im·c) *)
    psi_re.(m) <- Sub (Mul (z_re, c), Mul (z_im, s));
    psi_im.(m) <- Add (Mul (z_re, s), Mul (z_im, c))
  done;
  (* Step 3: IFFT_{N/2} unnormalized backward c2c on psi. *)
  let ifft_re, ifft_im =
    Dft.dft ~sign:`Bwd half
      (fun k -> psi_re.(k))
      (fun k -> psi_im.(k))
  in
  (* Step 4 + 5: post-twiddle and extract.
   * Z[k'] = (2cos + 2i·sin) · (ifft_re + i·ifft_im)
   *       = (2cos·ifft_re - 2sin·ifft_im) + i(2cos·ifft_im + 2sin·ifft_re)
   * with cos/sin = cos/sin(π(4k'+1)/(4N)). *)
  let out = Array.make n (Const 0.0) in
  for kp = 0 to half - 1 do
    let phi = pi *. float_of_int (4 * kp + 1) /. (4.0 *. float_of_int n) in
    let c = Const (2.0 *. cos phi) in
    let s = Const (2.0 *. sin phi) in
    let z_re = Sub (Mul (c, ifft_re.(kp)), Mul (s, ifft_im.(kp))) in
    let z_im = Add (Mul (c, ifft_im.(kp)), Mul (s, ifft_re.(kp))) in
    out.(2 * kp)         <- z_re;
    out.(n - 1 - 2 * kp) <- z_im
  done;
  out

let dft_expand_dct4 (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let out = dft_dct4 n input_re in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out.(k)) :: !acc
  done;
  List.rev !acc

(* === HC2HC: middle-stage Hermitian-packed cascade codelet ===
 *
 * Port of FFTW's gen_hc2hc.ml (lines 67-104). Operates in-place on
 * Hermitian-packed data — the inter-stage data format that exploits
 * X[N-k] = conj(X[k]) to halve the work per stage.
 *
 * Algorithm (DIT, per gen_hc2hc.ml line 97):
 *   output = (sym1 n) @@ (sym2 n) (Fft.dft sign n (byw input))
 *
 * - byw input: pre-twiddle each input position by W_M^{m*k} (inter-stage
 *   twiddle from external table)
 * - Fft.dft: standard c2c DFT-n on the twiddled inputs
 * - sym2: post-rotate the upper half (i ≥ n/2) of the DFT output by +i
 * - sym1: combine Re(f(i)) with Im(f(n-1-i)) into the Hermitian-packed slot
 *
 * The sym1 ∘ sym2 chain folds the natural-order c2c output into the
 * Hermitian-packed convention so the next cascade stage can pick up
 * with the same format.
 *
 * Storage convention (both input and output):
 *   Position i (0 ≤ i < n) stores a complex value from the packed
 *   half-spectrum. Lower half (i < n/2) and upper half (i ≥ n/2)
 *   together encode the n/2+1 unique values.
 *
 * Reference: FFTW's gen_hc2hc.ml; sym1, sym2 are defined at lines 67-74.
 *)

(* sym2: post-rotate upper half by +i. (re, im) → (-im, re) for i ≥ n/2. *)
let sym2_arr (n : int) (re_arr : expr array) (im_arr : expr array)
    : expr array * expr array =
  let r2 = Array.make n (Const 0.0) in
  let i2 = Array.make n (Const 0.0) in
  for i = 0 to n - 1 do
    if 2 * i < n then begin
      r2.(i) <- re_arr.(i);
      i2.(i) <- im_arr.(i)
    end else begin
      r2.(i) <- Neg im_arr.(i);
      i2.(i) <- re_arr.(i)
    end
  done;
  (r2, i2)

(* sym2i: pre-rotate upper half by -i. (re, im) → (im, -re) for i ≥ n/2.
 * Used by the DIF dispatch. *)
let sym2i_arr (n : int) (re_arr : expr array) (im_arr : expr array)
    : expr array * expr array =
  let r2 = Array.make n (Const 0.0) in
  let i2 = Array.make n (Const 0.0) in
  for i = 0 to n - 1 do
    if 2 * i < n then begin
      r2.(i) <- re_arr.(i);
      i2.(i) <- im_arr.(i)
    end else begin
      r2.(i) <- im_arr.(i);
      i2.(i) <- Neg (re_arr.(i))
    end
  done;
  (r2, i2)

(* sym1: combine Re(f(i)) with Im(f(n-1-i)) at every position. *)
let sym1_arr (n : int) (re_arr : expr array) (im_arr : expr array)
    : expr array * expr array =
  let r1 = Array.make n (Const 0.0) in
  let i1 = Array.make n (Const 0.0) in
  for i = 0 to n - 1 do
    r1.(i) <- re_arr.(i);
    i1.(i) <- im_arr.(n - 1 - i)
  done;
  (r1, i1)

(* Main hc2hc primitive (DIT case).
 *   input_re/im k     : load packed input at position k (k = 0..n-1)
 *   tw_re/im k        : load twiddle for position k (position 0 has trivial W^0 = 1)
 * Returns (out_re, out_im) of length n in Hermitian-packed format. *)
let dft_hc2hc_dit ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
    (tw_re : int -> expr) (tw_im : int -> expr)
    : expr array * expr array =
  let conj = (sign = `Bwd) in
  (* Pre-twiddle. Position 0's twiddle is W^0 = 1, no multiply. *)
  let twiddled_re = Array.make n (Const 0.0) in
  let twiddled_im = Array.make n (Const 0.0) in
  twiddled_re.(0) <- input_re 0;
  twiddled_im.(0) <- input_im 0;
  for k = 1 to n - 1 do
    let (re, im) =
      Dft.cmul_pattern ~conj (input_re k) (input_im k) (tw_re k) (tw_im k)
    in
    twiddled_re.(k) <- re;
    twiddled_im.(k) <- im
  done;
  (* c2c DFT *)
  let re_arr, im_arr =
    Dft.dft ~sign n
      (fun k -> twiddled_re.(k))
      (fun k -> twiddled_im.(k))
  in
  (* Apply sym2 then sym1 to fold into Hermitian-packed output. *)
  let r2, i2 = sym2_arr n re_arr im_arr in
  sym1_arr n r2 i2

(* DIF dispatch:
 *   output = byw (Fft.dft sign n (((sym2i n) @@ (sym1 n)) input))
 * Pre-sym chain, DFT, post-twiddle. *)
let dft_hc2hc_dif ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
    (tw_re : int -> expr) (tw_im : int -> expr)
    : expr array * expr array =
  let conj = (sign = `Bwd) in
  let in_re = Array.init n input_re in
  let in_im = Array.init n input_im in
  let s1_re, s1_im = sym1_arr n in_re in_im in
  let s2i_re, s2i_im = sym2i_arr n s1_re s1_im in
  let re_arr, im_arr =
    Dft.dft ~sign n
      (fun k -> s2i_re.(k))
      (fun k -> s2i_im.(k))
  in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  out_re.(0) <- re_arr.(0);
  out_im.(0) <- im_arr.(0);
  for k = 1 to n - 1 do
    let (re, im) =
      Dft.cmul_pattern ~conj re_arr.(k) im_arr.(k) (tw_re k) (tw_im k)
    in
    out_re.(k) <- re;
    out_im.(k) <- im
  done;
  (out_re, out_im)

(* Assignment-list wrapper for hc2hc.
 * Inputs at Input(k, true/false), twiddles at Twiddle(k, true/false) for
 * k = 0..n-1 (twiddle k=0 is unused since position 0 has trivial W^0).
 * Outputs at Output(k, true/false) for k = 0..n-1 in packed format. *)
let dft_expand_hc2hc ?(sign = `Fwd) ?(direction = `Dit) (n : int)
    : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let tw_re k = Load (Twiddle (k, true)) in
  let tw_im k = Load (Twiddle (k, false)) in
  let out_re, out_im = match direction with
    | `Dit -> dft_hc2hc_dit ~sign n input_re input_im tw_re tw_im
    | `Dif -> dft_hc2hc_dif ~sign n input_re input_im tw_re tw_im
  in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc

(* === HC2C: last-stage cascade codelet (Hermitian-packed in, natural out) ===
 *
 * Port of FFTW's gen_hc2c.ml (lines 50-115). This is the cascade
 * terminator — it reads Hermitian-packed input from the previous
 * hc2hc stage and produces natural-order complex output, fusing what
 * would otherwise be a standalone "unpack butterfly" pass into the
 * final radix-n DFT.
 *
 * Algorithm (DIT, per gen_hc2c.ml line 107-109):
 *   output = Fft.dft sign n (byw (load_array_c n locri))
 *   stored via locpm
 *   sym applied to output
 *
 * - locri (input): interleaved load from Hermitian-packed input arrays;
 *   the previous hc2hc stage's output is split across two physical
 *   streams (one for "positive" half-spectrum, one for "negative")
 *   and locri reconstructs the complex sequence at positions 0..n-1.
 *   In our math layer this is just `Load(Input(k, true/false))` — the
 *   physical interleaving is an emission-layer concern.
 *
 * - byw: pre-twiddle (same as hc2hc).
 *
 * - Fft.dft: standard c2c DFT-n.
 *
 * - sym n f i = if (i < n-i) then f i else conj(f i):
 *   conjugates the upper half of the DFT output. After this, the data
 *   is in natural complex order (lower half = positive frequencies,
 *   upper half = conjugates of mirror positions).
 *
 * - locpm (store): positions [0, n/2) go to "positive" arrays (Rp, Ip),
 *   positions [n/2, n) go to "negative" arrays (Rm, Im) indexed by
 *   (n-1-i). This is a physical-layout convention; in our math layer
 *   we just emit Output(k, true/false) for k = 0..n-1 and let the
 *   emitter/executor handle the split.
 *
 * Reference: FFTW's gen_hc2c.ml; sym defined at line 50, locri at line 100,
 * locpm at line 101, DIT/DIF wiring at lines 106-114.
 *)

(* sym: conjugate upper half. (re, im) → (re, -im) for i ≥ n/2. *)
let sym_arr (n : int) (re_arr : expr array) (im_arr : expr array)
    : expr array * expr array =
  let r = Array.make n (Const 0.0) in
  let im = Array.make n (Const 0.0) in
  for i = 0 to n - 1 do
    if 2 * i < n then begin
      r.(i) <- re_arr.(i);
      im.(i) <- im_arr.(i)
    end else begin
      r.(i) <- re_arr.(i);
      im.(i) <- Neg im_arr.(i)
    end
  done;
  (r, im)

(* hc2c primitive (DIT case):
 *   output = sym n (Fft.dft sign n (byw input)) *)
let dft_hc2c_dit ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
    (tw_re : int -> expr) (tw_im : int -> expr)
    : expr array * expr array =
  let conj = (sign = `Bwd) in
  let twiddled_re = Array.make n (Const 0.0) in
  let twiddled_im = Array.make n (Const 0.0) in
  twiddled_re.(0) <- input_re 0;
  twiddled_im.(0) <- input_im 0;
  for k = 1 to n - 1 do
    let (re, im) =
      Dft.cmul_pattern ~conj (input_re k) (input_im k) (tw_re k) (tw_im k)
    in
    twiddled_re.(k) <- re;
    twiddled_im.(k) <- im
  done;
  let re_arr, im_arr =
    Dft.dft ~sign n
      (fun k -> twiddled_re.(k))
      (fun k -> twiddled_im.(k))
  in
  sym_arr n re_arr im_arr

(* hc2c DIF case:
 *   output = byw (Fft.dft sign n (sym n input)) *)
let dft_hc2c_dif ?(sign = `Fwd) (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
    (tw_re : int -> expr) (tw_im : int -> expr)
    : expr array * expr array =
  let conj = (sign = `Bwd) in
  let in_re = Array.init n input_re in
  let in_im = Array.init n input_im in
  let s_re, s_im = sym_arr n in_re in_im in
  let re_arr, im_arr =
    Dft.dft ~sign n
      (fun k -> s_re.(k))
      (fun k -> s_im.(k))
  in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  out_re.(0) <- re_arr.(0);
  out_im.(0) <- im_arr.(0);
  for k = 1 to n - 1 do
    let (re, im) =
      Dft.cmul_pattern ~conj re_arr.(k) im_arr.(k) (tw_re k) (tw_im k)
    in
    out_re.(k) <- re;
    out_im.(k) <- im
  done;
  (out_re, out_im)

(* Assignment-list wrapper for hc2c.
 * Inputs at Input(k, true/false), twiddles at Twiddle(k, true/false),
 * outputs at Output(k, true/false) for k = 0..n-1.
 * The split-pointer storage convention (Rp/Ip/Rm/Im) is handled at
 * the executor / dispatcher level, not here. *)
let dft_expand_hc2c ?(sign = `Fwd) ?(direction = `Dit) (n : int)
    : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let tw_re k = Load (Twiddle (k, true)) in
  let tw_im k = Load (Twiddle (k, false)) in
  let out_re, out_im = match direction with
    | `Dit -> dft_hc2c_dit ~sign n input_re input_im tw_re tw_im
    | `Dif -> dft_hc2c_dif ~sign n input_re input_im tw_re tw_im
  in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc
