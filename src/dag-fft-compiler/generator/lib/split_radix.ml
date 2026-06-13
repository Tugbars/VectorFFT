(* lib/split_radix.ml
 *
 * SPLIT-RADIX DECOMPOSITION FOR POWER-OF-TWO N
 * ============================================
 *
 * Classical references:
 *   - Yavne 1968 (first publication)
 *   - Sorensen, Heideman, Burrus 1986 (canonical exposition)
 *   - Johnson, Frigo 2007 (conjugate-pair refinement — future work)
 *
 * Split-radix uses an asymmetric decomposition that reduces multiplication
 * count by ~33% compared to vanilla radix-2 Cooley-Tukey. An N-point DFT
 * becomes one (N/2)-point DFT plus two (N/4)-point DFTs:
 *
 *     N → DFT(N/2) over even-indexed inputs x[2n]      (call it E)
 *       + DFT(N/4) over stride-4 inputs x[4n+1]        (call it O1)
 *       + DFT(N/4) over stride-4 inputs x[4n+3]        (call it O3)
 *
 * The 33% multiplication savings come from sharing intermediate values
 * across 4 output groups simultaneously, rather than computing
 * W^k · O[k] separately for each.
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
  (* TEST: tan-factored / K-shared form, mirror of Dft.const_cmul *)
  let abs_eq = abs_float cr = abs_float ci in
  if cr = 1.0 && ci = 0.0 then (xr, xi)
  else if cr = 0.0 && ci = 1.0 then (Neg xi, xr)
  else if cr = 0.0 && ci = -1.0 then (xi, Neg xr)
  else if cr = -1.0 && ci = 0.0 then (Neg xr, Neg xi)
  else if abs_eq && cr <> 0.0 then begin
    let k = abs_float cr in
    let k_e = Const k in
    let s = Add (xr, xi) in
    let d = Sub (xr, xi) in
    let ks = Mul (k_e, s) in
    let kd = Mul (k_e, d) in
    let sr = if cr > 0.0 then 1 else -1 in
    let si = if ci > 0.0 then 1 else -1 in
    let with_sign sgn e = if sgn > 0 then e else Neg e in
    match sr, si with
    | 1, 1   -> (kd, ks)
    | 1, -1  -> (ks, with_sign (-1) kd)
    | -1, 1  -> (with_sign (-1) ks, kd)
    | -1, -1 -> (with_sign (-1) kd, with_sign (-1) ks)
    | _ -> assert false
  end else begin
    let round_13 x = float_of_string (Printf.sprintf "%.13e" x) in
    let cr_r = round_13 cr in
    let ci_r = round_13 ci in
    let acr = abs_float cr_r in
    let aci = abs_float ci_r in
    let r_abs = (min acr aci) /. (max acr aci) in
    if acr >= aci then begin
      let tn = if (ci_r >= 0.0) = (cr_r >= 0.0) then r_abs else -. r_abs in
      let cr_e = Const cr_r in
      let tn_e = Const tn in
      let inner_re = Sub (xr, Mul (tn_e, xi)) in
      let inner_im = Add (xi, Mul (tn_e, xr)) in
      (Mul (cr_e, inner_re), Mul (cr_e, inner_im))
    end else begin
      let ct = if (cr_r >= 0.0) = (ci_r >= 0.0) then r_abs else -. r_abs in
      let ci_e = Const ci_r in
      let ct_e = Const ct in
      let inner_re = Sub (Mul (ct_e, xr), xi) in
      let inner_im = Add (xr, Mul (ct_e, xi)) in
      (Mul (ci_e, inner_re), Mul (ci_e, inner_im))
    end
  end

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


(* ============================================================================
 * NEWSPLIT — Johnson-Frigo / Van Buskirk scaled (tangent) conjugate-pair
 * split-radix, ported from FFTW genfft fft.ml (newsplit0/S/S2/S4).
 *
 * Sub-transforms are RESCALED by real factors so that twiddle multiplies
 * collapse toward real scalings; newsplit0 (top entry) produces UNSCALED
 * output. All scale factors (s, sinv, sdiv2, sdiv4, t) are compile-time
 * float constants here. Gated behind VFFT_NEWSPLIT.
 * ========================================================================== *)
let newsplit_core (sign : [`Fwd | `Bwd]) =
  let pi = 4.0 *. atan 1.0 in
  let sgn = match sign with `Fwd -> -1 | `Bwd -> 1 in
  let cexp_f nn i =
    let th = 2.0 *. pi *. float_of_int i /. float_of_int nn in (cos th, sin th) in
  let real_f (c, _) = c in
  let sec_f nn m = let (c, _) = cexp_f nn m in 1.0 /. c in
  let rec s_f nn k =
    if nn <= 4 then 1.0
    else
      let k4 = (abs k) mod (nn / 4) in
      let k4' = if k4 <= nn / 8 then k4 else (nn / 4 - k4) in
      s_f (nn / 4) k4' *. real_f (cexp_f nn k4') in
  let rec sinv_f nn k =
    if nn <= 4 then 1.0
    else
      let k4 = (abs k) mod (nn / 4) in
      let k4' = if k4 <= nn / 8 then k4 else (nn / 4 - k4) in
      sinv_f (nn / 4) k4' *. sec_f nn k4' in
  let sdiv2_f nn k = s_f nn k *. sinv_f (2 * nn) k in
  let sdiv4_f nn k =
    let k4 = (abs k) mod nn in
    sec_f (4 * nn) (if k4 <= nn / 2 then k4 else nn - k4) in
  let t_f nn k =
    let (er, ei) = cexp_f nn k in
    let sd = sdiv4_f (nn / 4) k in (er *. sd, ei *. sd) in
  (* complex ops on (re,im) expr pairs *)
  let ccmul (xr, xi) (cr, ci) =
    (* tan-factored (FFTW -fma form); trivial components fold via mk_const/
       mk_mul at lift. For the tangent-FFT (1,t)/(t,1) twiddles the outer
       Mul is by 1.0 and disappears, leaving 2 FMA-able ops. *)
    if ci = 0.0 then (Mul (xr, Const cr), Mul (xi, Const cr))
    else if cr = 0.0 then (Neg (Mul (xi, Const ci)), Mul (xr, Const ci))
    else begin
      let round_13 x = float_of_string (Printf.sprintf "%.13e" x) in
      let cr_r = round_13 cr and ci_r = round_13 ci in
      let acr = abs_float cr_r and aci = abs_float ci_r in
      let r_abs = (min acr aci) /. (max acr aci) in
      if acr >= aci then begin
        let tn = if (ci_r >= 0.0) = (cr_r >= 0.0) then r_abs else -. r_abs in
        (Mul (Const cr_r, Sub (xr, Mul (Const tn, xi))),
         Mul (Const cr_r, Add (xi, Mul (Const tn, xr))))
      end else begin
        let ct = if (cr_r >= 0.0) = (ci_r >= 0.0) then r_abs else -. r_abs in
        (Mul (Const ci_r, Sub (Mul (Const ct, xr), xi)),
         Mul (Const ci_r, Add (xr, Mul (Const ct, xi))))
      end
    end in
  let cscale (xr, xi) r = (Mul (xr, Const r), Mul (xi, Const r)) in
  let cadd (ar, ai) (br, bi) = (Add (ar, br), Add (ai, bi)) in
  let dft1 (f : int -> expr * expr) = [| f 0 |] in
  let dft2 (f : int -> expr * expr) =
    let (ar, ai) = f 0 and (br, bi) = f 1 in
    [| (Add (ar, br), Add (ai, bi)); (Sub (ar, br), Sub (ai, bi)) |] in
  let rec newsplit0 nn (f : int -> expr * expr) : (expr * expr) array =
    if nn = 1 then dft1 f else if nn = 2 then dft2 f
    else begin
      let m = nn / 4 and n2 = nn / 2 in
      let u  = newsplit0 n2 (fun i -> f (i * 2)) in
      let z  = newsplitS m  (fun i -> f (i * 4 + 1)) in
      let z' = newsplitS m  (fun i -> f ((nn + i * 4 - 1) mod nn)) in
      combine0 nn u z z'
    end
  (* Top-level (unscaled-output) combine, factored out so the blocked
     expansion can reuse it on reloaded E/O1/O3 values. GROUPED: only m
     base cmuls; outputs j+2m via -s0, j+m / j+3m via the exact identity
     t(k+m) = (-sgn)i * t(k). s_f m k has period m in k, so the scale
     constant is identical across the group of four. *)
  and combine0 nn (u : (expr * expr) array)
                  (z : (expr * expr) array)
                  (z' : (expr * expr) array) : (expr * expr) array =
      let m = nn / 4 and n2 = nn / 2 in
      let out = Array.make nn (Const 0.0, Const 0.0) in
      for j = 0 to m - 1 do
        let (er, ei) = cexp_f nn (sgn * j) in
        let sc = s_f m j in
        let (tr, ti) = (er *. sc, ei *. sc) in
        let w  = ccmul z.(j)  (tr, ti) in
        let w' = ccmul z'.(j) (tr, ~-. ti) in
        let (s0r, s0i) = cadd w w' in
        let (dr, di) = (Sub (fst w, fst w'), Sub (snd w, snd w')) in
        let (ur, ui) = u.(j) and (ur', ui') = u.(j + m) in
        out.(j)          <- (Add (ur, s0r), Add (ui, s0i));
        out.(j + n2)     <- (Sub (ur, s0r), Sub (ui, s0i));
        if sgn < 0 then begin
          (* Fwd: t(j+m) = -i t(j); w+w' group at j+m = (d.im, -d.re) *)
          out.(j + m)      <- (Add (ur', di), Sub (ui', dr));
          out.(j + 3 * m)  <- (Sub (ur', di), Add (ui', dr))
        end else begin
          out.(j + m)      <- (Sub (ur', di), Add (ui', dr));
          out.(j + 3 * m)  <- (Add (ur', di), Sub (ui', dr))
        end
      done;
      out
  and newsplitS nn (f : int -> expr * expr) : (expr * expr) array =
    if nn = 1 then dft1 f else if nn = 2 then dft2 f
    else begin
      let m = nn / 4 and n2 = nn / 2 in
      let u  = newsplitS2 n2 (fun i -> f (i * 2)) in
      let z  = newsplitS m  (fun i -> f (i * 4 + 1)) in
      let z' = newsplitS m  (fun i -> f ((nn + i * 4 - 1) mod nn)) in
      let out = Array.make nn (Const 0.0, Const 0.0) in
      for j = 0 to m - 1 do
        let (tr, ti) = t_f nn (sgn * j) in
        let w  = ccmul z.(j)  (tr, ti) in
        let w' = ccmul z'.(j) (tr, ~-. ti) in
        let (s0r, s0i) = cadd w w' in
        let (dr, di) = (Sub (fst w, fst w'), Sub (snd w, snd w')) in
        let (ur, ui) = u.(j) and (ur', ui') = u.(j + m) in
        out.(j)         <- (Add (ur, s0r), Add (ui, s0i));
        out.(j + n2)    <- (Sub (ur, s0r), Sub (ui, s0i));
        if sgn < 0 then begin
          out.(j + m)     <- (Add (ur', di), Sub (ui', dr));
          out.(j + 3 * m) <- (Sub (ur', di), Add (ui', dr))
        end else begin
          out.(j + m)     <- (Sub (ur', di), Add (ui', dr));
          out.(j + 3 * m) <- (Add (ur', di), Sub (ui', dr))
        end
      done;
      out
    end
  and newsplitS2 nn (f : int -> expr * expr) : (expr * expr) array =
    if nn = 1 then dft1 f else if nn = 2 then dft2 f
    else begin
      let m = nn / 4 and n2 = nn / 2 in
      let u  = newsplitS4 n2 (fun i -> f (i * 2)) in
      let z  = newsplitS m  (fun i -> f (i * 4 + 1)) in
      let z' = newsplitS m  (fun i -> f ((nn + i * 4 - 1) mod nn)) in
      let out = Array.make nn (Const 0.0, Const 0.0) in
      for j = 0 to m - 1 do
        let (tr, ti) = t_f nn (sgn * j) in
        let w  = ccmul z.(j)  (tr, ti) in
        let w' = ccmul z'.(j) (tr, ~-. ti) in
        let (s0r, s0i) = cscale (cadd w w') (sdiv2_f nn j) in
        let c1 = sdiv2_f nn (j + m) in
        let (dr, di) = (Sub (fst w, fst w'), Sub (snd w, snd w')) in
        let (q1r, q1i) = (Mul (di, Const c1), Mul (dr, Const c1)) in
        let (ur, ui) = u.(j) and (ur', ui') = u.(j + m) in
        out.(j)         <- (Add (ur, s0r), Add (ui, s0i));
        out.(j + n2)    <- (Sub (ur, s0r), Sub (ui, s0i));
        if sgn < 0 then begin
          out.(j + m)     <- (Add (ur', q1r), Sub (ui', q1i));
          out.(j + 3 * m) <- (Sub (ur', q1r), Add (ui', q1i))
        end else begin
          out.(j + m)     <- (Sub (ur', q1r), Add (ui', q1i));
          out.(j + 3 * m) <- (Add (ur', q1r), Sub (ui', q1i))
        end
      done;
      out
    end
  and newsplitS4 nn (f : int -> expr * expr) : (expr * expr) array =
    if nn = 1 then dft1 f
    else if nn = 2 then begin
      let d = dft2 f in
      Array.init 2 (fun k -> cscale d.(k) (sinv_f 8 k))
    end
    else begin
      let m = nn / 4 and n2 = nn / 2 in
      let u  = newsplitS2 n2 (fun i -> f (i * 2)) in
      let z  = newsplitS m  (fun i -> f (i * 4 + 1)) in
      let z' = newsplitS m  (fun i -> f ((nn + i * 4 - 1) mod nn)) in
      let out = Array.make nn (Const 0.0, Const 0.0) in
      for j = 0 to m - 1 do
        let (tr, ti) = t_f nn (sgn * j) in
        let w  = ccmul z.(j)  (tr, ti) in
        let w' = ccmul z'.(j) (tr, ~-. ti) in
        let (s0r, s0i) = cadd w w' in
        let (dr, di) = (Sub (fst w, fst w'), Sub (snd w, snd w')) in
        let (ur, ui) = u.(j) and (ur', ui') = u.(j + m) in
        let emit k (re, im) = out.(k) <- cscale (re, im) (sdiv4_f nn k) in
        emit j          (Add (ur, s0r), Add (ui, s0i));
        emit (j + n2)   (Sub (ur, s0r), Sub (ui, s0i));
        if sgn < 0 then begin
          emit (j + m)     (Add (ur', di), Sub (ui', dr));
          emit (j + 3 * m) (Sub (ur', di), Add (ui', dr))
        end else begin
          emit (j + m)     (Sub (ur', di), Add (ui', dr));
          emit (j + 3 * m) (Add (ur', di), Sub (ui', dr))
        end
      done;
      out
    end
  in
  (newsplit0, newsplitS, combine0)

let dft_newsplit ~(sign : [`Fwd | `Bwd]) (n : int)
    (input_re : int -> expr) (input_im : int -> expr) : expr array * expr array =
  let (ns0, _, _) = newsplit_core sign in
  let inp i = (input_re i, input_im i) in
  let out = ns0 n inp in
  (Array.map fst out, Array.map snd out)

(* ============================================================================
 * BLOCKED NEWSPLIT — top-level E/O1/O3 seam for the spill recipe.
 *
 * PASS 1: E = newsplit0(n/2) on even inputs, O1/O3 = newsplitS(n/4) on the
 * 4k+1 / 4k-1 input cosets. All three are mutually independent (disjoint
 * input cosets), so they form natural PASS-1 clusters.
 *
 * Spill-slot layout (n slots total, same count as CT's):
 *   E  outputs -> slots [0, n/2)          (spans 2 fake clusters of n/4)
 *   O1 outputs -> slots [n/2, 3n/4)       (cluster 2)
 *   O3 outputs -> slots [3n/4, n)         (cluster 3)
 *
 * PASS 2 = combine0 on the reloaded values: output group j reads slots
 * {j, j+n/4, n/2+j, 3n/4+j}; with the fake ct = (4, n/4) the existing
 * emit machinery assigns cluster_of_pass2_node = min_input_slot mod n/4
 * = j, giving per-group SU and per-group store flush for free.
 * ========================================================================== *)
let dft_newsplit_blocked ~(sign : [`Fwd | `Bwd]) (n : int)
    (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array * (int * expr * expr) list =
  let (ns0, nsS, combine0) = newsplit_core sign in
  let m = n / 4 and n2 = n / 2 in
  let inp i = (input_re i, input_im i) in
  let u  = ns0 n2 (fun i -> inp (2 * i)) in
  let z  = nsS m  (fun i -> inp (4 * i + 1)) in
  let z' = nsS m  (fun i -> inp ((n + 4 * i - 1) mod n)) in
  let markers = ref [] in
  Array.iteri (fun i  (r, im) -> markers := (i,           r, im) :: !markers) u;
  Array.iteri (fun j  (r, im) -> markers := (n2 + j,      r, im) :: !markers) z;
  Array.iteri (fun j  (r, im) -> markers := (n2 + m + j,  r, im) :: !markers) z';
  let out = combine0 n u z z' in
  (Array.map fst out, Array.map snd out, List.rev !markers)
