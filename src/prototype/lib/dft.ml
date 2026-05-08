(* Dft.ml — algorithm selection + Cooley-Tukey decomposition.
 *
 * This module sits between the user's request "compute DFT-N symbolically"
 * and the lower-level `Expr.dft_kernel` (which does direct sum-of-products
 * expansion). For composite N, it decomposes the DFT recursively via
 * Cooley-Tukey before falling through to direct expansion.
 *
 * Why this layer exists: direct expansion of DFT-N produces O(N²)
 * arithmetic per output equation. Algsimp's reassociation finds *some*
 * butterfly structure within each equation, but doesn't discover the
 * Cooley-Tukey decomposition that shares subexpressions ACROSS
 * equations. For R=8 specifically, direct expansion + algsimp produces
 * 134 ops; CT decomposition + algsimp should produce ~52-56 ops.
 *
 * Architecture matches FFTW genfft's fft.ml: hardcoded algorithm
 * selection based on number-theoretic properties of N. For the
 * prototype we implement only:
 *   - Direct DFT (prime n, falls through to dft_kernel)
 *   - Cooley-Tukey radix-2 DIT decomposition for even n
 *
 * Future variants to add (matching what genfft does):
 *   - Split-radix for power-of-2 ≥ 8 (slightly fewer ops than radix-2)
 *   - Prime-Factor (Good-Thomas) for coprime composite n
 *   - Rader for primes ≥ 13
 *)

open Expr

(* === ALGORITHM SELECTION === *)

type algorithm =
  | Direct                        (* prime n, or n=2: use dft_kernel *)
  | Cooley_Tukey of int * int     (* (n1, n2): split DFT-n into n1 columns of n2 *)

(* Pick algorithm based on n's number-theoretic properties.
 *
 * Match user's hand-coded factorizations:
 *   R=4  → CT(2, 2) — radix-2 inside radix-2
 *   R=8  → CT(2, 4) — radix-2 outer, radix-4 inner
 *   R=16 → CT(4, 4) — two passes of radix-4 (NOT 2×8)
 *
 * For other even n, fall back to radix-2 CT.
 * Odd n (and primes) use direct DFT.
 *
 * Why R=16 = 4×4 instead of 2×8: hand-coded uses two clean radix-4
 * passes with explicit spill between them. The 4×4 structure has a
 * lower-depth dependency tree than the 4-level recursion 2×8 would
 * produce, and matches the user's existing benchmark layout. *)
let pick_algorithm (n : int) : algorithm =
  if n <= 2 then Direct
  else match n with
    | 4 -> Cooley_Tukey (2, 2)
    | 8 -> Cooley_Tukey (2, 4)
    | 16 -> Cooley_Tukey (4, 4)
    | 32 -> Cooley_Tukey (4, 8)
    | 64 -> Cooley_Tukey (8, 8)
        (* Hand-coded R=64 uses CT(8, 8) — symmetric factorization that
         * splits the 64-point DFT into 8 sub-FFT-8s in PASS 1 and 8
         * sub-FFT-8s in PASS 2. Same convention as gen_radix64.py. *)
        (* User's gen_radix32.py uses CT(8, 4) in their (N1, N2) convention.
         * Their convention has input mapping n = N2*n1 + n2 (high digit n1),
         * ours has n = n1 + n2*N1 (low digit n1). The labels swap.
         *
         * Their PASS 1: 4 sub-FFTs of size 8 (radix-8 butterflies).
         * Their PASS 2: 8 sub-FFTs of size 4.
         *
         * In OUR convention, PASS 1 has N1 sub-FFTs of size N2, so we want
         * N1=4, N2=8. The math is equivalent; just the labeling swaps.
         *
         * Why this factorization: 4×8 keeps R=32's register pressure
         * manageable. Each PASS 1 sub-FFT has 8 live values (fits AVX-512's
         * 32 ZMM with room for twiddles); going wider (e.g. 2×16) would
         * blow the register budget. Going narrower (e.g. 8×4) is also valid
         * but produces deeper inner DFTs in PASS 2. *)
    | _ when n mod 2 = 0 -> Cooley_Tukey (2, n / 2)
    | _ -> Direct

(* Whether algsimp's reassoc pass is appropriate for the given n.
 *
 * Reassoc helps when the input is a flat sum-of-products from direct
 * DFT expansion — it discovers butterfly subsums by reassociating
 * binary Add chains. But when CT decomposition has structured the
 * input correctly, reassoc actively destroys that structure by
 * flattening across CT boundaries. So:
 *
 *   - Direct DFT: reassoc HELPS (finds butterflies)
 *   - CT-decomposed: reassoc HURTS (shreds the CT structure)
 *)
let needs_reassoc (n : int) : bool =
  match pick_algorithm n with
  | Direct -> true
  | Cooley_Tukey _ -> false

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
 * Returns (out_re, out_im) as Expr trees. *)
let const_cmul (xr : expr) (xi : expr) (cr : float) (ci : float) : expr * expr =
  let cr_e = Const cr in
  let ci_e = Const ci in
  (* (xr*cr - xi*ci, xr*ci + xi*cr) *)
  let out_re = Sub (Mul (xr, cr_e), Mul (xi, ci_e)) in
  let out_im = Add (Mul (xr, ci_e), Mul (xi, cr_e)) in
  (out_re, out_im)

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
let rec dft (n : int) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  match pick_algorithm n with
  | Direct -> dft_direct n input_re input_im
  | Cooley_Tukey (n1, n2) -> dft_ct n1 n2 input_re input_im

(* Direct DFT: matrix-vector form.
 *   X[k].re = Σ_n  a[n] * cos(-2πnk/n) - b[n] * sin(-2πnk/n)
 *   X[k].im = Σ_n  a[n] * sin(-2πnk/n) + b[n] * cos(-2πnk/n) *)
and dft_direct (n : int) (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  let pi = 4.0 *. atan 1.0 in
  let out_re = Array.make n (Const 0.0) in
  let out_im = Array.make n (Const 0.0) in
  for k = 0 to n - 1 do
    let re_sum = ref (Const 0.0) in
    let im_sum = ref (Const 0.0) in
    for nn = 0 to n - 1 do
      let theta = -2.0 *. pi *. float_of_int (nn * k) /. float_of_int n in
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
  (out_re, out_im)

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
and dft_ct (n1 : int) (n2 : int)
           (input_re : int -> expr) (input_im : int -> expr)
    : expr array * expr array =
  let n = n1 * n2 in
  let pi = 4.0 *. atan 1.0 in

  (* PASS 1: N1 sub-FFTs of size N2.
   * For each n1_idx in [0, N1), compute DFT-N2 on inputs at
   *   x[n1_idx], x[n1_idx + N1], x[n1_idx + 2·N1], ...
   * pass1[n1_idx][k2] = DFT-N2 result at output bin k2. *)
  let pass1_re = Array.make_matrix n1 n2 (Const 0.0) in
  let pass1_im = Array.make_matrix n1 n2 (Const 0.0) in
  for n1_idx = 0 to n1 - 1 do
    let inner_input_re k2 = input_re (n1_idx + k2 * n1) in
    let inner_input_im k2 = input_im (n1_idx + k2 * n1) in
    let r, i = dft n2 inner_input_re inner_input_im in
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
      let theta = -2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
      let cr = cos theta in
      let ci = sin theta in
      let (tr, ti) = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
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
    let r, i = dft n1 outer_input_re outer_input_im in
    for k1 = 0 to n1 - 1 do
      out_re.(k1 * n2 + k2) <- r.(k1);
      out_im.(k1 * n2 + k2) <- i.(k1)
    done
  done;

  (out_re, out_im)

(* === TWIDDLE POLICY ===
 *
 * Different codelet variants source their twiddles differently:
 *   TP_Flat — load all (R-1) twiddles directly from a flat buffer.
 *             Maximum loads, minimum FMA. Wins when FMA ports are
 *             saturated and twiddle bandwidth is plentiful.
 *
 *   TP_Log3 — load only base twiddles (R=4: just W^1; R=8: W^1, W^2, W^4)
 *             and derive the rest by complex multiplication.
 *             Saves twiddle loads, costs extra FMAs. Wins when L1
 *             twiddle pressure dominates (high K, deep transforms).
 *
 * The math-layer payoff: log3 is a *substitution* — wherever t1_dit
 * emits Load(Twiddle(j, ...)), log3 substitutes the derivation tree.
 * Algsimp's Cmul handling propagates; nothing else changes. *)

type twiddle_policy =
  | TP_Flat
  | TP_Log3

(* Build a complex multiplication (a + ib)·(c + id) as (out_re, out_im)
 * using the cmul pattern that Algsimp.of_expr will lift to Cmul nodes.
 * The pattern is Sub(Mul, Mul) for re, Add(Mul, Mul) for im. *)
let cmul_pattern (ar : expr) (ai : expr) (br : expr) (bi : expr)
    : expr * expr =
  let out_re = Sub (Mul (ar, br), Mul (ai, bi)) in
  let out_im = Add (Mul (ar, bi), Mul (ai, br)) in
  (out_re, out_im)

(* Compute the (re, im) Expr trees for the j-th twiddle (1-indexed)
 * under the given policy and radix. Memoization is critical: derived
 * twiddles like W^7 = W^3·W^4 reference W^3, which must be the SAME
 * expression tree (pointer-equal at the Expr level, hash-cons-equal
 * after lifting) so that the underlying cmul gets shared. *)
let twiddle_expr (policy : twiddle_policy) (n : int) (j : int)
    : expr * expr =
  let cache : (int, expr * expr) Hashtbl.t = Hashtbl.create 16 in
  let rec lookup j =
    match Hashtbl.find_opt cache j with
    | Some result -> result
    | None ->
      let result = compute j in
      Hashtbl.add cache j result;
      result
  and compute j =
    match policy with
    | TP_Flat ->
      (* Direct load from slot (j-1). *)
      (Load (Twiddle (j - 1, true)), Load (Twiddle (j - 1, false)))

    | TP_Log3 ->
      (match n, j with
       (* R=4 log3 chain: load W^1; derive W^2 = W^1², W^3 = W^2·W^1. *)
       | 4, 1 -> (Load (Twiddle (0, true)), Load (Twiddle (0, false)))
       | 4, 2 ->
         let (w1r, w1i) = lookup 1 in
         cmul_pattern w1r w1i w1r w1i
       | 4, 3 ->
         let (w1r, w1i) = lookup 1 in
         let (w2r, w2i) = lookup 2 in
         cmul_pattern w2r w2i w1r w1i

       (* R=8 log3 tree: load W^1, W^2, W^4 from slots 0, 1, 3.
        * Derive W^3 = W^1·W^2, W^5 = W^1·W^4, W^6 = W^2·W^4,
        * W^7 = W^3·W^4. *)
       | 8, 1 -> (Load (Twiddle (0, true)), Load (Twiddle (0, false)))
       | 8, 2 -> (Load (Twiddle (1, true)), Load (Twiddle (1, false)))
       | 8, 4 -> (Load (Twiddle (3, true)), Load (Twiddle (3, false)))
       | 8, 3 ->
         let (w1r, w1i) = lookup 1 in
         let (w2r, w2i) = lookup 2 in
         cmul_pattern w1r w1i w2r w2i
       | 8, 5 ->
         let (w1r, w1i) = lookup 1 in
         let (w4r, w4i) = lookup 4 in
         cmul_pattern w1r w1i w4r w4i
       | 8, 6 ->
         let (w2r, w2i) = lookup 2 in
         let (w4r, w4i) = lookup 4 in
         cmul_pattern w2r w2i w4r w4i
       | 8, 7 ->
         let (w3r, w3i) = lookup 3 in
         let (w4r, w4i) = lookup 4 in
         cmul_pattern w3r w3i w4r w4i

       | _ ->
         failwith (Printf.sprintf
           "TP_Log3 not implemented for n=%d j=%d" n j))
  in
  lookup j

(* === ASSIGNMENT-LIST WRAPPERS ===
 *
 * Replace the old dft_expand / dft_expand_twiddled with versions that
 * call into this module's recursive dft. The output is the same
 * Expr.assignment list shape as before. *)

let dft_expand (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let out_re, out_im = dft n input_re input_im in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  (* Reorder so each output's re comes immediately before its im. *)
  let pairs = List.rev !acc in
  (* `acc` was built (re, im, re, im, ...) in reverse, so List.rev
   * gives the correct order. *)
  let _ = pairs in
  List.rev !acc

(* Twiddled (t1_dit) form: pre-multiply inputs by runtime twiddles
 * (Cmul nodes), then run the (possibly Cooley-Tukey-decomposed) DFT
 * on the twiddled inputs.
 *
 * The cmul outputs use the SUB(MUL,MUL)/ADD(MUL,MUL) pattern that
 * Algsimp.of_expr's pattern detector recognizes and lifts to Cmul
 * opaque atoms. The CT decomposition then sees the cmul outputs as
 * leaf-like values in its own Add/Sub structure — algsimp won't
 * shred them because they're inside Cmul nodes after lifting. *)
let dft_expand_twiddled ?(policy = TP_Flat) (n : int) : Expr.assignment list =
  (* For each leg k in 1..n-1, pre-multiply input x[k] by twiddle W^k
   * (sourced via the policy). Leg 0 has no twiddle (W^0 = 1).
   *
   * The twiddle Expr trees may themselves be cmul derivation patterns
   * (TP_Log3) or simple Loads (TP_Flat). Either way, the per-leg
   * Sub(Mul,Mul)/Add(Mul,Mul) cmul pattern is preserved so that
   * Algsimp.of_expr can lift it to Cmul opaque atoms. *)
  let twiddled_re = Array.make n (Const 0.0) in
  let twiddled_im = Array.make n (Const 0.0) in
  twiddled_re.(0) <- Load (Input (0, true));
  twiddled_im.(0) <- Load (Input (0, false));
  for k = 1 to n - 1 do
    let xr = Load (Input (k, true)) in
    let xi = Load (Input (k, false)) in
    let (wr, wi) = twiddle_expr policy n k in
    let (out_re, out_im) = cmul_pattern xr xi wr wi in
    twiddled_re.(k) <- out_re;
    twiddled_im.(k) <- out_im
  done;
  let input_re k = twiddled_re.(k) in
  let input_im k = twiddled_im.(k) in
  let out_re, out_im = dft n input_re input_im in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true),  out_re.(k))  :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc

(* === SPILL-AWARE EXPANSION ===
 *
 * For codelets large enough to overflow the vector register budget
 * (R=32 with ~38 live values vs 32 ZMM), explicit spill management
 * beats GCC's automatic spilling. Hand-coded codelets at this size
 * declare a stack-resident spill_re/spill_im buffer, store all PASS 1
 * outputs to it, then reload at the start of PASS 2. The result is
 * organized memory traffic with predictable stride, vs GCC's scattered
 * spill choices.
 *
 * To support spill emission, the math layer needs to identify which
 * Expr.expr trees correspond to PASS 1 outputs of the outermost CT
 * decomposition. We do this by manually expanding the outermost CT
 * step instead of recursing through `dft`, capturing pass1_re/pass1_im
 * before they get fed into INTERNAL TWIDDLES + PASS 2. Inner sub-DFTs
 * (recursive calls below the outermost level) still use plain `dft`. *)

(* Per-output-bin spill marker. For pass1 output at (n1_idx, k2):
 *   slot     = n1_idx * N2 + k2
 *   re_expr  = the Expr.expr to materialize at spill_re[slot]
 *   im_expr  = the Expr.expr to materialize at spill_im[slot] *)
type spill_marker = {
  slot: int;
  re_expr: expr;
  im_expr: expr;
}

(* Spill-aware version of dft_expand_twiddled.
 *
 * Returns:
 *   - assignments: same shape as dft_expand_twiddled (output stores)
 *   - markers:     spill markers for PASS 1 outputs, empty if n has
 *                  no CT decomposition (Direct DFT case)
 *)
let dft_expand_twiddled_spill ?(policy = TP_Flat) (n : int)
    : Expr.assignment list * spill_marker list * (int * int) option =
  match pick_algorithm n with
  | Direct ->
    (* No CT structure → no spill boundary. Fall back to plain expansion. *)
    (dft_expand_twiddled ~policy n, [], None)
  | Cooley_Tukey (n1, n2) ->
    (* Same external twiddle pre-multiply as dft_expand_twiddled. *)
    let twiddled_re = Array.make n (Const 0.0) in
    let twiddled_im = Array.make n (Const 0.0) in
    twiddled_re.(0) <- Load (Input (0, true));
    twiddled_im.(0) <- Load (Input (0, false));
    for k = 1 to n - 1 do
      let xr = Load (Input (k, true)) in
      let xi = Load (Input (k, false)) in
      let (wr, wi) = twiddle_expr policy n k in
      let (out_re, out_im) = cmul_pattern xr xi wr wi in
      twiddled_re.(k) <- out_re;
      twiddled_im.(k) <- out_im
    done;
    let input_re k = twiddled_re.(k) in
    let input_im k = twiddled_im.(k) in

    (* MANUALLY drive the outermost CT step (instead of `dft n input_re input_im`)
     * so we can capture pass1_re/pass1_im as spill markers. The implementation
     * below is a copy of dft_ct's body — kept in sync with dft_ct. *)
    let pi = 4.0 *. atan 1.0 in

    (* PASS 1: N1 sub-FFTs of size N2 — same as dft_ct *)
    let pass1_re = Array.make_matrix n1 n2 (Const 0.0) in
    let pass1_im = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      let inner_input_re k2 = input_re (n1_idx + k2 * n1) in
      let inner_input_im k2 = input_im (n1_idx + k2 * n1) in
      let r, i = dft n2 inner_input_re inner_input_im in
      for k2 = 0 to n2 - 1 do
        pass1_re.(n1_idx).(k2) <- r.(k2);
        pass1_im.(n1_idx).(k2) <- i.(k2)
      done
    done;

    (* CAPTURE SPILL MARKERS: one per (n1_idx, k2) PASS 1 output bin. *)
    let markers = ref [] in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let slot = n1_idx * n2 + k2 in
        markers := { slot;
                     re_expr = pass1_re.(n1_idx).(k2);
                     im_expr = pass1_im.(n1_idx).(k2);
                   } :: !markers
      done
    done;

    (* INTERNAL TWIDDLES — same as dft_ct *)
    let twiddled_re_inner = Array.make_matrix n1 n2 (Const 0.0) in
    let twiddled_im_inner = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let theta = -2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
        let cr = cos theta in
        let ci = sin theta in
        let (tr, ti) = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
        twiddled_re_inner.(n1_idx).(k2) <- tr;
        twiddled_im_inner.(n1_idx).(k2) <- ti
      done
    done;

    (* PASS 2 — same as dft_ct *)
    let out_re = Array.make n (Const 0.0) in
    let out_im = Array.make n (Const 0.0) in
    for k2 = 0 to n2 - 1 do
      let outer_input_re n1_idx = twiddled_re_inner.(n1_idx).(k2) in
      let outer_input_im n1_idx = twiddled_im_inner.(n1_idx).(k2) in
      let r, i = dft n1 outer_input_re outer_input_im in
      for k1 = 0 to n1 - 1 do
        out_re.(k1 * n2 + k2) <- r.(k1);
        out_im.(k1 * n2 + k2) <- i.(k1)
      done
    done;

    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true),  out_re.(k))  :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    (List.rev !acc, List.rev !markers, Some (n1, n2))

(* Heuristic: should this radix size benefit from explicit spill emission?
 * Returns true when peak live-set is expected to exceed register budget.
 *
 * Estimate: peak live ≈ n (all PASS 1 outputs alive across boundary)
 * Plus some headroom for inputs and twiddles being consumed: ~6
 * If n + 6 > vec_regs, GCC will spill anyway; we should do it ourselves.
 *
 * At this estimation level, the math is:
 *   n=8:  14 < 32 ZMM  → no spill needed (and 14 < 16 YMM marginal)
 *   n=16: 22 < 32 ZMM  → no spill needed (22 > 16 YMM, marginal for AVX-2)
 *   n=32: 38 > 32 ZMM  → spill needed for both
 *   n=64: 70 > 32 ZMM  → spill very needed
 *
 * This is a conservative estimate; the actual peak depends on scheduler
 * decisions and is bounded above by n (PASS 1 width). *)
let should_spill (n : int) (vec_regs : int) : bool =
  n + 6 > vec_regs
