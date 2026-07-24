(* dft.ml — c2c math-layer facade: twiddle policy + expansion wrappers.
 *
 * Top of the Dft chain (Dft_select < Dft_recurse < Dft). `include`
 * re-exports both lower layers, so every Dft.X spelling keeps working:
 * Dft.pick_algorithm comes from the select layer, Dft.dft from the
 * recursion, and this file adds what codelet emission actually calls:
 *
 *   - twiddle_policy (TP_Flat / TP_Log3) + twiddle_expr, and the DIT /
 *     DIF direction type — how runtime twiddles enter the DAG;
 *   - cmul_pattern — the Sub(Mul,Mul) / Add(Mul,Mul) shape that
 *     Algsimp.of_expr lifts into opaque Cmul atoms so reassociation
 *     cannot shred complex multiplies;
 *   - the dft_expand family — assignment-list packaging of the
 *     recursion for every codelet variant (n1, t1 dit / dif, twidsq,
 *     il2, blocked n1, newsplit) including the spill variants that
 *     return PASS 1 / PASS 2 markers;
 *   - spill_marker and the blocked-recipe predicates should_spill /
 *     should_block_n1 / exceeds_register_budget that gen_main and
 *     codelet_oop query when deciding recipe defaults.
 * ------------------------------------------------------------------
 * MODULE CARD (dft.ml — grep "MODULE CARD" for the full set)
 * ROLE: Facade + twiddle policy + dft_expand wrappers + spill markers.
 * PIPELINE: gen_main / codelet_oop / dft_r2c call these wrappers; the
 * output feeds Algsimp.of_assignments.
 * PUBLIC SURFACE (measured; grep counts incl. comments): dft_r2c(35),
 * gen_main(31), codelet_oop(19), pipeline(6), algsimp(5) + bin
 * dbg_eval(4), dump_ir(2), facdrv(4). Hot names: dft, pick_algorithm,
 * TP_Flat / TP_Log3, dft_expand_twiddled_spill, should_block_n1.
 * DEPS: Dft_recurse via include; Expr (open); Split_radix(2).
 * ENV (via the chain): VFFT_SPLIT_RADIX, VFFT_NEWSPLIT, VFFT_CT_FACTOR.
 * ------------------------------------------------------------------
 *)

include Dft_recurse
open Expr

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
 *   TP_PowW1 — load ONLY W^1 (slot 0) and derive EVERY higher power
 *             in-DAG: W^(2m) = (W^m)² (squaring), W^j = W^p·W^(j-p)
 *             with p the highest pow2 ≤ j. The zil terminator /
 *             t2sp/t2sq shape (z_cascade_plan §4.99): critical path for
 *             R=8 is w1→w2→w4→{w5,w6,w7} = 3 cmul links vs 6 for a
 *             running-product chain. At emission the single Twiddle(0)
 *             slot renders as a per-lane VECTOR load (packed per-column
 *             w¹, e.g. zsplit.h's twq 16 B/col), so each lane derives
 *             its own column's powers elementwise; the math layer stays
 *             lane-agnostic.
 *
 * The math-layer payoff: log3/poww1 are *substitutions* — wherever t1_dit
 * emits Load(Twiddle(j, ...)), they substitute the derivation tree.
 * Algsimp's Cmul handling propagates; nothing else changes. *)

type twiddle_policy =
  | TP_Flat
  | TP_Log3
  | TP_PowW1

(* DIT vs DIF — duals of each other:
 *   DIT (Decimation-In-Time):   y = DFT(W ⋅ x)   — twiddle on INPUT, pre-butterfly
 *   DIF (Decimation-In-Frequency): y = W ⋅ DFT(x) — twiddle on OUTPUT, post-butterfly
 *
 * In a CT recursion, you typically pair DIT codelets at one level with
 * DIF codelets at the next so that the twiddle layer flips. FFTW emits
 * both styles for this reason. *)
type direction =
  | DIT
  | DIF

(* Build a complex multiplication as (out_re, out_im) using the cmul pattern
 * that Algsimp.of_expr will lift to Cmul nodes.
 *
 *   ~conj:false (default)  →  (a + ib) · (c + id)        (Fwd external twiddle)
 *   ~conj:true             →  (a + ib) · conj(c + id)   (Bwd external twiddle)
 *
 * Bwd codelets receive the SAME W array as fwd (caller stores forward
 * twiddles); the codelet conjugates internally via this flag.
 * Internal log3 derivations multiply two stored W values that share the
 * same convention (whatever it is) — those calls use ~conj:false.
 *)
let cmul_pattern ?(conj = false) (ar : expr) (ai : expr) (br : expr) (bi : expr)
  : expr * expr
  =
  if conj
  then (
    (* (a + ib) · (c - id) = (ac + bd) + i(bc - ad) *)
    let out_re = Add (Mul (ar, br), Mul (ai, bi)) in
    let out_im = Sub (Mul (ai, br), Mul (ar, bi)) in
    out_re, out_im)
  else (
    let out_re = Sub (Mul (ar, br), Mul (ai, bi)) in
    let out_im = Add (Mul (ar, bi), Mul (ai, br)) in
    out_re, out_im)
;;

(* Compute the (re, im) Expr trees for the j-th twiddle (1-indexed)
 * under the given policy and radix. Memoization is critical: derived
 * twiddles like W^7 = W^3·W^4 reference W^3, which must be the SAME
 * expression tree (pointer-equal at the Expr level, hash-cons-equal
 * after lifting) so that the underlying cmul gets shared. *)
let twiddle_expr (policy : twiddle_policy) (n : int) (j : int) : expr * expr =
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
      Load (Twiddle (j - 1, true)), Load (Twiddle (j - 1, false))
    | TP_Log3 ->
      (* Generalized log3: load only the power-of-2 twiddles W^(2^k),
       * derive everything else by binary decomposition.
       *
       * Slot indexing matches TP_Flat (slot = j - 1) so the bench
       * harness fills the same twiddle array regardless of policy —
       * TP_Log3 simply consults a sparse subset of slots:
       *   W^1 → slot 0, W^2 → slot 1, W^4 → slot 3,
       *   W^8 → slot 7, W^16 → slot 15, W^32 → slot 31.
       *
       * Total slot reads per kstep:
       *   R=16: 4 (vs 15 flat)
       *   R=32: 5 (vs 31 flat)
       *   R=64: 6 (vs 63 flat)
       *
       * Decomposition: split j = p + q where p is the highest power
       * of 2 ≤ j. With memoization, each W^k computed once; hash-cons
       * dedupes across legs.
       *
       * Cmul cost per derivation: 4 muls + 2 adds (6 flops). Total:
       *   R=16: 4 loads + 11 cmuls
       *   R=32: 5 loads + 26 cmuls
       *   R=64: 6 loads + 57 cmuls
       *
       * Tradeoff: log3 saves twiddle bandwidth at the cost of arith.
       * Whether it wins depends on which is the bottleneck. *)
      let is_pow2 x = x > 0 && x land (x - 1) = 0 in
      let highest_pow2_le j =
        let rec loop p = if p * 2 > j then p else loop (p * 2) in
        loop 1
      in
      if j < 1 || j >= n
      then failwith (Printf.sprintf "TP_Log3: j=%d out of range for n=%d" j n)
      else if is_pow2 j
      then
        (* Direct load — slot = j - 1, matching TP_Flat layout. *)
        Load (Twiddle (j - 1, true)), Load (Twiddle (j - 1, false))
      else (
        (* Split j = p + q, derive W^j = W^p · W^q. *)
        let p = highest_pow2_le j in
        let q = j - p in
        let wpr, wpi = lookup p in
        let wqr, wqi = lookup q in
        cmul_pattern wpr wpi wqr wqi)
    | TP_PowW1 ->
      (* Squaring-tree derivation from W^1 alone. Same binary split as
       * TP_Log3 for non-pow2 j, but pow2 powers are DERIVED by squaring
       * ((W^m)²) instead of loaded — the only Twiddle slot consulted is
       * slot 0 (= W^1). Memoization + hash-consing share sub-powers, so
       * for R=8 legs 2..7 cost exactly 6 cmuls at depth ≤ 3:
       *   w2=w1², w3=w2·w1, w4=w2², w5=w4·w1, w6=w4·w2, w7=w4·w3
       * — matching codelet_zil's index arrays [0;0;1;2;2;4;4;4] /
       * [0;0;1;1;2;1;2;3]. Bwd note: a conjugated w¹ table (zsplit.h
       * twqb) derives conj powers automatically (conj(a)·conj(b) =
       * conj(a·b)); combine with ~table_conj:true so the cmul layer
       * doesn't conjugate a second time. *)
      let is_pow2 x = x > 0 && x land (x - 1) = 0 in
      let highest_pow2_le j =
        let rec loop p = if p * 2 > j then p else loop (p * 2) in
        loop 1
      in
      if j < 1 || j >= n
      then failwith (Printf.sprintf "TP_PowW1: j=%d out of range for n=%d" j n)
      else if j = 1
      then Load (Twiddle (0, true)), Load (Twiddle (0, false))
      else if is_pow2 j
      then (
        let hr, hi = lookup (j / 2) in
        cmul_pattern hr hi hr hi)
      else (
        let p = highest_pow2_le j in
        let q = j - p in
        let wpr, wpi = lookup p in
        let wqr, wqi = lookup q in
        cmul_pattern wpr wpi wqr wqi)
  in
  lookup j
;;

(* === ASSIGNMENT-LIST WRAPPERS ===
 *
 * Replace the old dft_expand / dft_expand_twiddled with versions that
 * call into this module's recursive dft. The output is the same
 * Expr.assignment list shape as before. *)

let dft_expand ?(sign = `Fwd) (n : int) : Expr.assignment list =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let out_re, out_im = dft ~sign n input_re input_im in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out_re.(k)) :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  (* Reorder so each output's re comes immediately before its im. *)
  let pairs = List.rev !acc in
  (* `acc` was built (re, im, re, im, ...) in reverse, so List.rev
   * gives the correct order. *)
  let _ = pairs in
  List.rev !acc
;;

(* Twiddled (t1_dit) form: pre-multiply inputs by runtime twiddles
 * (Cmul nodes), then run the (possibly Cooley-Tukey-decomposed) DFT
 * on the twiddled inputs.
 *
 * The cmul outputs use the SUB(MUL,MUL)/ADD(MUL,MUL) pattern that
 * Algsimp.of_expr's pattern detector recognizes and lifts to Cmul
 * opaque atoms. The CT decomposition then sees the cmul outputs as
 * leaf-like values in its own Add/Sub structure — algsimp won't
 * shred them because they're inside Cmul nodes after lifting. *)
let dft_expand_twiddled
      ?(policy = TP_Flat)
      ?(direction = DIT)
      ?(sign = `Fwd)
      ?(table_conj = false)
      (n : int)
  : Expr.assignment list
  =
  (* Structurally, twiddled codelets have two options:
   *   - PRE-twiddle: multiply inputs by twiddles, then run DFT
   *   - POST-twiddle: run DFT, then multiply outputs by twiddles
   *
   * For forward direction the choice mirrors the algorithm name:
   *   DIT fwd: PRE-twiddle  (decimation-in-time)
   *   DIF fwd: POST-twiddle (decimation-in-frequency)
   *
   * For backward (sign=Bwd), the codelet must be the inverse of the
   * corresponding forward codelet. Since (T·B)⁻¹ = B⁻¹·T⁻¹ and twiddle
   * doesn't commute with butterfly, the inverse has the OPPOSITE order:
   *   DIT bwd = inverse of (T then B) = B⁻¹ then T⁻¹  → POST-twiddle structure
   *   DIF bwd = inverse of (B then T) = T⁻¹ then B⁻¹ → PRE-twiddle structure
   *
   * The inverse butterfly B⁻¹ is the +θ DFT kernel (handled by `dft ~sign`)
   * and T⁻¹ for unit-magnitude twiddles is conj(T) (handled by ~conj here).
   *
   * Twiddle Exprs may be cmul derivation patterns (TP_Log3) or simple
   * Loads (TP_Flat). The per-leg Sub(Mul,Mul)/Add(Mul,Mul) cmul pattern
   * is preserved so Algsimp.of_expr can lift it to Cmul opaque atoms.
   *
   * ~table_conj: the caller's runtime twiddle TABLE is already conjugated
   * (zsplit.h stores −sin in twspb/twqb for the bwd kinds), so the cmul
   * layer must NOT conjugate again — the PRE/POST structure still follows
   * (direction, sign), only the conj flag is suppressed. Applies only when
   * sign = `Bwd; a fwd table is never conjugated. *)
  let conj = sign = `Bwd && not table_conj in
  let pre_twiddle =
    match direction, sign with
    | DIT, `Fwd -> true (* DIT fwd: T then B *)
    | DIT, `Bwd -> false (* inverse of DIT fwd: B⁻¹ then T_conj  → POST *)
    | DIF, `Fwd -> false (* DIF fwd: B then T *)
    | DIF, `Bwd -> true (* inverse of DIF fwd: T_conj then B⁻¹  → PRE *)
  in
  if pre_twiddle
  then (
    (* PRE-twiddle: multiply inputs by twiddles (conj if Bwd), then DFT. *)
    let twiddled_re = Array.make n (Const 0.0) in
    let twiddled_im = Array.make n (Const 0.0) in
    twiddled_re.(0) <- Load (Input (0, true));
    twiddled_im.(0) <- Load (Input (0, false));
    for k = 1 to n - 1 do
      let xr = Load (Input (k, true)) in
      let xi = Load (Input (k, false)) in
      let wr, wi = twiddle_expr policy n k in
      let out_re, out_im = cmul_pattern ~conj xr xi wr wi in
      twiddled_re.(k) <- out_re;
      twiddled_im.(k) <- out_im
    done;
    let input_re k = twiddled_re.(k) in
    let input_im k = twiddled_im.(k) in
    let out_re, out_im = dft ~sign n input_re input_im in
    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true), out_re.(k)) :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    List.rev !acc)
  else (
    (* POST-twiddle: run DFT on raw inputs, then twiddle the outputs
     * (conj if Bwd). *)
    let input_re k = Load (Input (k, true)) in
    let input_im k = Load (Input (k, false)) in
    let raw_re, raw_im = dft ~sign n input_re input_im in
    let acc = ref [] in
    (* Output 0: trivial twiddle, store as-is. *)
    acc := (Output (0, true), raw_re.(0)) :: !acc;
    acc := (Output (0, false), raw_im.(0)) :: !acc;
    for k = 1 to n - 1 do
      let wr, wi = twiddle_expr policy n k in
      let out_re, out_im = cmul_pattern ~conj raw_re.(k) raw_im.(k) wr wi in
      acc := (Output (k, true), out_re) :: !acc;
      acc := (Output (k, false), out_im) :: !acc
    done;
    let sorted =
      List.sort
        (fun (a, _) (b, _) ->
           match a, b with
           | Output (ka, ra), Output (kb, rb) ->
             if ka <> kb then compare ka kb else compare (not ra) (not rb)
           | _ -> 0)
        !acc
    in
    sorted)
;;

(* === OUT-OF-PLACE TWIDSQ EXPANSION (FFTW-style intermediate codelet) ===
 *
 * Builds the assignment DAG for an n×n "twiddle square" codelet:
 *   - Input:  row-major n×n block of complex values (n² elements)
 *   - Operation: apply inter-stage twiddle W^{i*k} to each (i, k), then
 *                run an n-point DFT along the k dimension of each row,
 *                producing row-result Y[i, j] for i, j ∈ [0, n).
 *   - Output: transposed layout — physical slot j*n + i gets Y[i, j].
 *
 * This corresponds to FFTW's gen_twidsq.ml codelet. Used at intermediate
 * stages of a multi-stage cascade where the layout transformation between
 * stages would otherwise require a separate transpose pass.
 *
 * Indexing conventions:
 *   - Input(i*n + k, _)            ← element at row i, position k
 *   - Twiddle((i-1)*(n-1) + (k-1), _) ← W^{i*k} for i ∈ [1, n), k ∈ [1, n)
 *     (Row 0 and column 0 have trivial W^0 = 1; no twiddle slots needed.)
 *   - Output(j*n + i, _)           ← row i's j-th DFT output, transposed store
 *
 * Number of twiddle slots: (n-1)² distinct values.
 *
 * NOTE: For the initial prototype we support DIT, TP_Flat only. DIF and
 * TP_Log3 can be added by parallel construction once the DAG semantics
 * are validated. Spill markers are not yet generated — large twidsq sizes
 * will need a parallel _spill variant similar to dft_expand_twiddled_spill.
 *)
let dft_expand_twidsq ?(direction = DIT) ?(sign = `Fwd) (n : int) : Expr.assignment list =
  let conj = sign = `Bwd in
  match direction with
  | DIT ->
    let acc = ref [] in
    for i = 0 to n - 1 do
      (* Step 1: Apply inter-stage twiddle to row i.
       * For position k > 0 of row i > 0: multiply by W^{i*k}.
       * For row 0 or position 0: pass-through (W^0 = 1). *)
      let twiddled_re = Array.make n (Const 0.0) in
      let twiddled_im = Array.make n (Const 0.0) in
      for k = 0 to n - 1 do
        let xr = Load (Input ((i * n) + k, true)) in
        let xi = Load (Input ((i * n) + k, false)) in
        if i = 0 || k = 0
        then (
          twiddled_re.(k) <- xr;
          twiddled_im.(k) <- xi)
        else (
          let twiddle_slot = ((i - 1) * (n - 1)) + (k - 1) in
          let wr = Load (Twiddle (twiddle_slot, true)) in
          let wi = Load (Twiddle (twiddle_slot, false)) in
          let out_re, out_im = cmul_pattern ~conj xr xi wr wi in
          twiddled_re.(k) <- out_re;
          twiddled_im.(k) <- out_im)
      done;
      (* Step 2: Compute DFT-n on the twiddled row.
       * Reuses the existing dft machinery — math layer is buffer-agnostic. *)
      let input_re k = twiddled_re.(k) in
      let input_im k = twiddled_im.(k) in
      let out_re, out_im = dft ~sign n input_re input_im in
      (* Step 3: Store row i's outputs TRANSPOSED.
       * Y[i, j] goes to physical slot j*n + i (transposed layout). *)
      for j = n - 1 downto 0 do
        acc := (Output ((j * n) + i, true), out_re.(j)) :: !acc;
        acc := (Output ((j * n) + i, false), out_im.(j)) :: !acc
      done
    done;
    List.rev !acc
  | DIF -> failwith "dft_expand_twidsq: DIF direction not yet implemented"
;;

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
type spill_marker =
  { slot : int
  ; re_expr : expr
  ; im_expr : expr
  }

(* Spill-aware version of dft_expand_twiddled.
 *
 * Returns:
 *   - assignments: same shape as dft_expand_twiddled (output stores)
 *   - markers:     spill markers for PASS 1 outputs, empty if n has
 *                  no CT decomposition (Direct DFT case)
 *)
(* IL2 (section 72): TWO independent column instances concatenated at
 * the math layer. Instance 1 shifts Input/Output slots by +n and
 * Twiddle slots by +(n-1). The SU scheduler braids the two
 * independent DAGs by readiness — generator-level column
 * interleaving, the probe for the latency-bound diagnosis. *)
let dft_expand_twiddled_il2
      ?(policy = TP_Flat)
      ?(direction = DIT)
      ?(sign = `Fwd)
      (n : int)
  : Expr.assignment list
  =
  let base = dft_expand_twiddled ~policy ~direction ~sign n in
  let shift_ref = function
    | Expr.Input (k, p) -> Expr.Input (k + n, p)
    | Expr.Output (k, p) -> Expr.Output (k + n, p)
    | Expr.Twiddle (t, p) -> Expr.Twiddle (t + (n - 1), p)
  in
  let rec shift = function
    | Const c -> Const c
    | Load r -> Load (shift_ref r)
    | Neg e -> Neg (shift e)
    | Add (a, b) -> Add (shift a, shift b)
    | Sub (a, b) -> Sub (shift a, shift b)
    | Mul (a, b) -> Mul (shift a, shift b)
  in
  let inst2 = List.map (fun (lhs, e) -> shift_ref lhs, shift e) base in
  base @ inst2
;;

let dft_expand_twiddled_spill
      ?(policy = TP_Flat)
      ?(direction = DIT)
      ?(sign = `Fwd)
      (n : int)
  : Expr.assignment list * spill_marker list * (int * int) option
  =
  let conj = sign = `Bwd in
  let sgn =
    match sign with
    | `Fwd -> -1.0
    | `Bwd -> 1.0
  in
  match pick_algorithm n with
  | Direct ->
    (* No CT structure → no spill boundary. Fall back to plain expansion. *)
    dft_expand_twiddled ~policy ~direction ~sign n, [], None
  | Split_radix ->
    (* Split-radix doesn't have the same PASS 1 / PASS 2 cluster boundary
     * that the recipe machinery is calibrated against. SR's structure is
     * three recursive sub-DFTs (E of size N/2, O1 and O3 of size N/4) whose
     * outputs combine in a different topology than CT's symmetric two-pass.
     *
     * For this first PR we fall back to plain (non-recipe) expansion: the
     * codelet still generates correctly, just without spill markers. A
     * follow-up PR will design a recipe topology for SR that gives R=32/64
     * comparable spill management to what they get under CT today.
     *
     * This is the same fallback path Direct takes — both algorithms simply
     * lack a clean cluster boundary for the current recipe shape. *)
    dft_expand_twiddled ~policy ~direction ~sign n, [], None
  | Cooley_Tukey (n1, n2) ->
    (* Pre vs post twiddle, decided by (direction, sign) — see comment in
     * dft_expand_twiddled. Forward DIT and backward DIF use PRE-twiddle;
     * forward DIF and backward DIT use POST-twiddle. *)
    let pre_twiddle =
      match direction, sign with
      | DIT, `Fwd -> true
      | DIT, `Bwd -> false
      | DIF, `Fwd -> false
      | DIF, `Bwd -> true
    in
    let input_re, input_im =
      if pre_twiddle
      then (
        let twiddled_re = Array.make n (Const 0.0) in
        let twiddled_im = Array.make n (Const 0.0) in
        twiddled_re.(0) <- Load (Input (0, true));
        twiddled_im.(0) <- Load (Input (0, false));
        for k = 1 to n - 1 do
          let xr = Load (Input (k, true)) in
          let xi = Load (Input (k, false)) in
          let wr, wi = twiddle_expr policy n k in
          let out_re, out_im = cmul_pattern ~conj xr xi wr wi in
          twiddled_re.(k) <- out_re;
          twiddled_im.(k) <- out_im
        done;
        (fun k -> twiddled_re.(k)), fun k -> twiddled_im.(k))
      else (fun k -> Load (Input (k, true))), fun k -> Load (Input (k, false))
    in
    (* MANUALLY drive the outermost CT step (instead of `dft n input_re input_im`)
     * so we can capture pass1_re/pass1_im as spill markers. The implementation
     * below is a copy of dft_ct's body — kept in sync with dft_ct. *)
    let pi = 4.0 *. atan 1.0 in
    (* PASS 1: N1 sub-FFTs of size N2 — same as dft_ct *)
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
    (* CAPTURE SPILL MARKERS: one per (n1_idx, k2) PASS 1 output bin. *)
    let markers = ref [] in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let slot = (n1_idx * n2) + k2 in
        markers
        := { slot; re_expr = pass1_re.(n1_idx).(k2); im_expr = pass1_im.(n1_idx).(k2) }
           :: !markers
      done
    done;
    (* INTERNAL TWIDDLES — same as dft_ct, sign-flipped for Bwd *)
    let twiddled_re_inner = Array.make_matrix n1 n2 (Const 0.0) in
    let twiddled_im_inner = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let theta = sgn *. 2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
        let cr = cos theta in
        let ci = sin theta in
        let tr, ti = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
        twiddled_re_inner.(n1_idx).(k2) <- tr;
        twiddled_im_inner.(n1_idx).(k2) <- ti
      done
    done;
    (* PASS 2 — same as dft_ct *)
    let raw_re = Array.make n (Const 0.0) in
    let raw_im = Array.make n (Const 0.0) in
    for k2 = 0 to n2 - 1 do
      let outer_input_re n1_idx = twiddled_re_inner.(n1_idx).(k2) in
      let outer_input_im n1_idx = twiddled_im_inner.(n1_idx).(k2) in
      let r, i = dft ~sign n1 outer_input_re outer_input_im in
      for k1 = 0 to n1 - 1 do
        raw_re.((k1 * n2) + k2) <- r.(k1);
        raw_im.((k1 * n2) + k2) <- i.(k1)
      done
    done;
    (* Post-twiddle (if POST structure) — applied to legs 1..n-1. *)
    let out_re = Array.make n (Const 0.0) in
    let out_im = Array.make n (Const 0.0) in
    if pre_twiddle
    then (
      Array.blit raw_re 0 out_re 0 n;
      Array.blit raw_im 0 out_im 0 n)
    else (
      out_re.(0) <- raw_re.(0);
      out_im.(0) <- raw_im.(0);
      for k = 1 to n - 1 do
        let wr, wi = twiddle_expr policy n k in
        let or_, oi = cmul_pattern ~conj raw_re.(k) raw_im.(k) wr wi in
        out_re.(k) <- or_;
        out_im.(k) <- oi
      done);
    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true), out_re.(k)) :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    List.rev !acc, List.rev !markers, Some (n1, n2)
;;

(* No-twiddle (n1) variant with spill markers between PASS 1 and PASS 2.
 *
 * Doc 58 motivation: the monolithic n1 path (dft_expand on line ~684)
 * calls `dft ~sign n` which recursively expands to dft_ct, but builds
 * the result as one big DAG with no pass boundary visible to the
 * scheduler. At R=32+ on AVX-512 this produces peak_live 260-994
 * (8-31× over the 32-zmm budget) and vmovapd counts 2.8-3.4× hand's
 * NFUSE-blocked implementations.
 *
 * The fix is structural: mirror what dft_expand_twiddled_spill does for
 * t1 — manually drive the outermost CT step and emit spill markers
 * between PASS 1 and PASS 2 — but without any outer external-twiddle
 * Cmul layer. The internal twiddles between passes are constants
 * (const_cmul), known at codegen time.
 *
 * After this change, n1 codelets inherit the full SU+spill recipe
 * machinery that t1 already benefits from: bounded per-pass working
 * set, M-project pinning that actually has room to operate, and
 * selective pinning's auto-fusion freedom. Hand-NFUSE-level vmovapd
 * counts become achievable.
 *
 * Returns (assigns, markers, Some (n1, n2)). For sizes that don't
 * benefit from blocking (Direct primes, sizes ≤ threshold where
 * monolithic already fits or wins), falls back to plain dft_expand
 * with empty markers. *)
let dft_expand_n1_blocked ?(sign = `Fwd) (n : int)
  : Expr.assignment list * spill_marker list * (int * int) option
  =
  let pi = 4.0 *. atan 1.0 in
  let sgn =
    match sign with
    | `Fwd -> -1.0
    | `Bwd -> 1.0
  in
  match pick_algorithm n with
  | Direct ->
    (* Primes: no CT structure → no pass boundary → no blocking benefit.
     * Fall back to plain expansion with empty markers. *)
    dft_expand ~sign n, [], None
  | Split_radix ->
    (* SR has a different topology than CT's symmetric two-pass; the
     * spill recipe machinery isn't calibrated for it (same caveat as
     * dft_expand_twiddled_spill). Fall back. A follow-up could add an
     * SR-specific blocked path. *)
    dft_expand ~sign n, [], None
  | Cooley_Tukey (n1, n2) ->
    let input_re k = Load (Input (k, true)) in
    let input_im k = Load (Input (k, false)) in
    (* PASS 1: N1 sub-FFTs of size N2 — identical to dft_ct.
     * For each n1_idx in [0, N1), compute DFT-N2 on the strided slice
     *   x[n1_idx], x[n1_idx + N1], x[n1_idx + 2·N1], ...
     * Inner DFTs recurse via `dft ~sign n2` — uses CT or Direct as
     * pick_algorithm decides for n2. *)
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
    (* CAPTURE SPILL MARKERS: one per (n1_idx, k2) PASS 1 output bin.
     * Slot indexing matches dft_expand_twiddled_spill so emit_c's spill
     * info construction sees an identical-shape marker list. *)
    let markers = ref [] in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let slot = (n1_idx * n2) + k2 in
        markers
        := { slot; re_expr = pass1_re.(n1_idx).(k2); im_expr = pass1_im.(n1_idx).(k2) }
           :: !markers
      done
    done;
    (* INTERNAL TWIDDLES: multiply pass1[n1_idx][k2] by Ï_N^{n1_idxÂ·k2}.
     * Identical to dft_ct's twiddle stage (these are codegen-time
     * constants, not runtime loads). For (n1_idx=0 â¨ k2=0) the twiddle
     * is 1 and const_cmul folds away. *)
    let twiddled_re_inner = Array.make_matrix n1 n2 (Const 0.0) in
    let twiddled_im_inner = Array.make_matrix n1 n2 (Const 0.0) in
    for n1_idx = 0 to n1 - 1 do
      for k2 = 0 to n2 - 1 do
        let theta = sgn *. 2.0 *. pi *. float_of_int (n1_idx * k2) /. float_of_int n in
        let cr = cos theta in
        let ci = sin theta in
        let tr, ti = const_cmul pass1_re.(n1_idx).(k2) pass1_im.(n1_idx).(k2) cr ci in
        twiddled_re_inner.(n1_idx).(k2) <- tr;
        twiddled_im_inner.(n1_idx).(k2) <- ti
      done
    done;
    (* PASS 2: N2 sub-FFTs of size N1 — identical to dft_ct. *)
    let out_re = Array.make n (Const 0.0) in
    let out_im = Array.make n (Const 0.0) in
    for k2 = 0 to n2 - 1 do
      let outer_input_re n1_idx = twiddled_re_inner.(n1_idx).(k2) in
      let outer_input_im n1_idx = twiddled_im_inner.(n1_idx).(k2) in
      let r, i = dft ~sign n1 outer_input_re outer_input_im in
      for k1 = 0 to n1 - 1 do
        out_re.((k1 * n2) + k2) <- r.(k1);
        out_im.((k1 * n2) + k2) <- i.(k1)
      done
    done;
    let acc = ref [] in
    for k = n - 1 downto 0 do
      acc := (Output (k, true), out_re.(k)) :: !acc;
      acc := (Output (k, false), out_im.(k)) :: !acc
    done;
    List.rev !acc, List.rev !markers, Some (n1, n2)
;;

(* Doc-58-style blocked expansion for the NEWSPLIT construction (scaled
 * conjugate-pair split radix). Same marker contract as
 * dft_expand_n1_blocked, but the seam is split-radix's natural E/O1/O3
 * boundary instead of CT's symmetric two-pass. The fake ct = (4, n/4)
 * exists purely to drive the emitter's cluster arithmetic — see
 * Split_radix.dft_newsplit_blocked for the slot layout. *)
let dft_expand_newsplit_blocked ?(sign = `Fwd) (n : int)
  : Expr.assignment list * spill_marker list * (int * int) option
  =
  let input_re k = Load (Input (k, true)) in
  let input_im k = Load (Input (k, false)) in
  let out_re, out_im, raw_markers =
    Split_radix.dft_newsplit_blocked ~sign n input_re input_im
  in
  let markers =
    List.map (fun (slot, re_expr, im_expr) -> { slot; re_expr; im_expr }) raw_markers
  in
  let acc = ref [] in
  for k = n - 1 downto 0 do
    acc := (Output (k, true), out_re.(k)) :: !acc;
    acc := (Output (k, false), out_im.(k)) :: !acc
  done;
  List.rev !acc, markers, Some (4, n / 4)
;;

(* Cost-model rule: should this codelet use the full spill+SU recipe?
 *
 * The rule is empirically derived from benchmarks across (R, ISA, K)
 * combinations. See docs/12_avx2_finding.md for the supporting data.
 *
 * Two clauses:
 *   1. n + 6 > vec_regs:
 *      Peak live exceeds register budget. GCC will re-spill aggressively
 *      without our help. Spill+SU is necessary.
 *
 *   2. vec_regs >= 32 (AVX-512):
 *      Even when peak live fits, AVX-512's wider vectors mean spill
 *      overhead is amortized over more useful work, AND GCC's scheduling
 *      on our DAG produces sub-optimal code that explicit spill+SU fixes.
 *      The recipe wins or ties at every R≥4 we've measured on AVX-512.
 *
 * Empirical bench summary (median SU+Spill / Topo, lower = better):
 *
 *   AVX-512 (vec_regs=32)
 *     R=4:  ~tied (noise)
 *     R=8:  0.90-0.98  ← clause (2) catches this
 *     R=16: substantial wins
 *     R=32: 0.61-0.83
 *     R=64: 0.60-0.92
 *
 *   AVX2 (vec_regs=16)
 *     R=8:  1.00-1.08  ← regression — clause (1) excludes this
 *     R=16: 0.80-0.88  ← clause (1) catches this (22 > 16)
 *     R=32: 0.56-0.81  ← clause (1) catches this (38 > 16)
 *
 *   AVX2 R=5/R=7 update (post-doc-28 fma_lift fix):
 *     The R=8 1.00-1.08 regression noted above was measured WITH fma_lift
 *     unconditional — that's the same regression class doc 28 traced for
 *     composites (extra register pressure from explicit NK_Fma). With
 *     fma_lift gated to primes only, the recipe path is healthy on AVX2 for
 *     small primes too. R=5 and R=7 with the recipe win 5-18% on every
 *     variant; without it they lose 4-37%. So the threshold extends to n≥5
 *     unconditionally. (Clause (3) below.)
 *
 * The unified rule: use the recipe iff CT-decomposed AND any clause holds. *)
let should_spill (n : int) (vec_regs : int) : bool =
  n + 6 > vec_regs || vec_regs >= 32 || n >= 5
;;

(* Compatibility: callers may want just clause (1) for register-pressure
 * predictions independent of ISA-specific GCC behavior. *)
let exceeds_register_budget (n : int) (vec_regs : int) : bool = n + 6 > vec_regs

(* Doc 58: should the n1 (no-twiddle) codelet use blocked construction
 * (with spill markers between PASS 1 and PASS 2) instead of monolithic?
 *
 * Empirical map of monolithic n1 peak_live vs register budget:
 *   AVX-512 (32 zmm):
 *     R=16   peak_live ~40   mild overflow, monolithic wins (68 vmovapd
 *                            vs hand-blocked 85) — keep monolithic
 *     R=32   peak_live 260   8× over budget, vmovapd 403 (2.8× hand)
 *     R=64   peak_live 267   8× over budget, vmovapd 1363 (3.4× hand)
 *     R=128  peak_live 498   15× over budget, vmovapd explosion
 *     R=256  peak_live 994   31× over budget
 *   AVX2 (16 ymm):
 *     R=8    peak_live ~20   mild overflow, monolithic still wins
 *     R=16   peak_live ~40   borderline, blocking starts to help
 *
 * Crossover: monolithic wins while peak_live stays within ~1.5× budget;
 * above that, the spill cascade dominates and blocked wins. The
 * threshold maps to n ≥ 32 on AVX-512 and n ≥ 32 on AVX2 (where the
 * smaller budget pushes the crossover earlier in absolute n but later
 * relative to budget because the per-codelet sizes are smaller).
 *
 * Empirical threshold: n ≥ 25. The original threshold was n ≥ 32, on
 * the assumption that smaller sizes "already beat hand because the whole
 * DFT fits in registers". Verified for R≤16 (R=16 ties FFTW at 144 ops
 * with 0 muls; whole codelet fits in 32 ZMM registers). But R=25 has
 * 384 ops and a natural 5×5 CT split — its inter-pass live set (25
 * complex values) does NOT fit, so the monolithic emit thrashes
 * registers and produces 1128 vector instructions (450 reg-to-reg
 * moves, 434 stack spills) for an 8.50 ns/transform AVX-512 result.
 * The blocked emit gives 67.98 ns/call (4.7 ns/transform) at AVX-512
 * — a 47% speedup at AVX-512, 39% at AVX2, beating Tugbars's hand-
 * generated 5×5 codelet by 12%.
 *
 * Verified: R=25 AVX-512 default (monolithic) vs blocked numerically
 * identical to 8.88e-16 (machine epsilon); both match FFTW within
 * 1e-13. See /tmp/bench_r25/bench_4way.c.
 *
 * Threshold remains a lower bound. Sizes < 25 (which means R≤16 for
 * CT-decomposed radices given our pick_algorithm) keep monolithic. *)
let should_block_n1 (n : int) (vec_regs : int) : bool =
  (* ISA-aware threshold (section 34). Doc-58's n>=25 was calibrated
   * in the avx512 era; on 16-register targets the R=16 monolithic
   * peak is 35 live values, and blocking it measured: vec movs 83 to
   * 62 with ~zero overflow, arith unchanged, bit-exact, +16.5 percent
   * runtime (+33.6 stacked with the regalloc widening). Default is
   * now 16 when vec_regs <= 16, 25 otherwise; VFFT_N1_BLOCK_MIN
   * remains as an experiment/back-out override in both directions. *)
  let default_min = if vec_regs <= 16 then 16 else 25 in
  let block_min =
    match Sys.getenv_opt "VFFT_N1_BLOCK_MIN" with
    | Some s ->
      (try int_of_string s with
       | _ -> default_min)
    | None -> default_min
  in
  match pick_algorithm n with
  | Direct -> false (* primes: no CT structure to block *)
  | Split_radix -> false (* SR topology not calibrated for recipe *)
  | Cooley_Tukey _ -> n >= block_min
;;
