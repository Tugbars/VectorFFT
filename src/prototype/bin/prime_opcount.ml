(* prime_opcount.ml — measure op count after dft_direct + algsimp for primes 2,3,5,7,11,13,17,19. *)

open Vfft_v2

(* Numerical evaluator for an algsimp DAG. Used to verify that
 * transformations preserve semantics. Optionally fed twiddle inputs
 * for testing twiddled (t1_dit / t1_dif) variants. *)
let eval_dag (assigns : (Expr.elem_ref * Algsimp.t) list) (n : int)
    (input_re : int -> float) (input_im : int -> float)
    ?(tw_re = (fun _ -> 0.0)) ?(tw_im = (fun _ -> 0.0)) ()
    : (int -> float) * (int -> float) =
  let memo : (int, float) Hashtbl.t = Hashtbl.create 256 in
  let rec eval (e : Algsimp.t) : float =
    match Hashtbl.find_opt memo e.tag with
    | Some v -> v
    | None ->
      let v = match e.node with
        | Algsimp.NK_Const c -> c
        | Algsimp.NK_Load r ->
          (match r with
           | Expr.Input (i, true) -> input_re i
           | Expr.Input (i, false) -> input_im i
           | Expr.Twiddle (j, true) -> tw_re j
           | Expr.Twiddle (j, false) -> tw_im j
           | _ -> 0.0)
        | Algsimp.NK_Neg a -> -. eval a
        | Algsimp.NK_Add (a, b) -> eval a +. eval b
        | Algsimp.NK_Sub (a, b) -> eval a -. eval b
        | Algsimp.NK_Mul (a, b) -> eval a *. eval b
        | Algsimp.NK_CmulRe (xr, xi, wr, wi) ->
          eval xr *. eval wr -. eval xi *. eval wi
        | Algsimp.NK_CmulIm (xr, xi, wr, wi) ->
          eval xr *. eval wi +. eval xi *. eval wr
        | Algsimp.NK_Fma (a, b, c, neg_mul, neg_add) ->
          let mul_part = eval a *. eval b in
          let mul_signed = if neg_mul then -. mul_part else mul_part in
          let add_signed = if neg_add then -. eval c else eval c in
          mul_signed +. add_signed
      in
      Hashtbl.add memo e.tag v;
      v
  in
  let out_re = Array.make n 0.0 in
  let out_im = Array.make n 0.0 in
  List.iter (fun (oref, e) ->
    match oref with
    | Expr.Output (k, true) -> out_re.(k) <- eval e
    | Expr.Output (k, false) -> out_im.(k) <- eval e
    | _ -> ()
  ) assigns;
  (Array.get out_re, Array.get out_im)

(* Reference: direct forward DFT *)
let ref_dft n input_re input_im : (int -> float) * (int -> float) =
  let out_re = Array.make n 0.0 in
  let out_im = Array.make n 0.0 in
  let two_pi = 2.0 *. Float.pi in
  for k = 0 to n - 1 do
    let sr = ref 0.0 in
    let si = ref 0.0 in
    for j = 0 to n - 1 do
      let theta = -. two_pi *. float_of_int (k * j) /. float_of_int n in
      let c = cos theta in
      let s = sin theta in
      sr := !sr +. input_re j *. c -. input_im j *. s;
      si := !si +. input_re j *. s +. input_im j *. c
    done;
    out_re.(k) <- !sr;
    out_im.(k) <- !si
  done;
  (Array.get out_re, Array.get out_im)

(* For twiddled variants:
 *   t1_dit:  output[k] = DFT(twiddle[j] · input[j])
 *   t1_dif:  output[k] = twiddle[k] · DFT(input[j])
 * The bench harness fills tw[j-1] = exp(-2πi·j/n), j=1..n-1; tw_re[0]
 * conventionally holds twiddle for slot 0 = j=1 (slot indexing matches
 * dft.ml's twiddle_expr policy). *)
let ref_dft_twiddled ~direction n input_re input_im tw_re tw_im
    : (int -> float) * (int -> float) =
  match direction with
  | `DIT ->
    let pre_re = Array.init n (fun j ->
      if j = 0 then input_re 0
      else input_re j *. tw_re (j - 1) -. input_im j *. tw_im (j - 1))
    in
    let pre_im = Array.init n (fun j ->
      if j = 0 then input_im 0
      else input_re j *. tw_im (j - 1) +. input_im j *. tw_re (j - 1))
    in
    ref_dft n (Array.get pre_re) (Array.get pre_im)
  | `DIF ->
    let (raw_re, raw_im) = ref_dft n input_re input_im in
    let post_re k =
      if k = 0 then raw_re k
      else raw_re k *. tw_re (k - 1) -. raw_im k *. tw_im (k - 1)
    in
    let post_im k =
      if k = 0 then raw_im k
      else raw_re k *. tw_im (k - 1) +. raw_im k *. tw_re (k - 1)
    in
    (post_re, post_im)

let test_inputs n =
  let in_re i = sin (float_of_int (i * 7 + 1)) in
  let in_im i = cos (float_of_int (i * 11 + 3)) in
  let two_pi = 2.0 *. Float.pi in
  (* Slot j (0-indexed) corresponds to twiddle exp(-2πi·(j+1)/n)
   * (matching twiddle_expr's slot = j_in_W^j - 1 convention). *)
  let tw_re j =
    cos (-. two_pi *. float_of_int (j + 1) /. float_of_int n) in
  let tw_im j =
    sin (-. two_pi *. float_of_int (j + 1) /. float_of_int n) in
  (in_re, in_im, tw_re, tw_im)

let check_correctness label assigns n =
  let (in_re, in_im, _, _) = test_inputs n in
  let (got_re, got_im) = eval_dag assigns n in_re in_im () in
  let (ref_re, ref_im) = ref_dft n in_re in_im in
  let max_err = ref 0.0 in
  for k = 0 to n - 1 do
    let er = abs_float (got_re k -. ref_re k) in
    let ei = abs_float (got_im k -. ref_im k) in
    if er > !max_err then max_err := er;
    if ei > !max_err then max_err := ei
  done;
  if !max_err > 1e-10 then
    Printf.printf "  ⚠ %s R=%d max_err=%g  (numerical mismatch — bug!)\n"
      label n !max_err

let check_correctness_twiddled label ~direction assigns n =
  let (in_re, in_im, tw_re, tw_im) = test_inputs n in
  let (got_re, got_im) = eval_dag assigns n in_re in_im
                          ~tw_re ~tw_im () in
  let (ref_re, ref_im) = ref_dft_twiddled ~direction n in_re in_im tw_re tw_im in
  let max_err = ref 0.0 in
  for k = 0 to n - 1 do
    let er = abs_float (got_re k -. ref_re k) in
    let ei = abs_float (got_im k -. ref_im k) in
    if er > !max_err then max_err := er;
    if ei > !max_err then max_err := ei
  done;
  if !max_err > 1e-10 then
    Printf.printf "  ⚠ %s R=%d max_err=%g  (numerical mismatch — bug!)\n"
      label n !max_err

let pipeline ~aggressive ~reassoc raw =
  Algsimp.reset ();
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped = Algsimp.dedup_sub_pairs simplified in
  let factored = Algsimp.factor_common_muls ~aggressive deduped in
  let factored = Algsimp.factor_by_atom ~aggressive factored in
  let factored = Algsimp.dedup_sub_pairs factored in
  (* For direct primes, skip share_subsums + transpose FP loop —
   * see gen_radix.ml comment for rationale. *)
  let is_direct = aggressive in
  let shared =
    if is_direct then factored
    else Algsimp.share_subsums ~aggressive factored
  in
  let has_cmul =
    let st = Algsimp.stats_reachable (List.map snd shared) in
    st.Algsimp.cmuls > 0
  in
  let count_ops a =
    let st = Algsimp.stats_reachable (List.map snd a) in
    st.Algsimp.adds + st.subs + st.muls + st.negs + (2 * st.cmuls) + st.fmas
  in
  let post_trans =
    if aggressive && not has_cmul && not is_direct then begin
      let rec fp prev_assigns prev_count iter =
        if iter >= 6 then prev_assigns
        else begin
          let t1 = Algsimp.transpose prev_assigns in
          let t1f = Algsimp.factor_common_muls ~aggressive t1 in
          let t1f = Algsimp.factor_by_atom ~aggressive t1f in
          let t1f = Algsimp.dedup_sub_pairs t1f in
          let t1s = Algsimp.share_subsums ~aggressive t1f in
          let t2 = Algsimp.transpose t1s in
          let t2f = Algsimp.factor_common_muls ~aggressive t2 in
          let t2f = Algsimp.factor_by_atom ~aggressive t2f in
          let t2f = Algsimp.dedup_sub_pairs t2f in
          let t2s = Algsimp.share_subsums ~aggressive t2f in
          let new_count = count_ops t2s in
          if new_count >= prev_count then prev_assigns
          else fp t2s new_count (iter + 1)
        end
      in
      fp shared (count_ops shared) 0
    end else shared
  in
  Algsimp.fma_lift post_trans

let count assigns =
  let st = Algsimp.stats_reachable (List.map snd assigns) in
  st.Algsimp.adds + st.subs + st.muls + st.negs + (2 * st.cmuls) + st.fmas

let fma_count assigns =
  let st = Algsimp.stats_reachable (List.map snd assigns) in
  st.Algsimp.fmas

let _mul_count assigns =
  let st = Algsimp.stats_reachable (List.map snd assigns) in
  st.Algsimp.muls

let measure n =
  let aggressive = match Dft.pick_algorithm n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> false
    | Dft.Split_radix -> false in
  let reassoc = Dft.needs_reassoc n in

  (* n1 (no twiddle, plain DFT) *)
  let raw_n1 = Dft.dft_expand n in
  let final_n1 = pipeline ~aggressive ~reassoc raw_n1 in
  check_correctness "n1" final_n1 n;

  (* t1_dit (twiddle on input) *)
  let raw_dit = Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIT n in
  let final_dit = pipeline ~aggressive ~reassoc raw_dit in
  check_correctness_twiddled "t1_dit" ~direction:`DIT final_dit n;

  (* t1_dit_log3 *)
  let raw_dit_log3 = Dft.dft_expand_twiddled ~policy:Dft.TP_Log3 ~direction:Dft.DIT n in
  let final_dit_log3 = pipeline ~aggressive ~reassoc raw_dit_log3 in
  check_correctness_twiddled "t1_dit_log3" ~direction:`DIT final_dit_log3 n;

  (* t1_dif (twiddle on output) *)
  let raw_dif = Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIF n in
  let final_dif = pipeline ~aggressive ~reassoc raw_dif in
  check_correctness_twiddled "t1_dif" ~direction:`DIF final_dif n;

  (* t1_dif_log3 — DIF + log3 twiddle policy. Necessary for planner
   * paths that pick DIF for the whole transform AND want log3 layout
   * (saves twiddle bandwidth). DIT and DIF must offer matching coverage
   * because the planner picks one direction for the entire transform —
   * mixing DIT and DIF in a single recursion needs reformulation. *)
  let raw_dif_log3 = Dft.dft_expand_twiddled ~policy:Dft.TP_Log3 ~direction:Dft.DIF n in
  let final_dif_log3 = pipeline ~aggressive ~reassoc raw_dif_log3 in
  check_correctness_twiddled "t1_dif_log3" ~direction:`DIF final_dif_log3 n;

  Printf.printf "R=%2d (%s): n1=%-3d  t1_dit=%-3d  t1_dit_log3=%-3d  t1_dif=%-3d  t1_dif_log3=%-3d  (fma: %d/%d/%d/%d/%d)\n"
    n
    (if aggressive then "aggr" else "safe")
    (count final_n1)
    (count final_dit)
    (count final_dit_log3)
    (count final_dif)
    (count final_dif_log3)
    (fma_count final_n1) (fma_count final_dit) (fma_count final_dit_log3)
    (fma_count final_dif) (fma_count final_dif_log3)

let () =
  Printf.printf "=== Op count + correctness: dft + factor + share + transpose ===\n";
  Printf.printf "Variants: n1 / t1_dit / t1_dit_log3 / t1_dif / t1_dif_log3\n\n";
  Printf.printf "PRIMES (aggressive: factor + share + transpose for n1; factor + share for twiddled)\n";
  List.iter measure [2; 3; 5; 7; 11; 13; 17; 19];
  Printf.printf "\nCT-DECOMPOSED (safe: passes default to no-op)\n";
  List.iter measure [4; 8; 16; 32; 64]
