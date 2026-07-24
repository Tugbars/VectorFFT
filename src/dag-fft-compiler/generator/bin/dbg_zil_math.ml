(* dbg_zil_math.ml — numeric validation of the zil-port math-layer additions
 * (zil_pipeline_port.md P0): TP_PowW1 squaring-tree twiddles and the
 * ~table_conj bwd variant, evaluated on concrete values against direct
 * references. Follows the dbg_eval.ml instrument pattern.
 *
 * Checks, per radix R in {4, 8}:
 *   1. slot census   — TP_PowW1 consults ONLY Twiddle slot 0 (= W^1)
 *   2. fwd vs direct — dft_expand_twiddled(DIT, Fwd, TP_PowW1) with a
 *                      concrete w1 equals the directly-summed DFT of the
 *                      pre-twiddled inputs x'[l] = x[l]·w1^l
 *   3. roundtrip     — bwd(DIT, Bwd, TP_PowW1, table_conj:true) fed a
 *                      CONJUGATED w1 table undoes fwd up to ×R — the
 *                      production contract (zsplit.h: "bwd(fwd) = N*x")
 *   4. table_conj=false double-conj sentinel — same bwd DAG built WITHOUT
 *                      table_conj, fed the conjugated table, must FAIL the
 *                      roundtrip (proves the flag is load-bearing)
 *
 * Usage: dune exec bin/dbg_zil_math.exe *)

open Vfft_v2
open Expr

let pi = 4.0 *. atan 1.0

(* deterministic pseudo-random complex input, |R| elements *)
let mk_input n =
  Array.init n (fun i ->
    ( sin (0.37 *. float_of_int (i + 1)) +. (0.21 *. cos (2.3 *. float_of_int i))
    , cos (0.53 *. float_of_int (i + 2)) -. (0.11 *. sin (1.9 *. float_of_int i)) ))
;;

(* the concrete W^1 used for the runtime table: an arbitrary non-trivial
 * angle (NOT a root of unity of order R — the derivation tree must be
 * exact for any w1, as the terminator feeds a different w1 per column) *)
let w1_angle = -2.0 *. pi *. 3.0 /. 64.0

(* --- evaluator over the math-layer Expr with a twiddle table --- *)
let eval_expr (x : (float * float) array) (tw : (float * float) array) (e : expr) : float =
  let rec go = function
    | Const c -> c
    | Load (Input (k, true)) -> fst x.(k)
    | Load (Input (k, false)) -> snd x.(k)
    | Load (Twiddle (j, true)) -> fst tw.(j)
    | Load (Twiddle (j, false)) -> snd tw.(j)
    | Load (Output _) -> failwith "eval: Load(Output)"
    | Neg a -> -.go a
    | Add (a, b) -> go a +. go b
    | Sub (a, b) -> go a -. go b
    | Mul (a, b) -> go a *. go b
  in
  go e
;;

(* run an assignment list: returns the (re, im) output array *)
let run_assigns n (x : (float * float) array) (tw : (float * float) array)
      (al : assignment list)
  : (float * float) array
  =
  let out = Array.make n (0.0, 0.0) in
  List.iter
    (fun (r, e) ->
       match r with
       | Output (k, true) -> out.(k) <- (eval_expr x tw e, snd out.(k))
       | Output (k, false) -> out.(k) <- (fst out.(k), eval_expr x tw e)
       | _ -> failwith "run_assigns: non-Output lhs")
    al;
  out
;;

(* direct reference: y[k] = sum_l x'[l] · e^{-2πi·l·k/R},  x'[l] = x[l]·w1^l *)
let direct_fwd n (x : (float * float) array) (w1r, w1i) : (float * float) array =
  let xp = Array.make n (0.0, 0.0) in
  let wr = ref 1.0
  and wi = ref 0.0 in
  for l = 0 to n - 1 do
    let ar, ai = x.(l) in
    xp.(l) <- ((ar *. !wr) -. (ai *. !wi), (ar *. !wi) +. (ai *. !wr));
    let nr = (!wr *. w1r) -. (!wi *. w1i)
    and ni = (!wr *. w1i) +. (!wi *. w1r) in
    wr := nr;
    wi := ni
  done;
  Array.init n (fun k ->
    let sr = ref 0.0
    and si = ref 0.0 in
    for l = 0 to n - 1 do
      let th = -2.0 *. pi *. float_of_int (l * k) /. float_of_int n in
      let c = cos th
      and s = sin th in
      let ar, ai = xp.(l) in
      sr := !sr +. ((ar *. c) -. (ai *. s));
      si := !si +. ((ar *. s) +. (ai *. c))
    done;
    (!sr, !si))
;;

let max_err (a : (float * float) array) (b : (float * float) array) : float =
  let w = ref 0.0 in
  Array.iteri
    (fun i (ar, ai) ->
       let br, bi = b.(i) in
       w := Float.max !w (Float.max (Float.abs (ar -. br)) (Float.abs (ai -. bi))))
    a;
  !w
;;

(* collect the set of Twiddle slots an assignment list consults *)
let twiddle_slots (al : assignment list) : int list =
  let tbl = Hashtbl.create 8 in
  let rec walk = function
    | Const _ -> ()
    | Load (Twiddle (j, _)) -> Hashtbl.replace tbl j ()
    | Load _ -> ()
    | Neg a -> walk a
    | Add (a, b) | Sub (a, b) | Mul (a, b) ->
      walk a;
      walk b
  in
  List.iter (fun (_, e) -> walk e) al;
  List.sort compare (Hashtbl.fold (fun k () acc -> k :: acc) tbl [])
;;

let check label ok detail =
  Printf.printf "%-44s %s  %s\n" label (if ok then "PASS" else "FAIL") detail;
  ok
;;

let run_radix n =
  Printf.printf "=== R=%d ===\n" n;
  let x = mk_input n in
  let w1 = (cos w1_angle, sin w1_angle) in
  let w1c = (cos w1_angle, -.sin w1_angle) in
  let tw = [| w1 |] in
  let twc = [| w1c |] in
  let all_ok = ref true in
  (* 1. slot census *)
  let fwd = Dft.dft_expand_twiddled ~policy:Dft.TP_PowW1 ~direction:Dft.DIT n in
  let slots = twiddle_slots fwd in
  all_ok
  := check
       "TP_PowW1 fwd consults only slot 0"
       (slots = [ 0 ])
       (Printf.sprintf "slots=[%s]" (String.concat ";" (List.map string_of_int slots)))
     && !all_ok;
  (* 2. fwd vs direct *)
  let y = run_assigns n x tw fwd in
  let y_ref = direct_fwd n x w1 in
  let e2 = max_err y y_ref in
  all_ok := check "fwd(TP_PowW1) vs direct DFT" (e2 < 1e-12) (Printf.sprintf "%.3e" e2) && !all_ok;
  (* 3. roundtrip with conj table + table_conj:true *)
  let bwd =
    Dft.dft_expand_twiddled
      ~policy:Dft.TP_PowW1
      ~direction:Dft.DIT
      ~sign:`Bwd
      ~table_conj:true
      n
  in
  let slots_b = twiddle_slots bwd in
  all_ok
  := check
       "TP_PowW1 bwd consults only slot 0"
       (slots_b = [ 0 ])
       (Printf.sprintf "slots=[%s]" (String.concat ";" (List.map string_of_int slots_b)))
     && !all_ok;
  let z = run_assigns n y twc bwd in
  let scaled = Array.map (fun (r, i) -> (r *. float_of_int n, i *. float_of_int n)) x in
  let e3 = max_err z scaled in
  all_ok
  := check "roundtrip bwd(fwd) = N*x (table_conj)" (e3 < 1e-11) (Printf.sprintf "%.3e" e3)
     && !all_ok;
  (* 4. double-conj sentinel: WITHOUT table_conj the same conj table must fail *)
  let bwd_dc =
    Dft.dft_expand_twiddled ~policy:Dft.TP_PowW1 ~direction:Dft.DIT ~sign:`Bwd n
  in
  let z_dc = run_assigns n y twc bwd_dc in
  let e4 = max_err z_dc scaled in
  all_ok
  := check
       "double-conj sentinel FAILS as expected"
       (e4 > 1e-6)
       (Printf.sprintf "%.3e (must be large)" e4)
     && !all_ok;
  !all_ok
;;

let () =
  let ok4 = run_radix 4 in
  let ok8 = run_radix 8 in
  Printf.printf "\n%s\n" (if ok4 && ok8 then "OVERALL PASS" else "OVERALL FAIL");
  exit (if ok4 && ok8 then 0 else 1)
;;
