(* dbg_eval.ml — numeric evaluation of the dct1 DAG after every
 * simplification pass, against the brute-force DCT-I reference.
 * Section 56 instrument: locates the first pass whose OUTPUT evaluates
 * wrong. Usage: dune exec bin/dbg_eval.exe -- [N]   (default 9) *)

open Vfft_v2

let n = if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 9

(* deterministic pseudo-random input *)
let x = Array.init n (fun i ->
  sin (0.41 *. float_of_int i) +. 0.3 *. cos (1.7 *. float_of_int i))

let brute () =
  Array.init n (fun k ->
    let s = ref (x.(0) +. (if k land 1 = 1 then -. x.(n-1) else x.(n-1))) in
    for m = 1 to n - 2 do
      s := !s +. 2.0 *. x.(m)
           *. cos (Float.pi *. float_of_int (m * k) /. float_of_int (n - 1))
    done;
    !s)

(* --- evaluator over the math-layer Expr --- *)
let rec eval_expr (e : Expr.expr) : float =
  match e with
  | Expr.Const c -> c
  | Expr.Load (Expr.Input (k, true)) -> x.(k)
  | Expr.Load _ -> failwith "eval_expr: unexpected load"
  | Expr.Neg a -> -. (eval_expr a)
  | Expr.Add (a, b) -> eval_expr a +. eval_expr b
  | Expr.Sub (a, b) -> eval_expr a -. eval_expr b
  | Expr.Mul (a, b) -> eval_expr a *. eval_expr b

(* --- evaluator over hash-consed Algsimp.t (memoized by tag) --- *)
let eval_t (root : Algsimp.t) : float =
  let memo : (int, float) Hashtbl.t = Hashtbl.create 256 in
  let rec go (e : Algsimp.t) : float =
    match Hashtbl.find_opt memo e.Algsimp.tag with
    | Some v -> v
    | None ->
      let v =
        match e.Algsimp.node with
        | Algsimp.NK_Const c -> c
        | Algsimp.NK_Load (Expr.Input (k, true)) -> x.(k)
        | Algsimp.NK_Load _ -> failwith "eval_t: unexpected load"
        | Algsimp.NK_Neg a -> -. (go a)
        | Algsimp.NK_Add (a, b) -> go a +. go b
        | Algsimp.NK_Sub (a, b) -> go a -. go b
        | Algsimp.NK_Mul (a, b) -> go a *. go b
        | Algsimp.NK_Fma (a, b, c, nm, na) ->
          let ab = go a *. go b in
          (if nm then -. ab else ab) +. (let cv = go c in if na then -. cv else cv)
        | Algsimp.NK_CmulRe (xr, xi, wr, wi) -> go xr *. go wr -. go xi *. go wi
        | Algsimp.NK_CmulIm (xr, xi, wr, wi) -> go xr *. go wi +. go xi *. go wr
        | Algsimp.NK_Plus _ -> failwith "eval_t: NK_Plus"
      in
      Hashtbl.replace memo e.Algsimp.tag v; v
  in
  go root

let check_expr (label : string) (al : Expr.assignment list) =
  let ref_y = brute () in
  let worst = ref 0.0 in
  List.iter (fun (r, e) ->
    match r with
    | Expr.Output (k, true) when k < n ->
      let d = Float.abs (eval_expr e -. ref_y.(k)) in
      if d > !worst then worst := d
    | _ -> ()) al;
  Printf.printf "%-28s max|err| = %.3e  %s\n" label !worst
    (if !worst < 1e-10 then "PASS" else "FAIL")

let check_t (label : string) (al : (Expr.elem_ref * Algsimp.t) list) =
  let ref_y = brute () in
  let worst = ref 0.0 in
  List.iter (fun (r, e) ->
    match r with
    | Expr.Output (k, true) when k < n ->
      let d = Float.abs (eval_t e -. ref_y.(k)) in
      if d > !worst then worst := d
    | _ -> ()) al;
  Printf.printf "%-28s max|err| = %.3e  %s\n" label !worst
    (if !worst < 1e-10 then "PASS" else "FAIL")

let () =
  let policy_n = 2 * (n - 1) in
  let aggressive = match Dft.pick_algorithm policy_n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> false
    | Dft.Split_radix -> false in
  Printf.printf "=== dct1 N=%d per-pass evaluation (aggressive=%b, as gen_main) ===\n"
    n aggressive;
  let raw = Dft_r2c.dft_expand_dct1 n in
  check_expr "math layer (Expr)" raw;
  Algsimp.reset ();
  let s0 = Algsimp.of_assignments ~reassoc:false raw in
  check_t "of_assignments" s0;
  let s1 = Algsimp.dedup_sub_pairs s0 in
  check_t "dedup_sub_pairs" s1;
  let s2 = Algsimp.factor_common_muls ~aggressive s1 in
  check_t "factor_common_muls" s2;
  let s3 = Algsimp.factor_by_atom ~aggressive s2 in
  check_t "factor_by_atom" s3;
  let s4 = Algsimp.dedup_sub_pairs s3 in
  check_t "dedup_sub_pairs #2" s4;
  let s5 = Algsimp.collect_m s4 in
  check_t "collect_m" s5
