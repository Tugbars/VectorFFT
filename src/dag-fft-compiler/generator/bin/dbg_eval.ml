(* dbg_eval.ml — numeric evaluation of the dct1 DAG after every
 * simplification pass, against the brute-force DCT-I reference.
 * Section 56 instrument: locates the first pass whose OUTPUT evaluates
 * wrong. Usage: dune exec bin/dbg_eval.exe -- [N]   (default 9) *)


let n = if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 9

(* deterministic pseudo-random input *)
let x =
  Array.init n (fun i ->
    sin (0.41 *. float_of_int i) +. (0.3 *. cos (1.7 *. float_of_int i)))
;;

let brute () =
  Array.init n (fun k ->
    let s = ref (x.(0) +. if k land 1 = 1 then -.x.(n - 1) else x.(n - 1)) in
    for m = 1 to n - 2 do
      s
      := !s
         +. (2.0 *. x.(m) *. cos (Float.pi *. float_of_int (m * k) /. float_of_int (n - 1))
            )
    done;
    !s)
;;

(* --- evaluator over the math-layer Expr --- *)
let rec eval_expr (e : Expr.expr) : float =
  match e with
  | Expr.Const c -> c
  | Expr.Load (Expr.Input (k, true)) -> x.(k)
  | Expr.Load _ -> failwith "eval_expr: unexpected load"
  | Expr.Neg a -> -.eval_expr a
  | Expr.Add (a, b) -> eval_expr a +. eval_expr b
  | Expr.Sub (a, b) -> eval_expr a -. eval_expr b
  | Expr.Mul (a, b) -> eval_expr a *. eval_expr b
;;

(* --- evaluator over hash-consed Ir.t (memoized by tag) --- *)
let eval_t (root : Ir.t) : float =
  let memo : (int, float) Hashtbl.t = Hashtbl.create 256 in
  let rec go (e : Ir.t) : float =
    match Hashtbl.find_opt memo e.Ir.tag with
    | Some v -> v
    | None ->
      let v =
        match e.Ir.node with
        | Ir.NK_Const c -> c
        | Ir.NK_Load (Expr.Input (k, true)) -> x.(k)
        | Ir.NK_Load _ -> failwith "eval_t: unexpected load"
        | Ir.NK_Neg a -> -.go a
        | Ir.NK_Add (a, b) -> go a +. go b
        | Ir.NK_Sub (a, b) -> go a -. go b
        | Ir.NK_Mul (a, b) -> go a *. go b
        | Ir.NK_Fma (a, b, c, nm, na) ->
          let ab = go a *. go b in
          (if nm then -.ab else ab)
          +.
          let cv = go c in
          if na then -.cv else cv
        | Ir.NK_CmulRe (xr, xi, wr, wi) -> (go xr *. go wr) -. (go xi *. go wi)
        | Ir.NK_CmulIm (xr, xi, wr, wi) -> (go xr *. go wi) +. (go xi *. go wr)
        | Ir.NK_Plus _ -> failwith "eval_t: NK_Plus"
      in
      Hashtbl.replace memo e.Ir.tag v;
      v
  in
  go root
;;

let check_expr (label : string) (al : Expr.assignment list) =
  let ref_y = brute () in
  let worst = ref 0.0 in
  List.iter
    (fun (r, e) ->
       match r with
       | Expr.Output (k, true) when k < n ->
         let d = Float.abs (eval_expr e -. ref_y.(k)) in
         if d > !worst then worst := d
       | _ -> ())
    al;
  Printf.printf
    "%-28s max|err| = %.3e  %s\n"
    label
    !worst
    (if !worst < 1e-10 then "PASS" else "FAIL")
;;

let check_t (label : string) (al : (Expr.elem_ref * Ir.t) list) =
  let ref_y = brute () in
  let worst = ref 0.0 in
  List.iter
    (fun (r, e) ->
       match r with
       | Expr.Output (k, true) when k < n ->
         let d = Float.abs (eval_t e -. ref_y.(k)) in
         if d > !worst then worst := d
       | _ -> ())
    al;
  Printf.printf
    "%-28s max|err| = %.3e  %s\n"
    label
    !worst
    (if !worst < 1e-10 then "PASS" else "FAIL")
;;

let () =
  let policy_n = 2 * (n - 1) in
  let aggressive =
    match Dft_select.pick_algorithm policy_n with
    | Dft_select.Direct -> true
    | Dft_select.Cooley_Tukey _ -> false
    | Dft_select.Split_radix -> false
  in
  Printf.printf
    "=== dct1 N=%d per-pass evaluation (aggressive=%b, as gen_main) ===\n"
    n
    aggressive;
  let raw = Dft_r2c.dft_expand_dct1 n in
  check_expr "math layer (Expr)" raw;
  Ir.reset ();
  let s0 = Ir.of_assignments ~reassoc:false raw in
  check_t "of_assignments" s0;
  let s1 = Simplify.dedup_sub_pairs s0 in
  check_t "dedup_sub_pairs" s1;
  let s2 = Simplify.factor_common_muls ~aggressive s1 in
  check_t "factor_common_muls" s2;
  let s3 = Simplify.factor_by_atom ~aggressive s2 in
  check_t "factor_by_atom" s3;
  let s4 = Simplify.dedup_sub_pairs s3 in
  check_t "dedup_sub_pairs #2" s4;
  let s5 = Simplify.collect_m s4 in
  check_t "collect_m" s5;
  (* M11: cross-check the per-pass prefix against THE cascade — Pipeline
     is the sole owner; this probe steps INSIDE it for numeric diagnosis,
     so it must also verify the unified cascade agrees end-to-end. *)
  Ir.reset ();
  let pipe =
    Pipeline.prepare_codelet
      ~recipe:Pipeline.default_recipe
      ~raw_assigns:raw
      ~spill_markers_raw:[]
      ~spill_ct:None
      ~reassoc:false
      ~aggressive
      ~algorithm:(Dft_select.pick_algorithm policy_n)
      ~force_fma_lift:false
      ~disable_fma_lift:false
      ~build_spill_info:false
      ~fuse:0
  in
  check_t "Pipeline (sole cascade)" pipe.Pipeline.assigns
;;
