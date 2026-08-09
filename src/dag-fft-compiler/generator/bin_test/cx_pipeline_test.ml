(* cx_pipeline_test.ml — unit gate for the cx pass cascade.
 *
 * The production emitters contain ZERO dedup_sub_pairs sites (audited
 * 2026-08-09: 14,266 sub sites, 0 mirrors), so the pass's REWRITE path never
 * runs in the 183-case emission gate. This test exercises it directly:
 * builds a DAG with a mirrored sub pair, asserts the rewrite fires once,
 * shares the subtraction, and preserves the numerics exactly.
 *
 * Run:  dune build bin_test/cx_pipeline_test.exe && ./_build/default/bin_test/cx_pipeline_test.exe *)

open Vfft_v2
open Cx_ir

(* Tiny evaluator over packed complex (one lane suffices for semantics). *)
let rec eval (env : Complex.t array) (e : t) : Complex.t =
  let open Complex in
  match e.node with
  | CIn i -> env.(i)
  | CAdd (a, b) -> add (eval env a) (eval env b)
  | CSub (a, b) -> sub (eval env a) (eval env b)
  | CNeg a -> neg (eval env a)
  | CRotNI a -> mul { re = 0.0; im = -1.0 } (eval env a)
  | CRotPI a -> mul { re = 0.0; im = 1.0 } (eval env a)
  | CFmaC (c, x, acc) -> add (mul { re = c; im = 0.0 } (eval env x)) (eval env acc)
  | CFnmaC (c, x, acc) -> add (mul { re = -.c; im = 0.0 } (eval env x)) (eval env acc)
  | CTwC (c, s, x) -> mul { re = c; im = s } (eval env x)
  | CTwV _ | CTwL _ -> failwith "cx_pipeline_test: eval does not model table twiddles"
  | CLoad _ | CStore _ | CTurn _ | CLo _ | CHi _ ->
    failwith "cx_pipeline_test: eval does not model memory/lane nodes"
;;

let ceq a b =
  let dre = a.Complex.re -. b.Complex.re
  and dim = a.Complex.im -. b.Complex.im in
  Float.sqrt ((dre *. dre) +. (dim *. dim)) < 1e-15
;;

let fail fmt = Printf.ksprintf (fun s -> prerr_endline ("FAIL: " ^ s); exit 1) fmt

let () =
  reset ();
  (* a DAG holding BOTH (x0 - x1) and (x1 - x0), each feeding further work *)
  let x0 = cin 0
  and x1 = cin 1 in
  let d01 = csub x0 x1 in
  let d10 = csub x1 x0 in
  let o0 = cadd d01 (crot x1) in
  let o1 = cfma 0.75 d10 x0 in
  let o2 = ctw 0.6 0.8 d10 in
  let assigns = [ Expr.Output (0, true), o0; Expr.Output (1, true), o1; Expr.Output (2, true), o2 ] in

  let rewritten, n = Cx_pipeline.dedup_sub_pairs_cx assigns in
  if n <> 1 then fail "expected exactly 1 mirror rewrite, got %d" n;

  (* the mirror must now be CNeg of the SHARED lower-tagged sub *)
  let count_subs, count_negs = ref 0, ref 0 in
  Cx_pipeline.iter_reachable (List.map snd rewritten) (fun e ->
    match e.node with
    | CSub _ -> incr count_subs
    | CNeg _ -> incr count_negs
    | _ -> ());
  if !count_subs <> 1 then fail "expected 1 shared sub after rewrite, got %d" !count_subs;
  if !count_negs <> 1 then fail "expected 1 neg after rewrite, got %d" !count_negs;

  (* numerics preserved exactly (negation is sign-bit-exact) *)
  let env = [| { Complex.re = 0.8321; im = -1.117 }; { Complex.re = -0.25; im = 2.03 } |] in
  List.iter2
    (fun (_, before) (_, after) ->
       if not (ceq (eval env before) (eval env after))
       then fail "rewrite changed a value")
    assigns
    rewritten;

  (* zero-site DAGs must pass through UNTOUCHED (same physical assigns) *)
  reset ();
  let y0 = cin 0
  and y1 = cin 1 in
  let clean = [ Expr.Output (0, true), cadd y0 y1; Expr.Output (1, true), csub y0 y1 ] in
  let clean', n0 = Cx_pipeline.dedup_sub_pairs_cx clean in
  if n0 <> 0 then fail "clean DAG reported %d rewrites" n0;
  if not (clean == clean') then fail "clean DAG was rebuilt instead of passed through";

  print_endline "cx_pipeline_test: ALL PASS (mirror rewrite + sharing + numerics + identity path)"
;;
