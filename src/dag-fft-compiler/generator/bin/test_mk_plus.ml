(* test_mk_plus.ml — Self-tests for the NK_Plus smart constructor.
 *
 * Verifies all 8 invariants documented in algsimp.ml's mk_plus. Each test
 * builds a specific input shape and asserts the result has the expected
 * structural form.
 *
 * Run via: dune exec bin/test_mk_plus.exe
 *
 * These tests live in bin/ rather than test/ because we don't yet have a
 * test framework set up; this is a standalone executable. When we land
 * proper testing infrastructure, the assertions migrate to that. *)

open Vfft_v2.Algsimp
open Vfft_v2.Expr

let tests_run = ref 0
let tests_passed = ref 0

let check name (cond : bool) =
  incr tests_run;
  if cond then begin
    incr tests_passed;
    Printf.printf "  PASS: %s\n" name
  end
  else Printf.printf "  FAIL: %s\n" name

(* Print a node's structure for debugging. *)
let rec describe (n : t) : string =
  match n.node with
  | NK_Const c -> Printf.sprintf "Const(%g)" c
  | NK_Load _ -> Printf.sprintf "Load(t%d)" n.tag
  | NK_Neg x -> Printf.sprintf "Neg(%s)" (describe x)
  | NK_Add (a, b) -> Printf.sprintf "Add(%s, %s)" (describe a) (describe b)
  | NK_Sub (a, b) -> Printf.sprintf "Sub(%s, %s)" (describe a) (describe b)
  | NK_Mul (a, b) -> Printf.sprintf "Mul(%s, %s)" (describe a) (describe b)
  | NK_Plus terms ->
      let parts =
        List.map
          (fun (s, t) ->
            Printf.sprintf "%s%s" (if s > 0 then "+" else "-") (describe t))
          terms
      in
      Printf.sprintf "Plus[%s]" (String.concat ", " parts)
  | NK_CmulRe _ -> Printf.sprintf "CmulRe(t%d)" n.tag
  | NK_CmulIm _ -> Printf.sprintf "CmulIm(t%d)" n.tag
  | NK_Fma _ -> Printf.sprintf "Fma(t%d)" n.tag

(* Helper: build a Load to use as a "variable" in tests. *)
let var i = of_expr (Load (Input (i, true)))

let () =
  print_endline "=== mk_plus invariants ===";

  (* Invariant 1: empty list → Const 0.0 *)
  let e1 = mk_plus [] in
  check "empty list → Const(0)"
    (match e1.node with NK_Const 0.0 -> true | _ -> false);

  (* Invariant: single term (+1, x) → x *)
  let x = var 0 in
  let e2 = mk_plus [ (1, x) ] in
  check "single (+1, x) → x" (e2.tag = x.tag);

  (* Invariant: single term (-1, x) → Neg(x) *)
  let e3 = mk_plus [ (-1, x) ] in
  check "single (-1, x) → Neg(x)"
    (match e3.node with NK_Neg n -> n.tag = x.tag | _ -> false);

  (* Two distinct terms produce NK_Plus *)
  let y = var 1 in
  let e4 = mk_plus [ (1, x); (1, y) ] in
  check "Plus[+x, +y] → NK_Plus"
    (match e4.node with
    | NK_Plus [ (1, a); (1, b) ] -> a.tag <> b.tag
    | _ -> false);

  (* Invariant 6: terms sorted by tag *)
  let e5_ab = mk_plus [ (1, x); (1, y) ] in
  let e5_ba = mk_plus [ (1, y); (1, x) ] in
  check "term order doesn't matter (canonical sort)" (e5_ab.tag = e5_ba.tag);

  (* Invariant 8: opposite signs cancel *)
  let e6 = mk_plus [ (1, x); (-1, x) ] in
  check "(+1,x) + (-1,x) → 0"
    (match e6.node with NK_Const 0.0 -> true | _ -> false);

  (* Invariant 5: multiple constants combine *)
  let c2 = of_expr (Const 2.0) in
  let c3 = of_expr (Const 3.0) in
  let e7 = mk_plus [ (1, c2); (1, c3); (1, x) ] in
  check "(+2) + (+3) + x → Plus[+5, +x]"
    (match e7.node with
    | NK_Plus terms ->
        List.exists
          (fun (s, t) ->
            s = 1 && match t.node with NK_Const 5.0 -> true | _ -> false)
          terms
        && List.length terms = 2
    | _ -> false);

  (* Invariant 7: zero-coefficient terms dropped *)
  let c0 = of_expr (Const 0.0) in
  let e8 = mk_plus [ (1, x); (1, c0); (1, y) ] in
  check "(+x) + (+0) + (+y) → Plus[+x, +y]"
    (match e8.node with NK_Plus terms -> List.length terms = 2 | _ -> false);

  (* Invariant 4: NK_Neg absorbed into sign *)
  let neg_x = Vfft_v2.Algsimp.mk_neg x in
  let e9 = mk_plus [ (1, neg_x); (1, y) ] in
  check "(+Neg(x)) + (+y) → Plus[-x, +y]"
    (match e9.node with
    | NK_Plus terms ->
        List.exists (fun (s, t) -> s = -1 && t.tag = x.tag) terms
        && List.exists (fun (s, t) -> s = 1 && t.tag = y.tag) terms
    | _ -> false);

  (* Invariant 3: nested NK_Plus flattens *)
  let z = var 2 in
  let inner = mk_plus [ (1, x); (-1, y) ] in
  let e10 = mk_plus [ (1, inner); (1, z) ] in
  check "Plus[Plus[+x, -y], +z] flattens to Plus[+x, -y, +z]"
    (match e10.node with NK_Plus terms -> List.length terms = 3 | _ -> false);

  (* lower_plus inverse: lowering produces equivalent binary tree *)
  print_endline "";
  print_endline "=== lower_plus ===";
  let plus_xy = mk_plus [ (1, x); (1, y) ] in
  let lowered = lower_plus plus_xy in
  check "lower(Plus[+x,+y]) → Add(x, y)"
    (match lowered.node with
    | NK_Add (a, b) ->
        (a.tag = x.tag && b.tag = y.tag) || (a.tag = y.tag && b.tag = x.tag)
    | _ -> false);

  let plus_xny = mk_plus [ (1, x); (-1, y) ] in
  let lowered2 = lower_plus plus_xny in
  check "lower(Plus[+x,-y]) is binary"
    (match lowered2.node with
    | NK_Sub _ | NK_Add _ -> true
    | _ ->
        Printf.printf "  (got: %s)\n" (describe lowered2);
        false);

  (* Lowering an existing binary tree is a no-op *)
  let just_x = lower_plus x in
  check "lower(x where x has no NK_Plus) → x" (just_x.tag = x.tag);

  (* === collect_m smoke tests === *)
  print_endline "";
  print_endline "=== collect_m (with VFFT_COLLECT_M=1) ===";
  Unix.putenv "VFFT_COLLECT_M" "1";

  (* Build expr: c1*x + c2*x where c1, c2 are constants.
   * In our IR: mk_add (mk_mul c1 x) (mk_mul c2 x).
   * After collect_m: should become Mul(Const(c1+c2), x). *)
  let c2 = of_expr (Const 2.0) in
  let c3 = of_expr (Const 3.0) in
  let term1 = Vfft_v2.Algsimp.mk_mul c2 x in
  let term2 = Vfft_v2.Algsimp.mk_mul c3 x in
  let sum = Vfft_v2.Algsimp.mk_add term1 term2 in

  let collected = Vfft_v2.Algsimp.collect_m [ (Output (0, true), sum) ] in
  let result = snd (List.hd collected) in

  check "2*x + 3*x → 5*x (single Mul)"
    (match result.node with
    | NK_Mul (a, b) -> (
        match (a.node, b.node) with
        | NK_Const 5.0, _ when b.tag = x.tag -> true
        | _, NK_Const 5.0 when a.tag = x.tag -> true
        | _ -> false)
    | _ ->
        Printf.printf "  (got: %s)\n" (describe result);
        false);

  (* Three-way: 2x + 3x - x → 4x *)
  let term3 = Vfft_v2.Algsimp.mk_neg x in
  let sum3 = Vfft_v2.Algsimp.mk_add sum term3 in
  let collected3 = Vfft_v2.Algsimp.collect_m [ (Output (0, true), sum3) ] in
  let result3 = snd (List.hd collected3) in
  check "2*x + 3*x - x → 4*x"
    (match result3.node with
    | NK_Mul (a, b) -> (
        match (a.node, b.node) with
        | NK_Const 4.0, _ when b.tag = x.tag -> true
        | _, NK_Const 4.0 when a.tag = x.tag -> true
        | _ -> false)
    | _ ->
        Printf.printf "  (got: %s)\n" (describe result3);
        false);

  (* Cancellation: 2x - 2x → 0 *)
  let term_pos = Vfft_v2.Algsimp.mk_mul c2 x in
  let term_neg = Vfft_v2.Algsimp.mk_neg (Vfft_v2.Algsimp.mk_mul c2 x) in
  let canceling = Vfft_v2.Algsimp.mk_add term_pos term_neg in
  let collected_c =
    Vfft_v2.Algsimp.collect_m [ (Output (0, true), canceling) ]
  in
  let result_c = snd (List.hd collected_c) in
  check "2*x + (-2*x) → 0"
    (match result_c.node with NK_Const 0.0 -> true | _ -> false);

  (* No mergers: x + y unchanged *)
  let xy = Vfft_v2.Algsimp.mk_add x y in
  let xy_tag_before = xy.tag in
  let collected_xy = Vfft_v2.Algsimp.collect_m [ (Output (0, true), xy) ] in
  let result_xy = snd (List.hd collected_xy) in
  check "x + y → x + y (no collection)" (result_xy.tag = xy_tag_before);

  Printf.printf "\n=== Results ===\n";
  Printf.printf "%d/%d tests passed\n" !tests_passed !tests_run;
  if !tests_passed = !tests_run then exit 0 else exit 1
