(* Dump the IR of a small case to understand what shape we're dealing with. *)
open Vfft_v2

let () =
  (* Generate R=25 raw assignments. *)
  let raw = Dft.dft_expand 25 in
  Ir.reset ();
  let reassoc = Dft_select.needs_reassoc 25 in
  let simplified = Ir.of_assignments ~reassoc raw in
  (* Pick the FIRST assignment and print its structure. *)
  Printf.printf "Total assignments: %d\n" (List.length simplified);
  let depth_limit = 4 in
  let rec dump indent (n : Ir.t) =
    if indent > depth_limit
    then Printf.printf "%s..t%d (truncated)\n" (String.make indent ' ') n.tag
    else (
      let pad = String.make indent ' ' in
      match n.node with
      | Ir.NK_Const c -> Printf.printf "%sConst(%g)  [t%d]\n" pad c n.tag
      | Ir.NK_Load _ -> Printf.printf "%sLoad  [t%d]\n" pad n.tag
      | Ir.NK_Neg x ->
        Printf.printf "%sNeg  [t%d]\n" pad n.tag;
        dump (indent + 2) x
      | Ir.NK_Add (a, b) ->
        Printf.printf "%sAdd  [t%d]\n" pad n.tag;
        dump (indent + 2) a;
        dump (indent + 2) b
      | Ir.NK_Sub (a, b) ->
        Printf.printf "%sSub  [t%d]\n" pad n.tag;
        dump (indent + 2) a;
        dump (indent + 2) b
      | Ir.NK_Mul (a, b) ->
        Printf.printf "%sMul  [t%d]\n" pad n.tag;
        dump (indent + 2) a;
        dump (indent + 2) b
      | Ir.NK_CmulRe _ -> Printf.printf "%sCmulRe  [t%d]\n" pad n.tag
      | Ir.NK_CmulIm _ -> Printf.printf "%sCmulIm  [t%d]\n" pad n.tag
      | Ir.NK_Fma _ -> Printf.printf "%sFma  [t%d]\n" pad n.tag
      | Ir.NK_Plus _ -> Printf.printf "%sPlus  [t%d]\n" pad n.tag)
  in
  (* Just dump first 3 outputs. *)
  List.iteri
    (fun i (_lhs, e) ->
       if i < 3
       then (
         Printf.printf "\n--- Output %d ---\n" i;
         dump 0 e))
    simplified
;;

let _ = ()
