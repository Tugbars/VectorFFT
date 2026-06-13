(* Dump the IR of a small case to understand what shape we're dealing with. *)
open Vfft_v2

let () =
  (* Generate R=25 raw assignments. *)
  let raw = Dft.dft_expand 25 in
  Algsimp.reset ();
  let reassoc = Dft.needs_reassoc 25 in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  
  (* Pick the FIRST assignment and print its structure. *)
  Printf.printf "Total assignments: %d\n" (List.length simplified);
  
  let depth_limit = 4 in
  let rec dump indent (n : Algsimp.t) =
    if indent > depth_limit then Printf.printf "%s..t%d (truncated)\n" (String.make indent ' ') n.tag
    else
      let pad = String.make indent ' ' in
      match n.node with
      | Algsimp.NK_Const c -> Printf.printf "%sConst(%g)  [t%d]\n" pad c n.tag
      | Algsimp.NK_Load _ -> Printf.printf "%sLoad  [t%d]\n" pad n.tag
      | Algsimp.NK_Neg x ->
        Printf.printf "%sNeg  [t%d]\n" pad n.tag; dump (indent+2) x
      | Algsimp.NK_Add (a, b) ->
        Printf.printf "%sAdd  [t%d]\n" pad n.tag;
        dump (indent+2) a; dump (indent+2) b
      | Algsimp.NK_Sub (a, b) ->
        Printf.printf "%sSub  [t%d]\n" pad n.tag;
        dump (indent+2) a; dump (indent+2) b
      | Algsimp.NK_Mul (a, b) ->
        Printf.printf "%sMul  [t%d]\n" pad n.tag;
        dump (indent+2) a; dump (indent+2) b
      | Algsimp.NK_CmulRe _ -> Printf.printf "%sCmulRe  [t%d]\n" pad n.tag
      | Algsimp.NK_CmulIm _ -> Printf.printf "%sCmulIm  [t%d]\n" pad n.tag
      | Algsimp.NK_Fma _ -> Printf.printf "%sFma  [t%d]\n" pad n.tag
      | Algsimp.NK_Plus _ -> Printf.printf "%sPlus  [t%d]\n" pad n.tag
  in
  
  (* Just dump first 3 outputs. *)
  List.iteri (fun i (_lhs, e) ->
    if i < 3 then begin
      Printf.printf "\n--- Output %d ---\n" i;
      dump 0 e
    end
  ) simplified

let _ = ()
