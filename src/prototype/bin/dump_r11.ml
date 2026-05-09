(* Dump the full DAG for R=11 after all simplification passes *)
open Vfft_v2

let () =
  let n = 11 in
  Algsimp.reset ();
  let inp_re i = Expr.Load (Expr.Input (i, true)) in
  let inp_im i = Expr.Load (Expr.Input (i, false)) in
  let (out_re, out_im) = Dft.dft_direct ~sign:`Fwd n inp_re inp_im in
  let raw = List.init n (fun i ->
    [(Expr.Output (i, true), out_re.(i)); (Expr.Output (i, false), out_im.(i))]
  ) |> List.concat in
  let simplified = Algsimp.of_assignments ~reassoc:true raw in
  let deduped = Algsimp.dedup_sub_pairs simplified in
  let factored = Algsimp.factor_common_muls ~aggressive:true deduped in
  let factored = Algsimp.dedup_sub_pairs factored in
  let shared = Algsimp.share_subsums ~aggressive:true factored in
  let t1 = Algsimp.transpose shared in
  let t1f = Algsimp.factor_common_muls ~aggressive:true t1 in
  let t1f = Algsimp.dedup_sub_pairs t1f in
  let t1s = Algsimp.share_subsums ~aggressive:true t1f in
  let t2 = Algsimp.transpose t1s in
  let t2f = Algsimp.factor_common_muls ~aggressive:true t2 in
  let t2f = Algsimp.dedup_sub_pairs t2f in
  let final = Algsimp.share_subsums ~aggressive:true t2f in
  let final = Algsimp.fma_lift final in
  print_string (Algsimp.print_dag final)
