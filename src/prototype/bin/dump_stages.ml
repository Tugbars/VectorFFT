open Vfft_v2

let count_arith a =
  let st = Algsimp.stats_reachable (List.map snd a) in
  st.Algsimp.adds + st.subs + st.muls + st.negs + (2*st.cmuls) + st.fmas

let test_n n =
  Algsimp.reset ();
  let raw = Dft.dft_expand n in
  let reassoc = Dft.needs_reassoc n in
  let aggressive = match Dft.pick_algorithm n with
    | Dft.Direct -> true | _ -> false in

  let s0 = Algsimp.of_assignments ~reassoc raw in

  (* Path A: factor + dedup, NO share *)
  let s1 = Algsimp.dedup_sub_pairs s0 in
  let s2 = Algsimp.factor_common_muls ~aggressive s1 in
  let s3 = Algsimp.factor_by_atom ~aggressive s2 in
  let s4 = Algsimp.dedup_sub_pairs s3 in
  let liftedA = Algsimp.fma_lift s4 in

  (* Path B: full current *)
  let s5 = Algsimp.share_subsums ~aggressive s4 in
  let rec fp prev pcount iter =
    if iter >= 6 then prev
    else
      let t1 = Algsimp.transpose prev in
      let t1f = Algsimp.factor_common_muls ~aggressive t1 in
      let t1f = Algsimp.factor_by_atom ~aggressive t1f in
      let t1f = Algsimp.dedup_sub_pairs t1f in
      let t1s = Algsimp.share_subsums ~aggressive t1f in
      let t2 = Algsimp.transpose t1s in
      let t2f = Algsimp.factor_common_muls ~aggressive t2 in
      let t2f = Algsimp.factor_by_atom ~aggressive t2f in
      let t2f = Algsimp.dedup_sub_pairs t2f in
      let t2s = Algsimp.share_subsums ~aggressive t2f in
      let nc = count_arith t2s in
      if nc >= pcount then prev else fp t2s nc (iter + 1)
  in
  let s6 = fp s5 (count_arith s5) 0 in
  let liftedB = Algsimp.fma_lift s6 in

  Printf.printf "R=%2d  A_no_share=%-3d  B_current=%-3d  delta=%+d\n"
    n (count_arith liftedA) (count_arith liftedB)
    ((count_arith liftedA) - (count_arith liftedB))

let () =
  List.iter test_n [3; 5; 7; 11; 13]
