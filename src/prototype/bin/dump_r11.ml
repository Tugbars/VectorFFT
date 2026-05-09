open Vfft_v2

let () =
  let n = 11 in
  Algsimp.reset ();
  let raw = Dft.dft_expand n in
  let reassoc = Dft.needs_reassoc n in
  let aggressive = match Dft.pick_algorithm n with
    | Dft.Direct -> true | _ -> false in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped = Algsimp.dedup_sub_pairs simplified in
  let factored = Algsimp.factor_common_muls ~aggressive deduped in
  let factored = Algsimp.factor_by_atom ~aggressive factored in
  let factored = Algsimp.dedup_sub_pairs factored in
  let shared = Algsimp.share_subsums ~aggressive factored in
  let count_ops a =
    let st = Algsimp.stats_reachable (List.map snd a) in
    st.Algsimp.adds + st.subs + st.muls + st.negs + (2 * st.cmuls) + st.fmas in
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
      let nc = count_ops t2s in
      if nc >= pcount then prev else fp t2s nc (iter + 1)
  in
  let final = fp shared (count_ops shared) 0 in
  let final = Algsimp.fma_lift final in
  print_string (Algsimp.print_dag final)
