open Vfft_v2

let count_arith a =
  let st = Algsimp.stats_reachable (List.map snd a) in
  st.Algsimp.adds + st.subs + st.muls + st.negs + (2*st.cmuls) + st.fmas

let () =
  let n = 13 in
  Algsimp.reset ();
  let raw_dit = Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIT n in
  let raw_dif = Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIF n in
  let process a =
    let aggressive = match Dft.pick_algorithm n with Dft.Direct -> true | _ -> false in
    let reassoc = Dft.needs_reassoc n in
    let s = Algsimp.of_assignments ~reassoc a in
    let s = Algsimp.dedup_sub_pairs s in
    let s = Algsimp.factor_common_muls ~aggressive s in
    let s = Algsimp.factor_by_atom ~aggressive s in
    let s = Algsimp.dedup_sub_pairs s in
    Algsimp.fma_lift s
  in
  let dit = process raw_dit in
  let dif = process raw_dif in
  Printf.printf "R=%d DIT: %d ops; DIF: %d ops\n" n (count_arith dit) (count_arith dif);
  Printf.printf "\n=== DIT first 30 lines ===\n";
  let s = Algsimp.print_dag dit in
  let lines = String.split_on_char '\n' s in
  List.iteri (fun i l -> if i < 30 then print_endline l) lines;
  Printf.printf "\n=== DIF first 30 lines ===\n";
  let s = Algsimp.print_dag dif in
  let lines = String.split_on_char '\n' s in
  List.iteri (fun i l -> if i < 30 then print_endline l) lines
