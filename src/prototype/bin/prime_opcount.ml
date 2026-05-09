(* prime_opcount.ml — measure op count after dft_direct + algsimp for primes 3,5,7,11. *)

open Vfft_v2

let measure n =
  Algsimp.reset ();
  let raw = Dft.dft_expand n in
  let reassoc = Dft.needs_reassoc n in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped = Algsimp.dedup_sub_pairs simplified in
  let aggressive = match Dft.pick_algorithm n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> false in
  let factored = Algsimp.factor_common_muls ~aggressive deduped in
  let factored = Algsimp.dedup_sub_pairs factored in
  let shared = Algsimp.share_subsums ~aggressive factored in
  if n = 3 then begin
    Printf.eprintf "--- POST-SHARE DAG for R=3 ---\n";
    Printf.eprintf "%s" (Algsimp.print_dag shared);
    Printf.eprintf "--- end ---\n%!"
  end;
  let stats_no_factor =
    Algsimp.stats_reachable (List.map snd deduped) in
  let stats_with_factor =
    Algsimp.stats_reachable (List.map snd shared) in
  let vec a = a.Algsimp.adds + a.subs + a.muls + a.negs + (2 * a.cmuls) in
  Printf.printf "R=%2d (%s): before vec=%-4d (a=%d s=%d m=%d n=%d)  after vec=%-4d (a=%d s=%d m=%d n=%d)  %+d\n"
    n
    (if aggressive then "aggr" else "safe")
    (vec stats_no_factor)
    stats_no_factor.adds stats_no_factor.subs stats_no_factor.muls stats_no_factor.negs
    (vec stats_with_factor)
    stats_with_factor.adds stats_with_factor.subs stats_with_factor.muls stats_with_factor.negs
    (vec stats_with_factor - vec stats_no_factor)

let () =
  Printf.printf "=== Op count: dft_direct + algsimp ± factor_common_muls ===\n";
  Printf.printf "PRIMES (target: hand-coded gen_radix*.py)\n";
  List.iter measure [3; 5; 7; 11];
  Printf.printf "\nCT-DECOMPOSED (must not regress)\n";
  List.iter measure [4; 8; 16; 32; 64]
