(* emit_ztt_drivers.ml — emit the ZTURN-T fused driver TU (Ztt_drivers.emit_tu).
 *
 * The driver TU is DERIVED from the corpus cells (every admissible (N, chain)
 * x {fwd, bwd} x {dest, plane}), not a codelet: it lives in generated/ beside
 * the registries, produced by the same promote-rule mechanism, and the
 * registry (emit_ztt_registry.exe) is derived from the same cell list.
 *
 * Usage:
 *   dune exec bin/emit_ztt_drivers.exe -- --isa avx2 --uarch raptor_lake_avx2
 *     > generated/ztt_drivers_avx2.c
 *)

let () =
  let isa = ref "avx2"
  and uarch = ref "raptor_lake_avx2" in
  let rec parse = function
    | "--isa" :: v :: tl ->
      isa := v;
      parse tl
    | "--uarch" :: v :: tl ->
      uarch := v;
      parse tl
    | [] -> ()
    | t :: _ -> failwith ("emit_ztt_drivers: unknown flag " ^ t)
  in
  parse (List.tl (Array.to_list Sys.argv));
  print_string (Ztt_drivers.emit_tu ~isa:(Isa.of_name !isa) ~uarch:(Uarch.of_name !uarch))
;;
