(* dump_profile.ml — quick driver to print op counts for one (R, variant).
 *
 * Usage:
 *   dune exec bin/dump_profile.exe -- 16 n1
 *   dune exec bin/dump_profile.exe -- 32 t1
 *
 * Output:
 *   one CSV-format line matching profile_avx{2,512}.csv exactly. *)

let () =
  if Array.length Sys.argv < 3 then begin
    prerr_endline "usage: dump_profile <radix> <variant>";
    prerr_endline "  variant: n1 | t1";
    exit 1
  end;
  let n = int_of_string Sys.argv.(1) in
  let variant = Sys.argv.(2) in
  let assigns =
    match variant with
    | "n1" -> Vfft_v2.Dft.dft_expand n
    | "t1" -> Vfft_v2.Dft.dft_expand_twiddled n
    | other -> failwith ("unknown variant " ^ other)
  in
  Vfft_v2.Algsimp.reset ();
  let reassoc = Vfft_v2.Dft.needs_reassoc n in
  let simplified = Vfft_v2.Algsimp.of_assignments ~reassoc assigns in
  let deduped = Vfft_v2.Algsimp.dedup_sub_pairs simplified in
  let counts = Vfft_v2.Profile.count_ops deduped in
  print_endline Vfft_v2.Profile.csv_header;
  print_endline
    (Vfft_v2.Profile.csv_row
       ~radix:n ~variant ~present:true
       ~file:(Printf.sprintf "r%d_%s.c" n variant)
       counts)
