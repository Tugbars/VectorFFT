(* gen_radix — thin CLI wrapper. The whole generator pipeline lives in
 * lib/gen_main.ml (section 39) so the in-process driver gen_set can
 * call it without forking per codelet. *)
let () = Vfft_v2.Gen_main.run Sys.argv
