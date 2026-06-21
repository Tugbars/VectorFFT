let () =
  let open Vfft_v2 in
  List.iter
    (fun n ->
      match Dft.pick_algorithm n with
      | Dft.Direct -> Printf.printf "R=%d Direct\n" n
      | Dft.Split_radix -> Printf.printf "R=%d Split_radix\n" n
      | Dft.Cooley_Tukey (a, b) -> Printf.printf "R=%d CT(%d,%d)\n" n a b)
    [ 16; 32; 64 ]
