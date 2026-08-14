let () =
  List.iter
    (fun n ->
       match Dft_select.pick_algorithm n with
       | Dft_select.Direct -> Printf.printf "R=%d Direct\n" n
       | Dft_select.Split_radix -> Printf.printf "R=%d Split_radix\n" n
       | Dft_select.Cooley_Tukey (a, b) -> Printf.printf "R=%d CT(%d,%d)\n" n a b)
    [ 16; 32; 64 ]
;;
