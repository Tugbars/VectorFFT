(* dft_select.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

type algorithm =
  | Direct
  | Cooley_Tukey of int * int
  | Split_radix

val newsplit_enabled : unit -> bool
val split_radix_enabled : unit -> bool
val target_vec_regs : int ref
val factor_override : int -> algorithm option
val pick_algorithm : int -> algorithm
val needs_reassoc : int -> bool
