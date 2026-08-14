(* dft_recurse.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

val const_cmul :
  Expr.expr -> Expr.expr -> float -> float -> Expr.expr * Expr.expr
val dft :
  ?sign:[ `Bwd | `Fwd ] ->
  int ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
val dft_direct :
  ?sign:[ `Bwd | `Fwd ] ->
  int ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
val dft_direct_conjugate_pair :
  ?sign:[ `Bwd | `Fwd ] ->
  int ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
val dft_winograd5 :
  ?sign:[ `Bwd | `Fwd ] ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
val dft_winograd5_cnum :
  ?sign:[ `Bwd | `Fwd ] -> (int -> Cnum.cnum) -> Cnum.cnum array
val dft_winograd25 :
  ?sign:[ `Bwd | `Fwd ] ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
val dft_winograd7 :
  ?sign:[ `Bwd | `Fwd ] ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
val dft_ct :
  ?sign:[ `Bwd | `Fwd ] ->
  int ->
  int ->
  (int -> Expr.expr) ->
  (int -> Expr.expr) -> Expr.expr array * Expr.expr array
