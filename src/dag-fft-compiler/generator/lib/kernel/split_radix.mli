(* split_radix.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. Existing SR construction. PLACED, FROZEN, NEVER DEVELOPED (standing ban).
   Generated from the inferred signature; trim = later per-module work. *)

val const_cmul : Expr.expr -> Expr.expr -> float -> float -> Expr.expr * Expr.expr

type dft_callback =
  sign:[ `Bwd | `Fwd ]
  -> int
  -> (int -> Expr.expr)
  -> (int -> Expr.expr)
  -> Expr.expr array * Expr.expr array

val dft_split_radix
  :  dft_rec:dft_callback
  -> ?sign:[ `Bwd | `Fwd ]
  -> int
  -> (int -> Expr.expr)
  -> (int -> Expr.expr)
  -> Expr.expr array * Expr.expr array

val newsplit_core
  :  [ `Bwd | `Fwd ]
  -> (int -> (int -> Expr.expr * Expr.expr) -> (Expr.expr * Expr.expr) array)
     * (int -> (int -> Expr.expr * Expr.expr) -> (Expr.expr * Expr.expr) array)
     * (int
        -> (Expr.expr * Expr.expr) array
        -> (Expr.expr * Expr.expr) array
        -> (Expr.expr * Expr.expr) array
        -> (Expr.expr * Expr.expr) array)

val dft_newsplit
  :  sign:[ `Bwd | `Fwd ]
  -> int
  -> (int -> Expr.expr)
  -> (int -> Expr.expr)
  -> Expr.expr array * Expr.expr array

val dft_newsplit_blocked
  :  sign:[ `Bwd | `Fwd ]
  -> int
  -> (int -> Expr.expr)
  -> (int -> Expr.expr)
  -> Expr.expr array * Expr.expr array * (int * Expr.expr * Expr.expr) list
