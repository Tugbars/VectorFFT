(* cnum.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. Exact complex constants.
   Generated from the inferred signature; trim = later per-module work. *)

type cnum = { re : Expr.expr; im : Expr.expr; }
val cnum : Expr.expr -> Expr.expr -> cnum
val czero : cnum
val cone : cnum
val cof_re : Expr.expr -> cnum
val cconst : float -> float -> cnum
val cneg : cnum -> cnum
val cconj : cnum -> cnum
val cadd : cnum -> cnum -> cnum
val csub : cnum -> cnum -> cnum
val cscale : Expr.expr -> cnum -> cnum
val cmul : cnum -> cnum -> cnum
val cmul_i : cnum -> cnum
val cmul_negi : cnum -> cnum
val croot_of_unity_fwd : int -> int -> cnum
val croot_of_unity_bwd : int -> int -> cnum
val croot_of_unity : sign:[ `Bwd | `Fwd ] -> int -> int -> cnum
val signal_of_re_im : (int -> Expr.expr) -> (int -> Expr.expr) -> int -> cnum
val split_re_im : cnum array -> Expr.expr array * Expr.expr array
