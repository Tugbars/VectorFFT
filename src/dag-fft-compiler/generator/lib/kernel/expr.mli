(* expr.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. elem_ref + the symbolic expression type — the ONE type both compiler stacks share.
   Generated from the inferred signature; trim = later per-module work. *)

type elem_ref =
    Input of int * bool
  | Output of int * bool
  | Twiddle of int * bool
type expr =
    Const of float
  | Load of elem_ref
  | Neg of expr
  | Add of expr * expr
  | Sub of expr * expr
  | Mul of expr * expr
val mk_const : float -> expr
val mk_neg : expr -> expr
val mk_mul : expr -> expr -> expr
val mk_add : expr -> expr -> expr
val mk_sub : expr -> expr -> expr
type assignment = elem_ref * expr
val string_of_elem_ref : elem_ref -> string
val string_of_expr_prec : int -> expr -> string
val string_of_expr : expr -> string
val string_of_assignments : assignment list -> string
val dft_kernel : int -> (int -> expr) -> (int -> expr) -> assignment list
val dft_expand : int -> assignment list
val dft_expand_twiddled : int -> assignment list
