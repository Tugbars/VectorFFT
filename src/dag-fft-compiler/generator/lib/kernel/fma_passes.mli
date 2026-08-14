(* fma_passes.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

val factor_const_muls
  :  ?frozen_tags:(int, unit) Hashtbl.t option
  -> (Expr.elem_ref * Ir.t) list
  -> (Expr.elem_ref * Ir.t) list * (int, int) Hashtbl.t

val fma_lift
  :  ?frozen_tags:(int, unit) Hashtbl.t option
  -> (Expr.elem_ref * Ir.t) list
  -> (Expr.elem_ref * Ir.t) list

val multi_use_fma_lift
  :  ?frozen_tags:(int, unit) Hashtbl.t option
  -> (Expr.elem_ref * Ir.t) list
  -> (Expr.elem_ref * Ir.t) list * (int, int) Hashtbl.t

val fma_addend_factor
  :  ?frozen_tags:(int, unit) Hashtbl.t option
  -> (Expr.elem_ref * Ir.t) list
  -> (Expr.elem_ref * Ir.t) list * (int, int) Hashtbl.t

val flatten_fma_mul_addend
  :  ?frozen_tags:(int, unit) Hashtbl.t option
  -> (Expr.elem_ref * Ir.t) list
  -> (Expr.elem_ref * Ir.t) list * (int, int) Hashtbl.t
