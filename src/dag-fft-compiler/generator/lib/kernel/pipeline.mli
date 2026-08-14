(* pipeline.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. THE pass cascade (sole owner after M11; today one of three copies — see P4).
   Generated from the inferred signature; trim = later per-module work. *)

type prepared = {
  assigns : (Expr.elem_ref * Ir.t) list;
  spill_info : Algsimp.spill_info option;
}
val prepare_codelet :
  raw_assigns:(Expr.elem_ref * Expr.expr) list ->
  spill_markers_raw:Dft.spill_marker list ->
  spill_ct:(int * int) option ->
  reassoc:bool ->
  aggressive:bool ->
  algorithm:Dft_select.algorithm ->
  force_fma_lift:bool ->
  disable_fma_lift:bool -> build_spill_info:bool -> fuse:int -> prepared
