(* bb.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. Basic-block / budget analysis.
   Generated from the inferred signature; trim = later per-module work. *)

val preds_of : Ir.t -> Ir.t list
val compute_peak_live :
  subset:Ir.t list -> sinks:Ir.t list -> Ir.t list -> int
val compute_progress : Ir.t list -> (int, int) Hashtbl.t -> int
val bb_search :
  uarch:Uarch.t ->
  subset:Ir.t list ->
  sinks:Ir.t list ->
  initial_schedule:Ir.t list ->
  initial_peak:int -> time_budget_sec:float -> Ir.t list * int * int
val bb_schedule_subset :
  Uarch.t ->
  time_budget_sec:float -> subset:Ir.t list -> sinks:Ir.t list -> Ir.t list
