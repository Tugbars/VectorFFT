(* emit_render.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

module Scratch : sig
  type t =
    { mutable ls_mode : Isa.ls_mode
    ; mutable regalloc : Regalloc.allocation option
    ; mutable emit_position : int
    ; mutable fence_only : bool
    ; il_seen : (int, unit) Hashtbl.t
    ; il_pending : Buffer.t
    ; mutable il_stash : (int * string) option
    ; dup_barrier_tags : (int, unit) Hashtbl.t
    ; mutable unpin_candidates : (int, unit) Hashtbl.t option
    ; hoisted_const_tags : (int, unit) Hashtbl.t
    }

  val create : unit -> t
  val il_reset : t -> unit
  val il_take_pending : t -> string
end

module Cfg : sig
  type tw_source =
    | Tw_default
    | Tw_perpos
    | Tw_linear of int
    | Tw_zsplit of string

  val tw_linear_legs : tw_source -> int
  val tw_zsplit_off : tw_source -> string option

  type t =
    { r2r : bool
    ; r2cf : bool
    ; r2cb : bool
    ; hc_strided : bool
    ; n1_oop_strided : bool
    ; strided_il_in : bool
    ; strided_il_out : bool
    ; strided_ilo_nt : bool
    ; strided_r2c : bool
    ; strided_r2c_bwd : bool
    ; ip_il_in : bool
    ; ip_il_out : bool
    ; hc2c_natural : bool
    ; hc2c_natural_bwd : bool
    ; r2c_term : bool
    ; r2c_term_rt : bool
    ; r2c_term_ls : bool
    ; r2c_term_ls_r : int
    ; hc_ranged : bool
    ; hc_ranged_r : int
    ; hc2c_nat_r : int
    ; hc2c_nat_sstar : int
    ; store_on_compute : bool
    ; tw : tw_source
    }

  val default : t
end

val topo_sort_reachable : Ir.t list -> Ir.t list
val il_in_name : sc:Scratch.t -> Isa.t -> int -> bool -> string

val render_load
  :  sc:Scratch.t
  -> cfg:Cfg.t
  -> isa:Isa.t
  -> in_place:bool
  -> t1s:bool
  -> ?twidsq:bool
  -> ?twidsq_n:int
  -> ?strided:bool
  -> Expr.elem_ref
  -> string

val inline_max_depth : int
val compute_unpin_candidates : Ir.t list -> (int, unit) Hashtbl.t
val hoist_consts_enabled : bool ref
val render_hoisted_consts : sc:Scratch.t -> isa:Isa.t -> Ir.t list -> string

val render_node_def_core
  :  sc:Scratch.t
  -> cfg:Cfg.t
  -> ?no_declarator:bool
  -> ?inline_set:(int, unit) Hashtbl.t option
  -> ?twidsq:bool
  -> ?twidsq_n:int
  -> ?strided:bool
  -> isa:Isa.t
  -> in_place:bool
  -> t1s:bool
  -> Ir.t
  -> string

type scheduler =
  | Topological
  | Annotated_topological
  | SU of Uarch.t
  | Annotated_SU of Uarch.t

val compute_inline_set
  :  sc:Scratch.t
  -> (Expr.elem_ref * Ir.t) list
  -> (int, unit) Hashtbl.t

val is_spilled : Algsimp.spill_info -> int -> bool
val is_fused_slot : Algsimp.spill_info -> int -> bool

val codelet_metadata
  :  isa:Isa.t
  -> spill:Algsimp.spill_info option
  -> tw_broadcast:bool
  -> peak_live:int
  -> (Expr.elem_ref * Ir.t) list
  -> string

val is_fused_tag : Algsimp.spill_info -> int -> bool

val compute_min_slot_pass1
  :  Algsimp.spill_info
  -> Ir.t list
  -> (int, int) Hashtbl.t * Ir.t list

val cluster_split_schedule
  :  Algsimp.spill_info
  -> pass1_blocked_topo:Ir.t list
  -> min_slot:(int, int) Hashtbl.t
  -> schedule_cluster:(subset:Ir.t list -> sinks:Ir.t list -> Ir.t list)
  -> Ir.t list

val classify_passes
  :  Algsimp.spill_info
  -> Ir.t list
  -> (int, [ `Pass1 | `Pass2 ]) Hashtbl.t

val filter_inline_set_cross_pass
  :  sc:Scratch.t
  -> (Expr.elem_ref * Ir.t) list
  -> Algsimp.spill_info
  -> Ir.t list
  -> (int, unit) Hashtbl.t

val provenance_argv : string array option ref
val provenance_env_overrides : unit -> string
val provenance_block : family:string -> string list -> string

val render_node_def
  :  sc:Scratch.t
  -> cfg:Cfg.t
  -> ?no_declarator:bool
  -> ?inline_set:(int, unit) Hashtbl.t option
  -> ?twidsq:bool
  -> ?twidsq_n:int
  -> ?strided:bool
  -> isa:Isa.t
  -> in_place:bool
  -> t1s:bool
  -> Ir.t
  -> string

val body_preamble
  :  sc:Scratch.t
  -> isa:Isa.t
  -> spill:Algsimp.spill_info option
  -> ?consts:('a * Ir.t) list
  -> unit
  -> string
