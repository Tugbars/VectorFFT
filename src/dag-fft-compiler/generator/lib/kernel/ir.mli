(* ir.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. Hash-consed DAG + smart constructors.

   RESET CONTRACT (the four memo tables below are the library's only
   legitimate process-global state): Ir.reset runs ONCE PER CODELET —
   gen_set's warm process depends on it; next_tag:=0 per codelet means any
   tag-keyed table surviving a codelet boundary aliases different nodes.
   KNOWN EXEMPTION: Algsimp.fresh bypasses hcons_table on the spill-lift
   path (production), so hcons totality is NOT an invariant (doc §12.3).
   Generated from the inferred signature; trim = later per-module work. *)

type node_kind =
  | NK_Const of float
  | NK_Load of Expr.elem_ref
  | NK_Neg of t
  | NK_Add of t * t
  | NK_Sub of t * t
  | NK_Mul of t * t
  | NK_Plus of (int * t) list
  | NK_CmulRe of t * t * t * t
  | NK_CmulIm of t * t * t * t
  | NK_Fma of t * t * t * bool * bool

and t =
  { tag : int
  ; node : node_kind
  }

val preds : t -> t list
val topo_sort_reachable : t list -> t list
val nk_plus_unreachable : string -> 'a
val hcons_table : (node_kind, t) Hashtbl.t
val next_tag : int ref
val hashcons : node_kind -> t
val lookup_node : node_kind -> t option

module ExprPhysHash : sig
  type t = Expr.expr

  val equal : 'a -> 'a -> bool
  val hash : 'a -> int
end

module ExprMemo : sig
  type key = ExprPhysHash.t
  type 'a t = 'a Hashtbl.Make(ExprPhysHash).t

  val create : int -> 'a t
  val clear : 'a t -> unit
  val reset : 'a t -> unit
  val copy : 'a t -> 'a t
  val add : 'a t -> key -> 'a -> unit
  val remove : 'a t -> key -> unit
  val find : 'a t -> key -> 'a
  val find_opt : 'a t -> key -> 'a option
  val find_all : 'a t -> key -> 'a list
  val replace : 'a t -> key -> 'a -> unit
  val mem : 'a t -> key -> bool
  val iter : (key -> 'a -> unit) -> 'a t -> unit
  val filter_map_inplace : (key -> 'a -> 'a option) -> 'a t -> unit
  val fold : (key -> 'a -> 'acc -> 'acc) -> 'a t -> 'acc -> 'acc
  val length : 'a t -> int
  val stats : 'a t -> Hashtbl.statistics
  val to_seq : 'a t -> (key * 'a) Seq.t
  val to_seq_keys : 'a t -> key Seq.t
  val to_seq_values : 'a t -> 'a Seq.t
  val add_seq : 'a t -> (key * 'a) Seq.t -> unit
  val replace_seq : 'a t -> (key * 'a) Seq.t -> unit
  val of_seq : (key * 'a) Seq.t -> 'a t
end

val of_expr_memo : t ExprMemo.t
val const_ident : (string, t) Hashtbl.t
val reset : unit -> unit
val zero_threshold : float
val is_zero : t -> bool
val is_one : t -> bool
val is_neg_one : t -> bool
val mk_const : float -> t
val mk_load : Expr.elem_ref -> t
val flatten_sum : int -> t -> (int * t) list
val flatten_sum_through_fma : int -> t -> (int * t) list
val cancel_signs : (int * t) list -> (int * t) list
val split_interleaved : 'a list -> 'a list * 'a list
val mk_neg : t -> t
val mk_add : t -> t -> t
val mk_sub : t -> t -> t
val mk_mul : t -> t -> t
val mk_add_binary : t -> t -> t
val mk_sub_binary : t -> t -> t
val mk_plus : (int * t) list -> t
val lower_plus : t -> t
val lower_plus_terms : (int * t) list -> t
val mk_cmul : t -> t -> t -> t -> t * t
val emit_signed_term : int * t -> t
val combine_two : int * t -> int * t -> t
val emit_pair_fold : (int * t) list -> t
val of_expr : ?reassoc:bool -> ExprMemo.key -> t
val of_assignments : ?reassoc:bool -> Expr.assignment list -> (Expr.elem_ref * t) list
