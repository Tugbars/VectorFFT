(* dft.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

type twiddle_policy =
  | TP_Flat
  | TP_Log3
  | TP_PowW1

type direction =
  | DIT
  | DIF

val cmul_pattern
  :  ?conj:bool
  -> Expr.expr
  -> Expr.expr
  -> Expr.expr
  -> Expr.expr
  -> Expr.expr * Expr.expr

val twiddle_expr : twiddle_policy -> int -> int -> Expr.expr * Expr.expr
val dft_expand : ?sign:[ `Bwd | `Fwd ] -> int -> Expr.assignment list

val dft_expand_twiddled
  :  ?policy:twiddle_policy
  -> ?direction:direction
  -> ?sign:[ `Bwd | `Fwd ]
  -> ?table_conj:bool
  -> int
  -> Expr.assignment list

val dft_expand_twidsq
  :  ?direction:direction
  -> ?sign:[ `Bwd | `Fwd ]
  -> int
  -> Expr.assignment list

type spill_marker =
  { slot : int
  ; re_expr : Expr.expr
  ; im_expr : Expr.expr
  }

val dft_expand_twiddled_il2
  :  ?policy:twiddle_policy
  -> ?direction:direction
  -> ?sign:[ `Bwd | `Fwd ]
  -> int
  -> Expr.assignment list

val dft_expand_twiddled_spill
  :  ?policy:twiddle_policy
  -> ?direction:direction
  -> ?sign:[ `Bwd | `Fwd ]
  -> int
  -> Expr.assignment list * spill_marker list * (int * int) option

val dft_expand_n1_blocked
  :  ?sign:[ `Bwd | `Fwd ]
  -> int
  -> Expr.assignment list * spill_marker list * (int * int) option

val dft_expand_newsplit_blocked
  :  ?sign:[ `Bwd | `Fwd ]
  -> int
  -> Expr.assignment list * spill_marker list * (int * int) option

val should_spill : int -> int -> bool
val exceeds_register_budget : int -> int -> bool
val should_block_n1 : int -> int -> bool
