(* emit_body.mli — M8.5 (G7): the engine's interface.  §10's rule
   ("more than one consumer OR a layer boundary") applies: after M8 the
   engine serves exactly TWO consumers — Real (the r2c/c2r/hc/trig
   route) and C2c_split (the split-complex c2c route) — across the
   kernel/gen layer boundary.  gen_main no longer calls it directly. *)

(* FAMILY HOOKS — the seam between the shared engine and the feature
   modules.  Each field names a POSITION in the emission sequence; a
   family passes closures whose bodies are its moved arms (capturing
   cfg/isa/radix family-side).  The engine keeps the cfg dispatch tests
   and fails LOUDLY if a family-owned arm is reached without its hook.
   Field order matches emit_body.ml exactly (a record's .mli must). *)
type family_hooks =
  { strided_prologue : (Buffer.t -> unit) option
  ; strided_locals : (Buffer.t -> unit) option
  ; strided_load : (Buffer.t -> unit) option
  ; strided_store : (Buffer.t -> unit) option
  ; strided_idx : (Buffer.t -> unit) option
  ; strided_load_il : (Buffer.t -> unit) option
  ; strided_store_il : (Buffer.t -> unit) option
  ; trailer : (Buffer.t -> unit) option
  }

val no_hooks : family_hooks

val emit_codelet
  :  ?hooks:family_hooks
  -> sc:Emit_render.Scratch.t
  -> cfg:Emit_render.Cfg.t
  -> ?in_place:bool
  -> ?t1s:bool
  -> ?twidsq:bool
  -> ?twidsq_n:int
  -> ?strided:bool
  -> ?radix:int
  -> ?scheduler:Emit_render.scheduler
  -> ?isa:Isa.t
  -> ?gh:bool
  -> ?bb_budget:float option
  -> ?spill:Algsimp.spill_info option
  -> ?is_log3:bool
  -> (Expr.elem_ref * Ir.t) list
  -> name:string
  -> string

(* UnitLeg-path no-op survivor (M8.2 culled its failwith companions). *)
val emit_avx512_transpose_indices : Isa.t -> Buffer.t -> unit
