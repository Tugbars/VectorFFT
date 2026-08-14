(* knobs.mli — M6.0 (the M5 row's second half, landed beside its consumers).
   LAYER 0.  The env-var REGISTRY: every knob DECLARED once, READ once
   (lazily, at first use), consumed through here — never via a scattered
   `Sys.getenv` at the point of use.

   This stage keeps the RAW `string option` per key so each consumer's
   historical parse (`<> None`, `= Some "1"`, numeric) stays byte-identical
   in place; the typed per-consumer recipes of §11.5 arrive when Render.ctx
   threads (M6.1+).  Semantics change vs before: each key is read ONCE PER
   PROCESS instead of per call — the design intent (a snapshot), and inert
   unless something mutated the environment mid-run (nothing does).

   Trace = the pure-diagnostic keys (§11.5's split): they cannot change
   emitted bytes and stay ambient forever — never threaded. *)

(* ── byte-affecting knobs (this tranche: schedule / fma_passes / simplify) ── *)
val sched_order : unit -> string option (* VFFT_SCHED_ORDER *)
val sched_loads : unit -> string option (* VFFT_SCHED_LOADS *)
val load_pace : unit -> string option (* VFFT_LOAD_PACE *)
val su_tiebreak : unit -> string option (* VFFT_SU_TIEBREAK *)
val gh_threshold : unit -> string option (* VFFT_GH_THRESHOLD *)
val fma_multiuse : unit -> string option (* VFFT_FMA_MULTIUSE *)
val collect_m : unit -> string option (* VFFT_COLLECT_M *)
val deep_collect : unit -> string option (* VFFT_DEEP_COLLECT *)

module Trace : sig
  val factor : unit -> bool (* FACTOR_TRACE *)
  val mulift : unit -> bool (* MULIFT_TRACE *)
  val mulfma : unit -> bool (* MULFMA_TRACE *)
  val fma_addend : unit -> bool (* FMA_ADDEND_TRACE *)
  val flatten_fma_mul : unit -> bool (* FLATTEN_FMA_MUL_TRACE *)
  val flatten_fma_mul_verbose : unit -> bool (* FLATTEN_FMA_MUL_TRACE_VERBOSE *)
  val deep_collect : unit -> bool (* VFFT_DEEP_COLLECT_TRACE *)
  val sched_dump : unit -> string option (* VFFT_SCHED_DUMP (carries a value) *)
end
