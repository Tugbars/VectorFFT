(* abi.mli — M4 (generator_lib_architecture.md §11.2).  LAYER 3.
   A codelet's C SIGNATURE as DATA: one TOTAL constructor over the measured
   kind shapes, one renderer.  Depends on Isa + Layout only.

   M4 staging (the doc's own order): this module lands BESIDE the legacy
   13-arm ladder in emit_c; `VFFT_ABI_XCHECK=1` makes emit_codelet render
   BOTH and assert byte equality on every emission — 1,020+ independent
   equality proofs across the corpus — and only after the xcheck runs clean
   does the ladder get deleted (the following commit).

   Deliberately ABSENT (refuted during review, §11.2): `prologue` — the
   spill declarations + hoisted constants are functions of the regalloc
   result and the scheduled DAG, i.e. Render.body_preamble, not a field
   here.  [signature] ends at the opening brace. *)

type shape =
  | Strided of { il : [ `None | `In | `Out ]; r2c : [ `No | `Fwd | `Bwd ] }
  | In_place of { il : [ `None | `In | `Out ] }
  | Twidsq
  | R2cb
  | R2cf
  | R2c_term_ls
  | R2c_term of { rt : bool }
  | Hc2c_nat of { ranged : bool }
  | Hc2c_nat_bwd of { ranged : bool }
  | Hc_strided of { ranged : bool }
  | N1_oop_strided
  | R2r
  | Oop_generic

type t = private
  { symbol : string
  ; target_attr : string
  ; params : Layout.param list
  }

val make : symbol:string -> target_attr:string -> shape -> t
(** TOTAL over [shape] — the compiler's warning-8 enforcement replaces the
    ladder's source-order priority spec.  All data-plane params come from
    Layout (the anti-hybrid law holds by construction). *)

val signature : t -> string
(** The attribute line, `void <symbol>(`, the parameter list, `)` and the
    opening brace — byte-exact to the historical ladder:
    ["__attribute__((target(\"...\")))\nvoid f(\n    ...,\n    size_t vl)\n{\n"] *)
