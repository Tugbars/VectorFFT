(* codelet.mli — M5.  THE WORD FOR THE THING WE COMPILE (§10.2).

   Constructors are exposed CONCRETELY (G9): the totality laws downstream
   (`recipes` at M10, the shape dispatch) rely on warning-8 exhaustiveness,
   which an abstract type would silently disable.

   ROUND-TRIP CONTRACT (M5 acceptance, bin_test/argv_roundtrip.ml):
   `to_argv (of_argv l) == l` VERBATIM — flag order included — for every
   live recorded argv line (1,183 provenance + 221 derive = 1,404/1,404).
   The 16 orphaned rfft/avx512_regen files are excluded: dead-era flag
   order, 0/16 in the gate, pool-sunset candidates. *)

type direction =
  | Fwd
  | Bwd

type tw_table =
  | Flat
  | Log3

type modifiers =
  { dir : direction
  ; dif : bool
  ; table : tw_table
  ; t1s : bool
  ; su : bool
  }

type il3 =
  [ `None
  | `In
  | `Out
  ]

type sw3 =
  [ `No
  | `Il
  | `Il_sw
  ]

type oop_edge =
  | UG
  | UL

type oop_tw =
  | Tw_group
  | Tw_pos
  | Post_tw
  | Tw_linear

type trig8 =
  | Dct1
  | Dct2
  | Dct3
  | Dct4
  | Dst1
  | Dst2
  | Dst3
  | Dst4
  | Dht

type cil_form =
  | Cil_n1
  | Cil_n1c
  | Cil_t2c
  | Cil_n1t
  | Cil_t2

type cil_turn =
  | Turnst
  | Turnst_gs

type zs_kind =
  | Dts
  | Dtsn
  | Dtso
  | Dtt
  | Msd
  | Msg
  | Msgb
  | S0s
  | S0sb
  | S0t
  | S0tb
  | Stf
  | Stf2
  | Stfb
  | Stfbn
  | Stfn
  | Sterm
  | Sterm2
  | Stermb

type kind =
  | C2c_inplace_su of { il : il3 }
  | C2c_inplace_tw of { il : il3 }
  | C2c_oop of
      { load : oop_edge
      ; store : oop_edge
      ; tw : oop_tw option
      ; fuse : int option
      ; store_fused : bool
      ; strides : (int * int * int * int) option
      ; spec_named : bool
      ; il_in : sw3
      ; il_out : sw3
      }
  | R2cf
  | R2cb
  | Hc2hc of { ranged : bool }
  | Hc2c
  | Hc2c_nat of { ranged : bool }
  | R2c_term of
      { rt : bool
      ; k : int option
      }
  | R2c_term_ls of { r : int }
  | Trig of trig8
  | Strided of { il : [ `No | `In | `Out | `Out_nt ] }
  | Strided_r2c
  | N1_oop_strided
  | Cil of
      { form : cil_form
      ; tangent : bool (* --cil-tangent: tangent-scaled butterfly interior *)
      ; blocked : bool
      ; oddct : bool
      ; split : (int * int) option
      ; turn : cil_turn option
      ; pre_tw : bool
      ; form_tag : bool
        (* --cil-form-tag: name the FORM in the emitted symbol, so a split /
           tangent / wing variant is distinguishable without a post-emit sed *)
      }
  | Zsplit of
      { k : zs_kind
      ; r0 : int option
      ; sink : bool
      }
  | K1_mono of
      { r1 : int option
      ; il : bool
      ; sw : bool
      }

type t =
  { radix : int
  ; isa : string option
  ; uarch : string option
  ; kind : kind
  ; mods : modifiers
  ; emit_c : bool
  }

exception Parse_error of string

(** Order-insensitive parse over the corpus flag surface.  [strict] (default
    true) raises [Parse_error] on an unknown flag; [~strict:false] skips
    unknown tokens — the Driver's mode, since gen_main's full CLI carries
    knobs orthogonal to the descriptor. Validates the measured invariants
    (in-place su (+) twiddled; hc2c-nat bwd <=> dif). *)
val of_argv : ?strict:bool -> string list -> t

(** The canonical per-kind flag sequence — reproduces recorded provenance
    lines verbatim.  provenance == coverage == regen recipe, one fact. *)
val to_argv : t -> string list

val validate : t -> t
