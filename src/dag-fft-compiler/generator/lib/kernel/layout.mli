(* layout.mli — M3 (generator_lib_architecture.md §10.1, §12.1).  LAYER 0.
   THE ANTI-HYBRID LAW LIVES HERE AND NOWHERE ELSE.

   The law: a codelet signature never carries a split pair (X_re + X_im) AND an
   interleaved pointer FOR THE SAME SIDE.  It is enforced structurally:
   [param] is private — the only constructors are [pointers] (total on ONE
   plane per call), [scalar] (whose ctype is validated against a closed list
   of scalar C types), and [tw_pair].  A hybrid parameter list is therefore
   not expressible outside this module (the §12.1 exploit rebuild: the record
   literal does not compile; the scalar smuggle raises).

   Planes are the MEASURED corpus vocabulary (X2/OQ-1 census), including the
   one the design sketch missed: [Real] — a single real-data plane with a bare
   name (the strided-r2c family: `rio`, `out`).  A real plane beside a split
   pair is LEGAL (r2c: real in, split out) — the ban is z-beside-split on one
   side, and that remains unconstructible. *)

type plane =
  | Split (** two pointers: <p>_re, <p>_im *)
  | Inter (** one pointer: <p>_z (pairs re,im); optional silenced twin <p>_unused *)
  | Inter_sw (** one pointer, pairs (im,re) — the bwd-swap enabler; prints as Inter *)
  | Real (** one REAL-data pointer, bare name <p> (r2c/c2r strided family) *)

type buffers =
  | Rio of plane (** true in-place: ONE buffer *)
  | From_z (** in_z -> rio_re, rio_im  (ip_il_in) *)
  | To_z (** rio_re, rio_im -> out_z (ip_il_out) *)
  | Oop of
      { load : plane
      ; store : plane
      } (** all 4x4 legal, incl. boundary conversions *)

type param = private
  { ctype : string (** "const double * " or "double       * " — byte-exact *)
  ; name : string
  ; restrict_ : bool
  ; silence : bool (** body must (void) it — the frozen-ABI unused twins *)
  ; comment : string option (** rendered at column 48, e.g. "/* interleaved pairs */" *)
  }

(** THE data-plane constructor.  Total, ONE plane per call; no overload takes
    two planes, and no other function returns a data-plane [param]. *)
val pointers
  :  plane
  -> const:bool
  -> prefix:string
  -> twin:bool (** add the silenced <p>_unused partner (frozen ABIs) *)
  -> ?comment:string (** attached to the plane's FIRST pointer only *)
  -> unit
  -> param list

(** The only other pointer-free constructor; raises [Invalid_argument] unless
    [ctype] is in the closed scalar list (size_t, int, ptrdiff_t, uint32_t,
    double). *)
val scalar : ctype:string -> name:string -> param

(** The twiddle-table pair (tw_re, tw_im) — the third and last legal pointer
    source, distinct from data planes. *)
val tw_pair : restrict_:bool -> param list

(** One C parameter line, byte-exact to the historical printers:
    ["    const double * __restrict__ in_z,          /* interleaved pairs */\n"] *)
val render : param -> string

(** ["    (void)in_unused;\n"] for silenced params, [None] otherwise. *)
val silencer : param -> string option

(** The OOP family's four historical booleans -> one [Oop] value.  Raises
    [Invalid_argument] on il_in && il_in_sw (or the out pair) — the two
    previously UNGUARDED illegal states (§12.1). *)
val buffers_of_oop_bools
  :  il_in:bool
  -> il_in_sw:bool
  -> il_out:bool
  -> il_out_sw:bool
  -> buffers

(** The in-place pair -> [From_z]/[To_z]/[Rio Split].  Raises
    [Invalid_argument] on both — the banned hybrid combination, previously
    accepted silently and resolved by if/else order in two places. *)
val ip_buffers_of_bools : il_in:bool -> il_out:bool -> buffers
