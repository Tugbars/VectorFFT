(* isa.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. ISA record + VALUE intrinsics (all width branches internal) — the model shared module.
   Generated from the inferred signature; trim = later per-module work. *)

type t =
  { name : string
  ; vec_type : string
  ; vec_width : int
  ; vec_regs : int
  ; intrinsic_prefix : string
  ; target_attr : string
  ; loadu_pd : string
  ; storeu_pd : string
  ; set1_pd : string
  ; maskload_pd : string
  ; maskstore_pd : string
  }

type ls_mode =
  | LS_vector
  | LS_masked of string

val avx512 : t
val avx2 : t
val scalar : t
val sse2 : t
val of_name : string -> t
val intr : t -> string -> string
val mul_pd : t -> string -> string -> string
val add_pd : t -> string -> string -> string
val sub_pd : t -> string -> string -> string
val addsub_pd : t -> string -> string -> string
val xor_pd : t -> string -> string -> string
val fmadd_pd : t -> string -> string -> string -> string
val fnmadd_pd : t -> string -> string -> string -> string
val fmsub_pd : t -> string -> string -> string -> string
val fnmsub_pd : t -> string -> string -> string -> string
val set1_pd_str : t -> string -> string
val cflip_pd : t -> string -> string
val xor_mask_pd : t -> string -> string -> string
val im_mask_decl : t -> string -> string
val re_mask_decl : t -> string -> string
val loadu_pd : ?mode:ls_mode -> t -> string -> string
val storeu_pd : ?mode:ls_mode -> t -> string -> string -> string
val const_decl : t -> string -> string -> string
val pinned_reg_decl : t -> string -> string -> string -> string
val fenced_decl : t -> string -> string -> string
val forward_decl : t -> string list -> string
