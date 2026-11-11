(* isa.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. ISA record + VALUE intrinsics (all width branches internal) — the model shared module.
   Generated from the inferred signature; trim = later per-module work. *)

type prec =
  | F64
  | F32

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
  ; prec : prec
  }

type ls_mode =
  | LS_vector
  | LS_masked of string

val avx512 : t
val avx2 : t
val avx512vl256 : t
val scalar : t
val sse2 : t

(* FP32 targets (2026-11-10): vec_width counts floats *)
val avx2_f32 : t
val avx512_f32 : t
val sse_f32 : t
val avx512vl256_f32 : t

(* one complex float (64 bits, bits = 64): the FP32 cil tail's last rung *)
val sse_f32x1 : t
val of_name : string -> t
val is_f32 : t -> bool
val bits : t -> int
val cpv : t -> int
val sfx : t -> string
val elem : t -> string
val kmask_type : t -> string
val flit : t -> float -> string
val f32_todo : t -> string -> unit
val intr : t -> string -> string
val mul_pd : t -> string -> string -> string
val add_pd : t -> string -> string -> string
val sub_pd : t -> string -> string -> string
val addsub_pd : t -> string -> string -> string
val xor_pd : t -> string -> string -> string
val fmadd_pd : t -> string -> string -> string -> string
val fmaddsub_pd : t -> string -> string -> string -> string
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
val deint_ordered : t -> string -> string -> string * string
val reint_pre : t -> string -> string
val reint_ordered : t -> string -> string -> string * string
val shuffle_consts : t -> deint:bool -> reint:bool -> transpose:bool -> string
val transpose : t -> qid:string -> string array -> string array -> string

(* ZTURN-T plain class column-run z edge (FP32 2026-11-10; FP64 = the unpack pair) *)
val zcol_reint : t -> string -> string -> string * string
val zcol_deint : t -> string -> string -> string * string
val const_splat_decl : t -> string -> string -> string

val cx_deint_pd : t -> odd:bool -> string -> string -> string
val cx_part_pd : t -> string -> int -> string

(* FP32 corner-turn stores (2026-11-10): a complex lane is one 64-bit unit *)
val storeu_cx_lane : t -> string -> string -> int -> string
val storeu_cx_unit : t -> string -> string -> int -> string
val cx_half : t -> hi:bool -> string -> string -> string
val cx_unpack : t -> hi:bool -> string -> string -> string
val storeu_cx_prefix : t -> string -> string -> int -> string
val cx_target_attr : t -> string
