(* uarch.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. Microarch latency/port tables.
   Generated from the inferred signature; trim = later per-module work. *)

type t =
  { name : string
  ; isa : Isa.t
  ; fma_latency : int
  ; add_latency : int
  ; mul_latency : int
  ; load_l1_latency : int
  ; store_latency : int
  ; vec_regs : int
  ; pressure_threshold : int
  ; fma_throughput : int
  }

val sapphire_rapids_avx512 : t
val raptor_lake_avx512 : t
val raptor_lake_avx2 : t
val zen5_avx512 : t
val generic_avx512 : t
val generic_avx2 : t
val of_name : string -> t
