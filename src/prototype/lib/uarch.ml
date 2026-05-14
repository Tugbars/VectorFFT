(* uarch.ml — per-microarchitecture latency + register-count profiles for SU.
 *
 * Each profile pairs an Isa.t with FMA/add/mul/load/store latencies (used
 * in critical-path distance) and vec_regs count (used in pressure
 * heuristics). Sourced from Agner Fog instruction tables, Intel
 * Optimization Reference Manual, AMD Software Optimization Guide.
 *
 * Approximate, L1-resident assumption. Doesn't model port pressure, µop
 * fusion, or forwarding paths — the scheduler uses simple cp-priority,
 * not a full resource model. *)

type t = {
  name: string;
  isa: Isa.t;

  (* Instruction latencies in cycles. Critical-path computation uses these. *)
  fma_latency: int;          (* fmadd_pd / fnmadd_pd / fmsub_pd / fnmsub_pd *)
  add_latency: int;          (* add_pd / sub_pd *)
  mul_latency: int;          (* mul_pd *)
  load_l1_latency: int;      (* loadu_pd hitting L1 *)
  store_latency: int;        (* storeu_pd retire latency (mostly hidden) *)

  (* Capacity. *)
  vec_regs: int;             (* architectural vector registers *)
  pressure_threshold: int;   (* Below this many concurrent live values,
                                don't bother prioritizing pressure reduction. *)

  fma_throughput: int;       (* FMAs per cycle (rounded up) *)
}

(* === PROFILES === *)

let sapphire_rapids_avx512 = {
  name = "sapphire_rapids_avx512";
  isa = Isa.avx512;
  fma_latency = 4;
  add_latency = 4;
  mul_latency = 4;
  load_l1_latency = 7;
  store_latency = 1;
  vec_regs = 32;
  pressure_threshold = 24;
  fma_throughput = 2;
}

let raptor_lake_avx512 = {
  name = "raptor_lake_avx512";
  isa = Isa.avx512;
  fma_latency = 4;
  add_latency = 4;
  mul_latency = 4;
  load_l1_latency = 5;
  store_latency = 1;
  vec_regs = 32;
  pressure_threshold = 24;
  fma_throughput = 2;
}

let raptor_lake_avx2 = {
  name = "raptor_lake_avx2";
  isa = Isa.avx2;
  fma_latency = 4;
  add_latency = 4;
  mul_latency = 4;
  load_l1_latency = 5;
  store_latency = 1;
  vec_regs = 16;
  pressure_threshold = 12;
  fma_throughput = 2;
}

let zen5_avx512 = {
  name = "zen5_avx512";
  isa = Isa.avx512;
  fma_latency = 4;
  add_latency = 3;
  mul_latency = 3;
  load_l1_latency = 4;
  store_latency = 1;
  vec_regs = 32;
  pressure_threshold = 24;
  fma_throughput = 2;
}

let generic_avx512 = {
  name = "generic_avx512";
  isa = Isa.avx512;
  fma_latency = 5;
  add_latency = 4;
  mul_latency = 4;
  load_l1_latency = 6;
  store_latency = 1;
  vec_regs = 32;
  pressure_threshold = 24;
  fma_throughput = 2;
}

let generic_avx2 = {
  name = "generic_avx2";
  isa = Isa.avx2;
  fma_latency = 5;
  add_latency = 4;
  mul_latency = 4;
  load_l1_latency = 6;
  store_latency = 1;
  vec_regs = 16;
  pressure_threshold = 12;
  fma_throughput = 2;
}

let of_name (s : string) : t =
  match s with
  | "sapphire_rapids" | "sapphire_rapids_avx512" | "spr" -> sapphire_rapids_avx512
  | "raptor_lake" | "raptor_lake_avx512" -> raptor_lake_avx512
  | "raptor_lake_avx2" -> raptor_lake_avx2
  | "zen5" | "zen5_avx512" -> zen5_avx512
  | "generic" | "generic_avx512" -> generic_avx512
  | "generic_avx2" -> generic_avx2
  | other ->
    failwith (Printf.sprintf "unknown uarch: %s (try: sapphire_rapids, raptor_lake, raptor_lake_avx2, zen5, generic)" other)
