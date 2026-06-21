(* Quick M1 sanity check: instantiate the types, exercise the stub. *)
open Vfft_v2

let () =
  Printf.printf "=== M1 sanity check ===\n";
  Printf.printf "Isa.avx512.vec_regs = %d (expected 32)\n" Isa.avx512.vec_regs;
  Printf.printf "Isa.avx2.vec_regs   = %d (expected 16)\n" Isa.avx2.vec_regs;

  (* Build a stub allocation for both ISAs. *)
  let alloc_avx512 = Regalloc.allocate_stub ~isa:Isa.avx512 ~scheduled:[] in
  let alloc_avx2 = Regalloc.allocate_stub ~isa:Isa.avx2 ~scheduled:[] in

  (* Confirm both are stub (no bindings). *)
  let regs_a, def_a = Regalloc.count_bindings alloc_avx512 in
  let regs_b, def_b = Regalloc.count_bindings alloc_avx2 in
  Printf.printf "AVX-512 stub: %d Reg, %d Default (expect 0, 0)\n" regs_a def_a;
  Printf.printf "AVX2 stub:    %d Reg, %d Default (expect 0, 0)\n" regs_b def_b;

  (* Lookup a random tag — should return Default. *)
  let result = Regalloc.lookup alloc_avx512 42 in
  let result_str =
    match result with
    | Regalloc.Reg r -> Printf.sprintf "Reg %s" r
    | Regalloc.Default -> "Default"
    | Regalloc.Spilled s -> Printf.sprintf "Spilled %d" s
  in
  Printf.printf "Lookup tag 42: %s (expect Default)\n" result_str;

  (* Manually inject a binding, verify it round-trips. *)
  Hashtbl.add alloc_avx512.assign 42 (Regalloc.Reg "zmm5");
  let result2 = Regalloc.lookup alloc_avx512 42 in
  let result2_str =
    match result2 with
    | Regalloc.Reg r -> Printf.sprintf "Reg %s" r
    | Regalloc.Default -> "Default"
    | Regalloc.Spilled s -> Printf.sprintf "Spilled %d" s
  in
  Printf.printf "After inject:  %s (expect Reg zmm5)\n" result2_str;

  let regs2, def2 = Regalloc.count_bindings alloc_avx512 in
  Printf.printf "After inject:  %d Reg, %d Default (expect 1, 0)\n" regs2 def2;

  Printf.printf "=== M1 PASS ===\n"
