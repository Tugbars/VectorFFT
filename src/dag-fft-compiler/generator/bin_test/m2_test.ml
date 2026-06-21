open Vfft_v2

let () =
  Printf.printf "=== M2 sanity check: hand-built schedules ===\n\n";

  (* Use mk_load with distinct refs to avoid constant folding by mk_add. *)
  let a = Algsimp.mk_load (Expr.Input (0, true)) in
  let b = Algsimp.mk_load (Expr.Input (1, true)) in
  let c = Algsimp.mk_load (Expr.Input (2, true)) in
  let ab = Algsimp.mk_add a b in
  let ac = Algsimp.mk_add a c in
  let result = Algsimp.mk_add ab ac in

  Printf.printf "Tags: a=%d b=%d c=%d ab=%d ac=%d result=%d\n" a.tag b.tag c.tag
    ab.tag ac.tag result.tag;

  let schedule = [ a; b; c; ab; ac; result ] in
  let info = Regalloc.peak_live_analysis ~isa:Isa.avx512 ~scheduled:schedule in
  Printf.printf "Test 1: small DAG with distinct loads\n";
  Printf.printf "  Expected peak_live=4\n";
  Printf.printf "  Got: %s\n\n" (Regalloc.format_live_info info);
  assert (info.peak_live = 4);

  (* Test 2: linear chain. Peak should be 2 (each step uses one prev result + one new load). *)
  let inp k = Algsimp.mk_load (Expr.Input (k, true)) in
  let l0 = inp 100 in
  let l1 = inp 101 in
  let l2 = inp 102 in
  let l3 = inp 103 in
  let l4 = inp 104 in
  let n1 = Algsimp.mk_add l0 l1 in
  let n2 = Algsimp.mk_add n1 l2 in
  let n3 = Algsimp.mk_add n2 l3 in
  let n4 = Algsimp.mk_add n3 l4 in
  (* Interleave loads and adds — peak should stay small. *)
  let schedule2 = [ l0; l1; n1; l2; n2; l3; n3; l4; n4 ] in
  Printf.printf "Tags2: l0=%d l1=%d n1=%d l2=%d n2=%d l3=%d n3=%d l4=%d n4=%d\n"
    l0.tag l1.tag n1.tag l2.tag n2.tag l3.tag n3.tag l4.tag n4.tag;
  let info2 =
    Regalloc.peak_live_analysis ~isa:Isa.avx512 ~scheduled:schedule2
  in
  Printf.printf "Test 2: linear chain, interleaved loads+adds\n";
  Printf.printf "  Expected peak_live <= 3\n";
  Printf.printf "  Got: %s\n\n" (Regalloc.format_live_info info2);
  assert (info2.peak_live <= 3);

  (* Test 3: empty schedule. *)
  let info3 = Regalloc.peak_live_analysis ~isa:Isa.avx512 ~scheduled:[] in
  Printf.printf "Test 3: empty schedule\n";
  Printf.printf "  Got: %s\n" (Regalloc.format_live_info info3);
  assert (info3.peak_live = 0);

  Printf.printf "\n=== M2 sanity PASS ===\n"
