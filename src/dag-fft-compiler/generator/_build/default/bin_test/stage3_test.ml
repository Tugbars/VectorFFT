(* Stage 3 sanity check: prepare_for_simple_codelet dedupes correctly,
 * produces I1-compliant output, and round-trips through Regalloc.allocate
 * without firing the contract assertions.
 *
 * We build a tiny synthetic IR (a few hashcons'd nodes), a fake
 * "raw_scheduled" list with intentional duplicates, and verify the prep
 * dedupes them. *)

open Vfft_v2

let () = Algsimp.reset ()

(* Tiny synthetic IR — use mk_const / mk_add. *)
let t_a = Algsimp.mk_const 1.0
let t_b = Algsimp.mk_const 2.0
let t_c = Algsimp.mk_add t_a t_b
let t_d = Algsimp.mk_add t_c t_a

(* Synthetic raw_scheduled with duplicates (as if from SU output:
 * intermediate (None, e) followed by store sink (Some _, e)). *)
let raw_dup = [ t_a; t_b; t_c; t_d; t_c; t_d ]

let assigns : (Expr.elem_ref * Algsimp.t) list = []

let () =
  Printf.printf "=== Stage 3 sanity check ===\n";

  (* Test 1: prepare_for_simple_codelet dedupes correctly. *)
  let input = Regalloc.prepare_for_simple_codelet
    ~raw_scheduled:raw_dup ~assigns () in
  let n_raw = List.length raw_dup in
  let n_dedup = List.length input.scheduled in
  Printf.printf "raw length=%d, dedup length=%d (expect 4)\n" n_raw n_dedup;
  assert (n_dedup = 4);

  (* Test 2: I1 — each tag appears at most once. *)
  let seen = Hashtbl.create 8 in
  List.iter (fun (e : Algsimp.t) ->
    assert (not (Hashtbl.mem seen e.tag));
    Hashtbl.add seen e.tag ()
  ) input.scheduled;
  Printf.printf "I1 dedup: PASS (all %d entries unique)\n" n_dedup;

  (* Test 3: first occurrence preserved. *)
  let first_dedup_tag = (List.hd input.scheduled).tag in
  let first_raw_tag = (List.hd raw_dup).tag in
  assert (first_dedup_tag = first_raw_tag);
  Printf.printf "First occurrence preserved: PASS\n";

  (* Test 4: prep round-trips through allocate without firing assertions. *)
  (match Regalloc.allocate ~isa:Isa.avx512 ~scheduled:input.scheduled
           ~inline_set:input.inline_set
           ~force_last_use:input.force_last_use () with
   | Regalloc.Allocated _ ->
     Printf.printf "Regalloc.allocate on prep output: PASS\n"
   | Regalloc.Overflow _ ->
     Printf.printf "Regalloc.allocate: overflow (unexpected)\n"; exit 1);

  (* Test 5: passing raw_dup directly (without prep) should fire I1. *)
  (try
     let _ = Regalloc.allocate ~isa:Isa.avx512 ~scheduled:raw_dup () in
     Printf.printf "ERROR: I1 assertion did not fire on raw duplicates\n";
     exit 1
   with Failure msg ->
     if (try let _ = Str.search_forward (Str.regexp "I1") msg 0 in true
         with Not_found -> false)
     then Printf.printf "I1 fires on raw duplicates: PASS\n"
     else begin
       Printf.printf "ERROR: got Failure but not I1: %s\n" msg;
       exit 1
     end);

  (* Test 6: oref variant. *)
  let raw_oref = [
    (None, t_a); (None, t_b); (None, t_c); (None, t_d);
    (Some (Expr.Output (0, true)), t_c);
    (Some (Expr.Output (1, true)), t_d);
  ] in
  let input2 = Regalloc.prepare_for_simple_codelet_from_oref
    ~raw_scheduled:raw_oref ~assigns () in
  assert (List.length input2.scheduled = 4);
  Printf.printf "oref variant dedupes: PASS\n";

  Printf.printf "=== All Stage 3 tests passed ===\n"
