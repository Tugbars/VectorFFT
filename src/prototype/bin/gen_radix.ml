(* gen_radix.ml — CLI driver.
 *
 * Usage:
 *   gen_radix N                          show DAG and stats (no twiddle)
 *   gen_radix N --twiddled               t1_dit (TP_Flat: load all twiddles)
 *   gen_radix N --twiddled --log3        t1_dit_log3 (load base twiddles, derive rest)
 *   gen_radix N --twiddled --emit-c      emit C code with cost-model defaults
 *   gen_radix N --twiddled --emit-c --bisect   emit with Frigo's bisection scheduler
 *
 * Cost-model defaults (auto-applied unless --no-recipe is passed):
 *   Spill + SU (the "full recipe") auto-enables when the cost model says yes.
 *   Currently that's: CT-decomposed AND (n + 6 > vec_regs OR vec_regs >= 32).
 *
 * Override flags:
 *   --no-recipe        force Topo (disable auto spill+SU)
 *   --spill / --su     explicit on (overrides --no-recipe locally per flag)
 *)

let () =
  let n = ref 4 in
  let twiddled = ref false in
  let log3 = ref false in
  let emit_c = ref false in
  let in_place = ref false in
  let bisect = ref false in
  let annotate = ref false in
  let su = ref false in
  let spill = ref false in
  let fuse = ref 0 in
  let no_recipe = ref false in
  let t1s = ref false in
  let dif = ref false in
  let bwd = ref false in
  let gh = ref false in
  let bb = ref false in
  let bb_budget = ref 1.0 in
  let isa_name = ref "avx512" in
  let uarch_name = ref "sapphire_rapids" in
  let args = Array.to_list Sys.argv in
  let i = ref 0 in
  let arr = Array.of_list args in
  while !i < Array.length arr do
    let arg = arr.(!i) in
    (if !i = 0 then ()
     else if arg = "--twiddled" then twiddled := true
     else if arg = "--log3"     then log3 := true
     else if arg = "--emit-c"   then emit_c := true
     else if arg = "--in-place" then in_place := true
     else if arg = "--bisect"   then bisect := true
     else if arg = "--annotate" then annotate := true
     else if arg = "--su"       then su := true
     else if arg = "--spill"    then spill := true
     else if arg = "--no-recipe" then no_recipe := true
     else if arg = "--t1s"       then t1s := true
     else if arg = "--dif"       then dif := true
     else if arg = "--bwd"       then bwd := true
     else if arg = "--gh"        then gh := true
     else if arg = "--bb"        then bb := true
     else if arg = "--bb-budget" && !i + 1 < Array.length arr then begin
       bb_budget := float_of_string arr.(!i + 1);
       incr i
     end
     else if arg = "--fuse" && !i + 1 < Array.length arr then begin
       fuse := int_of_string arr.(!i + 1);
       incr i
     end
     else if arg = "--isa" && !i + 1 < Array.length arr then begin
       isa_name := arr.(!i + 1);
       incr i
     end
     else if arg = "--uarch" && !i + 1 < Array.length arr then begin
       uarch_name := arr.(!i + 1);
       incr i
     end
     else (try n := int_of_string arg with _ -> ()));
    incr i
  done;
  let isa = Vfft_v2.Isa.of_name !isa_name in
  let uarch = Vfft_v2.Uarch.of_name !uarch_name in

  let n = !n in
  let policy : Vfft_v2.Dft.twiddle_policy =
    if !log3 then TP_Log3 else TP_Flat
  in
  let direction : Vfft_v2.Dft.direction =
    if !dif then DIF else DIT
  in
  let sign = if !bwd then `Bwd else `Fwd in

  (* Cost-model auto-defaults.
   *
   * If --no-recipe is set: don't auto-enable anything.
   * Otherwise: when the rule says yes AND user didn't override with bisect
   * or annotate, turn on --spill --su automatically.
   *
   * Explicit --spill / --su flags always take effect regardless of the rule. *)
  let recipe_applicable =
    !twiddled
    && not !bisect
    && not !annotate
    && not !no_recipe
    && Vfft_v2.Dft.should_spill n isa.vec_regs
  in
  if recipe_applicable then begin
    if not !spill then spill := true;
    if not !su then su := true
  end;

  (* Auto-enable Goodman-Hsu mode switch when:
   *   - the recipe is engaged (--su is on), AND
   *   - vec_regs <= 16 (AVX2 or narrower), AND
   *   - n >= 32 (peak live likely to exceed threshold inside clusters)
   *
   * Empirically this delivers 4-8% over the base recipe on AVX2 R={32,64}
   * and is a no-op on AVX-512 (threshold=24 not reached with cluster-sequential).
   * The flag toggle is free at runtime — pressure mode only fires when needed. *)
  if !su && isa.vec_regs <= 16 && n >= 32 && not !no_recipe then begin
    if not !gh then gh := true
  end;

  (* Drive math layer with or without spill marker capture.
   * Spill is meaningful only for twiddled CT-decomposed codelets. *)
  let raw, spill_markers, spill_ct =
    if !spill && !twiddled then
      let assignments, markers, ct =
        Vfft_v2.Dft.dft_expand_twiddled_spill ~policy ~direction ~sign n in
      (assignments, markers, ct)
    else if !twiddled then
      (Vfft_v2.Dft.dft_expand_twiddled ~policy ~direction ~sign n, [], None)
    else
      (Vfft_v2.Dft.dft_expand n, [], None)
  in

  Vfft_v2.Algsimp.reset ();
  let reassoc = Vfft_v2.Dft.needs_reassoc n in
  let simplified = Vfft_v2.Algsimp.of_assignments ~reassoc raw in
  let deduped = Vfft_v2.Algsimp.dedup_sub_pairs simplified in

  (* Lift spill markers to algsimp tags, then build spill_info.
   * Must happen AFTER of_assignments so hash-consing has run on the
   * marker subtrees. *)
  let spill_info : Vfft_v2.Emit_c.spill_info option =
    if !spill && spill_markers <> [] then
      let tag_markers = Vfft_v2.Algsimp.lift_spill_markers ~reassoc spill_markers in
      Some (Vfft_v2.Emit_c.make_spill_info ?ct:spill_ct ~fuse:!fuse tag_markers)
    else
      None
  in

  if !emit_c then begin
    let suffix = if !in_place then "_inplace" else "" in
    let variant = if !log3 then "_log3" else "" in
    let t1s_infix = if !t1s then "s" else "" in
    let dir_suffix = if !dif then "dif" else "dit" in
    let sgn_suffix = if !bwd then "bwd" else "fwd" in
    let sched_suffix = match !bisect, !su, !annotate with
      | false, false, false -> ""
      | true,  false, false -> "_bisect"
      | false, true,  false -> "_su"
      | false, false, true  -> "_anno"
      | true,  false, true  -> "_bisect_anno"
      | false, true,  true  -> "_su_anno"
      | _ -> "_combo"
    in
    let spill_suffix =
      if !spill && spill_info <> None then
        if !fuse > 0 then Printf.sprintf "_spill_fuse%d" !fuse
        else "_spill"
      else ""
    in
    let name =
      if !twiddled then
        Printf.sprintf "radix%d_t1%s_%s%s_%s_%s_gen%s%s%s"
          n t1s_infix dir_suffix variant sgn_suffix isa.name suffix sched_suffix spill_suffix
      else
        Printf.sprintf "radix%d_n1_%s_%s_gen%s%s%s"
          n sgn_suffix isa.name suffix sched_suffix spill_suffix
    in
    let scheduler : Vfft_v2.Emit_c.scheduler = match !bisect, !su, !annotate with
      | false, false, false -> Topological
      | true,  false, false -> Bisection
      | false, true,  false -> SU uarch
      | false, false, true  -> Annotated_topological
      | true,  false, true  -> Annotated_bisection
      | false, true,  true  -> Annotated_SU uarch
      | _ -> Topological
    in
    let bb_budget_arg = if !bb then Some !bb_budget else None in
    print_string (Vfft_v2.Emit_c.emit_codelet
                    ~in_place:!in_place ~t1s:!t1s ~scheduler ~isa ~gh:!gh
                    ~bb_budget:bb_budget_arg ~spill:spill_info
                    deduped ~name)
  end else begin
    let variant = if !log3 then ", log3" else "" in
    let label =
      if !twiddled then Printf.sprintf "twiddled (t1_dit%s)" variant
      else "no-twiddle (n1)"
    in
    Printf.printf "================================================================\n";
    Printf.printf "  DFT-%d, %s — DAG\n" n label;
    Printf.printf "================================================================\n\n";
    print_string (Vfft_v2.Algsimp.print_dag deduped);
    Printf.printf "\n================================================================\n";
    Printf.printf "  Stats\n";
    Printf.printf "================================================================\n\n";
    let roots = List.map snd deduped in
    let stats = Vfft_v2.Algsimp.stats_reachable roots in
    print_string (Vfft_v2.Algsimp.string_of_stats stats)
  end
