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
  let twidsq = ref false in
  let r2c = ref false in
  let r2c_first = ref false in
  let rdft = ref false in
  let hc2hc = ref false in
  let hc2c = ref false in
  let dct2 = ref false in
  let dct2_trigII = ref false in
  let dct3 = ref false in
  let dht = ref false in
  let dst2 = ref false in
  let dst3 = ref false in
  let dct4 = ref false in
  let c2r = ref false in
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
     else if arg = "--twidsq"    then twidsq := true
     else if arg = "--r2c"       then r2c := true
     else if arg = "--r2c-first" then r2c_first := true
     else if arg = "--rdft"      then rdft := true
     else if arg = "--hc2hc"     then hc2hc := true
     else if arg = "--hc2c"      then hc2c := true
     else if arg = "--dct2"      then dct2 := true
     else if arg = "--dct2-trigII" then dct2_trigII := true
     else if arg = "--dct3"      then dct3 := true
     else if arg = "--dht"       then dht := true
     else if arg = "--dst2"      then dst2 := true
     else if arg = "--dst3"      then dst3 := true
     else if arg = "--dct4"      then dct4 := true
     else if arg = "--c2r"       then c2r := true
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
   * Spill is meaningful only for twiddled CT-decomposed codelets.
   *
   * --twidsq selects the n×n twidsq DAG (FFTW-style OOP codelet,
   * doc 43): apply inter-stage twiddle, run n parallel DFT-n's,
   * store transposed. Bypasses the regular twiddled / spill paths
   * (those are for in-place codelets). *)
  let raw, spill_markers, spill_ct =
    if !r2c then
      (Vfft_v2.Dft_r2c.dft_expand_r2c ~sign n, [], None)
    else if !r2c_first then
      (Vfft_v2.Dft_r2c.dft_expand_r2c_first ~sign n, [], None)
    else if !rdft then
      (Vfft_v2.Dft_r2c.dft_expand_rdft ~sign n, [], None)
    else if !hc2hc then
      let direction = if !dif then `Dif else `Dit in
      (Vfft_v2.Dft_r2c.dft_expand_hc2hc ~sign ~direction n, [], None)
    else if !hc2c then
      let direction = if !dif then `Dif else `Dit in
      (Vfft_v2.Dft_r2c.dft_expand_hc2c ~sign ~direction n, [], None)
    else if !dct2 then
      (Vfft_v2.Dft_r2c.dft_expand_dct2 n, [], None)
    else if !dct2_trigII then
      (Vfft_v2.Dft_r2c.dft_expand_dct2_trigII n, [], None)
    else if !dct3 then
      (Vfft_v2.Dft_r2c.dft_expand_dct3 n, [], None)
    else if !dht then
      (Vfft_v2.Dft_r2c.dft_expand_dht n, [], None)
    else if !dst2 then
      (Vfft_v2.Dft_r2c.dft_expand_dst2 n, [], None)
    else if !dst3 then
      (Vfft_v2.Dft_r2c.dft_expand_dst3 n, [], None)
    else if !dct4 then
      (Vfft_v2.Dft_r2c.dft_expand_dct4 n, [], None)
    else if !c2r then
      (Vfft_v2.Dft_r2c.dft_expand_c2r n, [], None)
    else if !twidsq then
      (Vfft_v2.Dft.dft_expand_twidsq ~direction ~sign n, [], None)
    else if !spill && !twiddled then
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
  let deduped_pre = Vfft_v2.Algsimp.dedup_sub_pairs simplified in

  (* Aggressive prime-only passes: distributive factoring of same-const
   * muls, subsum sharing for X[0] outputs, and Frigo network transposition.
   * For monolithic prime DFTs (R=3/5/7/11) these expose Winograd-style
   * structure invisible to the pair-fold simplifier. CT-decomposed
   * codelets are already in FMA-friendly form — these passes default to
   * no-op for them.
   *
   * For twiddled forms (t1_dit / t1_dif), the inner DFT is wrapped by
   * Cmul nodes; the factor/share passes still help the inner DFT
   * structure. Transposition skips Cmul nodes (they're nonlinear). *)
  let aggressive = match Vfft_v2.Dft.pick_algorithm n with
    | Vfft_v2.Dft.Direct -> true
    | Vfft_v2.Dft.Cooley_Tukey _ -> false
    | Vfft_v2.Dft.Split_radix -> false in
  (* For direct primes (n odd prime ≥ 3), the conjugate-pair construction
   * in dft_direct_conjugate_pair already produces optimal hash-cons-shared
   * intermediates (pair sums/diffs, p_re/p_im/q_re/q_im chains). The
   * share_subsums pass — which factors common addition subsums across
   * outputs — actively HURTS this layout: it materializes partial sums
   * that prevent fma_lift from recognizing the unified mixed-sign FMA
   * chain (e.g. it splits `x[0] + c1·s1 + c2·s2 - c3·s3 - c4·s4 - c5·s5`
   * into separate positive and negative sub-chains, costing 4 extra ops
   * per pair output block).
   *
   * Empirically: for R=11, full pipeline gives 172 ops; skipping share
   * gives 150 (matching hand). For R=13, 256 → 204 ops. For composite
   * Cooley-Tukey, share_subsums DOES help (pow2 butterflies have many
   * cross-output partial-sum overlaps), so it's only disabled here for
   * direct-prime mode. *)
  let is_direct = aggressive in  (* aggressive ↔ Direct algorithm in current setup *)
  let factored = Vfft_v2.Algsimp.factor_common_muls ~aggressive deduped_pre in
  let factored = Vfft_v2.Algsimp.factor_by_atom ~aggressive factored in
  let factored = Vfft_v2.Algsimp.dedup_sub_pairs factored in
  (* Sub(Neg(Mul(a,b)), c) → NK_Fma(a, b, c, true, true) (= fnmsub).
   *
   * Now handled as a peephole inside `mk_sub_binary` so the rewrite fires
   * at construction time (during dedup_sub_pairs' rebuild), not as a
   * standalone post-pass. Standalone post-pass approach orphaned spill
   * marker tags at R=32/R=64; peephole keeps the DAG self-consistent
   * because the rewrite happens before markers get captured.
   * See docs/30_sub_neg_mul_fnmsub.md for the diagnostic that motivated
   * this and docs/31_peephole_vs_post_pass.md for the bug analysis. *)
  let shared =
    if is_direct then factored
    else Vfft_v2.Algsimp.share_subsums ~aggressive factored
  in
  (* Transposition is correct only when the DAG is purely linear (no Cmul
   * nodes — those wrap symbolic twiddle loads, making the network
   * nonlinear in our representation). For twiddled prime forms (t1_dit,
   * t1_dif), we skip transposition; factor+share still apply to the
   * inner DFT structure. *)
  let has_cmul =
    let st = Vfft_v2.Algsimp.stats_reachable (List.map snd shared) in
    st.Vfft_v2.Algsimp.cmuls > 0
  in
  let count_ops a =
    let st = Vfft_v2.Algsimp.stats_reachable (List.map snd a) in
    st.Vfft_v2.Algsimp.adds + st.subs + st.muls + st.negs
    + (2 * st.cmuls) + st.fmas
  in
  let post_trans =
    if aggressive && not has_cmul && not is_direct then begin
      (* FP transpose loop. For direct primes, this is also disabled since
       * share_subsums (which the loop relies on between transpositions)
       * hurts here, and transpose without share doesn't help either —
       * empirically the construction is already optimal post-fma_lift. *)
      let rec fp prev_assigns prev_count iter =
        if iter >= 6 then prev_assigns
        else begin
          let t1 = Vfft_v2.Algsimp.transpose prev_assigns in
          let t1f = Vfft_v2.Algsimp.factor_common_muls ~aggressive t1 in
          let t1f = Vfft_v2.Algsimp.factor_by_atom ~aggressive t1f in
          let t1f = Vfft_v2.Algsimp.dedup_sub_pairs t1f in
          let t1s_pass = Vfft_v2.Algsimp.share_subsums ~aggressive t1f in
          let t2 = Vfft_v2.Algsimp.transpose t1s_pass in
          let t2f = Vfft_v2.Algsimp.factor_common_muls ~aggressive t2 in
          let t2f = Vfft_v2.Algsimp.factor_by_atom ~aggressive t2f in
          let t2f = Vfft_v2.Algsimp.dedup_sub_pairs t2f in
          let t2s = Vfft_v2.Algsimp.share_subsums ~aggressive t2f in
          let new_count = count_ops t2s in
          if new_count >= prev_count then prev_assigns
          else fp t2s new_count (iter + 1)
        end
      in
      fp shared (count_ops shared) 0
    end else shared
  in
  (* FMA lift was historically applied unconditionally with the comment
   * "GCC -O3 -mfma auto-fuses un-lifted Mul+Add patterns reliably, so
   * fma_lift is essentially a no-op for codegen perf." That claim turns
   * out to be false for composite (Cooley-Tukey) DAGs: explicit NK_Fma
   * nodes constrain GCC's register allocation more than auto-fused
   * mul+add chains, producing significantly more vmovapd reg-reg moves
   * (R=32 SU+Spill: loop-body 251 vmovapd with fma_lift vs ~100 without
   * — and total FP instructions 910 vs 717, where hand has 709).
   *
   * Empirical llvm-mca SKX cycles (R=32 t1_dit loop body):
   *   With fma_lift:    312 cycles
   *   Without fma_lift: 226 cycles  (beating hand 338 by 33%)
   *
   * For primes (Direct construction with conjugate pairs), fma_lift IS a
   * marginal win (~1-2% on R=13/R=17 cycles) — the prime DAG shape
   * exposes specific Add(Mul, c) patterns that benefit from explicit FMA
   * encoding. So gate fma_lift to aggressive (= Direct primes) only.
   *
   * See docs/28_composite_regression_fma_lift.md for full investigation. *)
  let deduped =
    if aggressive then Vfft_v2.Algsimp.fma_lift post_trans
    else post_trans
  in

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
      if !r2c then
        (* R2C forward codelet: radix{N}_r2c_{sgn}_{isa}_gen *)
        Printf.sprintf "radix%d_r2c_%s_%s_gen%s%s%s"
          n sgn_suffix isa.name suffix sched_suffix spill_suffix
      else if !r2c_first then
        (* R2C first-stage cascade codelet: radix{R}_r2c_first_{sgn}_{isa}_gen
         * The {R} here is the SUB-DFT radix, not the total transform size. *)
        Printf.sprintf "radix%d_r2c_first_%s_%s_gen%s%s%s"
          n sgn_suffix isa.name suffix sched_suffix spill_suffix
      else if !rdft then
        (* FFTW-style real-input DFT: radix{N}_rdft_{sgn}_{isa}_gen *)
        Printf.sprintf "radix%d_rdft_%s_%s_gen%s%s%s"
          n sgn_suffix isa.name suffix sched_suffix spill_suffix
      else if !hc2hc then
        (* Middle-stage Hermitian-packed cascade codelet:
         * radix{R}_hc2hc_{dir}_{sgn}_{isa}_gen *)
        Printf.sprintf "radix%d_hc2hc_%s_%s_%s_gen%s%s%s"
          n dir_suffix sgn_suffix isa.name suffix sched_suffix spill_suffix
      else if !hc2c then
        (* Last-stage cascade codelet: Hermitian-packed in, natural complex out:
         * radix{R}_hc2c_{dir}_{sgn}_{isa}_gen *)
        Printf.sprintf "radix%d_hc2c_%s_%s_%s_gen%s%s%s"
          n dir_suffix sgn_suffix isa.name suffix sched_suffix spill_suffix
      else if !dct2 then
        (* DCT-II via Makhoul's reduction: radix{N}_dct2_{isa}_gen *)
        Printf.sprintf "radix%d_dct2_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !dct2_trigII then
        (* DCT-II via FFTW trigII embedding: radix{N}_dct2_trigII_{isa}_gen *)
        Printf.sprintf "radix%d_dct2_trigII_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !dct3 then
        (* DCT-III via inverse-Makhoul: radix{N}_dct3_{isa}_gen *)
        Printf.sprintf "radix%d_dct3_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !dht then
        (* DHT (Discrete Hartley Transform): radix{N}_dht_{isa}_gen *)
        Printf.sprintf "radix%d_dht_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !dst2 then
        (* DST-II via DCT-II wrapper: radix{N}_dst2_{isa}_gen *)
        Printf.sprintf "radix%d_dst2_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !dst3 then
        (* DST-III via DCT-III wrapper: radix{N}_dst3_{isa}_gen *)
        Printf.sprintf "radix%d_dst3_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !dct4 then
        (* DCT-IV via Lee 1984: radix{N}_dct4_{isa}_gen *)
        Printf.sprintf "radix%d_dct4_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !c2r then
        (* C2R backward codelet: radix{N}_c2r_{isa}_gen
         * c2r is always backward, so no separate sgn_suffix is needed. *)
        Printf.sprintf "radix%d_c2r_%s_gen%s%s%s"
          n isa.name suffix sched_suffix spill_suffix
      else if !twidsq then
        (* Twidsq codelets use their own name pattern reflecting the
         * inter-stage role: radix{N}_twidsq_{dir}_{sgn}_{isa}_gen. *)
        Printf.sprintf "radix%d_twidsq_%s_%s_%s_gen%s%s%s"
          n dir_suffix sgn_suffix isa.name suffix sched_suffix spill_suffix
      else if !twiddled then
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
                    ~in_place:!in_place ~t1s:!t1s ~twidsq:!twidsq
                    ~twidsq_n:(if !twidsq then n else 0)
                    ~scheduler ~isa ~gh:!gh
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
