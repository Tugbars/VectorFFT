(* bin/sr_structural_diff.ml — Compare post-algsimp DAGs between CT and SR.
 *
 * Produces a structural fingerprint of each output expression (the final
 * algsimp tag's subtree), serialized as a canonical S-expression that does
 * not depend on hashcons tag numbers. Two DAGs are structurally identical
 * iff their per-output fingerprints match.
 *
 * Diff strategy:
 *   1. Run the gen_radix algsimp pipeline (without spill/recipe) on both
 *      VFFT_SPLIT_RADIX=0 (CT) and VFFT_SPLIT_RADIX=1 (SR) within ONE process.
 *      Since the picker is gated on the env var at call time, we set it
 *      programmatically via Unix.putenv between runs.
 *   2. For each run, compute a structural fingerprint per output assignment.
 *   3. Diff the two fingerprint maps.
 *
 * Caveat: we deliberately skip spill machinery and emit_c — purely
 * comparing the post-algsimp DAG, which is what feeds into scheduling. *)

open Vfft_v2

(* Canonical S-expression for an algsimp node. Recursive — uses a memo to
 * avoid re-walking shared subtrees. The serialization is purely structural:
 * Const value, Load elem_ref, and constructor names + recursive children.
 * Tag values are NOT included, so two different hashcons-instance DAGs with
 * the same structure produce identical fingerprints. *)
let fingerprint (root : Algsimp.t) : string =
  let memo : (int, string) Hashtbl.t = Hashtbl.create 1024 in
  let rec go (e : Algsimp.t) =
    match Hashtbl.find_opt memo e.tag with
    | Some s -> s
    | None ->
      let s = match e.node with
        | Algsimp.NK_Const c ->
          (* Use %.17g to make floats stable across runs. *)
          Printf.sprintf "C%.17g" c
        | Algsimp.NK_Load r ->
          (match r with
           | Expr.Output (k, re) -> Printf.sprintf "Lo%d%s" k (if re then "r" else "i")
           | Expr.Input  (k, re) -> Printf.sprintf "Li%d%s" k (if re then "r" else "i")
           | Expr.Twiddle (k, re) -> Printf.sprintf "Lt%d%s" k (if re then "r" else "i"))
        | Algsimp.NK_Neg a ->
          Printf.sprintf "(N %s)" (go a)
        | Algsimp.NK_Add (a, b) ->
          (* Add and Mul are commutative — canonicalize by sorted children. *)
          let sa = go a and sb = go b in
          let l, r = if sa <= sb then sa, sb else sb, sa in
          Printf.sprintf "(A %s %s)" l r
        | Algsimp.NK_Sub (a, b) ->
          Printf.sprintf "(S %s %s)" (go a) (go b)
        | Algsimp.NK_Mul (a, b) ->
          let sa = go a and sb = go b in
          let l, r = if sa <= sb then sa, sb else sb, sa in
          Printf.sprintf "(M %s %s)" l r
        | Algsimp.NK_CmulRe (a, b, c, d) ->
          Printf.sprintf "(CR %s %s %s %s)" (go a) (go b) (go c) (go d)
        | Algsimp.NK_CmulIm (a, b, c, d) ->
          Printf.sprintf "(CI %s %s %s %s)" (go a) (go b) (go c) (go d)
        | Algsimp.NK_Fma (a, b, c, nm, na) ->
          Printf.sprintf "(F%s%s %s %s %s)"
            (if nm then "N" else "")
            (if na then "S" else "")
            (go a) (go b) (go c)
      in
      Hashtbl.add memo e.tag s;
      s
  in
  go root

(* Run the algsimp pipeline mirroring bin/gen_radix.ml. *)
let pipeline_assigns (n : int) : (Expr.elem_ref * Algsimp.t) list =
  let raw = Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIT ~sign:`Fwd n in
  Algsimp.reset ();
  let reassoc = Dft.needs_reassoc n in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped_pre = Algsimp.dedup_sub_pairs simplified in
  let aggressive = match Dft.pick_algorithm n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> false
    | Dft.Split_radix -> false in
  let is_direct = aggressive in
  let factored = Algsimp.factor_common_muls ~aggressive deduped_pre in
  let factored = Algsimp.factor_by_atom ~aggressive factored in
  let factored = Algsimp.dedup_sub_pairs factored in
  let shared =
    if is_direct then factored
    else Algsimp.share_subsums ~aggressive factored
  in
  let has_cmul =
    let st = Algsimp.stats_reachable (List.map snd shared) in
    st.Algsimp.cmuls > 0
  in
  let count_ops a =
    let st = Algsimp.stats_reachable (List.map snd a) in
    st.Algsimp.adds + st.subs + st.muls + st.negs + (2 * st.cmuls) + st.fmas
  in
  let post_trans =
    if aggressive && not has_cmul && not is_direct then begin
      let rec fp prev_assigns prev_count iter =
        if iter >= 6 then prev_assigns
        else
          let t1 = Algsimp.transpose prev_assigns in
          let t1f = Algsimp.factor_common_muls ~aggressive t1 in
          let t1f = Algsimp.factor_by_atom ~aggressive t1f in
          let t1f = Algsimp.dedup_sub_pairs t1f in
          let t1s = Algsimp.share_subsums ~aggressive t1f in
          let t2 = Algsimp.transpose t1s in
          let t2f = Algsimp.factor_common_muls ~aggressive t2 in
          let t2f = Algsimp.factor_by_atom ~aggressive t2f in
          let t2f = Algsimp.dedup_sub_pairs t2f in
          let t2s = Algsimp.share_subsums ~aggressive t2f in
          let new_count = count_ops t2s in
          if new_count >= prev_count then prev_assigns
          else fp t2s new_count (iter + 1)
      in
      fp shared (count_ops shared) 0
    end else shared
  in
  if aggressive then Algsimp.fma_lift post_trans else post_trans

(* Key an assignment by its output elem_ref for deterministic mapping. *)
let key_of_ref (r : Expr.elem_ref) : string =
  match r with
  | Expr.Output (k, re) -> Printf.sprintf "out[%d].%s" k (if re then "re" else "im")
  | Expr.Input (k, re) -> Printf.sprintf "in[%d].%s" k (if re then "re" else "im")
  | Expr.Twiddle (k, re) -> Printf.sprintf "tw[%d].%s" k (if re then "re" else "im")

let build_fingerprint_map (n : int) : (string * string) list =
  let assigns = pipeline_assigns n in
  List.map (fun (r, t) -> (key_of_ref r, fingerprint t)) assigns
  |> List.sort (fun (a, _) (b, _) -> compare a b)

let () =
  let n = int_of_string Sys.argv.(1) in
  (* The picker uses split_radix_enabled() which reads VFFT_SPLIT_RADIX via
   * Sys.getenv_opt. Sys.putenv changes it for subsequent calls. *)
  Unix.putenv "VFFT_SPLIT_RADIX" "";
  let ct_map = build_fingerprint_map n in
  Unix.putenv "VFFT_SPLIT_RADIX" "1";
  let sr_map = build_fingerprint_map n in

  (* Pairwise diff per output key. *)
  let ct_tbl = Hashtbl.create 64 in
  List.iter (fun (k, v) -> Hashtbl.add ct_tbl k v) ct_map;
  let sr_tbl = Hashtbl.create 64 in
  List.iter (fun (k, v) -> Hashtbl.add sr_tbl k v) sr_map;

  let total = ref 0 and equal = ref 0 and diff = ref 0 in
  let diff_examples = ref [] in
  List.iter (fun (k, ct_fp) ->
    incr total;
    match Hashtbl.find_opt sr_tbl k with
    | None ->
      incr diff;
      if List.length !diff_examples < 3 then
        diff_examples := (k, ct_fp, "<MISSING>") :: !diff_examples
    | Some sr_fp ->
      if ct_fp = sr_fp then incr equal
      else begin
        incr diff;
        if List.length !diff_examples < 3 then
          diff_examples := (k, ct_fp, sr_fp) :: !diff_examples
      end
  ) ct_map;

  Printf.printf "N=%d: %d outputs, %d structurally identical, %d differ\n"
    n !total !equal !diff;
  (* Show which output bins match and which differ. *)
  let bins_match_re = ref [] and bins_diff_re = ref [] in
  let bins_match_im = ref [] and bins_diff_im = ref [] in
  List.iter (fun (k, ct_fp) ->
    match Hashtbl.find_opt sr_tbl k with
    | None -> ()
    | Some sr_fp ->
      (* Parse "out[N].re" or "out[N].im" *)
      try
        let s = Scanf.sscanf k "out[%d].%s" (fun n s -> (n, s)) in
        let bin, comp = s in
        let target_match, target_diff =
          if comp = "re" then bins_match_re, bins_diff_re
          else bins_match_im, bins_diff_im
        in
        if ct_fp = sr_fp then target_match := bin :: !target_match
        else target_diff := bin :: !target_diff
      with _ -> ()
  ) ct_map;
  let sorted l = List.sort compare l in
  Printf.printf "  re identical bins: %s\n" (String.concat "," (List.map string_of_int (sorted !bins_match_re)));
  Printf.printf "  re differ    bins: %s\n" (String.concat "," (List.map string_of_int (sorted !bins_diff_re)));
  Printf.printf "  im identical bins: %s\n" (String.concat "," (List.map string_of_int (sorted !bins_match_im)));
  Printf.printf "  im differ    bins: %s\n" (String.concat "," (List.map string_of_int (sorted !bins_diff_im)));
  if !diff > 0 then begin
    Printf.printf "\nFingerprint length comparison for first 5 differing outputs:\n";
    let shown = ref 0 in
    List.iter (fun (k, ct_fp) ->
      if !shown < 5 then
        match Hashtbl.find_opt sr_tbl k with
        | Some sr_fp when sr_fp <> ct_fp ->
          incr shown;
          Printf.printf "  %s: CT_len=%d SR_len=%d (%+.1f%%)\n"
            k (String.length ct_fp) (String.length sr_fp)
            (100.0 *. (float_of_int (String.length sr_fp - String.length ct_fp))
             /. float_of_int (String.length ct_fp))
        | _ -> ()
    ) ct_map
  end
