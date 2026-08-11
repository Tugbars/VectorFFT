(* cx_pipeline.ml — the optimizer pipeline of the full-IL (cil) family.
 *
 * The cx analog of pipeline.ml: ONE entry point that every cil emission
 * flows through, between DAG construction (Cx_math) and scheduling
 * (Cx_sched.Sched.su_schedule). This is what makes the cil family
 * PIPELINE-HOSTED in structure, not just in name: passes have a home, env
 * kill switches follow the house convention, and future passes land here
 * instead of inside an emitter.
 *
 * WHY THE CASCADE IS SHORT (audited 2026-08-09, docs/roadmap/
 * cil_optimizer_gap.md + the 183-case emission audit): cx_kind has no bare
 * Mul and no standalone Neg source node — constants ride pre-fused in
 * CFmaC/CTwC/CTwV — so the real-valued cascade's Mul-shaped passes
 * (fma_lift, factor_const_muls, multi_use_fma_lift, fma_addend_factor,
 * flatten_fma_mul_addend, collect_m) have NO PATTERN THAT CAN EXIST here.
 * They are not "not yet ported"; they are structurally inapplicable. The
 * passes that ARE meaningful over this algebra start with dedup_sub_pairs.
 * Hash-consing (CSE) is the constructor layer itself (cx_ir.mk).
 *
 * MODULE CARD
 * ROLE: pass cascade over the hash-consed cx DAG; entry prepare_codelet.
 * PIPELINE: dedup_sub_pairs_cx (VFFT_CX_NO_SUBDEDUP=1 skips) [-> future
 * passes]. VFFT_CX_STATS=1 prints a per-codelet DAG census to stderr.
 * DEPS: Cx_ir only (rebuilds through the smart constructors, so the
 * hash-cons table stays canonical).
 * GOTCHA 1: passes run BETWEEN build and su_schedule, on the SAME
 * hash-cons generation — never call Cx_ir.reset inside a pass.
 * GOTCHA 2: rewrites must go through the smart constructors (cadd/csub/
 * cneg/...) — hand-built records would bypass hash-consing and duplicate
 * tags (the zN C names). *)

open Cx_ir

(* ── DAG walk helpers ─────────────────────────────────────────────────── *)

let children (e : t) : t list =
  match e.node with
  | CIn _ | CLoad _ -> []
  | CNeg a | CRotNI a | CRotPI a | CLo a | CHi a -> [ a ]
  | CStore (_, v) -> [ v ]
  | CAdd (a, b) | CSub (a, b) | CRotAdd (a, b) -> [ a; b ]
  | CTurn (a, b, _) -> [ a; b ]
  | CFmaC (_, x, e) | CFnmaC (_, x, e) -> [ x; e ]
  | CTwC (_, _, x) | CTwV (_, x) | CTwL (_, x) -> [ x ]
;;

(* Reachable nodes from the assign roots, each visited once. *)
let iter_reachable (roots : t list) (f : t -> unit) : unit =
  let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let rec go e =
    if not (Hashtbl.mem seen e.tag)
    then (
      Hashtbl.replace seen e.tag ();
      List.iter go (children e);
      f e)
  in
  List.iter go roots
;;

(* ── PASS: dedup_sub_pairs (cx) ───────────────────────────────────────────
 * Mirrors the real side's dedup_sub_pairs (pipeline.ml step 2): when BOTH
 * CSub(a,b) and CSub(b,a) are reachable, rewrite the higher-tagged one as
 * CNeg of the lower-tagged one, so the subtraction is computed once and the
 * mirror costs an xor. Exact in FP (negation flips the sign bit; no
 * rounding), so BITID gates still apply to anything this pass touches.
 * As of 2026-08-09 the emitted set contains ZERO mirror pairs (14,266 sub
 * sites audited) — the pass exists as a leg, and as the guard that keeps
 * that fact true by construction rather than by luck. *)
let dedup_sub_pairs_cx (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref * t) list * int
  =
  let roots = List.map snd assigns in
  (* map (a_tag, b_tag) -> the CSub node computing a - b *)
  let subs : (int * int, t) Hashtbl.t = Hashtbl.create 64 in
  iter_reachable roots (fun e ->
    match e.node with
    | CSub (a, b) -> Hashtbl.replace subs (a.tag, b.tag) e
    | _ -> ());
  (* victims: the HIGHER-tagged member of each mirror pair *)
  let victim : (int, t) Hashtbl.t = Hashtbl.create 8 in
  Hashtbl.iter
    (fun (at, bt) e ->
       match Hashtbl.find_opt subs (bt, at) with
       | Some m when e.tag > m.tag -> Hashtbl.replace victim e.tag m
       | _ -> ())
    subs;
  if Hashtbl.length victim = 0
  then assigns, 0 (* the common case: nothing to do, assigns unchanged *)
  else (
    (* rebuild bottom-up through the smart constructors; memo on old tag *)
    let memo : (int, t) Hashtbl.t = Hashtbl.create 256 in
    let rec rw (e : t) : t =
      match Hashtbl.find_opt memo e.tag with
      | Some r -> r
      | None ->
        let r =
          match Hashtbl.find_opt victim e.tag with
          | Some canon -> cneg (rw canon)
          | None ->
            (match e.node with
             | CIn _ | CLoad _ -> e
             | CNeg a -> cneg (rw a)
             | CRotNI a -> crot (rw a)
             | CRotPI a -> crotp (rw a)
             | CLo a -> clo (rw a)
             | CHi a -> chi (rw a)
             | CStore (a, v) -> cstore a (rw v)
             | CTurn (a, b, imm) -> cturn (rw a) (rw b) imm
             | CAdd (a, b) -> cadd (rw a) (rw b)
             | CSub (a, b) -> csub (rw a) (rw b)
             | CRotAdd (a, b) -> crotadd (rw a) (rw b)
             | CFmaC (c, x, acc) -> cfma c (rw x) (rw acc)
             | CFnmaC (c, x, acc) -> cfnma c (rw x) (rw acc)
             | CTwC (c, s, x) -> ctw c s (rw x)
             | CTwV (w, x) -> ctwv w (rw x)
             | CTwL (l, x) -> ctwl l (rw x))
        in
        Hashtbl.replace memo e.tag r;
        r
    in
    List.map (fun (lbl, e) -> lbl, rw e) assigns, Hashtbl.length victim)
;;

(* ── critical path (cycles) of the DAG under the shared latency table ────
 * Longest latency-weighted path from any input to any output, using the
 * SAME Cx_sched.Node.latency the SR scheduler uses. This is the serial
 * floor of one pass: no schedule, no port count, no compiler can go below
 * it — the discriminator between "chain-bound" and "resource-bound". *)
let critical_path (uarch : Uarch.t) (roots : t list) : int =
  let memo : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let rec cp (e : t) : int =
    match Hashtbl.find_opt memo e.tag with
    | Some v -> v
    | None ->
      let v =
        Cx_sched.Node.latency uarch e
        + List.fold_left (fun acc c -> max acc (cp c)) 0 (children e)
      in
      Hashtbl.replace memo e.tag v;
      v
  in
  List.fold_left (fun acc r -> max acc (cp r)) 0 roots
;;

(* ── stats (VFFT_CX_STATS=1): per-codelet DAG census to stderr ─────────── *)
let print_stats ?(uarch : Uarch.t option) (who : string)
      (assigns : (Expr.elem_ref * t) list) : unit =
  let n_in = ref 0
  and n_add = ref 0
  and n_sub = ref 0
  and n_neg = ref 0
  and n_rot = ref 0
  and n_fma = ref 0
  and n_tw = ref 0
  and n_all = ref 0 in
  let n_st = ref 0 in
  iter_reachable (List.map snd assigns) (fun e ->
    incr n_all;
    match e.node with
    | CIn _ | CLoad _ -> incr n_in
    | CStore _ -> incr n_st
    | CAdd _ | CRotAdd _ -> incr n_add
    | CSub _ -> incr n_sub
    | CNeg _ -> incr n_neg
    | CRotNI _ | CRotPI _ | CTurn _ | CLo _ | CHi _ -> incr n_rot
    | CFmaC _ | CFnmaC _ -> incr n_fma
    | CTwC _ | CTwV _ | CTwL _ -> incr n_tw);
  let cp_str =
    match uarch with
    | Some u ->
      Printf.sprintf ", cp %dc" (critical_path u (List.map snd assigns))
    | None -> ""
  in
  Printf.eprintf
    "[cx_pipeline] %s: nodes %d (in %d, add %d, sub %d, neg %d, rot %d, fma %d, tw %d, st %d), outs %d%s\n%!"
    who !n_all !n_in !n_add !n_sub !n_neg !n_rot !n_fma !n_tw !n_st
    (List.length assigns) cp_str
;;

(* ── THE ENTRY POINT ──────────────────────────────────────────────────────
 * Every cil emission (monolithic body, each blocked pass, each fused-K=1
 * pass) hands its labeled assigns through here before scheduling. Default
 * env = every existing kernel byte-identical (the cascade's passes find
 * zero sites on the current builders; kill switches skip them outright). *)
let prepare_codelet ?(who = "cil") ?(uarch : Uarch.t option)
      (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref * t) list
  =
  let assigns, ndedup =
    if Sys.getenv_opt "VFFT_CX_NO_SUBDEDUP" = Some "1"
    then assigns, 0
    else dedup_sub_pairs_cx assigns
  in
  if ndedup > 0
  then Printf.eprintf "[cx_pipeline] %s: dedup_sub_pairs rewrote %d mirror(s)\n%!" who ndedup;
  if Sys.getenv_opt "VFFT_CX_STATS" = Some "1" then print_stats ?uarch who assigns;
  assigns
;;
