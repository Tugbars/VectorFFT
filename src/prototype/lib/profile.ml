(* profile.ml — static op-count extraction from the hash-consed DAG.
 *
 * Produces a per-codelet feature vector with the same shape as
 * tools/radix_profile/extract.py: arithmetic / memory / shuffle op
 * counts plus a decl-count proxy for register pressure. The output
 * feeds src/prototype/cost_model/generated/radix_profile.h.
 *
 * Unlike the production regex-over-emitted-C extractor, this walks
 * the Algsimp.t DAG directly. No text parsing, no post-hoc analysis;
 * the counts are exact at the IR level, modulo the small set of
 * emit-time inlining decisions we re-apply here. *)

open Algsimp

(* Op counts per codelet. Field set + naming match the production CSV
 * (tools/radix_profile/profile_avx{2,512}.csv) so the prototype's
 * radix_profile.h can be diff-compared against production's for sanity. *)
type op_counts = {
  n_add   : int;    (* NK_Add + NK_Sub — both compile to vaddpd/vsubpd *)
  n_mul   : int;    (* NK_Mul — vmulpd. Cmul mul outputs counted here too. *)
  n_fma   : int;    (* NK_Fma + the fmadd/fmsub emitted by NK_Cmul* *)
  n_load  : int;    (* NK_Load — _mm{256,512}_loadu_pd *)
  n_store : int;    (* count of Output assignments — _mm{256,512}_storeu_pd *)
  n_blend : int;    (* unused in current OCaml emit; kept for CSV parity *)
  n_set1  : int;    (* distinct NK_Const float values — _mm{256,512}_set1_pd *)
  n_xor   : int;    (* NK_Neg of non-constant — _mm{256,512}_xor_pd(.,-0.0) *)
  n_decls : int;    (* reachable nodes that get a `const __m{256,512}d tN`
                       declaration: not in inline_set, not pure Load (Load
                       still gets named — see emit_c render_node_def),
                       not pure Const (materialized as set1 at use). *)
}

let zero_counts = {
  n_add = 0; n_mul = 0; n_fma = 0; n_load = 0; n_store = 0;
  n_blend = 0; n_set1 = 0; n_xor = 0; n_decls = 0;
}

(* Re-implementation of emit_c's compute_inline_set, scoped here to
 * avoid a circular dependency between Profile and Emit_c. The criteria
 * mirror emit_c.ml ~line 405 exactly: single-use, non-sink, kind-inlinable
 * (Load / Const / Cmul are excluded — they have special emit paths). *)
let compute_inline_set
    (assigns : (Expr.elem_ref * t) list) : (int, unit) Hashtbl.t =
  let roots = List.map snd assigns in
  (* Topological walk reachable from roots. Hash-cons tag order is
   * already topological. *)
  let seen : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag e;
      List.iter visit (Algsimp.preds e)
    end
  in
  List.iter visit roots;
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let bump tag =
    let c = try Hashtbl.find use_count tag with Not_found -> 0 in
    Hashtbl.replace use_count tag (c + 1)
  in
  Hashtbl.iter (fun _ n ->
    List.iter (fun (p : t) -> bump p.tag) (Algsimp.preds n)
  ) seen;
  List.iter (fun (_, e) -> bump e.tag) assigns;
  let sink_tags = Hashtbl.create 32 in
  List.iter (fun (_, e) -> Hashtbl.replace sink_tags e.tag ()) assigns;
  let result = Hashtbl.create 256 in
  Hashtbl.iter (fun _ (n : t) ->
    let count = try Hashtbl.find use_count n.tag with Not_found -> 0 in
    let is_sink = Hashtbl.mem sink_tags n.tag in
    let kind_inlinable = match n.node with
      | NK_Load _ | NK_CmulRe _ | NK_CmulIm _ | NK_Const _ -> false
      | _ -> true
    in
    if count = 1 && not is_sink && kind_inlinable then
      Hashtbl.add result n.tag ()
  ) seen;
  result

(* Walk reachable nodes once, accumulating per-kind counts. *)
let count_ops (assigns : (Expr.elem_ref * t) list) : op_counts =
  let roots = List.map snd assigns in
  let seen : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag e;
      List.iter visit (Algsimp.preds e)
    end
  in
  List.iter visit roots;
  let inline_set = compute_inline_set assigns in
  (* Distinct constant values seen (n_set1 = number of distinct set1
   * broadcasts emit_c would emit). *)
  let const_vals : (float, unit) Hashtbl.t = Hashtbl.create 16 in
  let n_add   = ref 0 in
  let n_mul   = ref 0 in
  let n_fma   = ref 0 in
  let n_load  = ref 0 in
  let n_xor   = ref 0 in
  let n_decls = ref 0 in
  Hashtbl.iter (fun _ (n : t) ->
    (* Per-kind contributions to the arithmetic / memory counters. *)
    (match n.node with
     | NK_Add _ | NK_Sub _ -> incr n_add
     | NK_Mul _ -> incr n_mul
     | NK_Fma _ -> incr n_fma
     | NK_Load _ -> incr n_load
     | NK_CmulRe _ | NK_CmulIm _ ->
       (* Each Cmul output materializes as one mul + one fmadd/fmsub
        * (see render_node_def). Cmul.re ≈ mul + fnmadd, Cmul.im ≈
        * mul + fmadd. Count both contributions. *)
       incr n_mul; incr n_fma
     | NK_Neg inner ->
       (match inner.node with
        | NK_Const _ -> ()  (* Neg(Const c) → set1(-c), no XOR emitted *)
        | _ -> incr n_xor)
     | NK_Const c ->
       Hashtbl.replace const_vals c ());
    (* Decl contribution: reachable AND not in inline_set AND not a
     * pure Const (materialized as set1 at use site, no decl). *)
    let no_decl_kind = match n.node with
      | NK_Const _ -> true
      | _ -> false
    in
    if not (Hashtbl.mem inline_set n.tag) && not no_decl_kind then
      incr n_decls
  ) seen;
  let n_store = List.length assigns in
  { n_add = !n_add; n_mul = !n_mul; n_fma = !n_fma;
    n_load = !n_load; n_store; n_blend = 0;
    n_set1 = Hashtbl.length const_vals;
    n_xor = !n_xor;
    n_decls = !n_decls }

(* CSV row formatter — matches the production header
 * `radix,variant,present,n_add,n_mul,n_fma,n_load,n_store,n_blend,n_set1,n_xor,n_decls,file`
 * exactly so diff-comparison against tools/radix_profile/profile_*.csv works. *)
let csv_header : string =
  "radix,variant,present,n_add,n_mul,n_fma,n_load,n_store,n_blend,n_set1,n_xor,n_decls,file"

let csv_row ~radix ~variant ~present ~file (c : op_counts) : string =
  Printf.sprintf "%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s"
    radix variant (if present then 1 else 0)
    c.n_add c.n_mul c.n_fma c.n_load c.n_store
    c.n_blend c.n_set1 c.n_xor c.n_decls file
