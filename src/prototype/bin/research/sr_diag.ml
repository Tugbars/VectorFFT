(* bin/sr_diag.ml — Diagnostic: count Mul/Add/Sub in the raw (pre-algsimp)
 * Expr tree and post-algsimp DAG, comparing CT and SR constructions side
 * by side. Discriminates between two hypotheses:
 *
 *   (A) Algsimp absorbs SR's algorithmic gains → pre-algsimp counts differ
 *       but post-algsimp counts converge.
 *   (B) Construction is wrong (SR ≡ CT at IR level) → pre-algsimp counts
 *       are already equal, so post-algsimp equality is unsurprising.
 *
 * Set VFFT_SPLIT_RADIX=1 in env to route through SR, otherwise CT. *)

open Vfft_v2

(* Walk a raw Expr tree and count node kinds, sharing-naive (a node is
 * counted each time it appears textually in the tree — not by hashcons).
 * This is the "as-constructed" count, what dft.ml or split_radix.ml emit
 * before of_assignments deduplicates via hash-consing. *)
type raw_counts = {
  mutable adds : int;
  mutable subs : int;
  mutable muls : int;
  mutable negs : int;
}

let walk_expr cnt e =
  let rec go (e : Expr.expr) = match e with
    | Expr.Const _ | Expr.Load _ -> ()
    | Expr.Neg a -> cnt.negs <- cnt.negs + 1; go a
    | Expr.Add (a, b) -> cnt.adds <- cnt.adds + 1; go a; go b
    | Expr.Sub (a, b) -> cnt.subs <- cnt.subs + 1; go a; go b
    | Expr.Mul (a, b) -> cnt.muls <- cnt.muls + 1; go a; go b
  in go e

(* Count nodes across ALL output expressions in the assignment list. This
 * is the "tree-textual" count — over-counts shared sub-expressions. The
 * difference between this and the post-of_assignments (hashcons'd DAG)
 * count tells us how much sharing the raw tree had implicitly. *)
let count_raw (assigns : Expr.assignment list) =
  let c = { adds = 0; subs = 0; muls = 0; negs = 0 } in
  List.iter (fun (_, e) -> walk_expr c e) assigns;
  c

(* Hash-cons the raw tree (mimics of_assignments) and count UNIQUE nodes.
 * This gives the post-hashcons count BEFORE any algsimp pass runs. *)
let count_unique (assigns : Expr.assignment list) =
  let tbl = Hashtbl.create 1024 in
  let adds = ref 0 and subs = ref 0 and muls = ref 0 and negs = ref 0 in
  let rec go (e : Expr.expr) =
    if not (Hashtbl.mem tbl e) then begin
      Hashtbl.add tbl e ();
      match e with
      | Expr.Const _ | Expr.Load _ -> ()
      | Expr.Neg a -> incr negs; go a
      | Expr.Add (a, b) -> incr adds; go a; go b
      | Expr.Sub (a, b) -> incr subs; go a; go b
      | Expr.Mul (a, b) -> incr muls; go a; go b
    end
  in
  List.iter (fun (_, e) -> go e) assigns;
  { adds = !adds; subs = !subs; muls = !muls; negs = !negs }

let () =
  let n = int_of_string Sys.argv.(1) in
  let raw_assigns =
    Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIT ~sign:`Fwd n in
  let raw = count_raw raw_assigns in
  let unique = count_unique raw_assigns in
  Printf.printf "  textual (over-counts shared)  : adds=%d subs=%d muls=%d negs=%d  total=%d\n"
    raw.adds raw.subs raw.muls raw.negs (raw.adds + raw.subs + raw.muls + raw.negs);
  Printf.printf "  unique (post-hashcons, pre-algsimp): adds=%d subs=%d muls=%d negs=%d  total=%d\n"
    unique.adds unique.subs unique.muls unique.negs
    (unique.adds + unique.subs + unique.muls + unique.negs)
