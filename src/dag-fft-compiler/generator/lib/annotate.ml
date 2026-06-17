(* annotate.ml — block-level lifetime annotation for scheduled DAGs.
 *
 * What this layer does:
 *   Takes a scheduled list of instructions (e.g. output of bisection
 *   or topological emission) and produces a NESTED-BLOCK structure.
 *   Variables with short live ranges get declared in inner blocks;
 *   variables with long live ranges get declared in outer blocks.
 *
 * Why it matters:
 *   When emitted as C, this becomes:
 *     {
 *       __m512d t1 = ...;  // long-lived: outer scope
 *       {
 *         __m512d t2 = ...;  // short-lived: inner block
 *         __m512d t3 = ... t1 ... t2 ...;
 *         /* t2 dies at this `}` — register reusable */
 *       }
 *       /* t3 also dead now */
 *     }
 *   GCC's register allocator sees these scope boundaries and reuses
 *   registers more aggressively. Without annotate, all SSA bindings
 *   appear to live until function end (from a syntactic perspective)
 *   and GCC must do its own liveness analysis, which is harder.
 *
 * Frigo's approach uses Seq trees from the bisection scheduler. We
 * recover scope structure post-hoc by recursively bisecting the
 * already-scheduled list at midpoints. This way annotate works on
 * the output of any scheduler — topological order, bisection,
 * future SU, etc.
 *)

open Algsimp

(* === SCHEDULED ENTRY ===
 *
 * The unit operated on. Matches the output type of
 * Schedule.bisection_schedule and our flat topological order. *)

type entry = {
  output_for: Expr.elem_ref option;  (* None = intermediate; Some = output store *)
  alg_node: Algsimp.t;
}

(* === SCOPE TREE ===
 *
 * The annotated structure. Each Block has:
 *   - decls: tags whose definitions belong at THIS scope's level
 *            (their values are computed by entries in `body`, but
 *            their declarations are emitted at this scope's open)
 *   - body: a sequence of Leaf entries and nested Block scopes
 *
 * For a Leaf entry that is a *definition* (output_for = None), the
 * variable belongs to whichever Block declared it. The leaf itself
 * is just the position where the value gets computed. *)

type scope =
  | Leaf of entry
  | Block of {
      decls: int list;       (* tags declared at this scope level *)
      body: scope list;
    }

(* === LIFETIME ANALYSIS === *)

(* Compute (first_def, last_use) for each tag in the schedule.
 *
 * first_def[tag] = index of the entry that first DEFINES tag (output_for=None).
 *   Loads and Consts are also "defined" at their first appearance as a
 *   non-output entry — they need a declaration too.
 *
 * last_use[tag] = the highest index where tag is referenced.
 *   Referenced by: another intermediate's preds, or an output entry's alg_node. *)
let compute_lifetimes (entries : entry array)
    : (int, int) Hashtbl.t * (int, int) Hashtbl.t =
  let first_def = Hashtbl.create 256 in
  let last_use = Hashtbl.create 256 in
  let n = Array.length entries in
  for i = 0 to n - 1 do
    let entry = entries.(i) in
    (match entry.output_for with
     | None ->
       (* This entry defines the alg_node as a new variable. *)
       if not (Hashtbl.mem first_def entry.alg_node.tag) then
         Hashtbl.add first_def entry.alg_node.tag i;
       (* It also USES its predecessors (which were defined earlier). *)
       List.iter (fun p ->
         Hashtbl.replace last_use p.tag i
       ) (preds entry.alg_node)
     | Some _ ->
       (* Output store entry: uses the alg_node (defined earlier). *)
       Hashtbl.replace last_use entry.alg_node.tag i)
  done;
  (* Variables that are defined but never used (rare — would be dead code,
   * but include them anyway for safety): set last_use = first_def. *)
  Hashtbl.iter (fun tag def_idx ->
    if not (Hashtbl.mem last_use tag) then
      Hashtbl.add last_use tag def_idx
  ) first_def;
  (first_def, last_use)

(* === RECURSIVE BISECTION FOR SCOPE TREE ===
 *
 * Take entries[lo..hi) and produce a scope. If the range is small
 * enough or has no nested structure to extract, return a flat Block.
 * Otherwise, split at the midpoint and recurse on each half.
 *
 * At each split, we compute which tags can be declared LOCALLY in
 * one of the children (their entire lifetime is inside that child)
 * vs which must be declared at THIS level (lifetime spans both
 * children, or includes the boundary).
 *)

(* Minimum range length below which we don't subdivide further.
 * Small scopes have overhead (extra braces, no real benefit) and
 * GCC handles small-scope register allocation fine. *)
let min_block_size = 8

let rec scope_range (entries : entry array)
                    (first_def : (int, int) Hashtbl.t)
                    (last_use : (int, int) Hashtbl.t)
                    (lo : int) (hi : int)
                    : scope =
  let lookup tbl tag default =
    match Hashtbl.find_opt tbl tag with Some v -> v | None -> default
  in
  let len = hi - lo in
  if len <= 0 then
    Block { decls = []; body = [] }
  else if len <= min_block_size then
    (* Leaf block: declare tags whose lifetime is entirely in [lo, hi).
     * Tags that escape are handled by the enclosing scope. *)
    let local_decls = ref [] in
    for i = lo to hi - 1 do
      let e = entries.(i) in
      if e.output_for = None then begin
        let tag = e.alg_node.tag in
        let def_i = lookup first_def tag i in
        let use_i = lookup last_use tag def_i in
        if def_i >= lo && use_i < hi then
          local_decls := tag :: !local_decls
      end
    done;
    let body = ref [] in
    for i = hi - 1 downto lo do
      body := Leaf entries.(i) :: !body
    done;
    Block { decls = List.rev !local_decls; body = !body }
  else
    let mid = (lo + hi) / 2 in
    let left = scope_range entries first_def last_use lo mid in
    let right = scope_range entries first_def last_use mid hi in
    (* Cross-decls at THIS level: tags defined in [lo, mid), first used at
     * or after mid, last used before hi. They span my children but don't
     * escape me, so I declare them. *)
    let cross_decls = ref [] in
    for i = lo to mid - 1 do
      let e = entries.(i) in
      if e.output_for = None then begin
        let tag = e.alg_node.tag in
        let def_i = lookup first_def tag i in
        let use_i = lookup last_use tag def_i in
        if def_i >= lo && def_i < mid
           && use_i >= mid && use_i < hi then
          cross_decls := tag :: !cross_decls
      end
    done;
    Block {
      decls = List.rev !cross_decls;
      body = [left; right];
    }

(* === PUBLIC API === *)

let annotate (entries : (Expr.elem_ref option * Algsimp.t) list) : scope =
  let arr = Array.of_list (List.map (fun (oref, n) ->
    { output_for = oref; alg_node = n }
  ) entries) in
  let (first_def, last_use) = compute_lifetimes arr in
  scope_range arr first_def last_use 0 (Array.length arr)

(* === EMIT HELPERS ===
 *
 * Walk the scope tree and produce nested C-block output. The caller
 * provides renderers for definitions and stores; we emit the right
 * brace structure around them. *)

(* Walk the scope, computing for each Block which definitions to emit
 * INLINE at their position vs hoist to scope opening.
 *
 * Strategy: emit all definitions inline at their position (where the
 * Leaf appears). The scope tree's nesting still gives us the nested
 * `{ ... }` structure that communicates lifetimes to GCC; we don't
 * need to physically re-order definitions.
 *
 * The only effect of `decls` at this level is documentary. Future
 * versions could hoist long-lived defs to scope-opening if it helps,
 * but for the first pass we just rely on nested braces. *)

(* === EMIT WITH FORWARD DECLARATIONS ===
 *
 * The challenge: a Leaf at a deep nested scope might define a tag that
 * an outer scope needs (because some sibling block uses it). Naive
 * inline `const VECTYPE t<tag> = ...;` emission would put the decl in
 * the inner scope, where it goes out of scope before the user can see it.
 *
 * Fix: at each Block open, FORWARD-DECLARE all tags that this scope
 * "owns" (their lifetime is fully within this scope, but they may be
 * defined inside a deeper-nested child of this scope). Then emit
 * ASSIGNMENTS (not const-decls) at Leaf positions. The forward decl
 * makes the variable visible throughout this scope, including all
 * nested children.
 *
 * Tracking: we maintain a mutable set of "forward-declared" tags. When
 * a Leaf intermediate is hit, we check if its tag was forward-declared:
 *   - If yes: emit `t<tag> = expr;` (assignment only)
 *   - If no:  emit `const VECTYPE t<tag> = expr;` (decl+init, scoped here) *)

(* Used internally — strip the `const VECTYPE ` prefix from a render_node_def
 * output, leaving just `        t<tag> = expr;`. We accept the ISA so we
 * can match its specific vec_type. *)
let strip_const_prefix (isa : Isa.t) (line : string) : string =
  let prefix = Printf.sprintf "const %s " isa.vec_type in
  match String.index_opt line 'c' with
  | None -> line
  | Some i ->
    let rest = String.sub line i (String.length line - i) in
    if String.length rest >= String.length prefix
       && String.sub rest 0 (String.length prefix) = prefix
    then
      let leading = String.sub line 0 i in
      let after = String.sub rest (String.length prefix)
                    (String.length rest - String.length prefix) in
      leading ^ after
    else
      line

let emit_scope (isa : Isa.t)
               (buf : Buffer.t)
               (render_intermediate : Algsimp.t -> string)
               (render_store : Expr.elem_ref -> Algsimp.t -> string)
               (s : scope) : unit =
  let forward_declared : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let render_decls tags =
    Isa.forward_decl isa (List.map (fun t -> Printf.sprintf "t%d" t) tags)
  in
  let rec walk indent s =
    match s with
    | Leaf entry ->
      Buffer.add_string buf indent;
      (match entry.output_for with
       | None ->
         let line = render_intermediate entry.alg_node in
         if Hashtbl.mem forward_declared entry.alg_node.tag then
           Buffer.add_string buf (strip_const_prefix isa line)
         else
           Buffer.add_string buf line
       | Some oref -> Buffer.add_string buf (render_store oref entry.alg_node));
      Buffer.add_char buf '\n'
    | Block { decls; body } ->
      let new_decls = List.filter (fun t ->
        not (Hashtbl.mem forward_declared t)
      ) decls in
      List.iter (fun t -> Hashtbl.add forward_declared t ()) new_decls;
      Buffer.add_string buf indent;
      Buffer.add_string buf "{\n";
      if new_decls <> [] then begin
        Buffer.add_string buf indent;
        Buffer.add_string buf "  ";
        Buffer.add_string buf (render_decls new_decls);
        Buffer.add_char buf '\n'
      end;
      let inner_indent = indent ^ "  " in
      List.iter (walk inner_indent) body;
      Buffer.add_string buf indent;
      Buffer.add_string buf "}\n";
      List.iter (fun t -> Hashtbl.remove forward_declared t) new_decls
  in
  match s with
  | Block { decls; body } ->
    let new_decls = List.filter (fun t ->
      not (Hashtbl.mem forward_declared t)
    ) decls in
    List.iter (fun t -> Hashtbl.add forward_declared t ()) new_decls;
    if new_decls <> [] then begin
      Buffer.add_string buf "        ";
      Buffer.add_string buf (render_decls new_decls);
      Buffer.add_char buf '\n'
    end;
    List.iter (walk "        ") body
  | Leaf _ -> walk "        " s
