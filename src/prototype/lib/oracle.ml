(* oracle.ml — numerical CSE-miss detector via Schwartz-Zippel hashing.
 *
 * Evaluates every post-algsimp DAG node with deterministic random doubles
 * at the Load leaves, buckets by numerical hash (rounded to 12 digits to
 * absorb FP non-associativity). Buckets with >1 distinct tag flag
 * algebraically-equivalent nodes algsimp didn't dedup. Diagnostic only;
 * not in the codegen path. *)

open Algsimp

(* Deterministic random value for each Load. Hash by node_kind so loads
 * with the same elem_ref get the same value. *)
let load_value (r : Expr.elem_ref) : float =
  let seed = match r with
    | Expr.Input (j, true)    -> j * 7 + 1
    | Expr.Input (j, false)   -> j * 7 + 2
    | Expr.Twiddle (j, true)  -> j * 7 + 3
    | Expr.Twiddle (j, false) -> j * 7 + 4
    | Expr.Output _ ->
      failwith "Oracle.load_value: Output cannot be a load source"
  in
  (* sin(seed * golden_ratio) * 1000 — irrational multiplier ensures
   * distinct seeds give well-separated values; *1000 scales to ranges
   * where small algebraic differences are easier to detect than
   * near-zero. *)
  sin (float_of_int seed *. 0.61803398875) *. 1000.0

(* Evaluate a node, memoized by tag. *)
let eval_table : (int, float) Hashtbl.t = Hashtbl.create 1024
let reset () = Hashtbl.clear eval_table

let rec eval (e : t) : float =
  match Hashtbl.find_opt eval_table e.tag with
  | Some v -> v
  | None ->
    let v = match e.node with
      | NK_Const c        -> c
      | NK_Load r         -> load_value r
      | NK_Neg a          -> -. (eval a)
      | NK_Add (a, b)     -> eval a +. eval b
      | NK_Sub (a, b)     -> eval a -. eval b
      | NK_Mul (a, b)     -> eval a *. eval b
      | NK_CmulRe (a, b, c, d) -> (eval a) *. (eval c) -. (eval b) *. (eval d)
      | NK_CmulIm (a, b, c, d) -> (eval a) *. (eval d) +. (eval b) *. (eval c)
      | NK_Fma (a, b, c, neg_prod, neg_acc) ->
        let prod = (eval a) *. (eval b) in
        let prod = if neg_prod then -. prod else prod in
        let acc = if neg_acc then -. (eval c) else eval c in
        prod +. acc
    in
    Hashtbl.add eval_table e.tag v;
    v

(* Bucket key: relative-precision hash.
 *
 * Naive scaling (v * 1e12) overflows int64 for |v| > 9e6, lumping all
 * large values into one bucket. Use frexp to separate mantissa from
 * exponent: mantissa ∈ [0.5, 1), exponent is the binary order of magnitude.
 * Round the mantissa to 40 bits (≈ 12 decimal digits) and combine with
 * the exponent. This gives ~12 digits of relative precision at every
 * magnitude, which is above the ~1-2 ULP FP-noise floor. *)
let bucket_key (v : float) : int64 =
  if Float.is_nan v then 0L
  else if Float.is_infinite v then
    if v > 0.0 then 1L else -1L
  else if v = 0.0 then 2L
  else
    let abs_v = Float.abs v in
    if abs_v < 1e-300 then 2L
    else
      let mantissa, exponent = Float.frexp abs_v in  (* mantissa ∈ [0.5, 1) *)
      let scaled = mantissa *. 1099511627776.0 in     (* 2^40 ≈ 1.1e12 *)
      let mantissa_bits = Int64.of_float (Float.round scaled) in
      let sign_bit = if v < 0.0 then 0x4000000000000000L else 0L in
      let exp_bits = Int64.shift_left (Int64.of_int (exponent + 2000)) 41 in
      Int64.logor (Int64.logor mantissa_bits exp_bits) sign_bit

(* Walk a list of root nodes, collect ALL reachable nodes. *)
let reachable (roots : t list) : t list =
  let seen : (int, unit) Hashtbl.t = Hashtbl.create 1024 in
  let result = ref [] in
  let rec walk e =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag ();
      result := e :: !result;
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a -> walk a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> walk a; walk b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        walk a; walk b; walk c; walk d
      | NK_Fma (a, b, c, _, _) -> walk a; walk b; walk c
    end
  in
  List.iter walk roots;
  !result

(* Diagnostic report. Returns (num_buckets_with_collision, total_collisions,
 * largest_bucket_size). A "collision" = two different tags producing the
 * same Oracle hash, i.e. an algebraic equivalence our structural CSE
 * missed. *)
type diag = {
  total_nodes : int;
  unique_values : int;
  collision_buckets : int;
  total_collisions : int;
  largest_bucket : int;
  examples : (int64 * t list) list;  (* top-N buckets with collisions *)
}

let diagnose (roots : t list) : diag =
  reset ();
  let nodes = reachable roots in
  let n = List.length nodes in
  (* Skip Const and Load — those are leaves; sharing is hash-consing's job. *)
  let interior = List.filter (fun e ->
    match e.node with NK_Const _ | NK_Load _ -> false | _ -> true) nodes
  in
  let buckets : (int64, t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun e ->
    let key = bucket_key (eval e) in
    let cur = try Hashtbl.find buckets key with Not_found -> [] in
    Hashtbl.replace buckets key (e :: cur)
  ) interior;
  let collision_buckets = ref 0 in
  let total_collisions = ref 0 in
  let largest = ref 0 in
  let examples = ref [] in
  Hashtbl.iter (fun key lst ->
    let k = List.length lst in
    if k > 1 then begin
      incr collision_buckets;
      total_collisions := !total_collisions + (k - 1);
      if k > !largest then largest := k;
      examples := (key, lst) :: !examples
    end
  ) buckets;
  let sorted_examples = List.sort (fun (_, a) (_, b) ->
    compare (List.length b) (List.length a)) !examples in
  {
    total_nodes = n;
    unique_values = Hashtbl.length buckets;
    collision_buckets = !collision_buckets;
    total_collisions = !total_collisions;
    largest_bucket = !largest;
    examples = (match sorted_examples with
      | _ :: _ :: _ :: _ :: _ :: _ -> List.filteri (fun i _ -> i < 5) sorted_examples
      | xs -> xs);
  }

(* Pretty-print a node's structure for diagnostic output. *)
let rec describe (e : t) : string =
  let short s = if String.length s > 60 then String.sub s 0 57 ^ "..." else s in
  short (match e.node with
  | NK_Const c -> Printf.sprintf "Const(%g)" c
  | NK_Load (Expr.Input (j, true))    -> Printf.sprintf "Input_re[%d]" j
  | NK_Load (Expr.Input (j, false))   -> Printf.sprintf "Input_im[%d]" j
  | NK_Load (Expr.Twiddle (j, true))  -> Printf.sprintf "Twid_re[%d]" j
  | NK_Load (Expr.Twiddle (j, false)) -> Printf.sprintf "Twid_im[%d]" j
  | NK_Load _ -> "Load(?)"
  | NK_Neg a -> Printf.sprintf "-(%s)" (describe a)
  | NK_Add (a, b) -> Printf.sprintf "(%s + %s)" (describe a) (describe b)
  | NK_Sub (a, b) -> Printf.sprintf "(%s - %s)" (describe a) (describe b)
  | NK_Mul (a, b) -> Printf.sprintf "(%s * %s)" (describe a) (describe b)
  | NK_CmulRe (a, b, c, d) ->
    Printf.sprintf "CmRe(%s,%s,%s,%s)" (describe a) (describe b) (describe c) (describe d)
  | NK_CmulIm (a, b, c, d) ->
    Printf.sprintf "CmIm(%s,%s,%s,%s)" (describe a) (describe b) (describe c) (describe d)
  | NK_Fma (a, b, c, _, _) ->
    Printf.sprintf "Fma(%s,%s,%s)" (describe a) (describe b) (describe c))

let print_diag (d : diag) : unit =
  Printf.printf "─────────────────────────────────────────────────────────────\n";
  Printf.printf "Oracle CSE diagnostic\n";
  Printf.printf "─────────────────────────────────────────────────────────────\n";
  Printf.printf "  Total reachable nodes:       %d\n" d.total_nodes;
  Printf.printf "  Unique Oracle-hash buckets:  %d\n" d.unique_values;
  Printf.printf "  Buckets with >1 distinct tag: %d  (= missed CSE chances)\n" d.collision_buckets;
  Printf.printf "  Total missed shares:         %d  (= sum of (bucket_size - 1))\n" d.total_collisions;
  Printf.printf "  Largest collision bucket:    %d distinct tags\n" d.largest_bucket;
  if d.total_collisions = 0 then begin
    Printf.printf "\n  Conclusion: structural CSE is COMPLETE for this DAG.\n";
    Printf.printf "  Oracle finds no algebraic equivalences our CSE missed.\n"
  end else begin
    Printf.printf "\n  Top collision buckets (showing structure of equivalent nodes):\n";
    List.iter (fun (_, nodes) ->
      Printf.printf "\n  Bucket [%d nodes, value ≈ %g]:\n"
        (List.length nodes) (eval (List.hd nodes));
      List.iter (fun n ->
        Printf.printf "    tag %4d: %s\n" n.tag (describe n)
      ) nodes
    ) d.examples
  end;
  Printf.printf "─────────────────────────────────────────────────────────────\n"
