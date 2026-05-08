(* expr.ml — the symbolic math IR.
 *
 * This is the "math layer" output: a tree-structured representation of
 * the arithmetic that computes a DFT. No SIMD, no microarchitecture, no
 * register allocation — just expressions with concrete numeric constants.
 *
 * --- OCaml lesson #1: variant types ---
 *
 * `type expr = ...` below defines an algebraic data type. Each `|` clause
 * is one possible shape an `expr` value can have. This is the natural way
 * to represent tree-structured data in OCaml: each node is a variant,
 * and pattern matching on the variant is exhaustive (the compiler warns
 * if you forget a case).
 *
 * Compare to Python: you'd use a class hierarchy (BaseExpr, Add(BaseExpr),
 * Mul(BaseExpr), ...) and `isinstance` checks, with no compiler-enforced
 * exhaustiveness. The OCaml version catches "I added a Sub case but
 * forgot to handle it in simplify()" at compile time.
 *)

(* A reference to an input or output element of the transform.
 * `Input(0, true)` means input element 0's real component;
 * `Output(2, false)` means output element 2's imaginary component.
 *
 * `type t = ... [@@deriving ...]` would give us automatic equality, hash,
 * etc. — but we'll write them explicitly for clarity in the prototype. *)
type elem_ref =
  | Input  of int * bool   (* (index, is_real) *)
  | Output of int * bool
  | Twiddle of int * bool  (* (twiddle index, is_real) — for t1_dit codelet *)

(* The math IR. An `expr` is a tree of arithmetic operations whose leaves
 * are constants or input element references.
 *
 * Note: `Const` carries a float because at the math layer, twiddles are
 * concrete numbers. For radix-4, all twiddles fold to ±1 or 0; for
 * radix-8 they're ±1, 0, ±sqrt(2)/2 etc. Carrying actual floats lets
 * us do constant folding in algsimp.ml.
 *)
type expr =
  | Const of float
  | Load  of elem_ref       (* read an input/twiddle element *)
  | Neg   of expr
  | Add   of expr * expr
  | Sub   of expr * expr
  | Mul   of expr * expr

(* A complete codelet's math layer output: a list of (output_ref, expr)
 * pairs, one per output element computed.
 *
 * For DFT-4 t1_dit (with twiddles, complex inputs/outputs) this list has
 * 8 entries: 4 complex outputs × 2 (real, imag) components. *)
type assignment = elem_ref * expr

(* --- OCaml lesson #2: structural equality and pretty-printing ---
 *
 * Polymorphic equality `(=)` works on these types out of the box because
 * variant types support structural comparison. We add a dedicated
 * `equal` function anyway because relying on `(=)` for cyclic or
 * float-containing data is brittle (NaN != NaN under `(=)`, for one).
 *
 * `string_of_*` functions are written manually here. In production
 * OCaml you'd use [@@deriving show] from the ppx_show package, but
 * for a sketch we keep dependencies minimal.
 *)

let string_of_elem_ref (e : elem_ref) : string =
  match e with
  | Input  (i, true)  -> Printf.sprintf "x[%d].re" i
  | Input  (i, false) -> Printf.sprintf "x[%d].im" i
  | Output (i, true)  -> Printf.sprintf "X[%d].re" i
  | Output (i, false) -> Printf.sprintf "X[%d].im" i
  | Twiddle (i, true)  -> Printf.sprintf "tw[%d].re" i
  | Twiddle (i, false) -> Printf.sprintf "tw[%d].im" i

(* Recursive pretty-printer with parentheses inserted minimally.
 * `prec` is the precedence of the surrounding context: 0 = top, 1 = +/-,
 * 2 = unary minus, 3 = *. We add parens when our op's precedence is
 * lower than the surrounding context's. *)
let rec string_of_expr_prec (prec : int) (e : expr) : string =
  match e with
  | Const c ->
    if c < 0.0 then Printf.sprintf "(%g)" c   (* parenthesize negatives *)
    else Printf.sprintf "%g" c
  | Load r -> string_of_elem_ref r
  | Neg e1 ->
    let s = "-" ^ string_of_expr_prec 2 e1 in
    if prec > 2 then "(" ^ s ^ ")" else s
  | Add (a, b) ->
    let s = string_of_expr_prec 1 a ^ " + " ^ string_of_expr_prec 1 b in
    if prec > 1 then "(" ^ s ^ ")" else s
  | Sub (a, b) ->
    let s = string_of_expr_prec 1 a ^ " - " ^ string_of_expr_prec 2 b in
    if prec > 1 then "(" ^ s ^ ")" else s
  | Mul (a, b) ->
    string_of_expr_prec 3 a ^ " * " ^ string_of_expr_prec 3 b

let string_of_expr (e : expr) : string = string_of_expr_prec 0 e

(* Print a complete assignment list, one per line, for debugging.
 * `Buffer.create` + `Buffer.add_string` is the OCaml idiom for building
 * up strings without quadratic-time concatenation. *)
let string_of_assignments (al : assignment list) : string =
  let buf = Buffer.create 1024 in
  List.iter (fun (lhs, rhs) ->
    Buffer.add_string buf (string_of_elem_ref lhs);
    Buffer.add_string buf " = ";
    Buffer.add_string buf (string_of_expr rhs);
    Buffer.add_char buf '\n'
  ) al;
  Buffer.contents buf

(* === Symbolic DFT-N expansion ===
 *
 * The mathematical core. Given N, build the list of output expressions
 * for the forward DFT:
 *
 *   X[k] = sum_{n=0..N-1} x[n] * exp(-2*pi*i*n*k/N)
 *
 * Splitting into real/imaginary parts (where x[n] = a[n] + i*b[n] and
 * the twiddle is c[n,k] + i*s[n,k] with c=cos, s=-sin since we're doing
 * forward DFT with the e^{-i...} convention):
 *
 *   X[k].re = sum_n  a[n]*c[n,k] - b[n]*s[n,k]
 *   X[k].im = sum_n  a[n]*s[n,k] + b[n]*c[n,k]
 *
 * For the t1_dit codelet (twiddle DIT), the inputs are pre-multiplied
 * by twiddles before this butterfly: x'[n] = x[n] * w[n] for n=1..N-1,
 * with x'[0] = x[0]. The codelet then runs the size-N no-twiddle DFT
 * on the twiddled inputs. We express this as composition of a
 * `cmul_premultiply` pass and the `dft_kernel` core, sharing the latter
 * between the no-twiddle and twiddled variants.
 *)

(* Internal: build the DFT-N output expressions given the input
 * expressions as a function of n. For n=0..N-1, `input_re n` should
 * return the real-part expression for input n (similarly imag).
 *
 * This separation lets us share the DFT body between:
 *   - `dft_expand`:           input n -> Load(Input(n, _))     (no twiddle)
 *   - `dft_expand_twiddled`:  input n -> precomputed cmul(x[n], w[n])
 * *)
let dft_kernel (n : int) (input_re : int -> expr) (input_im : int -> expr)
    : assignment list =
  let pi = 4.0 *. atan 1.0 in
  let assignments = ref [] in
  for k = 0 to n - 1 do
    let re_sum = ref (Const 0.0) in
    let im_sum = ref (Const 0.0) in
    for nn = 0 to n - 1 do
      let theta = -2.0 *. pi *. float_of_int (nn * k) /. float_of_int n in
      let c = cos theta in
      let s = sin theta in
      let a_nn = input_re nn in
      let b_nn = input_im nn in
      let term_re = Sub (Mul (a_nn, Const c), Mul (b_nn, Const s)) in
      let term_im = Add (Mul (a_nn, Const s), Mul (b_nn, Const c)) in
      re_sum := Add (!re_sum, term_re);
      im_sum := Add (!im_sum, term_im)
    done;
    assignments := (Output (k, true),  !re_sum) :: !assignments;
    assignments := (Output (k, false), !im_sum) :: !assignments
  done;
  List.rev !assignments

(* No-twiddle DFT-N: inputs are raw `x[n]`. Used for the n1 codelet path. *)
let dft_expand (n : int) : assignment list =
  let input_re nn = Load (Input (nn, true)) in
  let input_im nn = Load (Input (nn, false)) in
  dft_kernel n input_re input_im

(* Twiddled DFT-N: inputs are pre-multiplied by twiddles.
 *
 *   x'[0] = x[0]                    (no twiddle on leg 0)
 *   x'[n] = x[n] * w[n-1]           (n=1..N-1, twiddles indexed 0..N-2)
 *
 * Complex multiplication for split-complex with x = (a, b) and w = (c, d):
 *   (x * w).re = a*c - b*d
 *   (x * w).im = a*d + b*c
 *
 * The cmul expressions are built as Expr trees and substituted into the
 * DFT kernel. After algsimp:
 *   - The 4 muls + 2 adds/subs per twiddled leg become FMA-shaped after
 *     SIMD lowering (a*c - b*d is one FMS, a*d + b*c is one FMA).
 *   - The cmul outputs are referenced once per output equation (k=0..N-1),
 *     so hash-consing shares them across all output computations.
 *
 * The math layer doesn't know about log3 / t1s / buf — those are
 * lowering-time twiddle access patterns that substitute the
 * `Load(Twiddle(j, ...))` nodes with derived expressions. At this level,
 * twiddles are just loads; the lowering pass decides where they come from.
 *)
let dft_expand_twiddled (n : int) : assignment list =
  (* Per-leg cmul expressions, computed once (hash-consing handles
   * sharing across output equations later). For leg 0, no cmul: the
   * input is just x[0]. For legs 1..N-1, build the complex multiply. *)
  let twiddled_re = Array.make n (Const 0.0) in
  let twiddled_im = Array.make n (Const 0.0) in
  twiddled_re.(0) <- Load (Input (0, true));
  twiddled_im.(0) <- Load (Input (0, false));
  for nn = 1 to n - 1 do
    let a = Load (Input (nn, true)) in        (* x[n].re *)
    let b = Load (Input (nn, false)) in       (* x[n].im *)
    (* Twiddle index: w[nn-1] since leg 0 has no twiddle. So tw_re slot
     * (nn-1) corresponds to leg nn's twiddle. *)
    let c = Load (Twiddle (nn - 1, true)) in  (* w[n-1].re *)
    let d = Load (Twiddle (nn - 1, false)) in (* w[n-1].im *)
    twiddled_re.(nn) <- Sub (Mul (a, c), Mul (b, d));   (* a*c - b*d *)
    twiddled_im.(nn) <- Add (Mul (a, d), Mul (b, c))    (* a*d + b*c *)
  done;
  let input_re nn = twiddled_re.(nn) in
  let input_im nn = twiddled_im.(nn) in
  dft_kernel n input_re input_im
(* Note: `let xs = ref [] in ... xs := v :: !xs ... !xs` is OCaml's
 * idiomatic way to build a list inside a loop. `ref` makes `xs` mutable;
 * `!xs` dereferences (reads the current value); `xs := newval` writes.
 * We `List.rev` at the end because we built the list in reverse. *)
