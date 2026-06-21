(* cnum.ml — Symbolic complex number combinator layer.
 *
 * Inspired by FFTW's genfft/complex.ml (Frigo 2003), this module provides
 * a typed `cnum` = (real, imag) representation that flows through the
 * algebra of DFT codelets and pushes simplification opportunities down to
 * the Expr smart constructors.
 *
 * --- Why a separate layer? ---
 *
 * Before this module, the DFT code built `re` and `im` arrays separately
 * with raw `Expr.Mul`, `Expr.Add` constructors. That works, but when you
 * need to multiply by a constant scalar (e.g., a W5 internal constant)
 * a complex value that itself came from a cmul, the constants flow through
 * `re` and `im` independently — and Expr.mk_mul's rotation rule fires
 * naturally when an outer Const meets an inner Const that was placed there
 * by the previous cmul. That's how we close the algebraic gap with FFTW
 * (which gets 236 ops on R=25 vs our 383 pre-Cnum).
 *
 * --- Sign convention ---
 *
 * `cmul a b` computes the complex product (a.re + i·a.im)(b.re + i·b.im).
 * For forward DFTs (ω = exp(-2πi/N)), pass twiddles via [croot_of_unity_fwd].
 * For backward DFTs (ω = exp(+2πi/N)), use [croot_of_unity_bwd].
 *
 * --- Structure choice for cmul ---
 *
 * FFTW's cmul builds the result as `Plus[Times(a,c); Uminus(Times(b,d))]`
 * (n-ary Plus of two Times nodes). We use binary `Sub(Mul, Mul)` which is
 * equivalent — algsimp.ml flattens both into the same canonical form.
 *
 * The crucial design choice: keep the two Mul nodes at the leaves of the
 * Sub. This is what enables downstream `mk_mul k (.re)` to find a
 * `Mul(Const, Mul(Const, x))` pattern after distribution through Sub.
 * Compare to the older `const_cmul`'s tan-factored form
 * `Mul(cr, Sub(xr, Mul(ratio, xi)))`, which buries the constants inside
 * an outer Mul and prevents downstream simplification.
 *)

open Expr

type cnum = { re : expr; im : expr }

(* Constructors *)
let cnum re im = { re; im }
let czero = { re = Const 0.0; im = Const 0.0 }
let cone = { re = Const 1.0; im = Const 0.0 }
let cof_re x = { re = x; im = Const 0.0 }

(* Const complex: a + bi where a and b are floats.
 *   cconst 1.0 0.0  = one
 *   cconst 0.0 1.0  = i (imaginary unit)
 *)
let cconst (cr : float) (ci : float) : cnum = { re = Const cr; im = Const ci }

(* Negation: -(a + bi) = -a + (-b)i *)
let cneg (c : cnum) : cnum = { re = mk_neg c.re; im = mk_neg c.im }

(* Conjugation: conj(a + bi) = a - bi *)
let cconj (c : cnum) : cnum = { re = c.re; im = mk_neg c.im }

(* Addition: (a + bi) + (c + di) = (a+c) + (b+d)i *)
let cadd (a : cnum) (b : cnum) : cnum =
  { re = mk_add a.re b.re; im = mk_add a.im b.im }

(* Subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i *)
let csub (a : cnum) (b : cnum) : cnum =
  { re = mk_sub a.re b.re; im = mk_sub a.im b.im }

(* Scalar multiplication (real scalar * complex):
 *   k * (a + bi) = k·a + k·b·i
 *
 * Each component goes through mk_mul, which means if `a` or `b` is itself
 * a Mul(Const, _) (because cnum came from a previous cmul or another
 * cscale), the rotation rule in Expr.mk_mul fires and folds the constants
 * IF k is itself a Const. The savings compound through chains. *)
let cscale (k : expr) (c : cnum) : cnum =
  { re = mk_mul k c.re; im = mk_mul k c.im }

(* Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
 *
 * Built as Plus-of-Times form (the FFTW choice). The Mul nodes are at the
 * leaves of the Sub/Add, which is what enables the rotation rule to fire
 * when a downstream consumer multiplies by another Const. *)
let cmul (a : cnum) (b : cnum) : cnum =
  {
    re = mk_sub (mk_mul a.re b.re) (mk_mul a.im b.im);
    im = mk_add (mk_mul a.re b.im) (mk_mul a.im b.re);
  }

(* Multiplication by imaginary unit: i·(a + bi) = -b + ai
 * Free — just swaps components and negates the new real part. *)
let cmul_i (c : cnum) : cnum = { re = mk_neg c.im; im = c.re }

(* Multiplication by -i: -i·(a + bi) = b - ai *)
let cmul_negi (c : cnum) : cnum = { re = c.im; im = mk_neg c.re }

(* Twiddle: ω_N^k for forward DFT (= exp(-2πi·k/N)).
 *
 * Special-cased for trivial twiddles (k=0 → 1, and rotations by N/4 →
 * pure ±1 or ±i) so callers don't pay for spurious multiplications. *)
let croot_of_unity_fwd (k : int) (n : int) : cnum =
  let k' = ((k mod n) + n) mod n in
  (* normalize to [0, n) *)
  if k' = 0 then cone
  else if n mod 4 = 0 && k' = n / 4 then { re = Const 0.0; im = Const (-1.0) }
  else if n mod 2 = 0 && k' = n / 2 then { re = Const (-1.0); im = Const 0.0 }
  else if n mod 4 = 0 && k' = 3 * n / 4 then { re = Const 0.0; im = Const 1.0 }
  else begin
    let two_pi = 8.0 *. atan 1.0 in
    let theta = -1.0 *. two_pi *. float_of_int k' /. float_of_int n in
    { re = Const (cos theta); im = Const (sin theta) }
  end

(* Twiddle for backward DFT (= exp(+2πi·k/N)).
 * Same magnitudes, conjugated imaginary parts. *)
let croot_of_unity_bwd (k : int) (n : int) : cnum =
  cconj (croot_of_unity_fwd k n)

(* Convenience: pick the right twiddle based on a sign tag. *)
let croot_of_unity ~(sign : [ `Fwd | `Bwd ]) (k : int) (n : int) : cnum =
  match sign with
  | `Fwd -> croot_of_unity_fwd k n
  | `Bwd -> croot_of_unity_bwd k n

(* Build a cnum signal from a pair of (re, im) input-providing functions.
 * Used to adapt the (input_re, input_im) calling convention of the existing
 * DFT functions to the cnum-based ones. *)
let signal_of_re_im (input_re : int -> expr) (input_im : int -> expr) :
    int -> cnum =
 fun i -> { re = input_re i; im = input_im i }

(* Inverse: extract (re, im) arrays from a cnum array. *)
let split_re_im (xs : cnum array) : expr array * expr array =
  (Array.map (fun c -> c.re) xs, Array.map (fun c -> c.im) xs)
