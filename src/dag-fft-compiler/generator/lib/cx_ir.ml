(* cx_ir.ml — the packed-complex IR of the full-IL (cil) family.
 *
 * Split out of codelet_cil.ml (Phase 0 decomposition, 2026-08-09,
 * byte-identity gated). One node type, hash-consed: structural equality
 * becomes tag equality, which is CSE (mirrors Ir.hashcons). Also hosts the
 * emission-STATE refs (tw_log3 / tw_pre / st_turn / st_turn_gs) — they are
 * render/store-form state, not IR state, but they live beside the IR so
 * every cx_* module sees one copy.
 * MODULE CARD
 * ROLE: cx_kind + t + hash-cons (mk/reset) + smart constructors + state refs.
 * GOTCHA 1: `reset ()` MUST run before each codelet/pass — table and tag
 * counter are module-global, exactly like Algsimp.reset (tags name the zN
 * C locals, so a stale table leaks names across brace scopes). *)

(* ═══════════════════════════════════════════════════════════════
 *  THE COMPLEX IR
 * ═══════════════════════════════════════════════════════════════ *)

(* ── Symbolic addresses — the memory world, as DATA ─────────────────────
 * One constructor per address FORM the cil family emits. The runtime names
 * (k, Ls, OLs, OGs, twp) are fixed by the frozen z ABI, so a form + its
 * compile-time ints IS the address; rendering to a C string is cx_render's
 * job. `col` on the turned forms selects column k (0) or k+1 (1) — the
 * corner-turn writes two columns per iteration. *)
type caddr =
  | AZinLeg of int (* zin [2*((size_t)l*Ls  + k)]                  *)
  | AZoutLeg of int (* zout[2*((size_t)l*OLs + k)]                  *)
  | AZoutTurn of int * int (* (l, col)  zout[2*(((size_t)k+c)*OLs + l)]     *)
  | AZoutTurnG of int * int (* (l, col)  ... + (size_t)l*OGs)]  t2tg scatter  *)
  | AS of int (* S[i]  — the blocked spill plane, flat doubles *)
  | AP of int (* P[i]  — emit_k1's stage plane                 *)
  | AZinAbs of int (* zin [i] — emit_k1 absolute (no k)             *)
  | AZoutAbs of int (* zout[i] — emit_k1 absolute                    *)

type cx_kind =
  | CIn of int (* input leg i (a packed-complex load) *)
  | CLoad of caddr
  (* a load with its ADDRESS in the DAG — the complete-IR replacement for
     CIn + the hand load edge. Same is_load/latency treatment as CIn. *)
  | CStore of caddr * t
  (* a store node: address + the value it sinks. First-class so the
     scheduler CAN see stores (Node.is_store, the B2 hook) — whether it
     SCHEDULES them is the placement policy's choice, not the IR's. *)
  | CTurn of t * t * int
  (* permute2f128(a, b, imm) — the corner-turn lane regroup (0x20/0x31).
     In the DAG so turned stores are data, not a hand-printed edge. *)
  | CLo of t
  | CHi of t
  (* 128-bit halves (castpd256_pd128 / extractf128 1) — the odd-leg and
     leg-strided scatter halves. *)
  | CAdd of t * t
  | CSub of t * t
  | CNeg of t
  (* -x : negate BOTH lanes (complex negation). The algebraic atom the
     rewrite passes need (mirrors Ir's NK_Neg): dedup_sub_pairs rewrites
     Sub(b,a) into Neg(Sub(a,b)) so mirrored subtractions share one node.
     Never constructed by the math builders — pass-introduced only, so its
     absence keeps every existing kernel byte-identical. *)
  | CRotNI of t (* x * (-i) : cflip then negate the IM lane *)
  | CRotPI of t
  (* x * (+i) : cflip then negate the RE lane. The backward twin of
     CRotNI — an inverse transform's quarter-turn goes the other way.
     (a+bi)*(+i) = -b + ai, so from cflip = [b,a] we negate lane 0. *)
  | CFmaC of float * t * t (* c*x + e,  real scalar c *)
  | CFnmaC of float * t * t (* -c*x + e, real scalar c *)
  | CTwC of float * float * t (* x * (c + i*s), emit-time constants *)
  | CTwV of (float * float) array * t
  (* x * w, with a DIFFERENT emit-time constant per complex lane. The K=1
     fused kernel needs this: one vector holds two DIFFERENT output indices
     k1 of the same transform, so their twiddles w_N^{k1*j2} differ. Still
     zero runtime cost — the whole [c,c,c',c'][-s,+s,-s',+s'] pair is a
     file-scope VLIT, exactly like CTwC's. Array length = vec_width/2. *)
  | CTwL of int * t
  (* x * w[leg], w LOADED from the streamed VTW2 table — the bailey2 t2
     mid. Same BYTW2 shape as CTwC, but cvec/svec come from the runtime
     cursor `twp` instead of file-scope VLIT constants. The int is the
     LEG index; the record offset is (leg-1)*2*VW because leg 0 is
     untwiddled and each record is [c×VW][s×VW] (cos-first, sign-folded
     — one data-side shuffle, zero table-side work). *)

and t =
  { tag : int
  ; node : cx_kind
  }

(* Hash-consing: structural equality becomes tag equality, which is what
 * gives us CSE for free (the shared ±i rotations and repeated subsums in a
 * radix-8 body dedup automatically). Mirrors Ir.hashcons. *)
(* Twiddle-sourcing policy for the streamed VTW2 table, consulted by the
   CTwL renderer. A ref rather than a threaded parameter because it is
   emission state, not IR state — the DAG is identical either way; only how
   each leg's record is OBTAINED changes. Default false keeps every existing
   kernel byte-identical. *)
let tw_log3 = ref false

(* PRE-TWIDDLE ON A BACKWARD T2 (the "T2P" kind).
   Twiddle POSITION is normally derived from DIRECTION: forward pre-twiddles
   (w . x, then DFT), backward post-twiddles (IDFT, then conj(w) . y). Those
   are the only two combinations the emitter could express.

   The pure-IL two-pass INVERSE needs the third: PRE-twiddle with a BACKWARD
   butterfly. Without it the diagonal has to run as a separate scalar sweep
   over the whole scratch plane, which measures 26-56% of the backward's total
   time (build_tuned/benches/il2p_bwd_gate.c). Fusing it here does not remove
   the multiply -- it removes the extra read+write of the plane and does the
   arithmetic in-register, vectorized.

   Position and direction are INDEPENDENT properties of the kernel; tying them
   to `dir` was the accident. Default false keeps every existing kernel
   byte-identical. *)
let tw_pre = ref false

(* CORNER-TURNED STORE ON A T2 (the "T2T" kind).
   Store FORM is normally derived from KIND: N1/T2 store leg-major (straight),
   N1T fuses the four-step transpose into its stores. Like twiddle position,
   that coupling is an accident — the two are independent.

   The pure-IL inverse decomposition (transform R1 first, then R2) needs
   a kernel that carries the twiddle AND turns on store — t2t, THE canonical
   backward flat codelet. Default false keeps every existing kernel
   byte-identical. *)
let st_turn = ref false

(* LEG-STRIDED TURNED STORE (the "T2TG" kind, symbol tag `g`).
   The plain turned store hard-codes legs at stride 1: (leg p, col k) ->
   zout[2*(k*OLs + p)]. The 3-STAGE odd-chain BACKWARD (docs/roadmap/
   il_odd_chain.md) needs its middle stage to interleave leg groups from
   DIFFERENT calls: the clean l' = e + A*f split forces legs at stride A,
   i.e. zout[2*(k*OLs + p*OGs)] — so the g variant wires the otherwise
   `(void)`'d OGs argument as the turned store's LEG STRIDE. OGs=1
   reproduces t2t's addressing (but t2t stays its own emission — this flag
   emits a SEPARATE symbol precisely so every existing kernel remains
   byte-identical). Implies st_turn. *)
let st_turn_gs = ref false

let hcons : (cx_kind, t) Hashtbl.t = Hashtbl.create 256
let next_tag = ref 0

let reset () =
  Hashtbl.reset hcons;
  next_tag := 0
;;

let mk (nk : cx_kind) : t =
  match Hashtbl.find_opt hcons nk with
  | Some e -> e
  | None ->
    let e = { tag = !next_tag; node = nk } in
    incr next_tag;
    Hashtbl.add hcons nk e;
    e
;;

let cin i = mk (CIn i)
let cload a = mk (CLoad a)
let cstore a v = mk (CStore (a, v))
let cturn a b imm = mk (CTurn (a, b, imm))
let clo a = mk (CLo a)
let chi a = mk (CHi a)
let cadd a b = mk (CAdd (a, b))
let csub a b = mk (CSub (a, b))
let cneg a = mk (CNeg a)
let crot a = mk (CRotNI a)
let crotp a = mk (CRotPI a)
let cfma c x e = mk (CFmaC (c, x, e))
let cfnma c x e = mk (CFnmaC (c, x, e))
let ctw c s x = mk (CTwC (c, s, x))
let ctwv w x = mk (CTwV (w, x))
let ctwl leg x = mk (CTwL (leg, x))
