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
 * counter are module-global, exactly like Ir.reset (tags name the zN
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
  | ATw of int (* twp [i] — the T2 streamed VTW2 cursor          *)

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
  | CRotAdd of t * t
  (* a + i*y in ONE fused step: AVX2/SSE2 render = addsub(a, cflip y) —
     one shuffle + one vaddsubpd, no mask, legal in FWD kernels. Introduced
     for the wing construction's +i-side combines (the cadd/crot
     composition costs one extra uop). Never built by classic math paths,
     so flag-off emissions stay byte-identical. *)
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
(* ── M12a: THE PER-EMISSION CONTEXT ──
   The five emission-POLICY cells that lived here as module refs (tw_log3,
   tw_pre, st_turn, st_turn_gs, mono_spill_slots) plus the cx_math/
   cx_render env knobs are now ONE record, created per emission by the
   driver (C2c_il.emit / emit_k1) and threaded to the readers.  The old
   refs were set by the driver and NEVER reset — harmless in one-shot
   gen_radix, a leak the moment cil enters the warm gen_set process (the
   M12a precondition for the corpus entry).  Field semantics unchanged:
   tw_log3 = VTW2 sourcing for the CTwL renderer; tw_pre = pre-twiddle on
   a backward T2 (T2P); st_turn = corner-turned store (T2T); st_turn_gs =
   leg-strided turned store (T2TG, implies st_turn); mono_spill_slots =
   Belady S[] slots for the current MONO codelet (mutable — set mid-
   emission once the spill plan exists).  tangent / w32_combine /
   wing_enabled / rotfma capture their VFFT_CX_* envs at ctx creation
   (the kernel Knobs snapshot intent; tangent also ORs the --cil-tangent
   CLI flag the driver passes). *)
type ctx =
  { tw_log3 : bool
  ; tw_group : bool
    (* t2c: CTwL sources from the _wc/_ws names a GROUP prologue binds
       (per-(d,leg) records hoisted out of the column loop — the z-T1S
       broadcast strategy, il_native_design.md §6c). Same naming as log3,
       no derivation; the two are mutually exclusive by construction. *)
  ; tw_pre : bool
  ; st_turn : bool
  ; st_turn_gs : bool
  ; mutable mono_spill_slots : int
  ; tangent : bool
  ; w32_combine : bool
  ; wing_enabled : bool
  ; rotfma : bool
  }

let make_ctx ~tw_group ~tw_log3 ~tw_pre ~st_turn ~st_turn_gs ~tangent =
  { tw_log3
  ; tw_group
  ; tw_pre
  ; st_turn
  ; st_turn_gs
  ; mono_spill_slots = 0
  ; tangent = tangent || Sys.getenv_opt "VFFT_CX_TANGENT" = Some "1"
  ; w32_combine = Sys.getenv_opt "VFFT_CX_W32TG" = Some "1"
  ; wing_enabled = Sys.getenv_opt "VFFT_CX_WING" = Some "1"
  ; rotfma = Sys.getenv_opt "VFFT_CX_ROTFMA" = Some "1"
  }
;;

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
let crotadd a b = mk (CRotAdd (a, b))
let cfma c x e = mk (CFmaC (c, x, e))
let cfnma c x e = mk (CFnmaC (c, x, e))
let ctw c s x = mk (CTwC (c, s, x))
let ctwv w x = mk (CTwV (w, x))
let ctwl leg x = mk (CTwL (leg, x))
