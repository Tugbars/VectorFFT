(* emit_state.ml — the emitter's mutable mode surface.
 *
 * Every ref here is set by a driver (gen_main / codelet_oop /
 * emit_codelet itself) before body emission and read by the render
 * layer. Two blocks: the codelet-family signature flags, and the
 * M-project / per-pass emission mode refs. Physically single —
 * re-exported unchanged through Emit_render into Emit_c.
 *
 * Why refs and not parameters: the renderers have many call sites
 * scattered through emit_codelet's scheduler and spill variants;
 * threading ~20 mode parameters through each would dwarf the feature
 * code. The cost is the usual one — generation must stay
 * single-threaded, and every flag a driver sets must be reset before
 * the next codelet (gen_set generates the whole tree in one process).
 * ------------------------------------------------------------------
 * MODULE CARD (emit_state.ml — grep "MODULE CARD" for the full set)
 * ROLE: Signature-family flags (r2r / r2cf / r2cb / hc2c natural and
 * ranged / r2c_term family / strided) + tail ls_mode + M-project mode
 * refs (current_regalloc, fence-only, store-on-compute, emit
 * position).
 * PIPELINE: drivers set these; Emit_render reads them per node.
 * PUBLIC SURFACE (measured): zero direct Emit_state.X references —
 * all access goes through the Emit_c facade chain (heaviest setters:
 * gen_main's family wiring and codelet_oop's ls_mode handling).
 * DEPS: Isa(4), Regalloc(1) for types only.
 * GOTCHA: process-global single-threaded state; a flag left set by
 * one codelet silently changes the next codelet's ABI.
 * ------------------------------------------------------------------
 *)

(* t1p (per-position broadcast) twiddles: one scalar per (leg, position),
   broadcast across the vec_width batch lanes. Table is (R-1)*(me/VW) scalars,
   loaded with set1, vs t1's per-(position,lane) vector load (VW× larger).
   Set by emit_codelet for PerPositionTwiddles codelets, read below. *)
let current_tw_perpos : bool ref = ref false

(* LINEAR twiddle layout (row_major_engine.md §12.4 item 4a — the MKL pattern:
   one streaming cursor instead of (R-1) parallel strided rows). When > 0 the
   value is NLEGS = R-1 and the per-group twiddle address becomes
   tw_re[b*NLEGS + j*VW] — the table is packed in CONSUMPTION order (per
   group-quad: all legs' 4-vectors contiguous). Only valid for configs with
   no rem tail (UnitLeg load in the OOP family: the tail passes would index
   the flat layout). Set by emit_codelet from Codelet_oop.current_oop_tw_linear. *)
let current_tw_linear : int ref = ref 0

(* zil BLOCK-SPLIT twiddle records (codelet_zsplit; zil_pipeline_port.md §3).
   The zsplit families keep BOTH halves of a twiddle record in tw_re — the
   z ABI's tw_im slot is dead. A record is [c×VW][s×VW] (2·VW doubles):
   splat-pair sets for the mids (one record per leg, group-constant), packed
   per-column w¹ for the terminator (one record per VW columns, streamed).
   When Some off, Twiddle(j, re) renders loadu(&tw_re[off + j*2VW]) and
   Twiddle(j, im) renders loadu(&tw_re[off + j*2VW + VW]); `off` is a C
   expression prefix ("" for table-start records, "2*k" for the terminator's
   column-indexed stream). Set/cleared by Codelet_zsplit.emit_codelet only;
   default None leaves every existing family byte-identical. *)
let current_tw_zsplit : string option ref = ref None

(* Lean real-to-real ABI (notebook section 51, step 2): when true, the
 * OOP signature is `(const double *in, double *out, size_t K)` — the
 * convention dct.h's fast paths consume (python n8 vintage). Set by
 * gen_main for the real->real trig family {dct2, dct2_trigII, dct3,
 * dct4, dht}; rdft keeps the generic signature (complex output). The
 * body's array names map in_re->in, out_re->out; in_im/out_im/tw are
 * never referenced by these DAGs. *)
let r2r_signature : bool ref = ref false
let r2cf_signature : bool ref = ref false

(* r2cb (section 62): backward real leaf. Halfcomplex INPUT (in_re, in_im) ->
 * real OUTPUT (out_re only). Distinct from r2cf, which is real-in / split-out
 * with a reversed im stream. *)
let r2cb_signature : bool ref = ref false
let hc_strided : bool ref = ref false

(* n1_oop_strided (pack fix, section: r2c stage-0 copy deletion): true
 * strided-OOP n1 signature matching vfft_proto_n1_fn exactly —
 * (in_re, in_im, out_re, out_im, size_t is, size_t os, size_t vl),
 * legs j at in[j*is+v] / out[j*os+v]. Exists so the r2c fused first
 * stage (the one in!=out caller) doesn't pay the registry wrapper's
 * memcpy-then-in-place bridge. n1-only: no twiddles in the DAG. *)
let n1_oop_strided : bool ref = ref false

(* Interleaved-boundary strided variants (il_architecture.md section 6a4+):
   il_out — fwd rows codelet stores (re,im) pairs to out_z instead of split
   rows (interleave fused into the inverse-transpose postamble, plain
   stores). il_in — bwd rows codelet loads pairs (load-side lattice; not
   yet wired). *)
let strided_il_out : bool ref = ref false
let strided_r2c : bool ref = ref false
let strided_r2c_bwd : bool ref = ref false
let strided_il_in : bool ref = ref false

(* il_out with non-temporal (streaming) stores: z is written once and never
   re-read inside the transform, so NT bypasses RFO and cache pollution.
   Emits _mm_sfence() before function exit for cross-thread visibility. *)
let strided_ilo_nt : bool ref = ref false

(* Inplace-family IL boundary modes (il_architecture.md 6a11): the z lattice
   enters the emitter as render-level memoized emission — first scheduled
   touch of either side of input index j emits the shared z loads + both
   deinterleave permutes (lazy placement preserved); stores defer the re side
   one statement and fuse the (re,im) pair into full-width z stores. *)
let ip_il_in : bool ref = ref false
let ip_il_out : bool ref = ref false

(* memo: per-emit_body-pass. il_pending accumulates lattice statements to be
   flushed as a prefix of the next rendered node definition. *)
(* M6.1: il_seen moved into Emit_render.Scratch (per-emission). *)
(* M6.1: il_pending moved into Emit_render.Scratch (per-emission). *)
(* M6.1: il_stash moved into Emit_render.Scratch (per-emission). *)

(* M6.1: il_reset moved into Emit_render.Scratch (per-emission). *)

(* M6.1: il_take_pending moved into Emit_render.Scratch (per-emission). *)

(* D2 (section 69): hc2c NATURAL-split terminator. Sub-mode of
 * hc_strided (inputs/twiddles identical); overrides the signature
 * (FFTW khc2c-shaped 4-pointer outputs) and the output slot map.
 * sstar = constant regime boundary (design doc lemma): slots
 * s <= sstar are direct rows (Rp/Ip + s*osp), slots s > sstar are
 * conjugate-mirror rows (Rm/Im + (r-1-s)*osm). *)
let hc2c_natural : bool ref = ref false

(* hc2c_natural_bwd (c2r initiator, the INVERSE of hc2c_natural): the backward
 * natural codelet reads the SPLIT half-spectrum (Rp/Ip direct rows, Rm/Im
 * conjugate-mirror rows — the same sstar map as the forward, but on the INPUT
 * side) and writes the 2 PACKED cascade columns (out_re/out_im). I.e. the
 * forward natural's 2-packed-in / 4-split-out ABI, flipped: 4 split const-in /
 * 2 packed-out. The butterfly is dft_hc2c_dif ~sign:`Bwd (c2r's DIF-backward).
 * This lets a split-input c2r run the fast packed cascade with NO repack — the
 * mirror of how rfft_execute_fwd_natural gives split output at packed speed. *)
let hc2c_natural_bwd : bool ref = ref false

(* r2c_term (step-2 fusion): the fused forward terminator codelet. Dual-output
 * 2-input shape: reads Z[k] (input 0) and Z[m] (input 1), writes X[k]
 * (Xp pair) and X[m] (Xm pair), vectorized over vl lanes. Simpler than
 * hc2c_natural's r-way sstar split — exactly 2 outputs. *)
let r2c_term_signature : bool ref = ref false

(* r2c_term runtime-twiddle variant (slice 3b): one codelet for all
 * frequencies, loading W^f from a tw pointer. Adds tw_re/tw_im to the
 * signature and addresses Twiddle(0) as tw_re[0]/tw_im[0] (the executor
 * passes tw_re+f per call, so slot 0 is the current frequency's twiddle). *)
let r2c_term_rt : bool ref = ref false

(* model (b): the last-stage-fused terminator. r-leg inputs for two columns
 * (col k legs 0..r-1 at in[s*is_leg], col m-k legs r..2r-1 at in[(r+j)*is_leg]
 * via the is_leg stride and the is column offset), packed twiddle table (3r
 * slots, broadcast), 2r outputs: even slot s -> Xp[s*osp], odd -> Xm[s*osm]. *)
let r2c_term_laststage : bool ref = ref false
let r2c_term_ls_r : int ref = ref 0 (* radix r, for output slot count 2r *)

(* T1 (section 70, FFTW-structural): RANGED variant — the column
 * loop lives INSIDE the codelet (one call walks kcount columns,
 * pointers and twiddles advancing), amortizing call overhead and
 * making the twiddle walk a linear stream. Wraps the v-loop;
 * render layer unchanged (parameter pointers are mutated). *)
let hc_ranged : bool ref = ref false
let hc_ranged_r : int ref = ref 0
let hc2c_nat_sstar : int ref = ref 0
let hc2c_nat_r : int ref = ref 0

(* Arbitrary-K tail (docs/roadmap/arbitrary_k_vectorization): the mode the
 * body emitter is currently rendering under. Defaults to LS_vector so every
 * existing path (and the bulk loop) is byte-identical to before. The in-place
 * c2c tail sets this to (LS_masked m) around the masked remainder pass; it is
 * read by render_load (rio + per-lane twiddle loads) and emit_store. Broadcast
 * twiddles / constants go through set1_pd_str and are never masked. *)
(* M6.1: current_ls_mode moved into Emit_render.Scratch (per-emission). *)

(* === M3a regalloc consumption point ===
 *
 * Top-level mutable ref that carries the active register allocation
 * from emit_codelet (which computes it per scheduling site) to
 * render_node_def (which consumes it when emitting declarations).
 *
 * Why a top-level ref instead of a parameter: render_node_def has
 * ~10 call sites scattered through emit_codelet (different scheduler
 * paths, spill structure variants). Threading a parameter through all
 * of them would touch a lot of code for an opt-in feature. The ref is
 * set by emit_codelet at each scheduling site and reset to None at
 * exit, so other code paths see None.
 *
 * Threading note: ocaml refs are not thread-safe. Codelet generation
 * is single-threaded (gen_radix emits one codelet at a time), so this
 * is fine in practice. Documented limitation. *)
(* M6.1: current_regalloc moved into Emit_render.Scratch (per-emission). *)

(* === Fence-only emission mode ===
 *
 * When true, the non-pinned emission path uses Isa.fenced_decl
 * (`register __m512d tN = expr; asm volatile("" : "+v"(tN))`) instead
 * of Isa.const_decl (`const __m512d tN = expr;`). The fenced form
 * preserves the inline-asm scheduling barrier that constrains GCC's
 * scheduler to honor the codelet generator's SU+GH ordering, while
 * letting GCC pick the register freely.
 *
 * Set per-codelet at the top of emit_codelet based on the fence/pin
 * policy. See `docs/fence_pin_decomposition.md` for the empirical
 * basis. *)
(* M6.1: current_fence_only moved into Emit_render.Scratch (per-emission). *)

(* Duplication-clone barrier set (doc 65 §8): tags of un-CSE clones
 * produced by Algsimp.duplicate_uncse. The render layer must emit each
 * as a NON-const declaration with a "+x" compiler barrier (or gcc
 * re-CSEs the clone back into the original at -O3) and must never
 * inline them. Set by gen_main when VFFT_DUP=1; empty by default. *)
(* M6.1: dup_barrier_tags moved into Emit_render.Scratch (per-emission). *)

(* Store-on-compute (in-place 2-pass path). When true, each PASS 2 output
 * is stored to the output buffer the moment its value node is emitted,
 * with force_last_use set to that same position, instead of being held
 * live until its sub-DFT cluster flushes. Output value nodes are sinks
 * (compute_inline_set never inlines them), so their only use is the
 * store; storing at def is the natural last use and frees the register
 * immediately, lowering peak live in PASS 2. Safe because PASS 2 only
 * writes the output buffer, never reads it (all input reads retired in
 * PASS 1). PASS 1 outputs are left at end-of-PASS-1 (PASS 1 still reads
 * the input buffer, so an early write could alias an unread input). *)
let current_store_on_compute : bool ref = ref false

(* M5: schedule position of the node currently being emitted by
 * render_node_def. Set by the pass walkers in emit_c just before each
 * render_node_def call. Used by `v` (operand name renderer) to look
 * up name_overrides for reloaded values. *)
(* M6.1: current_emit_position moved into Emit_render.Scratch (per-emission). *)
