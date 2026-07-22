#!/usr/bin/env python3
"""emit_jit.py - single-plan executor emitter for the VectorFFT JIT path.

Cross-platform (Windows + Linux). Given ONE plan (N, K, factors, variants, dir,
orient, isa), prints a standalone C translation unit with one exported executor
`vfft_proto_jit_exec`. The body is a sequence of VFFT_PROTO_STAGE_* macro calls,
byte-identical to what the OCaml build-time emitter bakes into plan_executors.h
(same macros -> same machine code -> same speed). The heavy lifting lives in
jit_prelude.h (the extracted STAGE_* macros + codelet externs + plan types).

Mapping (verified against generator/generated/plan_executors.h):
  DIT fwd: stage 0 -> STAGE_OUTER; stages 1..n-1 -> STAGE_{FLAT,LOG3,T1S}[v]
  DIF fwd: stages 0..n-2 -> STAGE_DIF_{FLAT,LOG3}[v]; last -> STAGE_DIF_OUTER
  DIT bwd: stages n-1..0 (REVERSE) -> STAGE_BWD       (variant-agnostic)
  DIF bwd: stages n-1..0 (REVERSE) -> STAGE_DIF_BWD   (variant-agnostic)
"""
import argparse

# 6a21: variant-bound DIF bwd. LOG3 twin of plan_executors.h's
# VFFT_PROTO_STAGE_DIF_BWD, extracted verbatim at patch time (only the codelet
# symbol swapped). Emitted into the TU when a DIF-bwd plan carries a LOG3
# stage, so the generated header stays untouched. Version-coupled via
# VFFT_PROTO_JIT_VERSION (5).
DIF_BWD_LOG3_MACRO = r'''
#define VFFT_PROTO_STAGE_DIF_BWD_LOG3(S, R, ISA) \
    if (start_stage <= (S)) { \
        const stride_stage_t *st = &plan->stages[S]; \
        const int    num_groups  = st->num_groups; \
        const size_t stride      = st->stride; \
        const int    *needs_tw    = st->needs_tw; \
        const size_t *group_base  = st->group_base; \
        double      **grp_tw_re   = st->grp_tw_re; \
        double      **grp_tw_im   = st->grp_tw_im; \
        for (int g = 0; g < num_groups; g++) { \
            double *base_re = re + group_base[g]; \
            double *base_im = im + group_base[g]; \
            if (!needs_tw[g]) { \
                radix##R##_n1_bwd_##ISA(base_re, base_im, NULL, NULL, \
                                          stride, slice_K); \
                continue; \
            } \
            /* K-split safe: block-broadcast the constant-per-leg grp_tw at \
             * this_K stride (see DIF_FLAT). t1_dif_bwd conjugates internally. */ \
            double tw_buf_re[((R)-1) * VFFT_PROTO_TW_BLOCK_K]; \
            double tw_buf_im[((R)-1) * VFFT_PROTO_TW_BLOCK_K]; \
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) { \
                size_t this_K = (slice_K - kb < VFFT_PROTO_TW_BLOCK_K) \
                                ? (slice_K - kb) : VFFT_PROTO_TW_BLOCK_K; \
                for (int j = 0; j < ((R)-1); j++) \
                    _stride_broadcast_2(tw_buf_re + (size_t)j * this_K, \
                                        tw_buf_im + (size_t)j * this_K, this_K, \
                                        grp_tw_re[g][(size_t)j * full_K], \
                                        grp_tw_im[g][(size_t)j * full_K]); \
                radix##R##_t1_dif_log3_bwd_##ISA(base_re + kb, base_im + kb, \
                                              tw_buf_re, tw_buf_im, stride, this_K); \
            } \
        } \
    }
'''


DIT_VARIANT = {0: "FLAT", 1: "LOG3", 2: "T1S"}      # 3=BUF falls back upstream
DIF_VARIANT = {0: "DIF_FLAT", 1: "DIF_LOG3"}        # DIF has no T1S/BUF

# Codelet symbol infix per variant (DIT forward). Matches plan_executors.h externs:
#   OUTER -> radix{R}_n1_{dir}_{isa},  FLAT -> radix{R}_t1_dit_{dir}_{isa},
#   LOG3  -> radix{R}_t1_dit_log3_{dir}_{isa},  T1S -> radix{R}_t1s_dit_{dir}_{isa}
DIT_CODELET_INFIX = {0: "t1_dit", 1: "t1_dit_log3", 2: "t1s_dit"}
DIF_CODELET_INFIX = {0: "t1_dif", 1: "t1_dif_log3"}   # DIF: FLAT/LOG3 only (no T1S)


def codelet_names(factors, variants, orient, direction, isa):
    """Unique codelet symbols the executor calls — emitted as self-contained
    externs so a COLD plan (radixes absent from plan_executors.h's baked extern
    set) compiles. EVERY stage references its n1 codelet (the STAGE macros call
    radix{R}_n1 for needs_tw=0 groups), PLUS the per-stage twiddle codelet:
    DIT twiddles stages 1..n-1; DIF twiddles stages 0..n-2 (OUTER = last)."""
    nf = len(factors)
    names = [f"radix{factors[s]}_n1_{direction}_{isa}" for s in range(nf)]   # n1 every stage
    if direction == "bwd":
        # Backward is VARIANT-AGNOSTIC: STAGE_BWD always calls t1s_dit_bwd (+ a
        # leg0-conj macro, not a codelet); STAGE_DIF_BWD always calls t1_dif_bwd.
        if orient != "dit" and direction == "bwd" and 1 in variants:
            for r in set(f for f, v in zip(factors, variants) if v == 1):
                names.append(f"radix{r}_t1_dif_log3_bwd_{isa}")
        # Both read the canonical twiddle layout (tw_scalar / grp_tw) regardless
        # of the forward variant, so the fwd `variants` don't pick the codelet.
        # Every stage's macro references the tw codelet (compiled, not just the
        # taken branch), so emit it for ALL stages — mirrors the baked bwd.
        tw = "t1s_dit" if orient == "dit" else "t1_dif"
        for s in range(nf):
            names.append(f"radix{factors[s]}_{tw}_bwd_{isa}")
    elif orient == "dit":
        for s in range(1, nf):
            names.append(f"radix{factors[s]}_{DIT_CODELET_INFIX[variants[s]]}_{direction}_{isa}")
    else:  # dif fwd
        for s in range(nf - 1):
            names.append(f"radix{factors[s]}_{DIF_CODELET_INFIX[variants[s]]}_{direction}_{isa}")
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def extern_decls(names):
    # Unnamed params; re-declares plan_executors.h's named externs harmlessly
    # (same types) for baked codelets, and supplies the missing ones for cold cells.
    return [f"extern void {n}(double *, double *, const double *, "
            f"const double *, size_t, size_t);" for n in names]


def stage_line(suffix, s, r, isa):
    name = ("VFFT_PROTO_STAGE_" + suffix).ljust(22)
    return f"    {name}({s}, {r:2d}, {isa})"


def emit_body(factors, variants, orient, isa, direction="fwd"):
    nf = len(factors)
    lines = ["    (void)full_K;"]
    if direction == "bwd":
        # Backward walks stages in REVERSE (output -> input), mirroring the
        # generic/baked bwd executors. DIT -> STAGE_BWD (fused t1s_dit_bwd +
        # leg0-conj; no generic variant counterpart, stays plain). DIF bwd is
        # VARIANT-BOUND since 6a21: LOG3 stages emit STAGE_DIF_BWD_LOG3
        # (matches the generic tier's st->t1_bwd binding; deletes the
        # DIT->jit/DIF->core selection rule).
        macro = "BWD" if orient == "dit" else "DIF_BWD"
        for s in range(nf - 1, -1, -1):
            m = macro
            if orient != "dit" and variants[s] == 1:
                m = "DIF_BWD_LOG3"
            lines.append(stage_line(m, s, factors[s], isa))
        return lines
    if orient == "dit":
        lines.append(stage_line("OUTER", 0, factors[0], isa))
        for s in range(1, nf):
            lines.append(stage_line(DIT_VARIANT[variants[s]], s, factors[s], isa))
    else:  # dif: inner stages first, OUTER last
        for s in range(0, nf - 1):
            lines.append(stage_line(DIF_VARIANT[variants[s]], s, factors[s], isa))
        lines.append(stage_line("DIF_OUTER", nf - 1, factors[nf - 1], isa))
    return lines




def guard(body):
    """Wrap each STAGE line in an `if (S < stop_stage)` for the range flavor.
    S is the literal stage index already present in the line."""
    out = []
    for ln in body:
        t = ln.strip()
        if t.startswith("VFFT_PROTO_STAGE_"):
            s_idx = t.split("(", 1)[1].split(",", 1)[0].strip()
            out.append(f"    if ({s_idx} < stop_stage) {{ {t} }}")
        else:
            out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--factors", required=True, help="comma list, e.g. 4,4,4,32,64")
    ap.add_argument("--variants", default="", help="comma codes; default all T1S(dit)/FLAT(dif)")
    ap.add_argument("--isa", default="avx2")
    ap.add_argument("--dir", default="fwd", choices=["fwd", "bwd"])
    ap.add_argument("--orient", default="dit", choices=["dit", "dif"])
    ap.add_argument("--prelude", default="jit_prelude.h")
    ap.add_argument("--out", default="", help="write to this file instead of stdout")
    ap.add_argument("--body-only", action="store_true", help="emit only the macro body (for diffs)")
    a = ap.parse_args()

    def emit(text):
        if a.out:
            with open(a.out, "w", newline="\n") as fh:
                fh.write(text + "\n")
        else:
            print(text)

    factors = [int(x) for x in a.factors.split(",")]
    nf = len(factors)
    if a.variants:
        variants = [int(x) for x in a.variants.split(",")]
    else:
        variants = [(2 if a.orient == "dit" else 0)] * nf
    assert len(variants) == nf, "variants length must match factors"

    body = emit_body(factors, variants, a.orient, a.isa, a.dir)
    if a.body_only:
        emit("\n".join(body))
        return

    names = codelet_names(factors, variants, a.orient, a.dir, a.isa)
    out = [
        "/* AUTO-GENERATED by emit_jit.py - single-plan JIT executor. */",
        f'#include "{a.prelude}"',
        *( [DIF_BWD_LOG3_MACRO]
           if (a.dir == "bwd" and a.orient == "dif" and 1 in variants) else [] ),
        "",
        "/* Codelet externs (self-contained: covers cold plans absent from plan_executors.h). */",
        *extern_decls(names),
        "",
        "#if defined(_WIN32)",
        "#define VFFT_JIT_EXPORT __declspec(dllexport)",
        "#else",
        '#define VFFT_JIT_EXPORT __attribute__((visibility("default")))',
        "#endif",
        "",
        "/* Range flavor: run stages S with start_stage <= S < stop_stage, in the",
        " * direction's natural walk order. The per-line stop guard composes with",
        " * the start gate INSIDE the STAGE macros; both are compile-time-constant",
        " * comparisons per line. Classic 6-arg symbol = stop_stage-saturated call",
        " * (cache-compatible: old .so files simply lack this symbol). */",
        "VFFT_JIT_EXPORT void vfft_proto_jit_exec_range(const stride_plan_t *plan,",
        "                                               double *re, double *im,",
        "                                               size_t slice_K, size_t full_K,",
        "                                               int start_stage, int stop_stage)",
        "{",
    ]
    out.extend(guard(body))
    out.extend([
        "}",
        "",
        "VFFT_JIT_EXPORT void vfft_proto_jit_exec(const stride_plan_t *plan,",
        "                                         double *re, double *im,",
        "                                         size_t slice_K, size_t full_K,",
        "                                         int start_stage)",
        "{",
        "    vfft_proto_jit_exec_range(plan, re, im, slice_K, full_K,",
        "                              start_stage, 0x7fffffff);",
        "}",
    ])
    emit("\n".join(out))


if __name__ == "__main__":
    main()
