#!/usr/bin/env python3
"""emit_rfft_jit.py — single-plan JIT executor for the rfft (real FFT) forward.

The rfft sibling of emit_jit.py (c2c). Given ONE plan (N, K, factors, variants, isa,
mode), emit a standalone C TU with one exported executor — a specialization of rfft.h's
forward executor where:

  * the stage loop is UNROLLED (one block per combine stage),
  * structural constants (r, m, np, Q, kmax, has_mid, S, leaf_r) are BAKED as literals
    (computed here exactly as rfft_plan_create_ex computes them),
  * the leaf / k0 / interior / terminator codelets are called by DIRECT SYMBOL instead
    of the plan's function pointers (p->leaf / st->k0 / st->hc / p->hcn).

Twiddle DATA (st->tw_re/tw_im, st->mid_c/mid_s) and the ping-pong planes are read from
the plan at runtime, so this is a pure EXECUTOR specialization (codelets link from the
rfft objects, same as the generic path).

Modes:
  packed  -> rfft_jit_exec        (packed halfcomplex out)   [rfft_execute_fwd_packed]
  natural -> rfft_jit_exec        (split out_re/out_im)       [rfft_execute_fwd_natural]
(one exported symbol name regardless of mode; the resolver picks the .c per mode.)

Validity: default Kb == K (lane-blocking OFF), where the slab + (q,lane) fold collapse
to a single iteration (vlf = Q*K). Other Kb -> generic fallback (in the prelude).

Variant -> interior codelet (matches rfft_plan_create_ex): 1 -> log3, else flat.
"""
import argparse


def stage_params(N, factors):
    """Per-stage (d, r, Q, np, m, kmax, has_mid), order nf-2..0. Mirrors plan_create_ex."""
    nf = len(factors)
    leaf_r = factors[nf - 1]
    S = N // leaf_r
    stages = []
    for d in range(nf - 2, -1, -1):
        r = factors[d]
        Q = 1
        for i in range(d):
            Q *= factors[i]
        npv = N // Q
        m = npv // r
        kmax = (m // 2 - 1) if (m % 2 == 0) else ((m - 1) // 2)
        has_mid = (m % 2 == 0)
        stages.append((d, r, Q, npv, m, kmax, has_mid))
    return leaf_r, S, stages


def hc_infix(v):    return "hc2hc_dit_log3_fwd" if v == 1 else "hc2hc_dit_fwd"
# Natural terminator: rfft_plan_create_ex sets p->hcn = hc2c_log3[r0] ? : hc2c[r0]
# (log3-PREFERRED, ignoring the variant array). Match it -> always log3; if a radix
# lacks hc2c_log3 the JIT link fails and the caller falls back to the generic path.
NAT_INFIX = "hc2c_nat_log3_fwd"


def codelet_externs(N, factors, variants, isa, mode):
    leaf_r, _S, stages = stage_params(N, factors)
    r2cf = set([leaf_r])
    hc, nat = set(), set()
    for (d, r, Q, npv, m, kmax, has_mid) in stages:
        r2cf.add(r)                                   # st->k0 is r2cf[r]
        if mode == "natural" and d == 0:
            if (m - 1) // 2 >= 1: nat.add((r, NAT_INFIX))
        elif kmax >= 1:
            hc.add((r, hc_infix(variants[d])))
    lines = []
    for r in sorted(r2cf):
        lines.append(f"extern void radix{r}_r2cf_{isa}(const double*, double*, double*, "
                     f"ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t);")
    for (r, inf) in sorted(hc):
        lines.append(f"extern void radix{r}_{inf}_{isa}(const double*, const double*, "
                     f"double*, double*, const double*, const double*, "
                     f"ptrdiff_t, ptrdiff_t, size_t);")
    for (r, inf) in sorted(nat):
        lines.append(f"extern void radix{r}_{inf}_{isa}(const double*, const double*, "
                     f"double*, double*, double*, double*, const double*, const double*, "
                     f"ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t);")
    return lines


def stage_block(d, r, Q, npv, m, kmax, has_mid, inf, isa, cur, nxt):
    """Packed combine stage (also the natural intermediate stages): k0 r2cf +
    interior hc2hc + mid, plane -> plane (or -> out at d==0 in packed)."""
    L = [f"    /* stage d={d}: radix {r} (m={m}, np={npv}, Q={Q}, kmax={kmax}) */",
         "    {",
         f"        const rfft_stage_t *st = &p->st[{d}];",
         f"        const ptrdiff_t QK  = (ptrdiff_t)((size_t){Q} * K);",
         f"        const ptrdiff_t QmK = (ptrdiff_t)((size_t){Q} * (size_t){m} * K);",
         f"        const size_t vlf = (size_t){Q} * K;",
         f"        radix{r}_r2cf_{isa}({cur}, {nxt}, {nxt} + NK, QK, QmK, -QmK, vlf);"]
    if kmax >= 1:
        L += [f"        for (int k = 1; k <= {kmax}; k++)",
              f"            radix{r}_{inf}_{isa}(",
              f"                {cur} + ((size_t){Q}*(size_t)({r}*k))*K,",
              f"                {cur} + ((size_t){Q}*(size_t)({r}*({m}-k)))*K,",
              f"                {nxt} + ((size_t){Q}*(size_t)k)*K,",
              f"                {nxt} + ((size_t){Q}*(size_t)({m}-k))*K,",
              f"                st->tw_re + (size_t)(k-1)*{r}, st->tw_im + (size_t)(k-1)*{r},",
              "                QK, QmK, vlf);"]
    if has_mid:
        L += [f"        rfft_mid_column({r}, {m}, {npv}, (size_t){Q}, K, vlf,",
              f"            {cur} + ((size_t){Q}*(size_t)({r}*({m}/2)))*K,",
              f"            st->mid_c, st->mid_s, 0, 0, {nxt}, NULL);"]
    L.append("    }")
    return L


def emit_body_packed(N, factors, variants, isa):
    leaf_r, S, stages = stage_params(N, factors)
    nf = len(factors)
    L = ["    if (p->Kb != p->K) { rfft_execute_fwd_packed(p, x, out); return; }",
         "    const size_t K = p->K;",
         f"    const size_t NK = (size_t){N} * K;"]
    leaf_dst = "out" if nf == 1 else "p->planeA"
    L += ["    /* LEAF (folded, vl = S*K) */", "    {",
          f"        const ptrdiff_t SK = (ptrdiff_t)((size_t){S} * K);",
          f"        radix{leaf_r}_r2cf_{isa}(x, {leaf_dst}, {leaf_dst} + NK, SK, SK, -SK, (size_t){S} * K);",
          "    }"]
    if nf == 1:
        return L
    cur = "p->planeA"
    for (d, r, Q, npv, m, kmax, has_mid) in stages:
        nxt = "out" if d == 0 else ("p->planeB" if cur == "p->planeA" else "p->planeA")
        L += stage_block(d, r, Q, npv, m, kmax, has_mid, hc_infix(variants[d]), isa, cur, nxt)
        cur = nxt
    return L


def emit_body_natural(N, factors, variants, isa):
    leaf_r, S, stages = stage_params(N, factors)
    nf = len(factors)
    nh = N // 2
    L = ["    if (p->Kb != p->K) { rfft_execute_fwd_natural(p, x, out_re, out_im); return; }",
         "    const size_t K = p->K;",
         f"    const size_t NK = (size_t){N} * K;",
         f"    const size_t nh = (size_t){nh};"]
    if nf == 1:
        L += [f"    radix{leaf_r}_r2cf_{isa}(x, p->planeA, p->planeA + NK, (ptrdiff_t)K, (ptrdiff_t)K, -(ptrdiff_t)K, K);",
              "    memcpy(out_re, p->planeA, K*8); memset(out_im, 0, K*8);",
              f"    for (size_t f = 1; f < (size_t){(N + 1)//2}; f++) {{",
              "        memcpy(out_re + f*K, p->planeA + f*K, K*8);",
              f"        memcpy(out_im + f*K, p->planeA + ((size_t){N}-f)*K, K*8);",
              "    }"]
        if N % 2 == 0:
            L.append("    memcpy(out_re + nh*K, p->planeA + nh*K, K*8); memset(out_im + nh*K, 0, K*8);")
        return L
    L += ["    /* LEAF (folded, vl = S*K) */", "    {",
          f"        const ptrdiff_t SK = (ptrdiff_t)((size_t){S} * K);",
          f"        radix{leaf_r}_r2cf_{isa}(x, p->planeA, p->planeA + NK, SK, SK, -SK, (size_t){S} * K);",
          "    }"]
    # intermediate stages d = nf-2 .. 1 (terminator is d==0, handled below)
    cur = "p->planeA"
    for (d, r, Q, npv, m, kmax, has_mid) in stages:
        if d == 0:
            break
        nxt = "p->planeB" if cur == "p->planeA" else "p->planeA"
        L += stage_block(d, r, Q, npv, m, kmax, has_mid, hc_infix(variants[d]), isa, cur, nxt)
        cur = nxt
    # stage 0: natural terminator
    d0, r0, Q0, np0, m0, kmax0, mid0 = stages[-1]
    kmaxT = (m0 - 1) // 2
    L += [f"    /* stage 0: natural terminator (radix {r0}, m={m0}) */", "    {",
          "        const rfft_stage_t *st = &p->st[0];",
          f"        const ptrdiff_t mK = (ptrdiff_t)((size_t){m0} * K);",
          f"        radix{r0}_r2cf_{isa}({cur}, p->nat_k0, p->nat_k0 + (size_t){r0}*K, (ptrdiff_t)K, (ptrdiff_t)K, -(ptrdiff_t)K, K);",
          "        memcpy(out_re, p->nat_k0, K*8); memset(out_im, 0, K*8);",
          f"        for (int sI = 1; sI < {(r0 + 1)//2}; sI++) {{",
          f"            memcpy(out_re + (size_t)sI*(size_t){m0}*K, p->nat_k0 + (size_t)sI*K, K*8);",
          f"            memcpy(out_im + (size_t)sI*(size_t){m0}*K, p->nat_k0 + (size_t)({r0}-sI)*K, K*8);",
          "        }"]
    if r0 % 2 == 0:
        L.append(f"        memcpy(out_re + nh*K, p->nat_k0 + (size_t){r0 // 2}*K, K*8); memset(out_im + nh*K, 0, K*8);")
    if kmaxT >= 1:
        L += [f"        for (int k = 1; k <= {kmaxT}; k++)",
              f"            radix{r0}_{NAT_INFIX}_{isa}(",
              f"                {cur} + ((size_t)({r0}*k))*K, {cur} + ((size_t)({r0}*({m0}-k)))*K,",
              "                out_re + (size_t)k*K, out_im + (size_t)k*K,",
              f"                out_re + (size_t)({m0}-k)*K, out_im + (size_t)({m0}-k)*K,",
              f"                st->tw_re + (size_t)(k-1)*{r0}, st->tw_im + (size_t)(k-1)*{r0},",
              "                (ptrdiff_t)K, mK, mK, K);"]
    if mid0:
        L += [f"        rfft_mid_column({r0}, {m0}, {np0}, 1, K, K,",
              f"            {cur} + ((size_t)({r0}*({m0}/2)))*K, st->mid_c, st->mid_s, 1, nh, out_re, out_im);"]
    L.append("    }")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--factors", required=True, help="comma list, leaf last (e.g. 16,16)")
    ap.add_argument("--variants", default="", help="comma codes (0/2 flat, 1 log3); default all flat")
    ap.add_argument("--isa", default="avx2")
    ap.add_argument("--mode", default="packed", choices=["packed", "natural"])
    ap.add_argument("--prelude", default="rfft_jit_prelude.h")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    factors = [int(x) for x in a.factors.split(",")]
    nf = len(factors)
    variants = ([int(x) for x in a.variants.split(",")] if a.variants else [0] * nf)
    assert len(variants) == nf, "variants length must match factors"

    if a.mode == "natural":
        sig = ("VFFT_JIT_EXPORT void rfft_jit_exec(const rfft_plan_t *p, const double *x,\n"
               "                                   double *out_re, double *out_im)")
        body = emit_body_natural(a.N, factors, variants, a.isa)
    else:
        sig = "VFFT_JIT_EXPORT void rfft_jit_exec(const rfft_plan_t *p, const double *x, double *out)"
        body = emit_body_packed(a.N, factors, variants, a.isa)

    out = [
        f"/* AUTO-GENERATED by emit_rfft_jit.py — single-plan rfft {a.mode}-forward JIT. */",
        f'#include "{a.prelude}"',
        "",
        "/* Codelet externs (self-contained: covers cold plans). */",
        *codelet_externs(a.N, factors, variants, a.isa, a.mode),
        "",
        "#if defined(_WIN32)",
        "#define VFFT_JIT_EXPORT __declspec(dllexport)",
        "#else",
        '#define VFFT_JIT_EXPORT __attribute__((visibility("default")))',
        "#endif",
        "",
        sig,
        "{",
        *body,
        "}",
    ]
    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w", newline="\n") as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
