#!/usr/bin/env python3
"""emit_c2r_jit.py — single-plan JIT executor for the c2r (inverse real FFT).

The inverse sibling of emit_rfft_jit.py. Given ONE plan (N, K, factors, variants, isa),
emit a specialization of c2r.h's c2r_execute_packed where the stage loop is UNROLLED,
structural constants are BAKED, and the codelets are called by DIRECT SYMBOL:

  backward: stages d = 0 .. nf-2 (DIF backward) then leaf (r2cb) LAST.
    stage d: stage_dc (r2cb DC) + interior hc2hc_dif_bwd + inverse-mid; plane -> plane
    leaf:    r2cb (halfcomplex -> real) reading the final plane -> out

Stage params/twiddles come from p->base (an rfft_plan_t: st[d].tw_re/tw_im, planeA/B);
the inverse-mid tables come from p->mid_inv[d] (c2r-specific). Both read at runtime, so
this is a pure executor specialization (codelets link from the c2r objects).

Validity: default Kb == K (folded, vlf = Q*K). Other Kb -> generic c2r_execute_packed.
Variant -> interior codelet: 1 -> hc2hc_dif_log3_bwd, else hc2hc_dif_bwd.
"""
import argparse


def stages_asc(N, factors):
    """Combine stages d = 0 .. nf-2 (ascending), each (d, r, Q, np, m, kmax, has_mid)."""
    nf = len(factors)
    leaf_r = factors[nf - 1]
    S = N // leaf_r
    out = []
    for d in range(0, nf - 1):
        r = factors[d]
        Q = 1
        for i in range(d):
            Q *= factors[i]
        npv = N // Q
        m = npv // r
        kmax = (m // 2 - 1) if (m % 2 == 0) else ((m - 1) // 2)
        out.append((d, r, Q, npv, m, kmax, (m % 2 == 0)))
    return leaf_r, S, out


def hc_infix(v): return "hc2hc_dif_log3_bwd" if v == 1 else "hc2hc_dif_bwd"


def codelet_externs(N, factors, variants, isa):
    leaf_r, _S, st = stages_asc(N, factors)
    r2cb = set([leaf_r])
    hc = set()
    for (d, r, Q, npv, m, kmax, has_mid) in st:
        r2cb.add(r)                              # stage_dc is r2cb[r]
        if kmax >= 1: hc.add((r, hc_infix(variants[d])))
    lines = []
    for r in sorted(r2cb):
        lines.append(f"extern void radix{r}_r2cb_{isa}(const double*, const double*, double*, "
                     f"ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t);")
    for (r, inf) in sorted(hc):
        lines.append(f"extern void radix{r}_{inf}_{isa}(const double*, const double*, "
                     f"double*, double*, const double*, const double*, "
                     f"ptrdiff_t, ptrdiff_t, size_t);")
    return lines


def emit_body(N, factors, variants, isa):
    leaf_r, S, st = stages_asc(N, factors)
    nf = len(factors)
    L = ["    const rfft_plan_t *b = p->base;",
         "    if (p->Kb != b->K) { c2r_execute_packed(p, in, out); return; }",
         "    const size_t K = b->K;",
         f"    const size_t NK = (size_t){N} * K;"]
    if nf == 1:
        L += [f"    {{ const ptrdiff_t SK = (ptrdiff_t)((size_t){S} * K);",
              f"      radix{leaf_r}_r2cb_{isa}(in, in + NK, out, SK, -SK, SK, (size_t){S} * K); }}"]
        return L
    src = "in"
    for (d, r, Q, npv, m, kmax, has_mid) in st:
        dst = "b->planeA" if (d % 2 == 0) else "b->planeB"
        inf = hc_infix(variants[d])
        L += [f"    /* stage d={d}: radix {r} (m={m}, np={npv}, Q={Q}, kmax={kmax}) */",
              "    {",
              f"        const rfft_stage_t *st = &b->st[{d}];",
              f"        const ptrdiff_t QK  = (ptrdiff_t)((size_t){Q} * K);",
              f"        const ptrdiff_t QmK = (ptrdiff_t)((size_t){Q} * (size_t){m} * K);",
              f"        const size_t vlf = (size_t){Q} * K;",
              f"        radix{r}_r2cb_{isa}({src}, {src} + NK, {dst}, QmK, -QmK, QK, vlf);"]
        if kmax >= 1:
            L += [f"        for (int k = 1; k <= {kmax}; k++)",
                  f"            radix{r}_{inf}_{isa}(",
                  f"                {src} + ((size_t){Q}*(size_t)k)*K,",
                  f"                {src} + ((size_t){Q}*(size_t)({m}-k))*K,",
                  f"                {dst} + ((size_t){Q}*(size_t)({r}*k))*K,",
                  f"                {dst} + ((size_t){Q}*(size_t)({r}*({m}-k)))*K,",
                  f"                st->tw_re + (size_t)(k-1)*{r}, st->tw_im + (size_t)(k-1)*{r},",
                  "                QmK, QK, vlf);"]
        if has_mid:
            L += [f"        c2r_mid_inv_column({r}, {m}, (size_t){Q}, K, vlf, {src}, p->mid_inv[{d}],",
                  f"            {dst} + ((size_t){Q}*(size_t)({r}*({m}/2)))*K);"]
        L.append("    }")
        src = dst
    L += [f"    {{ const ptrdiff_t SK = (ptrdiff_t)((size_t){S} * K);",
          f"      radix{leaf_r}_r2cb_{isa}({src}, {src} + NK, out, SK, -SK, SK, (size_t){S} * K); }}"]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--factors", required=True, help="comma list, leaf last")
    ap.add_argument("--variants", default="", help="comma codes (0/2 flat, 1 log3); default all flat")
    ap.add_argument("--isa", default="avx2")
    ap.add_argument("--prelude", default="c2r_jit_prelude.h")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    factors = [int(x) for x in a.factors.split(",")]
    nf = len(factors)
    variants = ([int(x) for x in a.variants.split(",")] if a.variants else [0] * nf)
    assert len(variants) == nf, "variants length must match factors"

    out = [
        "/* AUTO-GENERATED by emit_c2r_jit.py — single-plan c2r (inverse real FFT) JIT. */",
        f'#include "{a.prelude}"',
        "",
        "/* Codelet externs (self-contained: covers cold plans). */",
        *codelet_externs(a.N, factors, variants, a.isa),
        "",
        "#if defined(_WIN32)",
        "#define VFFT_JIT_EXPORT __declspec(dllexport)",
        "#else",
        '#define VFFT_JIT_EXPORT __attribute__((visibility("default")))',
        "#endif",
        "",
        "VFFT_JIT_EXPORT void c2r_jit_exec(const c2r_plan_t *p, const double *in, double *out)",
        "{",
        *emit_body(a.N, factors, variants, a.isa),
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
