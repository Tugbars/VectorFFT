#!/usr/bin/env python3
"""
gen_radix32_notw.py — Twiddle-less DFT-32 codegen for AVX-512 and AVX2

Pure DFT-32 butterfly with no inter-stage twiddles.
Use as first/last stage where external twiddles are all 1,
or for standalone batch DFT-32.

8×4 decomposition, fwd + bwd.
  AVX-512: NFUSE=8 (entire last sub-FFT in registers, zero spills)
  AVX2:    NFUSE=2 (12/16 ymm at pass-2 peak)

Usage:
  python3 gen_radix32_notw.py avx512 > fft_radix32_avx512_notw.h
  python3 gen_radix32_notw.py avx2   > fft_radix32_avx2_notw.h
"""

import math, sys

N, N1, N2 = 32, 8, 4

# ── Twiddle analysis (internal W32 only — part of DFT-32 itself) ──

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8*e) % tN == 0:
        o = (8*e)//tN
        t = ['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True, t[o%8]
    return False, 'cmul'

def collect_internal_twiddles():
    tw = set()
    for n2 in range(1,N2):
        for k1 in range(1,N1):
            e=(n2*k1)%N
            _,t=twiddle_is_trivial(e,N)
            if t=='cmul': tw.add((e,N))
    return tw


# ── Target configuration ──

class Target:
    def __init__(self, isa):
        if isa == 'avx512':
            self.isa = 'avx512'
            self.vtype = '__m512d'
            self.vw = 8           # doubles per vector
            self.nfuse = 8        # entire last sub-FFT in regs
            self.align = 64
            self.prefix = '_mm512'
            self.attr = 'avx512f,avx512dq,fma'
            self.suffix = 'avx512'
            self.ld_macro = 'R32N5_LD'
            self.st_macro = 'R32N5_ST'
            self.guard = 'FFT_RADIX32_AVX512_NOTW_H'
        elif isa == 'avx2':
            self.isa = 'avx2'
            self.vtype = '__m256d'
            self.vw = 4
            self.nfuse = 2        # 12/16 ymm at peak
            self.align = 32
            self.prefix = '_mm256'
            self.attr = 'avx2,fma'
            self.suffix = 'avx2'
            self.ld_macro = 'R32NA_LD'
            self.st_macro = 'R32NA_ST'
            self.guard = 'FFT_RADIX32_AVX2_NOTW_H'
        else:
            raise ValueError(f"Unknown ISA: {isa}")


# ── Emitter ──

class E:
    def __init__(self, tgt):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
        self.t = tgt
        self.vt = tgt.vtype
        self.p = tgt.prefix
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def add(s,a,b): return f"{s.p}_add_pd({a},{b})"
    def sub(s,a,b): return f"{s.p}_sub_pd({a},{b})"
    def mul(s,a,b): return f"{s.p}_mul_pd({a},{b})"
    def neg(s,a):   return f"{s.p}_xor_pd({a},sign_flip)"
    def fma(s,a,b,c): return f"{s.p}_fmadd_pd({a},{b},{c})"
    def fms(s,a,b,c): return f"{s.p}_fmsub_pd({a},{b},{c})"

    def emit_load(s,v,n,k_expr="k"):
        s.o(f"{v}_re = LD(&in_re[{n}*K+{k_expr}]);")
        s.o(f"{v}_im = LD(&in_im[{n}*K+{k_expr}]);")
    def emit_store(s,v,m,k_expr="k"):
        s.o(f"ST(&out_re[{m}*K+{k_expr}],{v}_re);")
        s.o(f"ST(&out_im[{m}*K+{k_expr}],{v}_im);")
    def emit_spill(s,v,slot):
        s.o(f"{s.p}_store_pd(&spill_re[{slot}*{s.t.vw}],{v}_re);")
        s.o(f"{s.p}_store_pd(&spill_im[{slot}*{s.t.vw}],{v}_im);")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = {s.p}_load_pd(&spill_re[{slot}*{s.t.vw}]);")
        s.o(f"{v}_im = {s.p}_load_pd(&spill_im[{slot}*{s.t.vw}]);")
        s.reload_c+=1

    def emit_radix8(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        s.o(f"{{ {s.vt} e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        s.o(f"  {s.vt} t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={s.add(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={s.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t2r={s.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={s.add(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={s.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  e0r={s.add('t0r','t2r')}; e0i={s.add('t0i','t2i')};")
        s.o(f"  e2r={s.sub('t0r','t2r')}; e2i={s.sub('t0i','t2i')};")
        ja, js = ('add','sub') if fwd else ('sub','add')
        s.o(f"  e1r={getattr(s,ja)('t1r','t3i')}; e1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  e3r={getattr(s,js)('t1r','t3i')}; e3i={getattr(s,ja)('t1i','t3r')};")
        s.o(f"  {s.vt} o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        s.o(f"  o1r={getattr(s,ja)('t1r','t3i')}; o1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  o3r={getattr(s,js)('t1r','t3i')}; o3i={getattr(s,ja)('t1i','t3r')};")
        if fwd:
            s.o(f"  t0r={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  t0i={s.mul(s.sub('o1i','o1r'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r=o2i; t0i={s.neg('o2r')};")
            s.o(f"  o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.mul(s.sub('o3i','o3r'),'sqrt2_inv')};")
            s.o(f"  t0i={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        else:
            s.o(f"  t0r={s.mul(s.sub('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  t0i={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r={s.neg('o2i')}; t0i=o2r;")
            s.o(f"  o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  t0i={s.mul(s.sub('o3r','o3i'),'sqrt2_inv')};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        for i,j in [(0,4),(1,5),(2,6),(3,7)]:
            s.o(f"  {v[i]}_re={s.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={s.add(f'e{i}i',f'o{i}i')};")
            s.o(f"  {v[j]}_re={s.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={s.sub(f'e{i}i',f'o{i}i')};")
        s.o(f"}}")

    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ {s.vt} t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{a}_re',f'{c}_re')}; t0i={s.add(f'{a}_im',f'{c}_im')};")
        s.o(f"  t1r={s.sub(f'{a}_re',f'{c}_re')}; t1i={s.sub(f'{a}_im',f'{c}_im')};")
        s.o(f"  t2r={s.add(f'{b}_re',f'{dd}_re')}; t2i={s.add(f'{b}_im',f'{dd}_im')};")
        s.o(f"  t3r={s.sub(f'{b}_re',f'{dd}_re')}; t3i={s.sub(f'{b}_im',f'{dd}_im')};")
        s.o(f"  {a}_re={s.add('t0r','t2r')}; {a}_im={s.add('t0i','t2i')};")
        s.o(f"  {c}_re={s.sub('t0r','t2r')}; {c}_im={s.sub('t0i','t2i')};")
        if fwd:
            s.o(f"  {b}_re={s.add('t1r','t3i')}; {b}_im={s.sub('t1i','t3r')};")
            s.o(f"  {dd}_re={s.sub('t1r','t3i')}; {dd}_im={s.add('t1i','t3r')};")
        else:
            s.o(f"  {b}_re={s.sub('t1r','t3i')}; {b}_im={s.add('t1i','t3r')};")
            s.o(f"  {dd}_re={s.add('t1r','t3i')}; {dd}_im={s.sub('t1i','t3r')};")
        s.o(f"}}")

    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN)
        fwd=(d=='fwd')
        if typ=='one':
            if dst!=src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ=='neg_one':
            s.o(f"{dst}_re={s.neg(f'{src}_re')}; {dst}_im={s.neg(f'{src}_im')};")
        elif typ=='neg_j':
            if fwd: s.o(f"{{ {s.vt} t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
            else:   s.o(f"{{ {s.vt} t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ=='pos_j':
            if fwd: s.o(f"{{ {s.vt} t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:   s.o(f"{{ {s.vt} t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
        elif typ=='w8_1':
            s.o(f"{{ {s.vt} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='w8_3':
            s.o(f"{{ {s.vt} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:   s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='neg_w8_1':
            s.o(f"{{ {s.vt} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ=='neg_w8_3':
            s.o(f"{{ {s.vt} tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ {s.vt} tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={s.fms(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fma('tr',f'tw_{label}_im',s.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
            else:
                s.o(f"  {dst}_re={s.fma(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fms(f'{src}_im',f'tw_{label}_re',s.mul('tr',f'tw_{label}_im'))}; }}")


# ── Kernel emitter ──

def emit_notw_kernel(em, d, itw_set):
    """Emit a twiddle-less DFT-32 kernel."""
    tgt = em.t
    nfuse = tgt.nfuse
    vt = tgt.vtype
    vw = tgt.vw
    sfx = tgt.suffix
    spill_slots = 32  # max needed (reduced by nfuse)
    spill_doubles = spill_slots * vw

    em.L.append(f'static __attribute__((target("{tgt.attr}"))) void')
    em.L.append(f"radix32_notw_dit_kernel_{d}_{sfx}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o(f"const {vt} sign_flip = {em.p}_set1_pd(-0.0);")
    em.o(f"const {vt} sqrt2_inv = {em.p}_set1_pd(0.70710678118654752440);")
    em.b()
    em.o(f"__attribute__((aligned({tgt.align}))) double spill_re[{spill_doubles}];")
    em.o(f"__attribute__((aligned({tgt.align}))) double spill_im[{spill_doubles}];")
    em.b()
    em.o(f"{vt} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o(f"{vt} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    # Declare s-registers for NFUSE
    if nfuse > 0:
        slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
        em.o(f"{vt} {slist};")
    em.b()

    if itw_set:
        em.c(f"Hoisted internal W32 broadcasts [{d}]")
        for (e,tN) in sorted(itw_set):
            label=wN_label(e,tN)
            em.o(f"const {vt} tw_{label}_re = {em.p}_set1_pd({label}_re);")
            em.o(f"const {vt} tw_{label}_im = {em.p}_set1_pd({label}_im);")
        em.b()

    em.o(f"for (size_t k = 0; k < K; k += {vw}) {{")
    em.ind += 1

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    last_n2 = N2 - 1

    # PASS 1: 4 radix-8 sub-FFTs (no twiddles — just load + butterfly)
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}: load + radix-8")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()

        if is_last and nfuse > 0:
            em.c(f"FUSED: save x0..x{nfuse-1} to s-regs" +
                 (f", spill x{nfuse}..x{N1-1}" if nfuse < N1 else " (no spills!)"))
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        em.b()

    # PASS 2: 8 radix-4 combines (with internal W32 twiddles)
    em.c(f"PASS 2: radix-4 combines with internal W32 twiddles"); em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")

        if k1 < nfuse:
            # Reload from n2=0..last_n2-1, get n2=last from s-reg
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2*N1+k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
            for n2 in range(N2):
                em.emit_reload(f"x{n2}", n2*N1+k1)
        em.b()

        if k1 > 0:
            for n2 in range(1,N2):
                e=(n2*k1)%N
                em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
            em.b()
        em.emit_radix4(xv4,d,f"radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1+N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('avx512', 'avx2'):
        print(f"Usage: {sys.argv[0]} avx512|avx2", file=sys.stderr)
        sys.exit(1)

    tgt = Target(sys.argv[1])
    itw_set = collect_internal_twiddles()
    em = E(tgt)

    em.L.append("/**")
    em.L.append(f" * @file fft_radix32_{tgt.suffix}_notw.h")
    em.L.append(f" * @brief Twiddle-less DFT-32 {tgt.suffix.upper()} codelet")
    em.L.append(f" *")
    em.L.append(f" * Pure DFT-32 butterfly, no inter-stage twiddles.")
    em.L.append(f" * {tgt.vw}-wide doubles, NFUSE={tgt.nfuse}, 8x4 decomposition")
    em.L.append(f" * Generated by gen_radix32_notw.py {tgt.isa}")
    em.L.append(f" */")
    em.L.append("")
    em.L.append(f"#ifndef {tgt.guard}")
    em.L.append(f"#define {tgt.guard}")
    em.L.append("")
    em.L.append("#include <immintrin.h>")
    em.L.append("")

    by_tN = {}
    for (e,tN) in sorted(itw_set): by_tN.setdefault(tN,[]).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}"); em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr,wi=wN(e,tN); l=wN_label(e,tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif"); em.L.append("")

    em.L.append(f"#ifndef {tgt.ld_macro}")
    em.L.append(f"#define {tgt.ld_macro}(p) {tgt.prefix}_loadu_pd(p)")
    em.L.append(f"#endif")
    em.L.append(f"#ifndef {tgt.st_macro}")
    em.L.append(f"#define {tgt.st_macro}(p,v) {tgt.prefix}_storeu_pd((p),(v))")
    em.L.append(f"#endif")
    em.L.append(f"#define LD {tgt.ld_macro}")
    em.L.append(f"#define ST {tgt.st_macro}")
    em.L.append("")

    em.L.append(f"/* === TWIDDLE-LESS DFT-32 {tgt.suffix.upper()} (NFUSE={tgt.nfuse}) === */")
    em.L.append("")
    ff = emit_notw_kernel(em, 'fwd', itw_set)
    fb = emit_notw_kernel(em, 'bwd', itw_set)

    em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append(f"#endif /* {tgt.guard} */")

    print("\n".join(em.L))

    total = ff[0]+ff[1]
    baseline = 64  # 32sp + 32rl with NFUSE=0
    print(f"\n=== {tgt.suffix.upper()} TWIDDLE-LESS ===", file=sys.stderr)
    print(f"  NFUSE={tgt.nfuse}: {ff[0]}sp+{ff[1]}rl={total} mem ops (was {baseline}), saved {baseline-total}", file=sys.stderr)
    print(f"  No external twiddle loads (was 31 loads/k-step in twiddled)", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
