# w32tg_gen.py - generate radix-32 tangent-Givens kernels from the VALIDATED
# radix-16 tangent dataflow: DIT-32 = even-half DIT-16 + odd-half DIT-16 (both
# emitted from the proven listing interpretation, ingest slots remapped) + a
# new combining stage (16 rotated butterflies, W32^o in tangent form, signs
# folded into constants).
#   W16_LEAF=0 -> w32tg_kernel.c   radix32_z_w32tg_fwd_avx2  (mid: 31-record
#                 VTW2 ingest, natural leg-major stores)
#   W16_LEAF=1 -> w32tgL_kernel.c  radix32_z_w32tgL_fwd_avx2 (leaf: no ingest,
#                 corner-turn 128-bit stores)
import re, sys, os, math

LEAF = os.environ.get('W16_LEAF','0')=='1'
ASM = r'c:\Users\Tugbars\Desktop\highSpeedFFT\docs\research\mkl512_gap_campaign\asm\mkl512__w16_fwd_loop.asm'
MEMRE = re.compile(r'^(-?(?:0x[0-9a-fA-F]+|\d+))?\(([^)]*)\)$')
SITE_LEG=[8,0,4,12,14,10,6,2,1,13,9,5,15,11,7,3]
STORE_OUT=[6,2,10,14,5,13,11,3,8,4,0,12,9,15,7,1]

def split_ops(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch=='(':depth+=1
        if ch==')':depth-=1
        if ch==',' and depth==0: out.append(cur.strip()); cur=''
        else: cur+=ch
    if cur.strip(): out.append(cur.strip())
    return out
def base_of(m):
    mm=MEMRE.match(m); return (int(mm.group(1),0) if mm.group(1) else 0), [p.strip() for p in mm.group(2).split(',')][0]

insns=[]
for raw in open(ASM).read().splitlines():
    line=raw.split('#')[0].rstrip()
    if not line.strip(): continue
    m=re.match(r'^\s*[0-9a-fA-F]+\s+(\S+)\s*(.*)$', line)
    if not m: continue
    insns.append((m.group(1), split_ops(m.group(2)) if m.group(2) else []))

offs=sorted({base_of(o[0])[0] for mn,o in insns if o and '(' in o[0] and 'r8' in o[0]})
c_slot={offs[2*i]: i for i in range(15)}   # C offset -> interior slot (leg=slot+1)
s_slot={offs[2*i+1]: i for i in range(15)}

body=[]; n=0
def newv(expr):
    global n
    body.append(f'        const __m256d v{n} = {expr};')
    n+=1; return f'v{n-1}'

CN=['_TAN8','_R2','_CP8']

def emit_half(half):
    """Emit one DIT-16 interior over original legs {2k+half}, ingest per-leg
    (mid) or none (leaf), stores to S[8*o + 4*half_slot]... layout:
    E[o]->S[16*o], O[o]->S[16*o+8]  (per-iter S = 2 cols x 32 vals x 4 dbl?)
    Actually: value = __m256d (4 dbl, 2 cols). E[o] at S+8*o? Use stride:
    S[8*o + 4*half]  -> butterfly o reads S[8o] and S[8o+4] adjacent."""
    global n
    reg={}; spill={}; loadp=0; bcastp=0; storep=0
    for mn,ops in insns:
        if mn=='vmovupd':
            if ops[1].startswith('%ymm'):
                off,base=base_of(ops[0])
                if base=='%rsp':
                    reg[ops[1]] = '_MSK' if off==0xb0 else spill[ops[0]]
                else:
                    il=SITE_LEG[loadp]; loadp+=1
                    orig=2*il+half
                    v = newv(f'_mm256_loadu_pd(&zin[2*((size_t){orig}*Ls + k)])')
                    if (not LEAF) and orig!=0 and il==0:
                        # leg-0 site of the odd half: INJECT ingest (slot orig-1)
                        f = newv(f'_mm256_permute_pd({v}, 0x5)')
                        mres = newv(f'_mm256_mul_pd({f}, _mm256_loadu_pd(&twp[{(orig-1)*8+4}]))')
                        v = newv(f'_mm256_fmsub_pd({v}, _mm256_loadu_pd(&twp[{(orig-1)*8}]), {mres})')
                    reg[ops[1]] = v
            else:
                off,base=base_of(ops[1])
                if base=='%rsp': spill[ops[1]] = reg[ops[0]]
                else:
                    o=STORE_OUT[storep]; storep+=1
                    body.append(f'        _mm256_storeu_pd(&S[{8*o + 4*half}], {reg[ops[0]]});')
        elif mn=='vmovapd': reg[ops[1]] = reg[ops[0]]
        elif mn=='vbroadcastsd':
            reg[ops[1]] = CN[bcastp]; bcastp+=1
        elif mn=='vshufpd':
            src=reg[ops[1]]
            if src is None: reg[ops[3]]=None
            else: reg[ops[3]] = newv(f'_mm256_permute_pd({src}, 0x5)')
        elif mn=='vmulpd':
            if '(' in ops[0] and 'r8' in ops[0]:
                if LEAF: reg[ops[2]] = None
                else:
                    si=s_slot[base_of(ops[0])[0]]
                    orig=2*(si+1)+half
                    reg[ops[2]] = newv(f'_mm256_mul_pd({reg[ops[1]]}, _mm256_loadu_pd(&twp[{(orig-1)*8+4}]))')
            else:
                a = reg[ops[0]]; b = reg[ops[1]]
                x,y = (b,a) if isinstance(a,str) and a.startswith('_') else (a,b)
                # one of them is a constant name
                if isinstance(reg[ops[0]],str) and reg[ops[0]] in CN: cst,dat=reg[ops[0]],reg[ops[1]]
                elif isinstance(reg[ops[1]],str) and reg[ops[1]] in CN: cst,dat=reg[ops[1]],reg[ops[0]]
                else: cst,dat=None,None
                assert cst is not None
                reg[ops[2]] = newv(f'_mm256_mul_pd({dat}, {cst})')
        elif mn=='vaddpd':
            reg[ops[2]] = newv(f'_mm256_add_pd({reg[ops[1]]}, {reg[ops[0]]})')
        elif mn=='vsubpd':
            reg[ops[2]] = newv(f'_mm256_sub_pd({reg[ops[1]]}, {reg[ops[0]]})')
        elif mn=='vaddsubpd':
            reg[ops[2]] = newv(f'_mm256_addsub_pd({reg[ops[1]]}, {reg[ops[0]]})')
        elif mn=='vxorpd':
            a,b=reg[ops[0]],reg[ops[1]]
            x = b if a=='_MSK' else a
            reg[ops[2]] = newv(f'_mm256_xor_pd({x}, _MSK)')
        elif mn.startswith(('vfmadd','vfmsub','vfnmadd')):
            k3=mn[1:]
            if '(' in ops[0] and 'r8' in ops[0]:
                if LEAF:
                    reg[ops[2]] = reg[ops[1]]      # identity twiddle passthrough
                    continue
                si=c_slot[base_of(ops[0])[0]]
                orig=2*(si+1)+half
                A=f'_mm256_loadu_pd(&twp[{(orig-1)*8}])'
            else:
                A=reg[ops[0]]
            B=reg[ops[1]]; C=reg[ops[2]]
            if   '132' in k3: x,y,z = C,A,B
            elif '213' in k3: x,y,z = B,C,A
            else:             x,y,z = B,A,C
            f = ('_mm256_fnmadd_pd' if k3.startswith('fnmadd') else
                 '_mm256_fmsub_pd'  if 'sub' in k3 else '_mm256_fmadd_pd')
            reg[ops[2]] = newv(f'{f}({x}, {y}, {z})')
        elif mn in ('mov','add','cmp','jl'): pass
        else: raise SystemExit('unhandled '+mn)
    assert loadp==16 and storep==16, (loadp,storep)

# ---- pass B: 16 rotated butterflies  X[o]=E+W32^o*O, X[o+16]=E-W32^o*O ----
def emit_final():
    consts=[]
    def cvec(name, lane0, lane1):
        consts.append(f'static const __m256d {name} = {{ {lane0!r}, {lane1!r}, {lane0!r}, {lane1!r} }};')
        return name
    lines=[]
    for o in range(16):
        E=f'_mm256_loadu_pd(&S[{8*o}])'; O=f'_mm256_loadu_pd(&S[{8*o+4}])'
        lines.append(f'        {{ const __m256d e = {E}; const __m256d oo = {O};')
        if o==0:
            lines.append('          const __m256d lo = _mm256_add_pd(e, oo);')
            lines.append('          const __m256d hi = _mm256_sub_pd(e, oo);')
        elif o==8:
            # W=-i: -i*oo = [im,-re]: fold sign into butterfly consts [1,-1]
            lines.append('          const __m256d f = _mm256_permute_pd(oo, 0x5);')
            lines.append(f'          const __m256d lo = _mm256_fmadd_pd(_B8, f, e);')
            lines.append(f'          const __m256d hi = _mm256_fnmadd_pd(_B8, f, e);')
        else:
            oe = o if o<8 else o-8
            t=math.tan(math.pi*oe/16.0); c=math.cos(math.pi*oe/16.0)
            tn=cvec(f'_T{oe}', t, -t) if f'_T{oe}' not in ''.join(consts) else f'_T{oe}'
            lines.append('          const __m256d f = _mm256_permute_pd(oo, 0x5);')
            lines.append(f'          const __m256d sh = _mm256_fmadd_pd({tn}, f, oo);')
            if o<8:
                cn=cvec(f'_C{oe}', c, c)
                lines.append(f'          const __m256d lo = _mm256_fmadd_pd({cn}, sh, e);')
                lines.append(f'          const __m256d hi = _mm256_fnmadd_pd({cn}, sh, e);')
            else:
                # W32^o = -i * W32^{o-8}: -i*(c*sh) -> flip(sh), consts [c,-c]
                cn=cvec(f'_D{oe}', c, -c)
                lines.append('          const __m256d f2 = _mm256_permute_pd(sh, 0x5);')
                lines.append(f'          const __m256d lo = _mm256_fmadd_pd({cn}, f2, e);')
                lines.append(f'          const __m256d hi = _mm256_fnmadd_pd({cn}, f2, e);')
        if LEAF:
            lines.append(f'          _mm_storeu_pd(&zout[2*((size_t)k*OLs + {o})], _mm256_castpd256_pd128(lo));')
            lines.append(f'          _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + {o})], _mm256_extractf128_pd(lo, 1));')
            lines.append(f'          _mm_storeu_pd(&zout[2*((size_t)k*OLs + {o+16})], _mm256_castpd256_pd128(hi));')
            lines.append(f'          _mm_storeu_pd(&zout[2*((size_t)(k+1)*OLs + {o+16})], _mm256_extractf128_pd(hi, 1));')
        else:
            lines.append(f'          _mm256_storeu_pd(&zout[2*((size_t){o}*OLs + k)], lo);')
            lines.append(f'          _mm256_storeu_pd(&zout[2*((size_t){o+16}*OLs + k)], hi);')
        lines.append('        }')
    # dedupe consts (cvec may duplicate)
    seen=set(); cl=[]
    for c in consts:
        nm=c.split()[3]
        if nm not in seen: seen.add(nm); cl.append(c)
    return cl, lines

fn = 'w32tgL' if LEAF else 'w32tg'
sym = f'radix32_z_{fn}_fwd_avx2'
emit_half(0)
half0_end=len(body)
emit_half(1)
cl, fl = emit_final()

hdr = f'''/* {fn} - radix-32 tangent-Givens ({'leaf: no ingest, corner-turn stores' if LEAF else 'mid: 31-record VTW2 ingest (t2b48-compatible table, THEIR fold), natural stores'}).
 * DIT-32 = even-half DIT-16 (validated tangent interior) + odd-half DIT-16
 * + 16 rotated butterflies W32^o in tangent form (shear fmadd([t,-t],flip,x),
 * normalization+butterfly fused, -i composed via [c,-c] flip). Generated by
 * w32tg_gen.py. CONTRACT: count %% 2 == 0. */
#include <immintrin.h>
#include <stddef.h>

static const __m256d _MSK  = {{ -0.0, 0.0, -0.0, 0.0 }};
static const __m256d _TAN8 = {{ -0.41421356237309503, -0.41421356237309503, -0.41421356237309503, -0.41421356237309503 }};
static const __m256d _R2   = {{ -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476 }};
static const __m256d _CP8  = {{ -0.9238795325112867, -0.9238795325112867, -0.9238795325112867, -0.9238795325112867 }};
static const __m256d _B8   = {{ 1.0, -1.0, 1.0, -1.0 }};
'''+ '\n'.join(cl) + f'''

__attribute__((target("avx2,fma")))
void {sym}(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OGs;{' (void)tw_re;' if LEAF else ''}
    double S[256];
    for (size_t k = 0; k + 2 <= count; k += 2) {{
{'' if LEAF else '        const double *twp = tw_re + (k / 2) * (size_t)248;'}
'''
mid = '\n'.join(body[:half0_end]) + '\n' + '\n'.join(body[half0_end:]) + '\n' + '\n'.join(fl)
tail = '''    }
}
'''
out=f'w32tgL_kernel.c' if LEAF else 'w32tg_kernel.c'
open(out,'w').write(hdr+mid+'\n'+tail)
print(f'emitted {out}: {n} SSA values + final stage, consts {len(cl)}')
