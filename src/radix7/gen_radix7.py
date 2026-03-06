import sys
# Quick patch: read gen_radix7.py source, apply two fixes, regenerate

exec(open('gen_radix7.py').read().replace(
    "if __name__ == '__main__':", "if False:"))

# Override the sine chain in gen_simd_butterfly and gen_tw_kernel
# and the twiddle multiply for bwd

def gen_sine_chain_simd(I, m, sd, se, sf, comp):
    """Generate S = sd*d + se*e + sf*f using fma chain starting from sd*d."""
    V = I['vtype']
    mul = I['mul']
    fma = I['fmadd']
    fnma = I['fnmadd']

    def cvar(val):
        av = abs(val)
        if av == S1: return 'vs1'
        if av == S2: return 'vs2'
        if av == S3: return 'vs3'
        assert False

    # sd is always positive (s1, s2, or s3)
    # Start: base = sd * d
    expr = f'{mul}({cvar(sd)}, d{comp})'

    # Accumulate se * e
    if se < 0:
        expr = f'{fnma}({cvar(se)}, e{comp}, {expr})'
    else:
        expr = f'{fma}({cvar(se)}, e{comp}, {expr})'

    # Accumulate sf * f
    if sf < 0:
        expr = f'{fnma}({cvar(sf)}, f{comp}, {expr})'
    else:
        expr = f'{fma}({cvar(sf)}, f{comp}, {expr})'

    return f'        const {V} S{m}{comp} = {expr};'


def gen_tw_mul_simd(I, direction):
    """Generate twiddle multiply for SIMD: fwd = normal, bwd = conjugate."""
    V = I['vtype']
    mul = I['mul']
    fma = I['fmadd']
    fnma = I['fnmadd']

    lines = []
    for n in range(1, 7):
        tw_off = f'{n-1}*K+k'
        lines.append(f'        const {V} tw{n}r = LD(&tw_re[{tw_off}]);')
        lines.append(f'        const {V} tw{n}i = LD(&tw_im[{tw_off}]);')
        if direction == 'fwd':
            # t = x * tw: tr = xr*twr - xi*twi, ti = xr*twi + xi*twr
            lines.append(f'        const {V} t{n}r = {fnma}(x{n}i_raw, tw{n}i, {mul}(x{n}r_raw, tw{n}r));')
            lines.append(f'        const {V} t{n}i = {fma}(x{n}r_raw, tw{n}i, {mul}(x{n}i_raw, tw{n}r));')
        else:
            # t = x * conj(tw): tr = xr*twr + xi*twi, ti = -xr*twi + xi*twr
            lines.append(f'        const {V} t{n}r = {fma}(x{n}i_raw, tw{n}i, {mul}(x{n}r_raw, tw{n}r));')
            lines.append(f'        const {V} t{n}i = {fnma}(x{n}r_raw, tw{n}i, {mul}(x{n}i_raw, tw{n}r));')
    return '\n'.join(lines)


def gen_tw_mul_scalar(direction):
    lines = []
    for n in range(1, 7):
        lines.append(f'        const double tw{n}r = tw_re[{n-1}*K+k], tw{n}i = tw_im[{n-1}*K+k];')
        if direction == 'fwd':
            lines.append(f'        const double t{n}r = x{n}r_raw*tw{n}r - x{n}i_raw*tw{n}i;')
            lines.append(f'        const double t{n}i = x{n}r_raw*tw{n}i + x{n}i_raw*tw{n}r;')
        else:
            lines.append(f'        const double t{n}r = x{n}r_raw*tw{n}r + x{n}i_raw*tw{n}i;')
            lines.append(f'        const double t{n}i = -x{n}r_raw*tw{n}i + x{n}i_raw*tw{n}r;')
    return '\n'.join(lines)


# Monkey-patch the generators
import types

old_gen_simd_butterfly = gen_simd_butterfly
old_gen_simd_twiddle_multiply = gen_simd_twiddle_multiply
old_gen_scalar_twiddle_multiply = gen_scalar_twiddle_multiply
old_gen_tw_kernel = gen_tw_kernel

def new_gen_simd_butterfly(isa_name, direction):
    """Fixed version with correct sine chain ordering."""
    I = ISA[isa_name]
    V = I['vtype']
    set1 = I['set1']
    add = I['add']
    sub = I['sub']
    mul = I['mul']
    fma = I['fmadd']
    fnma = I['fnmadd']

    lines = []
    lines.append(f'    const {V} vc1 = {set1}({fmt_const(C1)});')
    lines.append(f'    const {V} vc2 = {set1}({fmt_const(C2)});')
    lines.append(f'    const {V} vc3 = {set1}({fmt_const(C3)});')
    lines.append(f'    const {V} vs1 = {set1}({fmt_const(S1)});')
    lines.append(f'    const {V} vs2 = {set1}({fmt_const(S2)});')
    lines.append(f'    const {V} vs3 = {set1}({fmt_const(S3)});')
    lines.append('')
    lines.append('    for (size_t k = 0; k < K; k += VL) {')

    for n in range(7):
        lines.append(f'        const {V} x{n}r = LD(&in_re[{n}*K+k]);')
        lines.append(f'        const {V} x{n}i = LD(&in_im[{n}*K+k]);')

    lines.append('')
    lines.append('        /* symmetric sums and antisymmetric diffs */')
    lines.append(f'        const {V} ar = {add}(x1r, x6r), ai = {add}(x1i, x6i);')
    lines.append(f'        const {V} dr = {sub}(x1r, x6r), di = {sub}(x1i, x6i);')
    lines.append(f'        const {V} br = {add}(x2r, x5r), bi = {add}(x2i, x5i);')
    lines.append(f'        const {V} er = {sub}(x2r, x5r), ei = {sub}(x2i, x5i);')
    lines.append(f'        const {V} cr = {add}(x3r, x4r), ci = {add}(x3i, x4i);')
    lines.append(f'        const {V} fr = {sub}(x3r, x4r), fi = {sub}(x3i, x4i);')

    lines.append('')
    lines.append(f'        ST(&out_re[k], {add}(x0r, {add}(ar, {add}(br, cr))));')
    lines.append(f'        ST(&out_im[k], {add}(x0i, {add}(ai, {add}(bi, ci))));')

    def cvar_cos(val):
        if val == C1: return 'vc1'
        if val == C2: return 'vc2'
        if val == C3: return 'vc3'
        assert False

    for (m, m2, ca, cb, cc, sd, se, sf) in PAIRS:
        lines.append(f'')
        lines.append(f'        /* outputs {m} and {m2} */')
        for comp in ['r', 'i']:
            lines.append(f'        const {V} R{m}{comp} = {add}(x0{comp}, {fma}({cvar_cos(ca)}, a{comp}, {fma}({cvar_cos(cb)}, b{comp}, {mul}({cvar_cos(cc)}, c{comp}))));')

        for comp in ['r', 'i']:
            lines.append(gen_sine_chain_simd(I, m, sd, se, sf, comp))

        if direction == 'fwd':
            lines.append(f'        ST(&out_re[{m}*K+k], {add}(R{m}r, S{m}i));')
            lines.append(f'        ST(&out_re[{m2}*K+k], {sub}(R{m}r, S{m}i));')
            lines.append(f'        ST(&out_im[{m}*K+k], {sub}(R{m}i, S{m}r));')
            lines.append(f'        ST(&out_im[{m2}*K+k], {add}(R{m}i, S{m}r));')
        else:
            lines.append(f'        ST(&out_re[{m}*K+k], {sub}(R{m}r, S{m}i));')
            lines.append(f'        ST(&out_re[{m2}*K+k], {add}(R{m}r, S{m}i));')
            lines.append(f'        ST(&out_im[{m}*K+k], {add}(R{m}i, S{m}r));')
            lines.append(f'        ST(&out_im[{m2}*K+k], {sub}(R{m}i, S{m}r));')

    lines.append('    }')
    return '\n'.join(lines)


# Patch the module-level functions
gen_simd_butterfly = new_gen_simd_butterfly
gen_simd_twiddle_multiply = gen_tw_mul_simd
gen_scalar_twiddle_multiply = gen_tw_mul_scalar

# Regenerate tw_kernel with fixed twiddle multiply
def new_gen_tw_kernel(isa_name, direction):
    I = ISA[isa_name]
    is_scalar = isa_name == 'scalar'
    V = I['vtype'] if not is_scalar else 'double'
    lines = []

    if is_scalar:
        lines.append(f'    const double c1 = {fmt_const(C1)};')
        lines.append(f'    const double c2 = {fmt_const(C2)};')
        lines.append(f'    const double c3 = {fmt_const(C3)};')
        lines.append(f'    const double s1 = {fmt_const(S1)};')
        lines.append(f'    const double s2 = {fmt_const(S2)};')
        lines.append(f'    const double s3 = {fmt_const(S3)};')
        lines.append('')
        lines.append('    for (size_t k = 0; k < K; k++) {')
        lines.append(f'        const double x0r = in_re[k], x0i = in_im[k];')
        for n in range(1, 7):
            lines.append(f'        const double x{n}r_raw = in_re[{n}*K+k], x{n}i_raw = in_im[{n}*K+k];')
        lines.append(gen_tw_mul_scalar(direction))
        for n in range(1, 7):
            lines.append(f'        const double x{n}r = t{n}r, x{n}i = t{n}i;')

        lines.append('')
        lines.append('        const double ar = x1r + x6r, ai = x1i + x6i;')
        lines.append('        const double dr = x1r - x6r, di = x1i - x6i;')
        lines.append('        const double br = x2r + x5r, bi = x2i + x5i;')
        lines.append('        const double er = x2r - x5r, ei = x2i - x5i;')
        lines.append('        const double cr = x3r + x4r, ci = x3i + x4i;')
        lines.append('        const double fr = x3r - x4r, fi = x3i - x4i;')
        lines.append('')
        lines.append('        out_re[k] = x0r + ar + br + cr;')
        lines.append('        out_im[k] = x0i + ai + bi + ci;')

        for (m, m2, ca, cb, cc, sd, se, sf) in PAIRS:
            lines.append(f'')
            lines.append(f'        const double R{m}r = x0r + {fmt_const(ca)}*ar + {fmt_const(cb)}*br + {fmt_const(cc)}*cr;')
            lines.append(f'        const double R{m}i = x0i + {fmt_const(ca)}*ai + {fmt_const(cb)}*bi + {fmt_const(cc)}*ci;')
            lines.append(f'        const double S{m}r = {fmt_const(sd)}*dr + {fmt_const(se)}*er + {fmt_const(sf)}*fr;')
            lines.append(f'        const double S{m}i = {fmt_const(sd)}*di + {fmt_const(se)}*ei + {fmt_const(sf)}*fi;')
            if direction == 'fwd':
                lines.append(f'        out_re[{m}*K+k] = R{m}r + S{m}i; out_re[{m2}*K+k] = R{m}r - S{m}i;')
                lines.append(f'        out_im[{m}*K+k] = R{m}i - S{m}r; out_im[{m2}*K+k] = R{m}i + S{m}r;')
            else:
                lines.append(f'        out_re[{m}*K+k] = R{m}r - S{m}i; out_re[{m2}*K+k] = R{m}r + S{m}i;')
                lines.append(f'        out_im[{m}*K+k] = R{m}i + S{m}r; out_im[{m2}*K+k] = R{m}i - S{m}r;')
    else:
        set1 = I['set1']
        add = I['add']
        sub = I['sub']
        mul = I['mul']
        fma = I['fmadd']
        fnma = I['fnmadd']
        def cvar_cos(val):
            if val == C1: return 'vc1'
            if val == C2: return 'vc2'
            if val == C3: return 'vc3'
            assert False

        lines.append(f'    const {V} vc1 = {set1}({fmt_const(C1)});')
        lines.append(f'    const {V} vc2 = {set1}({fmt_const(C2)});')
        lines.append(f'    const {V} vc3 = {set1}({fmt_const(C3)});')
        lines.append(f'    const {V} vs1 = {set1}({fmt_const(S1)});')
        lines.append(f'    const {V} vs2 = {set1}({fmt_const(S2)});')
        lines.append(f'    const {V} vs3 = {set1}({fmt_const(S3)});')
        lines.append('')
        lines.append('    for (size_t k = 0; k < K; k += VL) {')
        lines.append(f'        const {V} x0r = LD(&in_re[k]), x0i = LD(&in_im[k]);')
        for n in range(1, 7):
            lines.append(f'        const {V} x{n}r_raw = LD(&in_re[{n}*K+k]), x{n}i_raw = LD(&in_im[{n}*K+k]);')

        lines.append(gen_tw_mul_simd(I, direction))

        for n in range(1, 7):
            lines.append(f'        const {V} x{n}r = t{n}r, x{n}i = t{n}i;')

        lines.append('')
        lines.append(f'        const {V} ar = {add}(x1r, x6r), ai = {add}(x1i, x6i);')
        lines.append(f'        const {V} dr = {sub}(x1r, x6r), di = {sub}(x1i, x6i);')
        lines.append(f'        const {V} br = {add}(x2r, x5r), bi = {add}(x2i, x5i);')
        lines.append(f'        const {V} er = {sub}(x2r, x5r), ei = {sub}(x2i, x5i);')
        lines.append(f'        const {V} cr = {add}(x3r, x4r), ci = {add}(x3i, x4i);')
        lines.append(f'        const {V} fr = {sub}(x3r, x4r), fi = {sub}(x3i, x4i);')
        lines.append('')
        lines.append(f'        ST(&out_re[k], {add}(x0r, {add}(ar, {add}(br, cr))));')
        lines.append(f'        ST(&out_im[k], {add}(x0i, {add}(ai, {add}(bi, ci))));')

        for (m, m2, ca, cb, cc, sd, se, sf) in PAIRS:
            lines.append(f'')
            for comp in ['r', 'i']:
                lines.append(f'        const {V} R{m}{comp} = {add}(x0{comp}, {fma}({cvar_cos(ca)}, a{comp}, {fma}({cvar_cos(cb)}, b{comp}, {mul}({cvar_cos(cc)}, c{comp}))));')
            for comp in ['r', 'i']:
                lines.append(gen_sine_chain_simd(I, m, sd, se, sf, comp))

            if direction == 'fwd':
                lines.append(f'        ST(&out_re[{m}*K+k], {add}(R{m}r, S{m}i));')
                lines.append(f'        ST(&out_re[{m2}*K+k], {sub}(R{m}r, S{m}i));')
                lines.append(f'        ST(&out_im[{m}*K+k], {sub}(R{m}i, S{m}r));')
                lines.append(f'        ST(&out_im[{m2}*K+k], {add}(R{m}i, S{m}r));')
            else:
                lines.append(f'        ST(&out_re[{m}*K+k], {sub}(R{m}r, S{m}i));')
                lines.append(f'        ST(&out_re[{m2}*K+k], {add}(R{m}r, S{m}i));')
                lines.append(f'        ST(&out_im[{m}*K+k], {add}(R{m}i, S{m}r));')
                lines.append(f'        ST(&out_im[{m2}*K+k], {sub}(R{m}i, S{m}r));')

    lines.append('    }')
    return '\n'.join(lines)

gen_tw_kernel = new_gen_tw_kernel

# Now regenerate
for isa in ['avx512', 'avx2', 'scalar']:
    with open(f'fft_radix7_{isa}.h', 'w') as f:
        f.write(gen_file(isa))
    import subprocess
    n = int(subprocess.check_output(['wc', '-l', f'fft_radix7_{isa}.h']).split()[0])
    print(f'{isa}: {n} lines')
