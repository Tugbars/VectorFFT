#!/usr/bin/env python3
"""align_kernel_stacks.py -- every codelet that spills a vector register to the stack runs on a 64-byte aligned
frame (Windows x64, GCC). Both builds run it on the archive that holds the codelets: CMake on libvfft.a (and the
gauntlet's vfft_core), gauntlet/build.py on libdagcodelets.a.

THE PROBLEM. Win64 guarantees only a 16-byte aligned stack, and GCC cannot realign rsp in a Win64 (SEH) function,
so a kernel that runs out of vector registers spills them with UNALIGNED moves at [rsp+X]. Where rsp falls within a
64-byte line depends on the process and the call path, so the same kernel's spills split cache lines in one process
and not in the next: a 32-byte ymm spill splits at 1 of the 4 positions, a 64-byte zmm spill at 3 of 4. Measured on
the 14900KF (2026-10-29, 8 fresh processes per build): ZTURN-T lengths had one or two processes in 8 at 1.8-2x the
best (2352, 3024, 3840, 8192, 10000, 524288), N = 512 four of 16 at +7..29 %, N = 164 a median of 468 ns against
308. Races measure in whatever position their process drew, so a banked verdict can carry it too.

THE FIX. For each GLOBAL function of a codelet object that reads or writes a ymm/zmm slot at [rsp+X]:
  - the function is renamed f -> f__body in its object (objcopy --redefine-syms). Its code is untouched, so its
    output is bit-identical;
  - a wrapper named f keeps a frame pointer, aligns rsp to 64 and leaves the room that puts the body's rsp, after
    the body's own prologue, at the offset (0/16/32/48) that splits the fewest of the body's slots -- 0 for nearly
    every kernel. The body's frame T = 8 + 8 * pushes + sub is read from its disassembly. The wrapper copies the
    stack arguments the body reads (all 16 words when the body keeps a frame pointer, whose argument reads this
    script does not count) and calls it. One extra call, about 1 ns.
Static functions keep their place (their caller is a wrapped kernel, so their position is fixed too).

    align_kernel_stacks.py ARCHIVE --cc GCC [--objdump X --objcopy X --nm X --ar X] [--keep MEMBER ...] [--cache DIR]

  --cc       assembles the wrappers; the binutils default to its directory (x86_64-w64-mingw32-gcc -> ...-objdump)
  --keep     archive members left as they are (vfft.c's object: its functions are the library's own, varargs among them)
  --cache    processed members by content hash, so a rebuild that changed only vfft.c re-analyses nothing
An archive that already holds the wrapper member is left as it is. The archive is replaced atomically."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

WRAP_MEMBER = 'vfft_kernel_stack_align.o'   # the wrappers' object, and the "already processed" mark
BODY = '__body'
ALL_ARGS = 16          # stack arguments copied when the body's own reads cannot be counted
VERSION = 1            # bumped when the analysis or the wrapper changes: the cache keys on it

func_re = re.compile(r'^[0-9a-f]+ <([^>]+)>:')
push_re = re.compile(r'^\s*[0-9a-f]+:\s+push\s+\w+')
sub_re = re.compile(r'^\s*[0-9a-f]+:\s+sub\s+rsp,(0x[0-9a-f]+|rax)')
moveax_re = re.compile(r'^\s*[0-9a-f]+:\s+mov\s+eax,(0x[0-9a-f]+)')
framep_re = re.compile(r'^\s*[0-9a-f]+:\s+mov\s+rbp,rsp')
wide_re = re.compile(r'(YMMWORD|ZMMWORD) PTR \[rsp(?:\+(0x[0-9a-f]+))?\]')
any_re = re.compile(r'PTR \[rsp(?:\+(0x[0-9a-f]+))?\]')


def tool(cc, name):
    d, b = os.path.split(cc)
    b = re.sub(r'gcc(\.exe)?$', name + r'\1', b) if 'gcc' in b else name
    return os.path.join(d, b) if d else b


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', **kw)
    if r.returncode:
        sys.exit('align_kernel_stacks: %s failed:\n%s' % (os.path.basename(cmd[0]), r.stderr[-2000:]))
    return r.stdout


def analyse(objdump, paths):
    """per path: {function: {push, sub, T, wide: [(width, offset)], maxarg, fp}} from one objdump call"""
    out = run([objdump, '-d', '--no-show-raw-insn', '-M', 'intel'] + paths)
    res, cur, fn = {}, None, None
    for line in out.splitlines():
        if line.endswith('file format pe-x86-64'):
            cur = res.setdefault(os.path.basename(line.rsplit(':', 1)[0]), {})
            fn = None
            continue
        m = func_re.match(line)
        if m and cur is not None:
            fn = cur[m.group(1)] = {'push': 0, 'sub': None, 'eax': None, 'wide': [], 'maxarg': -1, 'fp': False}
            continue
        if fn is None:
            continue
        if framep_re.match(line):
            fn['fp'] = True
        if fn['sub'] is None:
            if push_re.match(line):
                fn['push'] += 1
            elif moveax_re.match(line):
                fn['eax'] = int(moveax_re.match(line).group(1), 16)
            elif sub_re.match(line):   # sub rsp,imm, or a large frame: mov eax,imm; call ___chkstk_ms; sub rsp,rax
                g = sub_re.match(line).group(1)
                fn['sub'] = fn['eax'] if g == 'rax' else int(g, 16)
                if fn['sub'] is not None:
                    fn['T'] = 8 + 8 * fn['push'] + fn['sub']
            continue
        if 'T' not in fn:
            continue
        for x in any_re.findall(line):
            off = int(x, 16) if x else 0
            if off >= fn['T'] + 32:   # the caller's stack argument (off - T - 32) / 8
                fn['maxarg'] = max(fn['maxarg'], (off - fn['T'] - 32) // 8)
        for w, x in wide_re.findall(line):
            fn['wide'].append((w, int(x, 16) if x else 0))
    return res


def splits(w, r):
    return r > 32 if w == 'YMMWORD' else r != 0


def plan(funcs, globs):
    """the wrapper specs of one object: the global functions with wide stack slots and a parsed frame"""
    specs, skipped = [], []
    for name in sorted(funcs):
        f = funcs[name]
        if not f['wide'] or name not in globs:
            continue
        if f.get('T') is None or f['T'] % 16:
            skipped.append(name)
            continue
        phase = min((0, 16, 32, 48), key=lambda a: (sum(splits(w, (a + x) % 64) for w, x in f['wide']), a))
        nargs = ALL_ARGS if f['fp'] else f['maxarg'] + 1
        specs.append({'name': name, 'T': f['T'], 'nargs': nargs, 'phase': phase})
    return specs, skipped


def wrapper_text(s):
    n, na = s['name'], s['nargs']
    room = ((32 + 8 * na + 63) // 64) * 64             # the body's home area and its stack arguments, whole lines
    y = room + (-(s['T'] + s['phase'])) % 64          # the body's rsp after its prologue: aligned - y - T = phase
    t = ['\t.p2align 4', '\t.globl %s' % n, '\t.def %s; .scl 2; .type 32; .endef' % n, '\t.seh_proc %s' % n,
         '%s:' % n, '\tpushq %rbp', '\t.seh_pushreg %rbp', '\tmovq %rsp, %rbp', '\t.seh_setframe %rbp, 0',
         '\t.seh_endprologue', '\tandq $-64, %rsp', '\tsubq $%d, %%rsp' % y]
    for i in range(na):
        t += ['\tmovq %d(%%rbp), %%rax' % (48 + 8 * i), '\tmovq %%rax, %d(%%rsp)' % (32 + 8 * i)]
    t += ['\tcall %s%s' % (n, BODY), '\tleaq 0(%rbp), %rsp', '\tpopq %rbp', '\tret', '\t.seh_endproc']
    return '\n'.join(t) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('archive')
    ap.add_argument('--cc', required=True)
    ap.add_argument('--objdump')
    ap.add_argument('--objcopy')
    ap.add_argument('--nm')
    ap.add_argument('--ar')
    ap.add_argument('--keep', action='append', default=[])
    ap.add_argument('--cache')
    a = ap.parse_args()
    objdump = a.objdump or tool(a.cc, 'objdump')
    objcopy = a.objcopy or tool(a.cc, 'objcopy')
    nm = a.nm or tool(a.cc, 'nm')
    ar = a.ar or tool(a.cc, 'ar')
    archive = os.path.abspath(a.archive)

    members = run([ar, 't', archive]).split()
    if WRAP_MEMBER in members:
        print('align_kernel_stacks: %s is already aligned' % os.path.basename(archive))
        return
    if len(set(members)) != len(members):
        sys.exit('align_kernel_stacks: %s has members of the same name; they cannot be extracted apart' % archive)
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)
    work = tempfile.mkdtemp(prefix='kstack_', dir=a.cache or os.path.dirname(archive))
    try:
        run([ar, 'x', archive], cwd=work)
        todo, specs, keys = [], {}, {}
        for m in members:
            if m in a.keep:
                continue
            with open(os.path.join(work, m), 'rb') as f:
                keys[m] = hashlib.sha256(f.read() + b'v%d' % VERSION).hexdigest()
            c = a.cache and os.path.join(a.cache, keys[m])
            if c and os.path.exists(c + '.json') and os.path.exists(c + '.o'):
                with open(c + '.json') as f:
                    specs[m] = json.load(f)
                shutil.copyfile(c + '.o', os.path.join(work, m))
            else:
                todo.append(m)

        def one_batch(batch):
            paths = [os.path.join(work, m) for m in batch]
            funcs = analyse(objdump, paths)
            out = {}
            for m in batch:
                globs = {l.split()[2] for l in run([nm, '-g', '--defined-only', os.path.join(work, m)]).splitlines()
                         if len(l.split()) == 3 and l.split()[1] == 'T'}
                out[m] = plan(funcs.get(m, {}), globs)
            return out

        skipped = []
        with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
            for res in ex.map(one_batch, [todo[i:i + 48] for i in range(0, len(todo), 48)]):
                for m, (sp, sk) in res.items():
                    specs[m] = sp
                    skipped += sk

        def rename(m):
            p = os.path.join(work, m)
            if specs[m]:
                syms = p + '.syms'
                with open(syms, 'w') as f:
                    f.writelines('%s %s%s\n' % (s['name'], s['name'], BODY) for s in specs[m])
                run([objcopy, '--redefine-syms=' + syms, p, p + '.new'])
                os.replace(p + '.new', p)
                os.remove(syms)
            if a.cache:   # write-then-rename: concurrent builds may share the cache
                c = os.path.join(a.cache, keys[m])
                shutil.copyfile(p, c + '.o.%d' % os.getpid())
                os.replace(c + '.o.%d' % os.getpid(), c + '.o')
                with open(c + '.json.%d' % os.getpid(), 'w') as f:
                    json.dump(specs[m], f)
                os.replace(c + '.json.%d' % os.getpid(), c + '.json')

        with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
            list(ex.map(rename, todo))

        wrapped = [s for m in members if m in specs for s in specs[m]]
        with open(os.path.join(work, 'wrappers.S'), 'w') as f:
            f.write('/* generated by src/tools/build/align_kernel_stacks.py: one wrapper per spilling kernel */\n'
                    '\t.text\n' + ''.join(wrapper_text(s) for s in wrapped))
        run([a.cc, '-c', os.path.join(work, 'wrappers.S'), '-o', os.path.join(work, WRAP_MEMBER)])
        rsp = os.path.join(work, 'members.rsp')
        with open(rsp, 'w') as f:
            f.write('\n'.join(os.path.join(work, m).replace('\\', '/') for m in members + [WRAP_MEMBER]))
        tmp = archive + '.%d.tmp' % os.getpid()
        if os.path.exists(tmp):
            os.remove(tmp)
        run([ar, 'rcs', tmp, '@' + rsp])
        os.replace(tmp, archive)
        print('align_kernel_stacks: %s: %d kernels aligned in %d of %d objects (%d analysed, %d from the cache)%s'
              % (os.path.basename(archive), len(wrapped), sum(1 for m in specs if specs[m]), len(specs), len(todo),
                 len(specs) - len(todo),
                 '; no frame found in %d spilling functions, left as they are: %s' % (len(skipped), ' '.join(skipped[:5]))
                 if skipped else ''))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
