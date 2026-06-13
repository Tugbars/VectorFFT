#!/usr/bin/env python3
# sched_analyze.py — A1 schedule quality analyzer (v1, disasm level).
#
# Parses objdump -d output, isolates the main loop body (largest
# backward-branch region), builds a dependence graph (registers exact;
# memory deps via literal rsp-offset matching, which captures spill
# store->load chains), weights edges with CLX-class latencies, and
# reports: critical path (cycles), port-pressure bounds per class,
# and the binding constraint with slack.
#
# v1 caveats (documented in docs/dag_scheduler_program.md):
#  - array store->load deps ignored (loads precede stores in these
#    kernels); rsp (spill) deps ARE tracked
#  - latencies are a fixed CLX-ish table, not per-form exact
#  - one iteration of the main loop analyzed; cross-iteration deps
#    via registers carried into next iteration are not chained
import re, sys, subprocess

LAT = {  # instruction-class latencies (Cascade Lake-ish)
    'fma': 4, 'mul': 4, 'add': 4, 'shuf': 1, 'perm_lane': 3,
    'load': 6, 'store': 0, 'bcast': 6, 'mov_rr': 1, 'other': 1,
}
# port widths per cycle: p01 fp (2), p5 shuffle (1), p23 load (2), p4 store (1)

RE_INS = re.compile(r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(\S+)\s*(.*)$')

def classify(mn, ops):
    if re.match(r'vf(n?m(add|sub)|maddsub|msubadd)', mn): return 'fma'
    if re.match(r'v(mul)p[ds]', mn): return 'mul'
    if re.match(r'v(add|sub)p[ds]', mn): return 'add'
    if re.match(r'vperm2|vperm[a-z]*\b', mn) and ('ymm' in ops or 'zmm' in ops): return 'perm_lane'
    if re.match(r'v(shuf|unpck|blend|insert|extract|movddup|movshdup|movsldup|perm)', mn): return 'shuf'
    if re.match(r'vbroadcast', mn): return 'bcast'
    if re.match(r'vmov(u|a)p[ds]', mn):
        if '(' in ops.split(',')[-1]: return 'store'
        if '(' in ops: return 'load'
        return 'mov_rr'
    return 'other'

def regs_of(s):
    return set(re.findall(r'%(?:zmm|ymm|xmm)\d+', s))

def rsp_slot(ops, kind):
    m = re.search(r'(-?0x[0-9a-f]+)?\(%rsp\)', ops)
    if not m: return None
    return m.group(1) or '0x0'

def parse(disfile):
    ins = []
    for line in open(disfile):
        m = RE_INS.match(line)
        if not m: continue
        addr = int(m.group(1), 16)
        ins.append((addr, m.group(2), m.group(3)))
    return ins

def main_loop_body(ins):
    # largest backward branch span
    best = None
    addr_index = {a: i for i, (a, _, _) in enumerate(ins)}
    for i, (a, mn, ops) in enumerate(ins):
        if mn.startswith(('jne', 'jb', 'jae', 'ja', 'jl', 'jg', 'jmp', 'jns', 'js', 'loop', 'dec')) or mn[0] == 'j':
            m = re.match(r'^([0-9a-f]+)', ops.strip())
            if m:
                tgt = int(m.group(1), 16)
                if tgt < a and tgt in addr_index:
                    span = i - addr_index[tgt]
                    if best is None or span > best[2]:
                        best = (addr_index[tgt], i, span)
    if best and best[2] >= 20:
        return ins[best[0]:best[1] + 1]
    return ins  # straight-line kernel

def analyze(disfile, label):
    body = main_loop_body(parse(disfile))
    n = len(body)
    last_writer = {}      # reg -> node id
    rsp_writer = {}       # slot -> node id
    finish = {}           # node -> critical-path finish time
    counts = dict(fma=0, mul=0, add=0, shuf=0, perm_lane=0, load=0,
                  store=0, bcast=0, mov_rr=0, other=0)
    cp = 0
    for i, (a, mn, ops) in enumerate(body):
        cls = classify(mn, ops)
        counts[cls] += 1
        lat = LAT[cls]
        srcs = []
        rs = regs_of(ops)
        parts = ops.split(',')
        dst_regs = regs_of(parts[-1]) if parts else set()
        src_regs = rs - dst_regs if cls in ('store',) else rs - (dst_regs if len(parts) > 1 else set())
        # register deps (read-after-write)
        for r in (rs if cls == 'store' else src_regs):
            if r in last_writer: srcs.append(last_writer[r])
        # FMA also reads its destination
        if cls == 'fma':
            for r in dst_regs:
                if r in last_writer: srcs.append(last_writer[r])
        # spill slot deps
        if cls == 'load':
            slot = rsp_slot(ops, 'load')
            if slot is not None and slot in rsp_writer:
                srcs.append(rsp_writer[slot])
        start = max((finish[s] for s in srcs), default=0)
        finish[i] = start + lat
        cp = max(cp, finish[i])
        if cls == 'store':
            slot = rsp_slot(ops, 'store')
            if slot is not None: rsp_writer[slot] = i
        else:
            for r in dst_regs: last_writer[r] = i
    flops = 2 * counts['fma'] + counts['mul'] + counts['add']
    p01 = counts['fma'] + counts['mul'] + counts['add']
    p5 = counts['shuf'] + counts['perm_lane']
    p23 = counts['load'] + counts['bcast']
    p4 = counts['store']
    bound_p01 = p01 / 2.0
    bound_p5 = p5 / 1.0
    bound_p23 = p23 / 2.0
    bound_p4 = p4 / 1.0
    port_bound = max(bound_p01, bound_p5, bound_p23, bound_p4)
    which = ['p01', 'p5', 'p23', 'p4'][[bound_p01, bound_p5, bound_p23, bound_p4].index(port_bound)]
    binding = 'LATENCY' if cp > port_bound else f'PORT({which})'
    print(f"{label:30s} body={n:4d} flops={flops:5d} | CP={cp:6.0f}cy  "
          f"port-bound={port_bound:6.1f}cy ({which})  binding={binding}  "
          f"slack={cp/max(port_bound,0.1):4.2f}")
    print(f"{'':30s} per-100-flop: CP={cp*100/max(flops,1):5.1f}  "
          f"p01={bound_p01*100/max(flops,1):5.1f}  p5={bound_p5*100/max(flops,1):5.1f}  "
          f"p23={bound_p23*100/max(flops,1):5.1f}  p4={bound_p4*100/max(flops,1):5.1f}")

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        label, path = arg.split('=', 1)
        analyze(path, label)
