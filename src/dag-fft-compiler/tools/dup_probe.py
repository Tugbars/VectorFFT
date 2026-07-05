#!/usr/bin/env python3
"""Selective-duplication probe (C-level): clone values with >=2 uses,
span>=S lines, defined by add/sub/mul over DIRECT leaf temps (loads /
set1 consts), immediately before their LAST use, with a per-clone
one-value asm barrier (plain C clones get merged straight back by
gcc's FRE — hash-consing is not the only enforcer of mandatory CSE).

Usage: dup_probe.py <codelet.c> <span_S> <cap>
Emits /tmp/dup.c + /tmp/dup.s and prints insns/spills/fma.

VALIDATED (container, gcc 13.3, -march=raptorlake, main-loop-only,
FMA-invariant, bit-exact vs baseline on R=13):
  R=11: 317/42 -> 269/17 (-60% spills)    R=13: 446/70 -> 377/31 (-56%,
  beats the annealer's converged 392/51)  R=17: 756/175 -> 705/118 (-33%)
  R=19: 876/192 -> 849/139 (-28%)
NEGATIVE: R=23 regresses at every dose; blocked pow2 (R>=25) regresses —
the pass structure + designed spill arrays already split long ranges
(doc 58); remaining blocked spills are dense within-pass pressure
(span median 13) that duplication only worsens. Duplication is a
PRIME/MONOLITHIC lever; race it per codelet, A4-style.

Productization scope (the real pass): algsimp-level clone with fresh
tag, re-point the last user, rebuild ancestors; fence ONLY the clone
via the existing value-fence machinery; selection = this probe's rule
(add/sub/mul over leaves, span>=~30, cap ~8-16); anneal AFTER dup so
the search sees the transformed DAG (dup-on-strict 377/31 already beat
dup-on-annealed 391/42 — the annealed order was tuned for the pre-dup
structure)."""
import re, sys, subprocess

SRC, S, CAP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = open(SRC).read().splitlines()
defre = re.compile(r'^(\s*)const __m256d (t\d+) = (.*);\s*$')
leaf, cheap, defs = set(), {}, {}
for i, l in enumerate(lines):
    m = defre.match(l)
    if not m: continue
    name, rhs = m.group(2), m.group(3)
    defs[name] = (i, rhs, m.group(1))
    if 'set1_pd' in rhs or 'loadu_pd' in rhs or 'load_pd' in rhs: leaf.add(name)
    elif re.match(r'_mm256_(add|sub|mul)_pd\((t\d+), ?(t\d+)\)$', rhs):
        ops = re.findall(r't\d+', rhs)
        cheap[name] = ops
uses = {n: [] for n in defs}
for i, l in enumerate(lines):
    for n in set(re.findall(r'\bt\d+\b', l)):
        if n in defs and i > defs[n][0]: uses[n].append(i)
cands = []
for n, ops in cheap.items():
    if len(uses[n]) >= 2 and all(o in leaf for o in ops):
        span = uses[n][-1] - defs[n][0]
        if span >= S: cands.append((span, n))
cands.sort(reverse=True)
chosen = [n for _, n in cands[:CAP]]
ins = {}   # line -> list of new defs to insert before it
for n in chosen:
    last = uses[n][-1]
    _, rhs, ind = defs[n]
    ins.setdefault(last, []).append(f"{ind}__m256d {n}_d = {rhs}; __asm__ volatile(\"\" : \"+x\"({n}_d));")
out = []
for i, l in enumerate(lines):
    if i in ins:
        out.extend(ins[i])
        for n in chosen:
            if uses[n][-1] == i:
                l = re.sub(rf'\b{n}\b', n + '_d', l)
    out.append(l)
open('/tmp/dup.c', 'w').write("\n".join(out) + "\n")
subprocess.run(['gcc','-O3','-mavx2','-mfma','-march=raptorlake','-w','-S','/tmp/dup.c','-o','/tmp/dup.s'], check=True)
ls=[l for l in open('/tmp/dup.s') if l.startswith('\t') and not l.lstrip().startswith('.')]
sp=sum(1 for l in ls if re.search(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)',l))
fm=sum(1 for l in ls if re.search(r'vf?n?m(add|sub)',l))
print(f"dups={len(chosen)} -> insns={len(ls)} spills={sp} fma={fm}")
