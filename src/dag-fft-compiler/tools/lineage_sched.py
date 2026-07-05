#!/usr/bin/env python3
"""Lineage/Belady-guided list scheduler (SU's successor prototype).
"""RACED AND LOST vs SU (doc 69): model traffic 110/98-class vs SU 30. Kept as a recorded negative."""
Priority (lexicographic) for ready node n against a SIMULATED regfile F:
  1. traffic(n) = reloads forced (operands not in F) + evictions of
     still-needed values caused by (missing operands + result) overflow
  2. -kills(n)   (operands whose last use is n)
  3. heir bonus  (n consumes the value defined in the previous step, or
     n is the max-height user of a value it kills — MRIS lineage)
  4. cp_dist DESC (issue tie-break only), 5. tag ASC.
Consts are register-free (rematerializable) and schedulable only when hot.
Eviction: DAG-Belady — evict the value whose nearest unscheduled user is
farthest from ready (max over F of min over its users of unsched-pred count).
Usage: lineage_sched.py dump kinds R_budget > order.txt"""
import sys
dump,kf,RB = sys.argv[1],sys.argv[2],int(sys.argv[3])
kind={}
for l in open(kf):
    t,k=l.split(); kind[int(t)]=k
preds={}; order0=[]
for l in open(dump):
    l=l.strip()
    if not l or l.startswith('#'): continue
    h,_,r=l.partition(':'); t=int(h); order0.append(t)
    preds[t]=[int(x) for x in r.split()] if r.strip() else []
users={t:[] for t in order0}
for t in order0:
    for p in preds[t]: users[p].append(t)
# heights (cp with unit-ish lat) and heirs
LAT={'L':5,'C':0}
h={}
for t in reversed(order0):
    h[t]=LAT.get(kind[t],4)+max((h[u] for u in users[t]),default=0)
heir={}
for t in order0:
    if users[t]: heir[t]=max(users[t],key=lambda u:(h[u],-u))
remu={t:len(users[t]) for t in order0}
unsp={t:len(preds[t]) for t in order0}
ready={t for t in order0 if unsp[t]==0}
sched=set(); F=set(); out=[]; last_def=None
def dist_to_ready(v):   # nearest unscheduled user's missing-pred count
    d=99
    for u in users[v]:
        if u in sched: continue
        m=sum(1 for p in preds[u] if p not in sched)
        if m<d: d=m
    return d
def hot_const(c):
    return any(unsp[u]==1 for u in users[c])
H=None
def prio(n):
    global H
    k=kind[n]
    miss=[p for p in preds[n] if kind[p]!='C' and p not in F]
    need=(1 if k!='C' and remu[n]>0 else 0)
    kills=sum(1 for p in preds[n] if kind[p]!='C' and remu[p]==1 and p in F)
    over=max(0,len(F)+len(miss)+need-kills-RB)
    ev=0
    if over>0:
        cands=sorted((v for v in F if v not in preds[n]),key=lambda v:-dist_to_ready(v))
        for v in cands[:over]:
            if remu[v]>0: ev+=1
    traffic=len(miss)+ev
    cont = 0 if n==H else (1 if H is not None and n in preds.get(H,[]) else 2)
    return (traffic,cont,-kills,-h[n],n)
while len(sched)<len(order0):
    H = heir.get(last_def) if last_def is not None else None
    if H is not None and H in sched: H=None
    cands=[n for n in ready if kind[n]!='C' or hot_const(n)
           or (H is not None and n in preds.get(H,[]))] or list(ready)
    n=min(cands,key=prio)
    k=kind[n]
    # apply: reload misses, evict, define
    for p in preds[n]:
        if kind[p]!='C' and p not in F: F.add(p)          # reload (traffic counted in prio)
    for p in preds[n]:
        if kind[p]!='C':
            remu[p]-=1
            if remu[p]==0: F.discard(p)
    if k!='C' and remu[n]>0:
        while len(F)>=RB:
            v=max((x for x in F),key=dist_to_ready); F.discard(v)
        F.add(n)
    sched.add(n); ready.discard(n); out.append(n)
    if k!='C': last_def=n
    for u in users[n]:
        unsp[u]-=1
        if unsp[u]==0: ready.add(u)
print("\n".join(map(str,out)))
