#!/usr/bin/env python3
"""Beam-Belady scheduler (v3): beam search over ready choices; state
"""RACED AND LOST vs SU (doc 69): best model traffic 90 vs SU 30. Kept as a recorded negative."""
scored by (belady_traffic_so_far, sum_live_so_far); expansion capped by
per-state candidate pruning (top-c by one-step key). Deterministic.
Usage: beam_sched.py dump kinds R_budget beam_width > order.txt"""
import sys, heapq
dump,kf,RB,BW = sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
CAND=8
kind={}
for l in open(kf):
    t,k=l.split(); kind[int(t)]=k
preds={}; nodes=[]
for l in open(dump):
    l=l.strip()
    if not l or l.startswith('#'): continue
    h,_,r=l.partition(':'); t=int(h); nodes.append(t)
    preds[t]=[int(x) for x in r.split()] if r.strip() else []
users={t:[] for t in nodes}
for t in nodes:
    for p in preds[t]: users[p].append(t)
hgt={}
for t in reversed(nodes):
    hgt[t]=(5 if kind[t]=='L' else 0 if kind[t]=='C' else 4)+max((hgt[u] for u in users[t]),default=0)
NU={t:len(users[t]) for t in nodes}
NP={t:sum(1 for p in preds[t]) for t in nodes}
def dist(v,remu,unsp,sched):
    d=99
    for u in users[v]:
        if u in sched: continue
        m=unsp[u]
        if m<d: d=m
    return d
def step(state,n):
    sched,ready,F,remu,unsp,traf,sl,out = state
    F=set(F); remu=dict(remu); unsp=dict(unsp); sched=set(sched); ready=set(ready)
    k=kind[n]; t=traf
    for p in preds[n]:
        if kind[p]!='C' and p not in F:
            F.add(p); t+=1                      # reload
    for p in preds[n]:
        if kind[p]!='C':
            remu[p]-=1
            if remu[p]==0: F.discard(p)
    if k!='C' and remu[n]>0:
        while len(F)>=RB:
            v=max(F,key=lambda x:(dist(x,remu,unsp,sched),x))
            F.discard(v)
            if remu[v]>0: t+=1                  # spill store (reload charged on return)
        F.add(n)
    sched.add(n); ready.discard(n)
    for u in users[n]:
        unsp[u]-=1
        if unsp[u]==0: ready.add(u)
    return (frozenset(sched),frozenset(ready),frozenset(F),
            tuple(sorted(remu.items())),tuple(sorted(unsp.items())),
            t, sl+len(F), out+(n,))
remu0={t:NU[t] for t in nodes}; unsp0={t:NP[t] for t in nodes}
ready0=frozenset(t for t in nodes if NP[t]==0)
init=(frozenset(),ready0,frozenset(),tuple(sorted(remu0.items())),
      tuple(sorted(unsp0.items())),0,0,())
beam=[init]
for _ in range(len(nodes)):
    nxt={}
    for st in beam:
        sched,ready,F,remuT,unspT,traf,sl,out=st
        remu=dict(remuT); unsp=dict(unspT)
        # one-step candidate pruning: cheap key
        def key1(n):
            miss=sum(1 for p in preds[n] if kind[p]!='C' and p not in F)
            kills=sum(1 for p in preds[n] if kind[p]!='C' and remu[p]==1 and p in F)
            return (miss,-kills,-hgt[n],n)
        cands=sorted(ready,key=key1)[:CAND]
        for n in cands:
            ns=step((set(sched),set(ready),F,remu,unsp,traf,sl,out),n)
            sig=(ns[0],)                          # dedupe on scheduled-set
            sc=(ns[5],ns[6])
            if sig not in nxt or sc<nxt[sig][0]:
                nxt[sig]=(sc,ns)
    beam=[v[1] for v in sorted(nxt.values(),key=lambda x:x[0])[:BW]]
best=min(beam,key=lambda s:(s[5],s[6]))
sys.stderr.write(f"traffic={best[5]} sumlive={best[6]}\n")
print("\n".join(map(str,best[7])))
