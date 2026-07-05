import sys
dump,kf,ordf=sys.argv[1],sys.argv[2],sys.argv[3]
kind={}
for l in open(kf): t,k=l.split(); kind[int(t)]=k
preds={}; su=[]
for l in open(dump):
    l=l.strip()
    if not l or l.startswith('#'):continue
    h,_,r=l.partition(':'); t=int(h); su.append(t)
    preds[t]=[int(x) for x in r.split()] if r.strip() else []
users={t:[] for t in su}
for t in su:
    for p in preds[t]: users[p].append(t)
order=[int(x) for x in open(ordf) if x.strip() and not x.startswith('#')]
pos={t:i for i,t in enumerate(order)}
nu={t:sorted(pos[u] for u in users[t]) for t in order}
cur={t:0 for t in order}; F=set(); stored=set(); tr=0; ml=0
def nxt(v,s):
    while cur[v]<len(nu[v]) and nu[v][cur[v]]<=s: cur[v]+=1
    return nu[v][cur[v]] if cur[v]<len(nu[v]) else 1<<30
for s,v in enumerate(order):
    for p in preds[v]:
        if kind[p]=='C':continue
        if p not in F:
            if len(F)>=16:
                ev=max((w for w in F if w not in preds[v]),key=lambda w:nxt(w,s))
                if nxt(ev,s)<(1<<30) and ev not in stored: tr+=1; stored.add(ev)
                F.discard(ev)
            F.add(p); tr+=1
    for p in preds[v]:
        if kind[p]!='C' and nxt(p,s)>=(1<<30): F.discard(p)
    if kind[v]!='C' and users[v]:
        if len(F)>=16:
            ev=max(F,key=lambda w:nxt(w,s))
            if nxt(ev,s)<(1<<30) and ev not in stored: tr+=1; stored.add(ev)
            F.discard(ev)
        F.add(v)
    ml=max(ml,len(F))
print(f"{tr} {ml}")
