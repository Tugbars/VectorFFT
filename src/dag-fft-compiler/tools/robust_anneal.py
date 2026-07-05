#!/usr/bin/env python3
"""Minimax (robust) schedule search: minimize the WORST regret across
targets T = {gcc, clang}, regret_t = spills_t / floor_t, with per-target
FMA-invariance gates. Seeded from the gcc-annealed order.
Floors: gcc order-floor 51 (annealed), clang best-known 126 (same order)."""
import os, re, sys, math, random, subprocess, hashlib
ITERS = int(sys.argv[1]) if len(sys.argv)>1 else 200
random.seed(31)
GENDIR="/home/claude/vectorfft/VectorFFT-dev-arbitraryTail/src/dag-fft-compiler/generator"
GEN=GENDIR+"/_build/default/bin/gen_radix.exe"
W="/home/claude/sched_search/robust_work"; os.makedirs(W,exist_ok=True)
FLOORS={"gcc":51.0,"clang":126.0}
def run_gen(env,out):
    e=dict(os.environ,VFFT_NO_ANYK_TAIL="1",**env)
    with open(out,"w") as f: subprocess.run([GEN,"13","--in-place","--isa","avx2","--su","--emit-c"],cwd=GENDIR,env=e,stdout=f,stderr=subprocess.DEVNULL)
def parse(sf):
    ins=sp=fm=0
    for l in open(sf):
        if l.startswith("\t") and not l.lstrip().startswith("."):
            ins+=1
            if re.search(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)',l):sp+=1
            if re.search(r'vf?n?m(add|sub)',l):fm+=1
    return ins,sp,fm
memo={}
def score(order):
    k=hashlib.md5(",".join(map(str,order)).encode()).hexdigest()
    if k in memo: return memo[k]
    of=f"{W}/o.txt"; open(of,"w").write("\n".join(map(str,order))+"\n")
    run_gen({"VFFT_SCHED_ORDER":of},f"{W}/c.c")
    res={}
    for cc in ("gcc","clang"):
        r=subprocess.run([cc,"-O3","-mavx2","-mfma","-march=raptorlake","-w","-S",f"{W}/c.c","-o",f"{W}/{cc}.s"],capture_output=True)
        res[cc]=parse(f"{W}/{cc}.s") if r.returncode==0 else None
    memo[k]=res; return res
# DAG from dump; seed = annealed order
dump=f"{W}/d.txt"; run_gen({"VFFT_SCHED_DUMP":dump},f"{W}/su.c")
preds={}; su=[]
for l in open(dump):
    l=l.strip()
    if not l or l.startswith("#"):continue
    h,_,r=l.partition(":"); t=int(h); su.append(t)
    preds[t]=[int(x) for x in r.split()] if r.strip() else []
succs={t:[] for t in su}
for t in su:
    for p in preds[t]: succs[p].append(t)
seed=[int(x) for x in open("best_r13_s1.txt") if x.strip() and not x.startswith("#")]
def reinsert(o):
    o=o[:]; i=random.randrange(len(o)); n=o.pop(i)
    pos={t:j for j,t in enumerate(o)}
    lo=max((pos[p] for p in preds[n] if p in pos),default=-1)+1
    hi=min((pos[s] for s in succs[n] if s in pos),default=len(o))
    o.insert(random.randint(lo,hi),n); return o
base=score(seed); bg,bc=base["gcc"],base["clang"]
def regret(r):
    if r["gcc"] is None or r["clang"] is None: return None
    if r["gcc"][2]!=bg[2] or r["clang"][2]!=bc[2]: return None   # FMA gates
    return max(r["gcc"][1]/FLOORS["gcc"], r["clang"][1]/FLOORS["clang"])
cur=seed; curR=regret(base); best,bestR,bestr=seed,curR,base
print(f"seed(annealed): gcc={bg[0]}/{bg[1]} clang={bc[0]}/{bc[1]}  max-regret={curR:.3f}",flush=True)
T=0.02
for it in range(ITERS):
    cand=reinsert(cur); r=score(cand); rr=regret(r)
    if rr is None: continue
    if rr<curR or random.random()<math.exp(-(rr-curR)/max(T,1e-6)):
        cur,curR=cand,rr
        if rr<bestR:
            best,bestR,bestr=cand,rr,r
            print(f"  it={it} NEW BEST max-regret={rr:.3f} gcc={r['gcc'][1]} clang={r['clang'][1]}",flush=True)
    T*=0.985
print(f"DONE minimax: max-regret={bestR:.3f}  gcc={bestr['gcc'][0]}/{bestr['gcc'][1]}  clang={bestr['clang'][0]}/{bestr['clang'][1]}",flush=True)
open("best_minimax_r13.txt","w").write("\n".join(map(str,best))+"\n")
