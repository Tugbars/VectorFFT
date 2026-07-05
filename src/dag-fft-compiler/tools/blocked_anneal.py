#!/usr/bin/env python3
"""Blocked-codelet schedule search: coordinate descent over clusters.

For blocked CT codelets (R>=25), su_schedule_subset runs once per cluster and
the injector is keyed per-subset ("<prefix>_<key>.txt"). Clusters are
independent for LEGALITY, but gcc allocates over the whole function, so local
optima need not compose — therefore: MOVES are per-cluster, SCORING is always
global (whole codelet compiled).

Objective (Phase-0-validated, unchanged): minimize global total insns,
gated on FMA invariance and spills never above the su baseline.

Extra step: isomorphic transfer. Clusters with identical node counts are
instances of the same sub-DFT; a winning index-permutation on one cluster is
tested once on each same-size sibling.

Usage: blocked_anneal.py R [iters_per_cluster] [seed]
"""
import os, re, sys, math, random, subprocess, hashlib, glob, shutil, time

R     = int(sys.argv[1]) if len(sys.argv) > 1 else 32
CITER = int(sys.argv[2]) if len(sys.argv) > 2 else 80
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 1
random.seed(SEED)

GENDIR = "/home/claude/vectorfft/VectorFFT-dev-arbitraryTail/src/dag-fft-compiler/generator"
GEN    = GENDIR + "/_build/default/bin/gen_radix.exe"
BASE   = f"/home/claude/sched_search/blocked_r{R}"
DUMPD  = BASE + "/dump"
ORDD   = BASE + "/ord"      # incumbent order files (what gets injected)
GCC    = ["gcc", "-O3", "-mavx2", "-mfma", "-march=raptorlake", "-w", "-S"]
GENARGS = [str(R), "--in-place", "--isa", "avx2", "--su", "--emit-c"]

SPILL_RE = re.compile(r'vmov(ap|up)[ds].*\((%rsp|%rbp)\)')
FMA_RE   = re.compile(r'vf?n?m(add|sub)')

def run_gen(env_extra, out_c):
    env = dict(os.environ, VFFT_NO_ANYK_TAIL="1", **env_extra)
    with open(out_c, "w") as f:
        subprocess.run([GEN] + GENARGS, cwd=GENDIR, env=env,
                       stdout=f, stderr=subprocess.DEVNULL)

def dump_subsets():
    # RESUME FIX (2026-07-02): ORDD is cross-run resume state — never
    # wipe it (a full BASE rmtree here made every seed restart from the
    # SU baseline; a 50-iter campaign was burned to this). Only the
    # dump scratch is cleared; stale per-key node-sets are detected and
    # replaced at load time.
    shutil.rmtree(DUMPD, ignore_errors=True)
    os.makedirs(DUMPD, exist_ok=True)
    os.makedirs(ORDD, exist_ok=True)
    run_gen({"VFFT_SCHED_DUMP": DUMPD + "/sub"}, BASE + "/su.c")
    clusters = {}   # key -> (order:list, preds:dict)
    for f in glob.glob(DUMPD + "/sub_*.txt"):
        key = os.path.basename(f)[4:-4]
        order, preds = [], {}
        for line in open(f):
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                if line.startswith("#dagsig "): DAGSIG[key] = line.split()[1]
                continue
            head, _, rest = line.partition(":")
            t = int(head)
            order.append(t)
            preds[t] = [int(x) for x in rest.split()] if rest.strip() else []
        clusters[key] = (order, preds)
        # RESUME FIX (2026-07-02): this used to unconditionally clobber
        # the shared incumbent dir with the SU order at startup, so a
        # new seed always restarted from baseline (we burned a 50-iter
        # campaign to this). Seed ORDD only when no incumbent exists;
        # existing files are the resume state. Node-set changes (e.g.
        # a DAG transform flag) are caught below: an incumbent whose
        # node set mismatches is replaced by SU.
        f = f"{ORDD}/ord_{key}.txt"
        if os.path.exists(f):
            o = [int(l) for l in open(f).read().splitlines()
                 if l.strip() and not l.startswith("#")]
            if sorted(o) == sorted(order):
                clusters[key] = (o, preds)
            else:
                write_ord(key, order)
        else:
            write_ord(key, order)
    return clusters

def write_ord(key, order):
    with open(f"{ORDD}/ord_{key}.txt", "w") as f:
        if key in DAGSIG: f.write(f"#dagsig {DAGSIG[key]}\n")
        f.write("\n".join(map(str, order)) + "\n")

memo = {}
DAGSIG = {}
def score_global(orders):
    """orders: dict key->order for ALL clusters. Writes all files, compiles
    whole codelet, returns (insns, spills, fma) or None."""
    sig = hashlib.md5(
        ";".join(k + ":" + ",".join(map(str, o))
                 for k, o in sorted(orders.items())).encode()).hexdigest()
    if sig in memo:
        return memo[sig]
    for k, o in orders.items():
        write_ord(k, o)
    cf, sf = BASE + "/cand.c", BASE + "/cand.s"
    if os.path.exists(sf): os.remove(sf)
    run_gen({"VFFT_SCHED_ORDER": ORDD + "/ord"}, cf)
    r = subprocess.run(GCC + [cf, "-o", sf], capture_output=True)
    if r.returncode != 0 or not os.path.exists(sf):
        memo[sig] = None
        return None
    insns = spills = fma = 0
    for l in open(sf):
        if l.startswith("\t") and not l.lstrip().startswith("."):
            insns += 1
            if SPILL_RE.search(l): spills += 1
            if FMA_RE.search(l):   fma += 1
    memo[sig] = (insns, spills, fma)
    return memo[sig]

# --- per-cluster move operators (same semantics as anneal_linux.py) ---
def reinsert(order, preds, succs):
    o = order[:]
    i = random.randrange(len(o))
    node = o.pop(i)
    pos = {t: j for j, t in enumerate(o)}
    lo = max((pos[p] for p in preds[node] if p in pos), default=-1) + 1
    hi = min((pos[s] for s in succs[node] if s in pos), default=len(o))
    o.insert(random.randint(lo, hi), node)
    return o

def block_move(order, preds, succs):
    n = len(order)
    if n < 4: return order
    i = random.randrange(n)
    j = min(n, i + random.randint(2, 6))
    block = order[i:j]; bset = set(block)
    rest = order[:i] + order[j:]
    pos = {t: k for k, t in enumerate(rest)}
    lo = -1; hi = len(rest)
    for nd in block:
        for p in preds[nd]:
            if p in pos and p not in bset: lo = max(lo, pos[p])
        for s in succs[nd]:
            if s in pos and s not in bset: hi = min(hi, pos[s])
    lo += 1
    if lo > hi: return order
    p = random.randint(lo, hi)
    return rest[:p] + block + rest[p:]

def segment_reverse(order, preds, succs):
    n = len(order)
    if n < 3: return order
    i = random.randrange(n)
    j = min(n, i + random.randint(2, 8))
    seg = order[i:j]; sset = set(seg)
    for nd in seg:
        if any(p in sset for p in preds[nd]):
            return order
    return order[:i] + seg[::-1] + order[j:]

def propose(order, preds, succs):
    r = random.random()
    if r < 0.6:    return reinsert(order, preds, succs)
    elif r < 0.85: return block_move(order, preds, succs)
    else:          return segment_reverse(order, preds, succs)

def apply_index_perm(target_su_order, source_su_order, source_best):
    """Map source cluster's best order onto an isomorphic sibling by rank:
    node at su-position i in source moved to rank j  =>  same move for the
    node at su-position i in target."""
    idx = {t: i for i, t in enumerate(source_su_order)}
    perm = [idx[t] for t in source_best]          # ranks in new order
    return [target_su_order[i] for i in perm]

def legal(order, preds):
    seen = set()
    pset = set(order)
    for t in order:
        if any(p in pset and p not in seen for p in preds[t]):
            return False
        seen.add(t)
    return True

def main():
    t0 = time.time()
    clusters = dump_subsets()
    su_orders = {k: o[:] for k, (o, _) in clusters.items()}
    succs_of = {}
    for k, (order, preds) in clusters.items():
        s = {t: [] for t in order}
        for t in order:
            for p in preds[t]:
                if p in s: s[p].append(t)
        succs_of[k] = s

    incumbent = {k: o[:] for k, o in su_orders.items()}
    warm = os.environ.get("VFFT_WARM")
    if warm:
        loaded = 0
        for k in incumbent:
            import glob as _g
            cands = _g.glob(f"{warm}/*_{k}.txt") + [f"{warm}/ord_{k}.txt"]
            f = next((c for c in cands if os.path.exists(c)), cands[-1])
            if os.path.exists(f):
                o = [int(l) for l in open(f).read().splitlines()
                     if l.strip() and not l.strip().startswith("#")]
                if sorted(o) == sorted(su_orders[k]) and legal(o, clusters[k][1]):
                    incumbent[k] = o
                    loaded += 1
        print(f"warm start: {loaded}/{len(incumbent)} cluster orders from {warm}")
    if warm:
        su_score = score_global({k: o[:] for k, o in su_orders.items()})
        print(f"su reference: insns={su_score[0]} spills={su_score[1]}")
    base = score_global(incumbent)
    if incumbent != su_orders and base:
        print(f"resume/warm base: insns={base[0]} spills={base[1]}")
    b_ins, b_sp, b_fma = base
    print(f"R={R} clusters={len(clusters)} "
          f"sizes={sorted((len(o) for o, _ in clusters.values()), reverse=True)}")
    print(f"su baseline: insns={b_ins} spills={b_sp} fma={b_fma}  "
          f"iters/cluster={CITER} seed={SEED}", flush=True)

    cur_ins, cur_sp = b_ins, b_sp
    order_keys = sorted(clusters, key=lambda k: -len(clusters[k][0]))
    cluster_best = {}   # key -> improved order (for isomorphic transfer)

    for ci, key in enumerate(order_keys):
        order, preds = clusters[key]
        succs = succs_of[key]
        n = len(order)
        T = max(2.0, cur_ins * 0.004)
        local = incumbent[key][:]
        local_ins = cur_ins
        improved = False
        for it in range(CITER):
            cand = propose(local, preds, succs)
            if cand == local: continue
            trial = dict(incumbent); trial[key] = cand
            r = score_global(trial)
            if r is None: continue
            ins, sp, fma = r
            if fma != b_fma or sp > b_sp:
                continue
            if ins < local_ins or random.random() < math.exp(-(ins - local_ins) / max(T, 1e-6)):
                local, local_ins = cand, ins
                if ins < cur_ins or (ins == cur_ins and sp < cur_sp):
                    incumbent[key] = cand
                    cur_ins, cur_sp = ins, sp
                    improved = True
                    print(f"  [{ci+1}/{len(order_keys)}] cluster {key} (n={n}) "
                          f"it={it:3d} GLOBAL BEST insns={ins} spills={sp}", flush=True)
            T *= 0.97
        if improved:
            cluster_best[key] = incumbent[key][:]
            # isomorphic transfer to same-size siblings
            for k2 in order_keys:
                if k2 == key or len(clusters[k2][0]) != n: continue
                cand2 = apply_index_perm(su_orders[k2], su_orders[key], incumbent[key])
                if not legal(cand2, clusters[k2][1]): continue
                trial = dict(incumbent); trial[k2] = cand2
                r = score_global(trial)
                if r is None: continue
                ins, sp, fma = r
                if fma == b_fma and sp <= b_sp and (ins < cur_ins or (ins == cur_ins and sp < cur_sp)):
                    incumbent[k2] = cand2
                    cur_ins, cur_sp = ins, sp
                    print(f"    transfer {key} -> {k2}: insns={ins} spills={sp}", flush=True)
        # keep incumbent files on disk consistent
        score_global(incumbent)

    win = cur_ins < b_ins
    strong = win and cur_sp < b_sp
    print(f"\nDONE in {time.time()-t0:.0f}s  su({b_ins}/{b_sp}) -> best({cur_ins}/{cur_sp})"
          f"  d_insns={b_ins-cur_ins} d_spills={b_sp-cur_sp}"
          f"  {'STRONG WIN' if strong else 'WIN' if win else 'no win'}", flush=True)
    if win:
        outd = f"best_blocked_r{R}_s{SEED}"
        shutil.rmtree(outd, ignore_errors=True)
        os.makedirs(outd)
        for k, o in incumbent.items():
            with open(f"{outd}/ord_{k}.txt", "w") as f:
                if k in DAGSIG: f.write(f"#dagsig {DAGSIG[k]}\n")
                f.write("\n".join(map(str, o)) + "\n")
        print(f"incumbent orders -> {outd}/", flush=True)

if __name__ == "__main__":
    main()
