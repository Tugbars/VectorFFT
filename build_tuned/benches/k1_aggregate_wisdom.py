#!/usr/bin/env python3
"""k1_aggregate_wisdom.py - median-of-runs K=1 wisdom writer (§13.1 stability
mandate): consume >=2 calibration dumps (cand,N,route,R1,R2,ns lines), take
the per-candidate MEDIAN across runs, recompute per-axis winners, and emit
kind-3 oop-wisdom lines (N 1 3 spr R1 R2 ilr iR1 iR2 ns).

Usage: python k1_aggregate_wisdom.py out_wisdom.txt dump1 [dump2 ...]
"""
import sys, re, statistics, collections

AXIS = {'3p': 0, '3p-ip': 1, '2pa-ip': 1, '2pb-ip': 1, 'twl-ip': 1,
        '3p-l3-ip': 1, '2pa-l3-ip': 1, 'mono': 1, 'mono-alt': 1,
        '2pb-spec': 1, '2pa-l3-spec': 1, 'jit-v3': 1,
        '3p-il': 2, '2p-il': 2, 'mono-il': 2}
SPID = {'3p-ip': 0, '3p': 0, '2pa-ip': 1, '2pb-ip': 2, 'twl-ip': 3,
        'mono': 4, 'mono-alt': 4, '2pa-l3-ip': 5, '3p-l3-ip': 6,
        '2pb-spec': 2, '2pa-l3-spec': 5}
# jit-v3 bakes a per-cell route (calibrate_k1.c's jr table); the wisdom line
# names the UNDERLYING route until the prefer_jit flag lands.
JIT_ROUTE = {512: 1, 1024: 2, 2048: 3, 4096: 5}
ILID = {'3p-il': 1, '2p-il': 2, 'mono-il': 3}
SP_CCOL = 7   # 'cc<code>' names: composed column, code = the chain digits


def axis_of(route):
    if route.startswith('cc'):
        return 1
    return AXIS.get(route)


def spid_of(n, route):
    """-> (spr, cc_code|None), or (None, None) if unmappable."""
    if route.startswith('cc'):
        try:
            return SP_CCOL, int(route[2:])
        except ValueError:
            return None, None
    if route == 'jit-v3':
        return JIT_ROUTE.get(n), None
    return SPID.get(route), None

def main():
    out_path, dumps = sys.argv[1], sys.argv[2:]
    samples = collections.defaultdict(list)   # (N, route, R1, R2) -> [ns...]
    for path in dumps:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                m = re.match(r'cand,(\d+),([\w\-]+),(\d+),(\d+),([\d.]+)', line)
                if m:
                    n, route, r1, r2, ns = m.groups()
                    samples[(int(n), route, int(r1), int(r2))].append(float(ns))
    cells = collections.defaultdict(list)     # N -> [(median, route, R1, R2)]
    for (n, route, r1, r2), v in samples.items():
        cells[n].append((statistics.median(v), route, r1, r2, len(v)))
    lines = []
    for n in sorted(cells):
        best = {}
        for med, route, r1, r2, cnt in cells[n]:
            ax = axis_of(route)
            if ax is None:
                continue
            # winners must be persistable: skip split candidates with no
            # route mapping (e.g. jit at an unlisted cell)
            if ax == 1 and spid_of(n, route)[0] is None:
                continue
            if ax not in best or med < best[ax][0]:
                best[ax] = (med, route, r1, r2, cnt)
        if 1 not in best:
            continue
        med, route, r1, r2, cnt = best[1]
        spr, cc_code = spid_of(n, route)
        if route == 'mono-alt':
            r1, r2 = 8, n // 8
        ilr, ir1, ir2 = 0, 0, 0
        if 2 in best:
            imed, iroute, ir1, ir2, icnt = best[2]
            ilr = ILID.get(iroute, 0)
        cc = f" {cc_code}" if spr == SP_CCOL else ""
        lines.append(f"{n} 1 3 {spr} {r1} {r2} {ilr} {ir1} {ir2}{cc} {med:.1f}")
        print(f"  N={n}: split={route} {r1}x{r2} ({med:.1f}ns, {cnt} runs)"
              + (f"  il={best[2][1]} {ir1}x{ir2} ({best[2][0]:.1f}ns)" if 2 in best else "  il=none"))
    with open(out_path, 'w') as f:
        f.write("# K=1 engine wisdom (kind 3): N 1 3 sp_route R1 R2 il_route iR1 iR2 ns\n")
        f.write("# written by k1_aggregate_wisdom.py (median of runs)\n")
        for ln in lines:
            f.write(ln + "\n")
    print(f"wrote {len(lines)} kind-3 lines -> {out_path}")

if __name__ == '__main__':
    main()
