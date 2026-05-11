import subprocess, re, statistics

NUM_RUNS = 5
data = {}  # (R, K) -> list of ratios

for run in range(NUM_RUNS):
    out = subprocess.run(['./regression_bench'], capture_output=True, text=True).stdout
    for line in out.split('\n'):
        m = re.match(r'R=(\d+)\s+K=(\d+)\s+Hand=\s*([0-9.]+) ns\s+OCaml=\s*([0-9.]+) ns\s+ratio=([0-9.]+)', line)
        if m:
            R, K = int(m.group(1)), int(m.group(2))
            ratio = float(m.group(5))
            data.setdefault((R, K), []).append(ratio)

print()
print("══════════════════════════════════════════════════════════════════════════")
print(f"  Stability summary across {NUM_RUNS} runs (ratio = OCaml time / Hand time)")
print("  ratio < 1 = OCaml WINS, ratio = 1 means TIE, ratio > 1 means OCaml slower")
print("══════════════════════════════════════════════════════════════════════════")
print(f"{'R':>4} {'K':>5} {'median':>8} {'min':>8} {'max':>8} {'spread':>8}  verdict")
for (R, K), ratios in sorted(data.items()):
    med = statistics.median(ratios)
    mn, mx = min(ratios), max(ratios)
    spread = mx - mn
    if med < 0.95: verdict = "OCaml WINS"
    elif med < 1.05: verdict = "TIE"
    elif med < 1.15: verdict = "OCaml SLOWER"
    else: verdict = "REGRESSION"
    print(f"{R:>4} {K:>5} {med:>8.3f} {mn:>8.3f} {mx:>8.3f} {spread:>8.3f}  {verdict}")
