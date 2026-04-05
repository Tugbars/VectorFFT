"""
VectorFFT Benchmark Plotter
Reads vfft_bench_results.csv + hardcoded prime data, generates PNG plots.

Usage:
    cd src/stride-fft/bench
    python plot_bench.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import os, sys

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

def is_pow2(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0

# ============================================================
# 1. Load main benchmark CSV
# ============================================================
csv_path = 'vfft_bench_results.csv'
if not os.path.exists(csv_path):
    csv_path = '../../build/bin/vfft_bench_results.csv'
if not os.path.exists(csv_path):
    print(f'ERROR: cannot find vfft_bench_results.csv'); sys.exit(1)

N_list, K_list, fac_list, ours_list, fftw_list, mkl_list = [], [], [], [], [], []
with open(csv_path) as f:
    f.readline()  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 8: continue
        try:
            N = int(parts[1])
            K = int(parts[2])
            fac = parts[3]
            ours = float(parts[4])
            # parts[5], parts[6] may be empty (heur_factors, heur_ns)
            # Collect all remaining floats
            nums = []
            for p in parts[5:]:
                p = p.strip()
                if p == '': continue
                try: nums.append(float(p))
                except ValueError: continue
            fftw = nums[0] if len(nums) >= 1 else float('nan')
            mkl  = nums[1] if len(nums) >= 2 else float('nan')
        except (ValueError, IndexError):
            continue
        N_list.append(N); K_list.append(K); fac_list.append(fac)
        ours_list.append(ours); fftw_list.append(fftw); mkl_list.append(mkl)

N_arr   = np.array(N_list, dtype=np.float64)
K_arr   = np.array(K_list, dtype=np.float64)
ours_ns = np.array(ours_list, dtype=np.float64)
fftw_ns = np.array(fftw_list, dtype=np.float64)
mkl_ns  = np.array(mkl_list, dtype=np.float64)
has_mkl = not np.all(np.isnan(mkl_ns))

print(f'Loaded {len(N_arr)} entries from {csv_path}, MKL={has_mkl}')

# ============================================================
# 2. Prime-N data (from bench_primes.c output)
# ============================================================
prime_data = [
    (29,   32,  'Rader',      1554.0,    4177.1,    4040.4),
    (29,   256, 'Rader',     15413.4,   31888.8,   34436.6),
    (61,   32,  'Rader',      3947.5,   15347.5,   21360.2),
    (61,   256, 'Rader',     52935.9,  136439.1,  176145.3),
    (97,   32,  'Rader',      6554.3,   21009.9,   21537.0),
    (97,   256, 'Rader',     70412.5,  158405.0,  180640.0),
    (127,  32,  'Rader',     10663.0,   25496.3,   19887.8),
    (127,  256, 'Rader',    101743.3,  213443.3,  170630.0),
    (181,  32,  'Rader',     15969.8,   55572.1,   44326.7),
    (181,  256, 'Rader',    185585.7,  422781.0,  353771.4),
    (251,  32,  'Rader',     21208.1,   50468.5,   43514.5),
    (251,  256, 'Rader',    252870.0,  397600.0,  356855.0),
    (337,  32,  'Rader',     30583.7,  101342.4,   98204.3),
    (337,  256, 'Rader',    314470.0,  744420.0,  810210.0),
    (401,  32,  'Rader',     36284.4,  113502.6,   90350.6),
    (401,  256, 'Rader',    386085.0,  839830.0,  723670.0),
    (449,  32,  'Rader',     44153.6,  114984.1,   87981.2),
    (449,  256, 'Rader',    466395.0,  916740.0,  783030.0),
    (53,   32,  'Rader',      3601.2,   10792.5,   15778.6),
    (53,   256, 'Rader',     41164.4,  101637.0,  127817.8),
    (131,  32,  'Rader',     10869.7,   29173.5,   31450.8),
    (131,  256, 'Rader',    111206.9,  261958.6,  270310.3),
    (263,  32,  'Bluestein',  65076.3,   67723.7,   71399.2),
    (263,  256, 'Bluestein', 589105.0,  518260.0,  592320.0),
    (509,  32,  'Bluestein', 123093.4,  106660.7,   96406.6),
    (509,  256, 'Bluestein',1204805.0,  860055.0,  824385.0),
    (1021, 32,  'Rader',    142653.3,  250993.3,  198966.7),
    (1021, 256, 'Rader',   1268530.0, 2134140.0, 1926505.0),
    (2053, 32,  'Rader',    333055.0,  678235.0,  729975.0),
    (2053, 256, 'Rader',   2901360.0, 7316630.0, 6512640.0),
]

p_N    = np.array([d[0] for d in prime_data], dtype=np.float64)
p_K    = np.array([d[1] for d in prime_data], dtype=np.float64)
p_meth = np.array([d[2] for d in prime_data])
p_ours = np.array([d[3] for d in prime_data], dtype=np.float64)
p_fftw = np.array([d[4] for d in prime_data], dtype=np.float64)
p_mkl  = np.array([d[5] for d in prime_data], dtype=np.float64)
p_vs_fftw = p_fftw / p_ours
p_vs_mkl  = p_mkl / p_ours

# ============================================================
# Plot 1: Throughput scatter
# ============================================================
data_bytes = 2.0 * N_arr * K_arr * 8
ours_gbs = data_bytes / ours_ns
fftw_gbs = data_bytes / fftw_ns
mkl_gbs  = data_bytes / mkl_ns

fig, ax = plt.subplots(figsize=(14, 7))
used = set()
for lib, gbs, marker, colors, base in [
    ('ours', ours_gbs, 'o', ['#e41a1c','#ff6b6b','#ffaaaa'], 'VectorFFT'),
    ('mkl',  mkl_gbs,  's', ['#377eb8','#74b3e8','#b3d4f0'], 'Intel MKL'),
    ('fftw', fftw_gbs, '^', ['#4daf4a','#8dd38b','#c4e8c3'], 'FFTW'),
]:
    for ki, Kv in enumerate([32, 256, 1024]):
        m = K_arr == Kv
        if not np.any(m): continue
        lbl = f'{base} K={Kv}'
        ax.scatter(N_arr[m], gbs[m], marker=marker, s=60, alpha=0.85,
                   color=colors[ki], edgecolors='black', linewidth=0.3,
                   label=lbl if lbl not in used else '_nolegend_',
                   zorder=5 if lib == 'ours' else 3)
        used.add(lbl)

ax.set_xlabel('FFT Length (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Throughput (GB/s)', fontsize=13, fontweight='bold')
ax.set_title('FP64 Batched 1D Split-Complex FFT -- Single Thread AVX2\n'
             'VectorFFT vs Intel MKL vs FFTW', fontsize=14, fontweight='bold')
ax.set_xscale('log'); ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('vfft_throughput.png', dpi=200, bbox_inches='tight')
print('Saved vfft_throughput.png')
plt.close()

# ============================================================
# Plot 2: Speedup bars per K
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ki, Kv in enumerate([32, 256, 1024]):
    ax = axes[ki]
    m = K_arr == Kv
    if not np.any(m): continue
    Ns = N_arr[m]; vf = fftw_ns[m]/ours_ns[m]; vm = mkl_ns[m]/ours_ns[m]
    o = np.argsort(Ns); Ns=Ns[o]; vf=vf[o]; vm=vm[o]
    x = np.arange(len(Ns)); w = 0.35
    ax.bar(x-w/2, vf, w, label='vs FFTW', color='#4daf4a', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x+w/2, vm, w, label='vs MKL',  color='#377eb8', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n):,}' for n in Ns], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'K = {Kv}', fontsize=13, fontweight='bold')
    ax.set_xlabel('FFT Length (N)')
    if ki == 0: ax.set_ylabel('Speedup (higher = better)')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

fig.suptitle('VectorFFT Speedup over FFTW and Intel MKL\n'
             'FP64 Split-Complex, Single Thread AVX2, i9-14900KF',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('vfft_speedup_bars.png', dpi=200, bbox_inches='tight')
print('Saved vfft_speedup_bars.png')
plt.close()

# ============================================================
# Plot 3: vs MKL scatter
# ============================================================
vs_mkl_all = mkl_ns / ours_ns
fig, ax = plt.subplots(figsize=(14, 7))
pow2_m = np.array([is_pow2(n) for n in N_arr], dtype=bool)

used = set()
for Kv, marker, sz in [(32,'o',80),(256,'s',80),(1024,'D',80)]:
    m = K_arr == Kv
    m2 = m & pow2_m
    if np.any(m2):
        lbl = f'pow2 K={Kv}'
        ax.scatter(N_arr[m2], vs_mkl_all[m2], marker=marker, s=sz, color='#e41a1c',
                   edgecolors='black', linewidth=0.5, alpha=0.9,
                   label=lbl if lbl not in used else '_nolegend_', zorder=5)
        used.add(lbl)
    mc = m & ~pow2_m
    if np.any(mc):
        lbl = f'composite K={Kv}'
        ax.scatter(N_arr[mc], vs_mkl_all[mc], marker=marker, s=sz, color='#377eb8',
                   edgecolors='black', linewidth=0.5, alpha=0.9,
                   label=lbl if lbl not in used else '_nolegend_', zorder=5)
        used.add(lbl)

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.fill_between([10, 200000], 0, 1, color='red', alpha=0.05)
ax.set_xlabel('FFT Length (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup over Intel MKL', fontsize=13, fontweight='bold')
ax.set_title('VectorFFT vs Intel MKL', fontsize=14, fontweight='bold')
ax.set_xscale('log'); ax.set_ylim(0, float(np.nanmax(vs_mkl_all)) * 1.15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('vfft_vs_mkl.png', dpi=200, bbox_inches='tight')
print('Saved vfft_vs_mkl.png')
plt.close()

# ============================================================
# Plot 4: Prime-N bars
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for ax, Kv, title in [(ax1, 32, 'K = 32'), (ax2, 256, 'K = 256')]:
    m = p_K == Kv
    o = np.argsort(p_N[m])
    Ns = p_N[m][o]; meths = p_meth[m][o]
    vf = p_vs_fftw[m][o]; vm = p_vs_mkl[m][o]

    x = np.arange(len(Ns)); w = 0.35
    c_fw = ['#4daf4a' if mt == 'Rader' else '#ff7f00' for mt in meths]
    c_mk = ['#377eb8' if mt == 'Rader' else '#e41a1c' for mt in meths]

    bars_fw = ax.bar(x - w/2, vf, w, color=c_fw, edgecolor='black', linewidth=0.5, alpha=0.8)
    bars_mk = ax.bar(x + w/2, vm, w, color=c_mk, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    labels = [f'{int(n)}\n({"R" if mt == "Rader" else "B"})' for n, mt in zip(Ns, meths)]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Prime N'); ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars_mk, vm):
        color = 'green' if val >= 1.0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=7,
                fontweight='bold', color=color)

ax1.set_ylabel('Speedup (higher = better)', fontsize=11)

legend_elements = [
    Patch(facecolor='#4daf4a', edgecolor='black', label='Rader (VectorFFT) vs FFTW'),
    Patch(facecolor='#377eb8', edgecolor='black', label='Rader (VectorFFT) vs MKL'),
    Patch(facecolor='#ff7f00', edgecolor='black', label='Bluestein (VectorFFT) vs FFTW'),
    Patch(facecolor='#e41a1c', edgecolor='black', label='Bluestein (VectorFFT) vs MKL'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='upper right')

fig.suptitle('VectorFFT Prime-N: Rader & Bluestein vs FFTW & Intel MKL\n'
             'FP64 Split-Complex, Single Thread AVX2, i9-14900KF\n'
             'R = Rader (smooth prime)  |  B = Bluestein (non-smooth prime)',
             fontsize=12, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('vfft_primes.png', dpi=200, bbox_inches='tight')
print('Saved vfft_primes.png')
plt.close()

# ============================================================
# Summary
# ============================================================
vs_fftw_all = fftw_ns / ours_ns
rader_m = p_meth == 'Rader'
blue_m  = p_meth == 'Bluestein'
pow2_m  = np.array([is_pow2(n) for n in N_arr], dtype=bool)

print('\n=== VectorFFT Benchmark Summary ===')
print(f'vs FFTW:  {vs_fftw_all.min():.2f}x - {vs_fftw_all.max():.2f}x  (mean {vs_fftw_all.mean():.2f}x)')
print(f'vs MKL:   {vs_mkl_all.min():.2f}x - {vs_mkl_all.max():.2f}x  (mean {vs_mkl_all.mean():.2f}x)')
print(f'Wins vs MKL: {np.sum(vs_mkl_all > 1.0)}/{len(vs_mkl_all)}')
print(f'\nComposite vs MKL: mean {vs_mkl_all[~pow2_m].mean():.2f}x')
print(f'Pow2 vs MKL:      mean {vs_mkl_all[pow2_m].mean():.2f}x')
print(f'\nRader primes:    {p_vs_mkl[rader_m].min():.2f}x - {p_vs_mkl[rader_m].max():.2f}x vs MKL (mean {p_vs_mkl[rader_m].mean():.2f}x)')
print(f'Bluestein primes: {p_vs_mkl[blue_m].min():.2f}x - {p_vs_mkl[blue_m].max():.2f}x vs MKL (mean {p_vs_mkl[blue_m].mean():.2f}x)')
print('\nDone. Generated: vfft_throughput.png, vfft_speedup_bars.png, vfft_vs_mkl.png, vfft_primes.png')
