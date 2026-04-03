"""
VectorFFT vs FFTW vs Intel MKL — Performance Comparison Plots

Split-complex batched 1D FFT (double precision, single-threaded AVX2)

Usage: python plot_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Raw benchmark data: (N, K, ours_ns, fftw_ns, mkl_ns)
data = [
    (256, 32, 5673.0, 16990.2, 12152.5),
    (256, 256, 57635.0, 143570.0, 103925.0),
    (256, 1024, 402075.0, 715240.0, 572265.0),
    (1024, 32, 31723.3, 70260.0, 57470.0),
    (1024, 256, 492590.0, 837665.0, 687885.0),
    (1024, 1024, 2492880.0, 3338950.0, 3155740.0),
    (4096, 32, 324230.0, 601395.0, 362305.0),
    (4096, 256, 3086035.0, 5588065.0, 3645745.0),
    (4096, 1024, 23391000.0, 38359380.0, 23078850.0),
    (16384, 32, 1806235.0, 5481325.0, 1956690.0),
    (16384, 256, 18436605.0, 101104860.0, 20619815.0),
    (60, 32, 784.4, 3239.8, 3544.4),
    (60, 256, 6993.8, 29001.5, 33958.5),
    (60, 1024, 41160.0, 129720.0, 148810.0),
    (200, 32, 4659.6, 12372.4, 13636.5),
    (200, 256, 34840.0, 108945.0, 114995.0),
    (200, 1024, 308150.0, 604695.0, 592895.0),
    (1000, 32, 30454.8, 75819.4, 80054.8),
    (1000, 256, 460385.0, 957270.0, 871445.0),
    (1000, 1024, 2326215.0, 4064430.0, 3982745.0),
    (5000, 32, 302330.0, 1121350.0, 670720.0),
    (5000, 256, 3876140.0, 10980025.0, 5602105.0),
    (5000, 1024, 21513075.0, 76916320.0, 34268740.0),
    (10000, 32, 656845.0, 3215560.0, 1398145.0),
    (10000, 256, 9070750.0, 61916520.0, 12500250.0),
    (10000, 1024, 51884655.0, 350036390.0, 76945505.0),
    (20000, 32, 1842905.0, 7983290.0, 3089815.0),
    (20000, 256, 23031115.0, 134442775.0, 31710880.0),
    (50000, 32, 5864495.0, 36718520.0, 8664815.0),
    (50000, 256, 62886105.0, 806097790.0, 89504545.0),
    (49, 32, 717.9, 3397.3, 3011.5),
    (49, 256, 6850.6, 27603.8, 28340.5),
    (143, 32, 3853.7, 14797.7, 12068.8),
    (143, 256, 42244.4, 92359.3, 99463.0),
    (875, 32, 26528.6, 91602.9, 77565.7),
    (875, 256, 357155.0, 1094155.0, 807425.0),
    (2401, 32, 116315.0, 306000.0, 252030.0),
    (2401, 256, 1254015.0, 4987360.0, 2724385.0),
    (8000, 256, 6804680.0, 20866825.0, 9484270.0),
    (12000, 256, 12823690.0, 57693790.0, 17381185.0),
    (25000, 256, 29877375.0, 174054100.0, 42920425.0),
    (40000, 256, 51255220.0, 632291525.0, 71515695.0),
    (100000, 32, 15887030.0, 109881560.0, 24456365.0),
]

N_arr = np.array([d[0] for d in data])
K_arr = np.array([d[1] for d in data])
ours_ns = np.array([d[2] for d in data])
fftw_ns = np.array([d[3] for d in data])
mkl_ns  = np.array([d[4] for d in data])

# Throughput: GB/s = (2 * N * K * 8 bytes) / time_ns
data_bytes = 2.0 * N_arr * K_arr * 8
ours_gbs = data_bytes / ours_ns
fftw_gbs = data_bytes / fftw_ns
mkl_gbs  = data_bytes / mkl_ns

def is_pow2(n):
    return n > 0 and (n & (n-1)) == 0

print(f'{len(data)} test cases loaded\n')

# ═══════════════════════════════════════════════════════════════
# Plot 1: Throughput (GB/s) vs FFT Length
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))

for lib, gbs, marker, base_color, label_base in [
    ('ours', ours_gbs, 'o', ['#e41a1c', '#ff6b6b', '#ffaaaa'], 'VectorFFT'),
    ('mkl',  mkl_gbs,  's', ['#377eb8', '#74b3e8', '#b3d4f0'], 'Intel MKL'),
    ('fftw', fftw_gbs, '^', ['#4daf4a', '#8dd38b', '#c4e8c3'], 'FFTW'),
]:
    for ki, K_val in enumerate([32, 256, 1024]):
        mask = K_arr == K_val
        if not np.any(mask):
            continue
        ax.scatter(N_arr[mask], gbs[mask],
                   marker=marker, s=60, alpha=0.85,
                   color=base_color[ki], edgecolors='black', linewidth=0.3,
                   label=f'{label_base} K={K_val}',
                   zorder=5 if lib == 'ours' else 3)

ax.set_xlabel('FFT Length (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Throughput (GB/s)', fontsize=13, fontweight='bold')
ax.set_title('FP64 Batched 1D Split-Complex FFT — Single Thread AVX2\n'
             'VectorFFT vs Intel MKL vs FFTW', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('vfft_throughput.png', dpi=200, bbox_inches='tight')
print('Saved: vfft_throughput.png')
plt.show()

# ═══════════════════════════════════════════════════════════════
# Plot 2: Speedup ratios — grouped bar chart per K
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ki, K_val in enumerate([32, 256, 1024]):
    ax = axes[ki]
    mask = K_arr == K_val
    Ns = N_arr[mask]
    vs_fftw = fftw_ns[mask] / ours_ns[mask]
    vs_mkl  = mkl_ns[mask] / ours_ns[mask]

    order = np.argsort(Ns)
    Ns = Ns[order]
    vs_fftw = vs_fftw[order]
    vs_mkl = vs_mkl[order]

    x = np.arange(len(Ns))
    w = 0.35

    bars1 = ax.bar(x - w/2, vs_fftw, w, label='vs FFTW', color='#4daf4a', alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, vs_mkl,  w, label='vs MKL',  color='#377eb8', alpha=0.8,
                   edgecolor='black', linewidth=0.5)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Break-even')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n):,}' for n in Ns], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'K = {K_val}', fontsize=13, fontweight='bold')
    ax.set_xlabel('FFT Length (N)', fontsize=11)
    if ki == 0:
        ax.set_ylabel('Speedup (higher = better)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, vs_mkl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=7, fontweight='bold')

fig.suptitle('VectorFFT Speedup over FFTW and Intel MKL\n'
             'FP64 Split-Complex, Single Thread AVX2, i9-14900KF',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('vfft_speedup_bars.png', dpi=200, bbox_inches='tight')
print('Saved: vfft_speedup_bars.png')
plt.show()

# ═══════════════════════════════════════════════════════════════
# Plot 3: Speedup vs MKL scatter
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))

vs_mkl_all = mkl_ns / ours_ns
pow2_mask = np.array([is_pow2(n) for n in N_arr])
composite_mask = ~pow2_mask

for K_val, marker, size in [(32, 'o', 80), (256, 's', 80), (1024, 'D', 80)]:
    mask = K_arr == K_val

    m = mask & pow2_mask
    if np.any(m):
        ax.scatter(N_arr[m], vs_mkl_all[m], marker=marker, s=size,
                   color='#e41a1c', edgecolors='black', linewidth=0.5,
                   alpha=0.9, label=f'pow2 K={K_val}', zorder=5)

    m = mask & composite_mask
    if np.any(m):
        ax.scatter(N_arr[m], vs_mkl_all[m], marker=marker, s=size,
                   color='#377eb8', edgecolors='black', linewidth=0.5,
                   alpha=0.9, label=f'composite K={K_val}', zorder=5)

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.fill_between([10, 200000], 0, 1, color='red', alpha=0.05)
ax.text(50, 0.85, 'MKL wins', fontsize=12, color='red', alpha=0.5, fontstyle='italic')
ax.text(50, 1.15, 'VectorFFT wins', fontsize=12, color='green', alpha=0.5, fontstyle='italic')

ax.set_xlabel('FFT Length (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup over Intel MKL', fontsize=13, fontweight='bold')
ax.set_title('VectorFFT vs Intel MKL — FP64 Batched Split-Complex FFT\n'
             'Single Thread AVX2, i9-14900KF', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_ylim(0, 5.5)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('vfft_vs_mkl.png', dpi=200, bbox_inches='tight')
print('Saved: vfft_vs_mkl.png')
plt.show()

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

vs_fftw_all = fftw_ns / ours_ns

print('\n' + '='*50)
print('VectorFFT vs Intel MKL:')
print(f'  Min:    {vs_mkl_all.min():.2f}x')
print(f'  Max:    {vs_mkl_all.max():.2f}x')
print(f'  Mean:   {vs_mkl_all.mean():.2f}x')
print(f'  Median: {np.median(vs_mkl_all):.2f}x')
print(f'  Wins:   {np.sum(vs_mkl_all > 1.0)}/{len(vs_mkl_all)}')
print()
print('VectorFFT vs FFTW:')
print(f'  Min:    {vs_fftw_all.min():.2f}x')
print(f'  Max:    {vs_fftw_all.max():.2f}x')
print(f'  Mean:   {vs_fftw_all.mean():.2f}x')
print(f'  Median: {np.median(vs_fftw_all):.2f}x')
print()
comp = np.array([not is_pow2(n) for n in N_arr])
print(f'Composite N vs MKL:  mean={vs_mkl_all[comp].mean():.2f}x  median={np.median(vs_mkl_all[comp]):.2f}x')
p2 = np.array([is_pow2(n) for n in N_arr])
print(f'Pow2 N vs MKL:       mean={vs_mkl_all[p2].mean():.2f}x  median={np.median(vs_mkl_all[p2]):.2f}x')
