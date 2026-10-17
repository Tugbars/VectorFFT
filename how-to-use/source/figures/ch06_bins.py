"""Figure 6.1: the bins of a VFFT_C2C spectrum of N = 8 points in natural order and the
frequency each holds: bin k <= N/2 holds frequency k, bin k > N/2 holds k - N; bin N/2 holds
both N/2 and -N/2."""
import style

N = 8
f = style.Figure(height=104)
x0, y0, w, h = 86.0, 34.0, 36.0, 20.0
xs = f.cells(x0, y0, N, w=w, h=h, labels=["$X[%d]$" % k for k in range(N)])
right = x0 + N * w
mid = x0 + (N // 2) * w
# the rule, above the cells: bins below N/2, the bin N/2, bins above N/2
f.brace(x0 + 1, mid - 1, y0 - 4, "frequency $k$", below=False)
f.text("$\\pm N/2$", mid + w / 2, y0 - 12)
f.brace(mid + w + 1, right - 1, y0 - 4, "frequency $k - N$", below=False)
# the rows under the cells: the bin, then the frequency it holds
ky, fy = y0 + h + 14.0, y0 + h + 33.0
f.text("bin $k$", x0 - 10, ky, anchor="r", color=style.MUTED)
f.text("frequency", x0 - 10, fy, anchor="r", color=style.MUTED)
freq = ["$%d$" % k for k in range(N // 2)] + ["$\\pm 4$"] + ["$%d$" % (k - N) for k in range(N // 2 + 1, N)]
for k, cx in enumerate(xs):
    f.text("$%d$" % k, cx, ky)
    f.text(freq[k], cx, fy)
f.line(x0, ky + 6.5, right, ky + 6.5, lw=style.THIN, color=style.RULE)
f.save()
