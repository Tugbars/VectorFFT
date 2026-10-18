"""The sample figure: the bins of the DFT of a real sequence of N = 8 samples. A VFFT_R2C plan
stores bins 0..N/2; the bins above N/2 are the conjugates of stored bins. Each bin's
frequency is written under it: bin k <= N/2 is frequency k, bin k > N/2 is frequency k - N.
The stored bins are blue; a bin that is not stored is an empty red outline, its text grey."""
import style

N = 8
f = style.Figure(height=100)
x0, y0, w, h = 66.0, 38.0, 40.0, 20.0
labels = ["$X[%d]$" % k for k in range(N // 2 + 1)] + ["$X^{*}[%d]$" % (N - k) for k in range(N // 2 + 1, N)]
stored = [k <= N // 2 for k in range(N)]
xs = f.cells(x0, y0, N, w=w, h=h, labels=labels,
             fills=[style.BLUE_T if s else None for s in stored],
             strokes=[style.BLUE if s else style.RED for s in stored],
             label_color=[style.INK if s else style.MUTED for s in stored])
right = x0 + N * w
cut = x0 + (N // 2 + 1) * w
f.brace(x0 + 1, cut - 1, y0 - 4, "stored: $N/2 + 1 = 5$ bins", below=False, color=style.BLUE)
f.brace(cut + 1, right - 1, y0 - 4, "not stored", below=False, color=style.RED)
# the rows under the cells: the index k, then the frequency it holds
ky, fy = y0 + h + 13.0, y0 + h + 31.0
f.text("bin $k$", x0 - 10, ky, anchor="r")
f.text("frequency", x0 - 10, fy, anchor="r")
freq = ["$%d$" % k for k in range(N // 2)] + ["$\\pm 4$"] + ["$-%d$" % (N - k) for k in range(N // 2 + 1, N)]
for k, cx in enumerate(xs):
    f.text("$%d$" % k, cx, ky)
    f.text(freq[k], cx, fy)
f.line(x0, ky + 6.5, right, ky + 6.5, lw=style.THIN, color=style.RULE)
f.save()
