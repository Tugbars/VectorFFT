── Benchmark: VectorFFT vs FFTW (ns, min-of-5, FFTW_ESTIMATE) ──
  N         factors            vfft_fwd   fftw_fwd   fwd_x    vfft_rt    fftw_rt    rt_x
  ──────────────────────────────────────────────────────────────────────────────────
  N=256     128x2             vfft_fwd=    470  fftw_fwd=    553  fwd_x=1.18  vfft_rt=   1318  fftw_rt=   1115  rt_x=0.85
  N=512     128x4             vfft_fwd=   1020  fftw_fwd=   1222  fwd_x=1.20  vfft_rt=   2516  fftw_rt=   2494  rt_x=0.99
  N=1024    128x8             vfft_fwd=   2688  fftw_fwd=   3049  fwd_x=1.13  vfft_rt=   5997  fftw_rt=   6628  rt_x=1.11
  N=2048    128x16            vfft_fwd=   6714  fftw_fwd=   6497  fwd_x=0.97  vfft_rt=  13510  fftw_rt=  13052  rt_x=0.97
  N=4096    128x32            vfft_fwd=  13367  fftw_fwd=  25138  fwd_x=1.88  vfft_rt=  27572  fftw_rt=  51407  rt_x=1.86
  N=8192    128x64            vfft_fwd=  35022  fftw_fwd=  50100  fwd_x=1.43  vfft_rt=  62786  fftw_rt= 111414  rt_x=1.77
  N=16384   128x128           vfft_fwd=  63473  fftw_fwd= 101863  fwd_x=1.60  vfft_rt= 147993  fftw_rt= 205391  rt_x=1.39
  N=32768   128x128x2         vfft_fwd= 152198  fftw_fwd= 293708  fwd_x=1.93  vfft_rt= 357315  fftw_rt= 557774  rt_x=1.56
  N=192     64x3              vfft_fwd=    343  fftw_fwd=    393  fwd_x=1.14  vfft_rt=    795  fftw_rt=    809  rt_x=1.02
  N=320     64x5              vfft_fwd=    595  fftw_fwd=    765  fwd_x=1.29  vfft_rt=   1263  fftw_rt=   1640  rt_x=1.30
  N=448     64x7              vfft_fwd=    875  fftw_fwd=   1443  fwd_x=1.65  vfft_rt=   2273  fftw_rt=   3004  rt_x=1.32
  N=384     128x3             vfft_fwd=    960  fftw_fwd=   1039  fwd_x=1.08  vfft_rt=   2144  fftw_rt=   2245  rt_x=1.05
  N=768     128x2x3           vfft_fwd=   1869  fftw_fwd=   2506  fwd_x=1.34  vfft_rt=   4216  fftw_rt=   4874  rt_x=1.16
  N=1536    128x4x3           vfft_fwd=   4271  fftw_fwd=   5871  fwd_x=1.37  vfft_rt=  10112  fftw_rt=  11961  rt_x=1.18
  N=3072    128x8x3           vfft_fwd=   9505  fftw_fwd=  12038  fwd_x=1.27  vfft_rt=  22209  fftw_rt=  25505  rt_x=1.15
  N=6144    128x16x3          vfft_fwd=  21073  fftw_fwd=  28060  fwd_x=1.33  vfft_rt=  45329  fftw_rt=  49968  rt_x=1.10
  N=200     8x5x5             vfft_fwd=    345  fftw_fwd=    500  fwd_x=1.45  vfft_rt=    703  fftw_rt=    979  rt_x=1.39
  N=400     16x5x5            vfft_fwd=    613  fftw_fwd=   1038  fwd_x=1.70  vfft_rt=   1299  fftw_rt=   2123  rt_x=1.63
  N=1000    8x5x5x5           vfft_fwd=   2408  fftw_fwd=   4097  fwd_x=1.70  vfft_rt=   5049  fftw_rt=   8114  rt_x=1.61
  N=2000    16x5x5x5          vfft_fwd=   4949  fftw_fwd=   8299  fwd_x=1.68  vfft_rt=  10870  fftw_rt=  15801  rt_x=1.45
  N=5000    8x5x5x5x5         vfft_fwd=  14758  fftw_fwd=  22325  fwd_x=1.51  vfft_rt=  32816  fftw_rt=  45283  rt_x=1.38
  N=10000   16x5x5x5x5        vfft_fwd=  30811  fftw_fwd=  51626  fwd_x=1.68  vfft_rt=  69882  fftw_rt= 114618  rt_x=1.64
  N=224     32x7              vfft_fwd=    344  fftw_fwd=    559  fwd_x=1.63  vfft_rt=    719  fftw_rt=   1274  rt_x=1.77
  N=896     128x7             vfft_fwd=   2651  fftw_fwd=   3161  fwd_x=1.19  vfft_rt=   5236  fftw_rt=   5966  rt_x=1.14
  N=1792    128x2x7           vfft_fwd=   5784  fftw_fwd=   6817  fwd_x=1.18  vfft_rt=  11365  fftw_rt=  13754  rt_x=1.21
  N=3584    128x4x7           vfft_fwd=  13455  fftw_fwd=  16502  fwd_x=1.23  vfft_rt=  28541  fftw_rt=  33234  rt_x=1.16
  N=120     8x5x3             vfft_fwd=    188  fftw_fwd=    252  fwd_x=1.34  vfft_rt=    406  fftw_rt=    604  rt_x=1.49
  N=240     16x5x3            vfft_fwd=    355  fftw_fwd=    575  fwd_x=1.62  vfft_rt=    706  fftw_rt=   1142  rt_x=1.62
  N=480     32x5x3            vfft_fwd=    775  fftw_fwd=   1505  fwd_x=1.94  vfft_rt=   1972  fftw_rt=   3374  rt_x=1.71
  N=960     64x5x3            vfft_fwd=   2604  fftw_fwd=   3450  fwd_x=1.32  vfft_rt=   6541  fftw_rt=   6489  rt_x=0.99
  N=1920    128x5x3           vfft_fwd=   6258  fftw_fwd=   7298  fwd_x=1.17  vfft_rt=  13082  fftw_rt=  14184  rt_x=1.08
  N=4800    64x5x5x3          vfft_fwd=  14731  fftw_fwd=  20177  fwd_x=1.37  vfft_rt=  30260  fftw_rt=  39939  rt_x=1.32
  N=88      8x11              vfft_fwd=    157  fftw_fwd=    222  fwd_x=1.42  vfft_rt=    327  fftw_rt=    441  rt_x=1.35
  N=704     64x11             vfft_fwd=   1800  fftw_fwd=   2405  fwd_x=1.34  vfft_rt=   3475  fftw_rt=   4858  rt_x=1.40
  N=5632    128x4x11          vfft_fwd=  21084  fftw_fwd=  29495  fwd_x=1.40  vfft_rt=  54631  fftw_rt=  66327  rt_x=1.21
  N=104     8x13              vfft_fwd=    177  fftw_fwd=    296  fwd_x=1.67  vfft_rt=    436  fftw_rt=    538  rt_x=1.23
  N=832     64x13             vfft_fwd=   2295  fftw_fwd=   2987  fwd_x=1.30  vfft_rt=   5299  fftw_rt=   7443  rt_x=1.40
  N=6656    128x4x13          vfft_fwd=  31836  fftw_fwd=  37646  fwd_x=1.18  vfft_rt=  67409  fftw_rt=  75153  rt_x=1.11
  N=136     8x17              vfft_fwd=    313  fftw_fwd=    580  fwd_x=1.86  vfft_rt=    585  fftw_rt=   1119  rt_x=1.91
  N=1088    64x17             vfft_fwd=   3692  fftw_fwd=   5843  fwd_x=1.58  vfft_rt=   7907  fftw_rt=  11843  rt_x=1.50
  N=152     8x19              vfft_fwd=    277  fftw_fwd=    857  fwd_x=3.10  vfft_rt=    653  fftw_rt=   1385  rt_x=2.12
  N=1216    64x19             vfft_fwd=   3497  fftw_fwd=   6904  fwd_x=1.97  vfft_rt=   7894  fftw_rt=  15184  rt_x=1.92
  N=184     8x23              vfft_fwd=    407  fftw_fwd=    900  fwd_x=2.21  vfft_rt=    885  fftw_rt=   1697  rt_x=1.92
  N=1472    64x23             vfft_fwd=   4530  fftw_fwd=   8872  fwd_x=1.96  vfft_rt=   9696  fftw_rt=  16292  rt_x=1.68
  N=12000   32x5x5x5x3        vfft_fwd=  39991  fftw_fwd=  67453  fwd_x=1.69  vfft_rt=  85610  fftw_rt= 141628  rt_x=1.65
  N=20000   32x5x5x5x5        vfft_fwd=  64694  fftw_fwd= 107404  fwd_x=1.66  vfft_rt= 149521  fftw_rt= 255588  rt_x=1.71
  N=40000   64x5x5x5x5        vfft_fwd= 161585  fftw_fwd= 246664  fwd_x=1.53  vfft_rt= 345580  fftw_rt= 500926  rt_x=1.45
The weak spots (sorted by fwd_x, lowest first):

N	factors	fwd_x	note
N=2048	128x16	0.97	only loss
N=256	128x2	1.18	small N, overhead dominated
N=512	128x4	1.20	same
N=6656	128x4x13	1.18	genfft R=13, large K strided
N=384	128x3	1.08	R=3 strided
N=1024	128x8	1.13	R=8 strided at large K
N=192	64x3	1.14	R=3 small
N=1920	128x5x3	1.17	R=3/5 strided
N=5632	128x4x11	1.40	genfft R=11
Pattern is clear: R=3 composites are consistently weak (1.08–1.37×), and large-K stages with genfft primes (R=11, R=13) plateau around 1.18–1.40×. The pow-2 sizes that are outliers (N=256/512/1024/2048) are just small-N overhead. N=2048 being <1× suggests the R=16 strided path at K=128 is particularly cache-unfriendly — that's exactly the case where pack+walk would help most.