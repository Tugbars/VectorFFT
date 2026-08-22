/* n2_zr2c_refute.c -- ADVERSARIAL verification of the claimed N=2 zr2c hole.
 * Claim: half==1 has no zr2c child => N=2 has no C2R and no in-place R2C. */
#include "vfft.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double dre_[64], sre_[64], zbuf_[64], ref_[64];

static void hdr(const char *t){ printf("\n===== %s =====\n", t); }

/* naive real DFT forward -> CCE (N/2+1 pairs) */
static void ref_r2c(const double *x, int N, double *X)
{
    for (int k = 0; k <= N/2; k++) {
        double re=0, im=0;
        for (int n = 0; n < N; n++) {
            double a = -2.0*M_PI*(double)k*(double)n/(double)N;
            re += x[n]*cos(a); im += x[n]*sin(a);
        }
        X[2*k]=re; X[2*k+1]=im;
    }
}
/* naive c2r backward from CCE, unnormalised (sum over full hermitian spectrum) */
static void ref_c2r(const double *X, int N, double *x)
{
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int k = 0; k < N; k++) {
            double re, im;
            if (k <= N/2) { re = X[2*k]; im = X[2*k+1]; }
            else          { re = X[2*(N-k)]; im = -X[2*(N-k)+1]; }
            double a = 2.0*M_PI*(double)k*(double)n/(double)N;
            s += re*cos(a) - im*sin(a);
        }
        x[n]=s;
    }
}

static void probe(vfft_transform_t tr, int N, int placement, int layout, int K)
{
    vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
    cfg.transform = tr;
    cfg.placement = (vfft_placement_t)placement;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = (size_t)K;
    cfg.layout = (vfft_layout_t)layout;
    cfg.wisdom_write = 0;
    const char *tn = tr==VFFT_R2C?"r2c":(tr==VFFT_C2R?"c2r":"c2c");
    printf("[%s N=%-4d K=%d %s %s] ", tn, N, K,
           placement?"IP ":"OOP", layout==VFFT_LAYOUT_INTERLEAVED?"IL   ":"SPLIT");
    fflush(stdout);
    vfft_plan h = vfft_create(&cfg);
    if (!h) { printf("-> CREATE REFUSED\n"); return; }
    struct vfft_plan_s *p = (struct vfft_plan_s *)h;
    printf("-> created (zr2c_child=%s route=%d rplan=%p c2rdisp=%p) ",
           p->zr2c_child?"yes":"NO ", p->zr2c_route,
           (void*)p->rplan, (void*)p->c2rdisp);
    fflush(stdout);
    /* numeric check only for the interleaved K=1 real shapes */
    if (layout == VFFT_LAYOUT_INTERLEAVED && K == 1 &&
        (tr == VFFT_R2C || tr == VFFT_C2R)) {
        for (int i=0;i<64;i++){ sre_[i]=dre_[i]=zbuf_[i]=ref_[i]=0; }
        if (tr == VFFT_R2C) {
            for (int i=0;i<N;i++) sre_[i] = 0.3 + 0.7*(double)(i+1);
            ref_r2c(sre_, N, ref_);
            if (placement == VFFT_INPLACE) {
                for (int i=0;i<N;i++) zbuf_[i]=sre_[i];
                vfft_execute(h, VFFT_FORWARD, zbuf_, NULL, zbuf_, NULL);
                memcpy(dre_, zbuf_, sizeof(double)*(size_t)(N+2));
            } else {
                vfft_execute(h, VFFT_FORWARD, sre_, NULL, dre_, NULL);
            }
            double e=0,n=0;
            for (int i=0;i<N+2;i++){ double d=dre_[i]-ref_[i]; e+=d*d; n+=ref_[i]*ref_[i]; }
            printf("rel=%.3g  got[", n>0?sqrt(e/n):sqrt(e));
            for (int i=0;i<N+2;i++) printf("%s%.4f", i?",":"", dre_[i]);
            printf("]");
        } else {
            for (int k=0;k<=N/2;k++){ zbuf_[2*k]=1.0+k; zbuf_[2*k+1]=(k==0||k==N/2)?0.0:(0.5*k); }
            ref_c2r(zbuf_, N, ref_);
            if (placement == VFFT_INPLACE) {
                vfft_execute(h, VFFT_BACKWARD, zbuf_, NULL, zbuf_, NULL);
                memcpy(dre_, zbuf_, sizeof(double)*(size_t)N);
            } else {
                vfft_execute(h, VFFT_BACKWARD, zbuf_, NULL, dre_, NULL);
            }
            double e=0,n=0;
            for (int i=0;i<N;i++){ double d=dre_[i]-ref_[i]; e+=d*d; n+=ref_[i]*ref_[i]; }
            printf("rel=%.3g  got[", n>0?sqrt(e/n):sqrt(e));
            for (int i=0;i<N;i++) printf("%s%.4f", i?",":"", dre_[i]);
            printf("] ref[");
            for (int i=0;i<N;i++) printf("%s%.4f", i?",":"", ref_[i]);
            printf("]");
        }
    }
    printf("\n");
    vfft_destroy(h);
}

static void c2cprobe(int N, int placement, int layout, int order)
{
    vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)placement;
    cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=(vfft_layout_t)layout; cfg.order=order;
    cfg.wisdom_write = 0;
    printf("[c2c N=%-3d %s %s ord=%d] ", N, placement?"IP ":"OOP",
           layout==VFFT_LAYOUT_INTERLEAVED?"IL   ":"SPLIT", order);
    fflush(stdout);
    vfft_plan h = vfft_create(&cfg);
    printf("-> %s\n", h?"created":"CREATE REFUSED");
    if (h) vfft_destroy(h);
}

int main(void)
{
    hdr("A: c2c N=1 (the zr2c child at half==1)");
    c2cprobe(1, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL);
    c2cprobe(1, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL);
    c2cprobe(1, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT);
    c2cprobe(1, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT);
    c2cprobe(2, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL);
    c2cprobe(2, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL);

    hdr("B: N=2 INTERLEAVED real, all four cells (the claim)");
    probe(VFFT_R2C, 2, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_R2C, 2, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_C2R, 2, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_C2R, 2, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, 1);

    hdr("C: N=4 control (same cells, one rung up)");
    probe(VFFT_R2C, 4, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_R2C, 4, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_C2R, 4, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_C2R, 4, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, 1);

    hdr("D: N=2 SPLIT layout (is the hole layout-specific or whole-N?)");
    probe(VFFT_R2C, 2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT, 1);
    probe(VFFT_C2R, 2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT, 1);
    probe(VFFT_R2C, 2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT, 4);
    probe(VFFT_C2R, 2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT, 4);

    hdr("E: N=6 (half=3, odd child) sanity");
    probe(VFFT_R2C, 6, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_C2R, 6, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, 1);
    probe(VFFT_C2R, 6, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, 1);
    return 0;
}
