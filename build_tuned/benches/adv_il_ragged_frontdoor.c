/* adv_il_ragged_frontdoor.c -- ADVERSARIAL verification probe.
 * Does the PUBLIC API (vfft_create) actually attach an il2p plan whose
 * blocked structural default was refused by the odd-partner parity gate? */
#include "vfft.c"
#include <stdio.h>
#include <stdlib.h>

static const char *cls(vfft_il2p_fn got, vfft_il2p_fn mono)
{ return got == mono ? "MONO" : (got ? "blocked" : "NULL"); }

static void probe(int N, int placement, int layout, int order)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)placement;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = (vfft_layout_t)layout;
    cfg.order  = order;
    cfg.wisdom_write = 0;
    vfft_plan h = vfft_create(&cfg);
    if (!h) { printf("N=%-5d pl=%d lay=%d ord=%d : create NULL\n", N, placement, layout, order); return; }
    struct vfft_plan_s *p = (struct vfft_plan_s *)h;
    if (!p->k1il2p) {
        printf("N=%-5d pl=%d lay=%d ord=%d : k1il2p=NULL  il_route=%d k1_on=%d k1il3p=%p k1ilpr=%p\n",
               N, placement, layout, order, p->k1_il_route, p->k1_on,
               (void*)p->k1il3p, (void*)p->k1ilpr);
        vfft_destroy(h); return;
    }
    vfft_il2p_plan_t *q = p->k1il2p;
    printf("N=%-5d pl=%d lay=%d ord=%d : il2p %dx%d  leaf_f=%-7s mid_f=%-7s t2t_b=%-7s n1_b_r2=%-7s"
           "  [blk-leaf-exists=%d blk-n1b-exists=%d]\n",
           N, placement, layout, order, q->R1, q->R2,
           cls(q->leaf_f, vfft_il2p_leaf_fn(q->R2,0)),
           cls(q->mid_f,  vfft_il2p_mid_fn(q->R1,0)),
           cls(q->t2t_b,  vfft_il2p_t2t_bwd_fn(q->R1)),
           cls(q->n1_b_r2,vfft_il2p_n1_bwd_fn(q->R2)),
           vfft_il2p_leaf_v_fn(q->R2, 2, 1) || vfft_il2p_leaf_v_fn(q->R2, 1, 1) ? 1 : 0,
           vfft_il2p_n1_bwd_v_fn(q->R2, 2, 1) || vfft_il2p_n1_bwd_v_fn(q->R2, 1, 1) ? 1 : 0);
    vfft_destroy(h);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int one = argc > 1 ? atoi(argv[1]) : 0;
    static const int NS_[] = {480, 800, 864, 1600, 1728, 512, 1024, 240, 96};
    int NSbuf[9]; const int *NS = NS_; unsigned nn = sizeof NS_/sizeof NS_[0];
    if (one) { NSbuf[0] = one; NS = NSbuf; nn = 1; }
    for (unsigned i = 0; i < nn; i++) {
        probe(NS[i], VFFT_INPLACE,   VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT);
        probe(NS[i], VFFT_OUTOFPLACE,VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT);
        probe(NS[i], VFFT_OUTOFPLACE,VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL);
    }
    return 0;
}
