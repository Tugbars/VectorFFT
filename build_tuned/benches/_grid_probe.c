#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static const char *TN[] = {"c2c","r2c","c2r","dct1","dct2","dct3","dct4",
                           "dst1","dst2","dst3","dht"};
static void probe(const char *tag, vfft_config_t *c)
{
    vfft_plan p = vfft_create(c);
    printf("%-52s %s\n", tag, p ? "CREATE-OK" : "REFUSE");
    fflush(stdout);
    if (p) vfft_destroy(p);
}
static void base(vfft_config_t *c){ memset(c,0,sizeof *c); c->howmany=1; c->dims=1; }

int main(int argc, char**argv)
{
    vfft_config_t c; int sec = argc>1 ? atoi(argv[1]) : 0;
    if (sec==0) {
      int Ns[] = {256, 255, 127, 129, 4106, 115, 202, 2048, 45, 50};
      for (int i=0;i<10;i++){
        for (int lay=0; lay<2; lay++)
          for (int pl=0; pl<2; pl++){
            char t[128];
            base(&c); c.transform=VFFT_C2C; c.n[0]=Ns[i];
            c.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
            c.placement=pl?VFFT_OUTOFPLACE:VFFT_INPLACE;
            snprintf(t,sizeof t,"1D c2c N=%d %s %s K=1", Ns[i],
                     lay?"IL":"SP", pl?"OOP":"IP");
            probe(t,&c);
          }
      }
    }
    if (sec==1) {
      const char *on[]={"DEF","NAT","SCR"};
      for (int o=0;o<3;o++){
        char t[128];
        base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.order=o;
        c.layout=VFFT_LAYOUT_INTERLEAVED; c.placement=VFFT_INPLACE;
        snprintf(t,sizeof t,"1D c2c 256 IL IP order=%s",on[o]); probe(t,&c);
        base(&c); c.transform=VFFT_R2C; c.n[0]=256; c.order=o;
        snprintf(t,sizeof t,"1D r2c 256 SP OOP order=%s",on[o]);
        c.placement=VFFT_OUTOFPLACE; probe(t,&c);
        base(&c); c.transform=VFFT_DCT2; c.n[0]=256; c.order=o;
        snprintf(t,sizeof t,"1D dct2 256 order=%s",on[o]); probe(t,&c);
        base(&c); c.transform=VFFT_C2C; c.dims=2; c.n[0]=64;c.n[1]=64; c.order=o;
        c.layout=VFFT_LAYOUT_INTERLEAVED;
        snprintf(t,sizeof t,"2D c2c 64x64 IL order=%s",on[o]); probe(t,&c);
        base(&c); c.transform=VFFT_C2C; c.dims=2; c.n[0]=64;c.n[1]=64; c.order=o;
        snprintf(t,sizeof t,"2D c2c 64x64 SP order=%s",on[o]); probe(t,&c);
        base(&c); c.transform=VFFT_R2C; c.dims=2; c.n[0]=64;c.n[1]=64; c.order=o;
        c.layout=VFFT_LAYOUT_INTERLEAVED; c.placement=VFFT_OUTOFPLACE;
        snprintf(t,sizeof t,"2D r2c 64x64 IL OOP order=%s",on[o]); probe(t,&c);
        base(&c); c.transform=VFFT_R2C; c.dims=2; c.n[0]=64;c.n[1]=64; c.order=o;
        c.placement=VFFT_OUTOFPLACE;
        snprintf(t,sizeof t,"2D r2c 64x64 SP OOP order=%s",on[o]); probe(t,&c);
        base(&c); c.transform=VFFT_C2C; c.dims=3; c.n[0]=8;c.n[1]=8;c.n[2]=8; c.order=o;
        snprintf(t,sizeof t,"3D c2c 8^3 SP order=%s",on[o]); probe(t,&c);
      }
    }
    if (sec==2) {
      for (int d=1; d<=4; d++)
        for (int tr=0; tr<3; tr++)
          for (int lay=0; lay<2; lay++)
            for (int pl=0; pl<2; pl++){
              char t[160];
              base(&c); c.dims=d; c.transform=(vfft_transform_t)tr;
              c.n[0]=8;c.n[1]=8;c.n[2]=8;c.n[3]=8;
              c.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
              c.placement=pl?VFFT_OUTOFPLACE:VFFT_INPLACE;
              snprintf(t,sizeof t,"%dD %s 8.. %s %s K=1",d,TN[tr],
                       lay?"IL":"SP", pl?"OOP":"IP");
              probe(t,&c);
            }
      for (int d=1; d<=3; d++){ char t[128];
        base(&c); c.dims=d; c.transform=VFFT_DCT2; c.n[0]=64;c.n[1]=64;c.n[2]=64;
        snprintf(t,sizeof t,"%dD dct2 64.. SP IP",d); probe(t,&c); }
    }
    if (sec==3) {
      int Ks[]={1,2,3,8};
      for (int i=0;i<4;i++) for (int bg=0; bg<3; bg++) for (int lay=0;lay<2;lay++){
        char t[160];
        base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.howmany=Ks[i];
        c.batch_geom=bg; c.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
        c.placement=VFFT_INPLACE;
        snprintf(t,sizeof t,"1D c2c 256 K=%d %s bg=%d IP",Ks[i],lay?"IL":"SP",bg);
        probe(t,&c);
      }
      for (int i=0;i<4;i++) for (int bg=0; bg<3; bg++) for (int pl=0;pl<2;pl++){
        char t[160];
        base(&c); c.transform=VFFT_R2C; c.n[0]=256; c.howmany=Ks[i];
        c.batch_geom=bg; c.layout=VFFT_LAYOUT_INTERLEAVED;
        c.placement=pl?VFFT_OUTOFPLACE:VFFT_INPLACE;
        snprintf(t,sizeof t,"1D r2c 256 K=%d IL bg=%d %s",Ks[i],bg,pl?"OOP":"IP");
        probe(t,&c);
      }
      for (int i=0;i<3;i++) for (int lay=0;lay<2;lay++) for (int tr=0;tr<3;tr++){
        char t[160];
        base(&c); c.dims=2; c.transform=(vfft_transform_t)tr;
        c.n[0]=64;c.n[1]=64; c.howmany=(size_t)(i+1);
        c.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
        c.placement=tr?VFFT_OUTOFPLACE:VFFT_INPLACE;
        snprintf(t,sizeof t,"2D %s 64x64 K=%d %s",TN[tr],i+1,lay?"IL":"SP");
        probe(t,&c);
      }
    }
    if (sec==4) {
      int Ns[]={256,255,127,129,254,6};
      for (int i=0;i<6;i++) for (int tr=1; tr<=2; tr++)
        for (int lay=0;lay<2;lay++) for(int pl=0;pl<2;pl++){
          char t[160];
          base(&c); c.transform=(vfft_transform_t)tr; c.n[0]=Ns[i];
          c.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
          c.placement=pl?VFFT_OUTOFPLACE:VFFT_INPLACE;
          snprintf(t,sizeof t,"1D %s N=%d %s %s K=1",TN[tr],Ns[i],
                   lay?"IL":"SP",pl?"OOP":"IP");
          probe(t,&c);
        }
      int Nt[]={256,255,127};
      for (int i=0;i<3;i++) for (int tr=3; tr<=10; tr++){
        char t[160];
        base(&c); c.transform=(vfft_transform_t)tr; c.n[0]=Nt[i];
        snprintf(t,sizeof t,"1D %s N=%d",TN[tr],Nt[i]); probe(t,&c);
      }
    }
    if (sec==5) {
      for (int tr=0; tr<11; tr++) for (int pl=0;pl<2;pl++){
        char t[160];
        base(&c); c.transform=(vfft_transform_t)tr; c.n[0]=256; c.howmany=7;
        c.owned_buffers=1; c.placement=pl?VFFT_OUTOFPLACE:VFFT_INPLACE;
        snprintf(t,sizeof t,"owned 1D %s 256 K=7 %s",TN[tr],pl?"OOP":"IP");
        probe(t,&c);
      }
      {
        base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.howmany=7; c.owned_buffers=1;
        c.layout=VFFT_LAYOUT_INTERLEAVED; probe("owned 1D c2c IL",&c);
        base(&c); c.transform=VFFT_C2C; c.dims=2; c.n[0]=64;c.n[1]=64; c.owned_buffers=1;
        probe("owned 2D c2c",&c);
        base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.howmany=7; c.owned_buffers=1;
        c.order=VFFT_ORDER_NATURAL; probe("owned 1D c2c order=NAT",&c);
        base(&c); c.transform=VFFT_R2C; c.n[0]=255; c.howmany=7; c.owned_buffers=1;
        c.placement=VFFT_OUTOFPLACE; probe("owned 1D r2c odd N=255",&c);
        base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.howmany=7; c.owned_buffers=1;
        c.batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS; probe("owned 1D c2c bg=TC",&c);
      }
    }
    if (sec==6) {
      int dims[][2] = {{127,100},{100,127},{101,129},{64,63},{63,64},{127,127},{6,6}};
      for (int i=0;i<7;i++) for (int tr=0;tr<3;tr++) for (int lay=0;lay<2;lay++){
        char t[160];
        base(&c); c.dims=2; c.transform=(vfft_transform_t)tr;
        c.n[0]=dims[i][0]; c.n[1]=dims[i][1];
        c.layout=lay?VFFT_LAYOUT_INTERLEAVED:VFFT_LAYOUT_SPLIT;
        c.placement=tr?VFFT_OUTOFPLACE:VFFT_INPLACE;
        snprintf(t,sizeof t,"2D %s %dx%d %s",TN[tr],dims[i][0],dims[i][1],lay?"IL":"SP");
        probe(t,&c);
      }
    }
    if (sec==7) {
      base(&c); c.transform=(vfft_transform_t)11; c.n[0]=256;
      probe("transform=11 (out of range)",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.placement=(vfft_placement_t)5;
      probe("placement=5",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.layout=(vfft_layout_t)5;
      probe("layout=5",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.order=9; probe("order=9",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.rigor=(vfft_rigor_t)9; probe("rigor=9",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.dims=5; probe("dims=5",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.dims=0; probe("dims=0 (== 1D)",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=0; probe("n[0]=0",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.howmany=0; probe("howmany=0",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.batch_geom=9; probe("batch_geom=9 K=1",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=256; c.howmany=4; c.batch_geom=9;
      probe("batch_geom=9 K=4",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=1; probe("N=1 c2c",&c);
      base(&c); c.transform=VFFT_C2C; c.n[0]=2; probe("N=2 c2c",&c);
      base(&c); c.transform=VFFT_DCT2; c.n[0]=256; c.layout=VFFT_LAYOUT_INTERLEAVED;
      probe("dct2 IL",&c);
      base(&c); c.transform=VFFT_R2C; c.n[0]=256; c.dims=0; c.placement=VFFT_INPLACE;
      c.layout=VFFT_LAYOUT_INTERLEAVED; probe("r2c dims=0 IP IL even N",&c);
    }
    (void)argv;
    return 0;
}
