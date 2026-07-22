/* #7 gate: wisdom free/save parity — leaks (ASAN) + save->load->save idempotence. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
static long fsize(const char*p){ FILE*f=fopen(p,"rb"); if(!f)return -1;
    fseek(f,0,SEEK_END); long n=ftell(f); fclose(f); return n; }
static int fdiff(const char*a,const char*b){
    long x=fsize(a),y=fsize(b); if(x!=y) return 1; if(x<0) return -1;
    FILE*fa=fopen(a,"rb"),*fb=fopen(b,"rb"); int d=0;
    for(long i=0;i<x;i++) if(fgetc(fa)!=fgetc(fb)){d=1;break;}
    fclose(fa);fclose(fb); return d; }
int main(void){
    int all=1;
    /* leak loop: 3x load+free — LSAN reports on exit if parity broken */
    for(int i=0;i<3;i++){ vfft_wisdom*t=vfft_wisdom_load("/tmp/wb3"); if(t) vfft_wisdom_free(t); }
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wb3");
    if(!w){ puts("load A FAIL"); return 1; }
    if(vfft_wisdom_save(w,"/tmp/wb7B")){ puts("save B rc FAIL"); all=0; }
    const char* files[]={"spike_wisdom.txt","fft3d_c2c_wisdom.txt","fft2d_c2c_wisdom.txt",
                         "fft2d_r2c_wisdom.txt","fft2d_c2r_wisdom.txt","bluestein_wisdom.txt",
                         "oop_wisdom.txt","rfft_wisdom.txt"};
    char pa[768],pb[768];
    for(int i=0;i<8;i++){ snprintf(pb,sizeof pb,"/tmp/wb7B/%s",files[i]);
        long n=fsize(pb);
        printf("  B/%-24s %s (%ld B)\n",files[i],n>=0?"exists":"**MISSING**",n);
        all &= (n>=0); }
    vfft_wisdom *w2=vfft_wisdom_load("/tmp/wb7B");
    if(!w2){ puts("load B FAIL"); return 1; }
    if(vfft_wisdom_save(w2,"/tmp/wb7C")){ puts("save C rc FAIL"); all=0; }
    for(int i=0;i<8;i++){ snprintf(pa,sizeof pa,"/tmp/wb7B/%s",files[i]);
        snprintf(pb,sizeof pb,"/tmp/wb7C/%s",files[i]);
        int d=fdiff(pa,pb);
        printf("  idempotence %-24s %s\n",files[i],d==0?"BYTE-EQ":"**DIFF**");
        all &= (d==0); }
    vfft_wisdom_free(w); vfft_wisdom_free(w2);
    puts(all?"W7 GATE: ALL PASS (check LSAN output above/below)":"W7 GATE: FAILURES");
    return all?0:1; }
