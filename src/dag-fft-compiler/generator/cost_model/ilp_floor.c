// ilp_floor.c — exhaustive/B&B ILP floor for small codelet DAGs.
//
// Model (Raptor Lake AVX2, from uarch.ml + schedule.ml port classes):
//   latencies: A(dd/sub)=4 N(eg)=4 M(ul)=4 F(ma)=4 L(oad)=5 C(onst)=0 X(cmul)=4,2uops
//   ports/cycle: mul+fma <= 2 (P0,P1); mul+fma+add/neg <= 3 (P0,P1,P5); loads <= 3
//   consts: free (no port, latency 0).
// Metric: FINISH makespan = max(issue + latency).
//
// In-order issue: instr i issues at max(operands-finished, issue[i-1],
// first cycle with class capacity). Window-W: per cycle, issue any ready
// instr among the first W un-issued in program order (oldest first),
// capacity-limited. W=INF ~= dataflow greedy.
//
// Usage: ilp_floor dump.txt dump.txt.kinds [bnb_node_cap]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXN 512
#define HOR  16384
static int N;
static int lat[MAXN]; static char kind[MAXN];
static int npred[MAXN], pred[MAXN][8];
static int nsucc[MAXN], succ[MAXN][16];
static int tag2i[1<<16], tags[MAXN];
static int cp_tail[MAXN];            // finish-distance to a sink, inclusive

static int lat_of(char k){ return k=='L'?5 : k=='C'?0 : 4; }
static int is_mf(char k){ return k=='M'||k=='F'||k=='X'; }
static int is_fp(char k){ return is_mf(k)||k=='A'||k=='N'; }
static int uops_mf(char k){ return k=='X'?2 : is_mf(k)?1:0; }
static int uops_fp(char k){ return k=='X'?2 : is_fp(k)?1:0; }

// ---- one full in-order simulation of a permutation ----
static unsigned gen;
static unsigned mf_g[HOR],fp_g[HOR],ld_g[HOR];
static unsigned char mf[HOR],fp[HOR],ld[HOR];
#define GET(a,t) ((a##_g[t]==gen)?a[t]:0)
#define ADD(a,t,v) do{ if(a##_g[t]!=gen){a##_g[t]=gen;a[t]=0;} a[t]+=v; }while(0)
static int sim_inorder(const int *ord){
  static int fin[MAXN];
  gen++;
  int prev_issue=0, mk=0;
  for(int s=0;s<N;s++){
    int v=ord[s], t=prev_issue;
    for(int j=0;j<npred[v];j++){ int f=fin[pred[v][j]]; if(f>t)t=f; }
    char k=kind[v];
    if(k=='C'){ fin[v]=t; if(t>mk)mk=t; continue; }
    if(k=='L'){ while(GET(ld,t)>=3)t++; ADD(ld,t,1); }
    else{
      int um=uops_mf(k), uf=uops_fp(k);
      while(GET(mf,t)+um>2 || GET(fp,t)+uf>3) t++;
      ADD(mf,t,um); ADD(fp,t,uf);
    }
    prev_issue=t; fin[v]=t+lat[v]; if(fin[v]>mk)mk=fin[v];
  }
  return mk;
}

// ---- window-W greedy OoO simulation of a permutation ----
static int sim_window(const int *ord, int W){
  static int fin[MAXN]; static char done[MAXN];
  memset(done,0,N); for(int i=0;i<N;i++) fin[i]=1<<29;
  int issued=0, head=0, t=0, mk=0;
  while(issued<N){
    int mfu=0,fpu=0,ldu=0;
    while(head<N && done[ord[head]]) head++;
    int lim = head+W; if(W<=0||lim>N) lim=N;
    for(int s=head; s<lim; s++){
      int v=ord[s]; if(done[v])continue;
      int rdy=1; for(int j=0;j<npred[v];j++) if(!done[pred[v][j]]||fin[pred[v][j]]>t){rdy=0;break;}
      if(!rdy)continue;
      char k=kind[v];
      if(k=='C'){ done[v]=1; fin[v]=t; issued++; if(t>mk)mk=t; continue; }
      if(k=='L'){ if(ldu>=3)continue; ldu++; }
      else{ int um=uops_mf(k),uf=uops_fp(k); if(mfu+um>2||fpu+uf>3)continue; mfu+=um; fpu+=uf; }
      done[v]=1; fin[v]=t+lat[v]; issued++; if(fin[v]>mk)mk=fin[v];
    }
    t++;
    if(t>=HOR){fprintf(stderr,"horizon\n");exit(1);}
  }
  return mk;
}

// ---- exhaustive linear-extension count (u64, capped) ----
static unsigned long long ext_count; static int ext_capped;
static int indeg[MAXN];
static void count_ext(int left){
  if(ext_capped)return;
  if(!left){ if(++ext_count>=(1ULL<<62))ext_capped=1; return; }
  for(int v=0;v<N;v++) if(indeg[v]==0){
    indeg[v]=-1; for(int j=0;j<nsucc[v];j++)indeg[succ[v][j]]--;
    count_ext(left-1);
    for(int j=0;j<nsucc[v];j++)indeg[succ[v][j]]++; indeg[v]=0;
    if(ext_count>200000000ULL){ext_capped=1;return;}
  }
}

// ---- B&B exact min in-order finish makespan ----
static int best; static long long bnb_nodes, bnb_cap; static int bnb_exact;
static int g_fin[MAXN]; static unsigned char g_mf[HOR],g_fp[HOR],g_ld[HOR];
static int rem_mf, rem_fp, rem_ld;   // remaining uop counts
static void bnb(int left,int prev_issue,int curmax){
  if(++bnb_nodes>bnb_cap){bnb_exact=0;return;}
  if(!left){ if(curmax<best)best=curmax; return; }
  // lower bounds
  int lb=curmax;
  for(int v=0;v<N;v++) if(indeg[v]==0){
    int r=prev_issue; for(int j=0;j<npred[v];j++){int f=g_fin[pred[v][j]];if(f>r)r=f;}
    int c=r+cp_tail[v]; if(c<lb&&0)c=c; if(r+cp_tail[v]>lb-0){} // keep simple below
  }
  { // CP-tail bound: some ready node's chain must finish
    int need=1<<29;
    for(int v=0;v<N;v++) if(indeg[v]==0){
      int r=prev_issue; for(int j=0;j<npred[v];j++){int f=g_fin[pred[v][j]];if(f>r)r=f;}
      int c=r+cp_tail[v]; if(c<need)need=c;
    }
    // every unscheduled sink chain must complete: max over ready of earliest completion is weak;
    // stronger: max over ALL unscheduled u of (opt_ready(u)+cp_tail) with opt_ready>=prev_issue
    int cp_lb=0;
    for(int u=0;u<N;u++) if(indeg[u]!=-1){
      int r=prev_issue;
      for(int j=0;j<npred[u];j++){int p=pred[u][j]; if(indeg[p]==-1&&g_fin[p]>r)r=g_fin[p];}
      int c=r+cp_tail[u]; if(c>cp_lb)cp_lb=c;
    }
    if(cp_lb>lb)lb=cp_lb;
    int pb=prev_issue + (rem_mf+1)/2; if(pb>lb)lb=pb;
    pb=prev_issue + (rem_fp+2)/3;     if(pb>lb)lb=pb;
    pb=prev_issue + (rem_ld+2)/3;     if(pb>lb)lb=pb;
  }
  if(lb>=best)return;
  for(int v=0;v<N;v++) if(indeg[v]==0){
    int t=prev_issue; for(int j=0;j<npred[v];j++){int f=g_fin[pred[v][j]];if(f>t)t=f;}
    char k=kind[v]; int um=uops_mf(k),uf=uops_fp(k),ul=(k=='L');
    int t0=t;
    if(k!='C'){
      if(ul){ while(g_ld[t]>=3)t++; g_ld[t]++; }
      else if(uf){ while(g_mf[t]+um>2||g_fp[t]+uf>3)t++; g_mf[t]+=um; g_fp[t]+=uf; }
    }
    int fin=t+lat[v], nmax=curmax>fin?curmax:fin;
    int np=(k=='C')?prev_issue:t;
    g_fin[v]=fin; indeg[v]=-1; for(int j=0;j<nsucc[v];j++)indeg[succ[v][j]]--;
    rem_mf-=um; rem_fp-=uf; rem_ld-=ul;
    bnb(left-1,np,nmax);
    rem_mf+=um; rem_fp+=uf; rem_ld+=ul;
    for(int j=0;j<nsucc[v];j++)indeg[succ[v][j]]++; indeg[v]=0;
    if(k!='C'){ if(ul)g_ld[t]--; else if(uf){g_mf[t]-=um;g_fp[t]-=uf;} }
    (void)t0;
    if(!bnb_exact)return;
  }
}

int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage: %s dump kinds [bnbcap]\n",argv[0]);return 1;}
  bnb_cap = argc>3? atoll(argv[3]) : 400000000LL;
  // kinds
  { FILE*f=fopen(argv[2],"r"); int t; char k;
    memset(tag2i,-1,sizeof tag2i);
    while(fscanf(f,"%d %c",&t,&k)==2){ tag2i[t]=N; tags[N]=t; kind[N]=k; lat[N]=lat_of(k); N++; }
    fclose(f); }
  // dump preds
  { FILE*f=fopen(argv[1],"r"); char line[512];
    while(fgets(line,sizeof line,f)){
      if(line[0]=='#'||line[0]=='\n')continue;
      int t=atoi(line); int i=tag2i[t]; char*p=strchr(line,':'); if(!p)continue; p++;
      while(*p){ while(*p==' ')p++; if(*p<'0'||*p>'9')break;
        int q=atoi(p); int pi=tag2i[q];
        pred[i][npred[i]++]=pi; succ[pi][nsucc[pi]++]=i;
        while(*p>='0'&&*p<='9')p++; }
    } fclose(f); }
  // cp_tail
  for(int i=N-1;i>=0;i--){ int m=0;
    for(int j=0;j<nsucc[i];j++){ int c=cp_tail[succ[i][j]]; if(c>m)m=c; }
    cp_tail[i]=lat[i]+m; }
  int cp=0; for(int i=0;i<N;i++) if(cp_tail[i]>cp)cp=cp_tail[i];
  int tmf=0,tfp=0,tld=0;
  for(int i=0;i<N;i++){ tmf+=uops_mf(kind[i]); tfp+=uops_fp(kind[i]); tld+=(kind[i]=='L'); }
  int pb_mf=(tmf+1)/2, pb_fp=(tfp+2)/3, pb_ld=(tld+2)/3;
  printf("N=%d  CP(finish)=%d  port-bounds(issue): mf=%d fp=%d ld=%d\n",N,cp,pb_mf,pb_fp,pb_ld);

  // SU order = dump order (identity permutation over our index space)
  static int ident[MAXN]; for(int i=0;i<N;i++)ident[i]=i;
  printf("SU order: in-order=%d  W4=%d W8=%d W16=%d W32=%d W64=%d W128=%d Winf=%d\n",
    sim_inorder(ident), sim_window(ident,4),sim_window(ident,8),
    sim_window(ident,16),sim_window(ident,32),sim_window(ident,64),sim_window(ident,128),sim_window(ident,0));

  // extension count
  memset(indeg,0,sizeof indeg);
  for(int i=0;i<N;i++) for(int j=0;j<npred[i];j++) indeg[i]++;
  if(N<=40){
    ext_count=0; ext_capped=0; count_ext(N);
    if(ext_capped) printf("linear extensions: >2e8 (capped)\n");
    else printf("linear extensions: %llu\n",ext_count);
  } else printf("linear extensions: skipped (N>40; >>2e8)\n");

  // sampling
  srand(12345);
  static int ord[MAXN]; static int rdy[MAXN];
  int smin=1<<29,smax=0; long long ssum=0; int worst_ord[MAXN];
  for(int it=0; it<1000000; it++){
    memset(indeg,0,sizeof indeg);
    for(int i=0;i<N;i++)for(int j=0;j<npred[i];j++)indeg[i]++;
    int nr=0; for(int i=0;i<N;i++) if(!indeg[i]) rdy[nr++]=i;
    for(int s=0;s<N;s++){
      int pick=rand()%nr, v=rdy[pick]; rdy[pick]=rdy[--nr]; ord[s]=v;
      for(int j=0;j<nsucc[v];j++) if(--indeg[succ[v][j]]==0) rdy[nr++]=succ[v][j];
    }
    int m=sim_inorder(ord);
    if(m<smin)smin=m;
    if(m>smax){smax=m;memcpy(worst_ord,ord,sizeof(int)*N);}
    ssum+=m;
  }
  printf("1e6 random orders (in-order): min=%d mean=%.1f max=%d\n",smin,(double)ssum/1e6,smax);
  printf("WORST order under windows: W4=%d W8=%d W16=%d W32=%d W64=%d W128=%d Winf=%d\n",
    sim_window(worst_ord,4),sim_window(worst_ord,8),sim_window(worst_ord,16),
    sim_window(worst_ord,32),sim_window(worst_ord,64),sim_window(worst_ord,128),sim_window(worst_ord,0));

  // B&B exact min
  memset(indeg,0,sizeof indeg);
  for(int i=0;i<N;i++)for(int j=0;j<npred[i];j++)indeg[i]++;
  memset(g_mf,0,HOR);memset(g_fp,0,HOR);memset(g_ld,0,HOR);
  rem_mf=tmf;rem_fp=tfp;rem_ld=tld;
  best=smin; bnb_nodes=0; bnb_exact=1;
  if(N<=120) bnb(N,0,0); else { bnb_exact=0; bnb_nodes=0; }
  printf("B&B min (in-order): %d  [%s, %lld nodes]\n",best,
         bnb_exact?"EXACT":"node-cap hit: best-found",bnb_nodes);
  return 0;
}
