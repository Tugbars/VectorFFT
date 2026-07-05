// pareto.c — bi-criteria characterization of a codelet DAG's schedules.
// For any legal order: MAXLIVE (schedule pressure, compiler-free),
// Belady-optimal spill counts under R registers (furthest-next-use,
// optimal for straight-line code), and machine cycles of the
// spill-EXPANDED sequence under the (ports, latencies, window) model.
// Modes:
//   pareto dump kinds map <order|-'SU'>     -> full metrics for one order
//   pareto dump kinds cloud <n>             -> sample n orders: Pareto set of (W1cyc,maxlive)
//   pareto dump kinds frontier <mmax>       -> exact min-makespan s.t. MAXLIVE<=m (B&B, small N)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXN 512
#define HOR 32768
#define MAXE 4096
static int N, lat[MAXN]; static char kind[MAXN];
static int npred[MAXN], pred[MAXN][8], nsucc[MAXN], succ[MAXN][16];
static int tag2i[1<<16], tags[MAXN];
static int lat_of(char k){return k=='L'?5:k=='C'?0:k=='S'?1:4;}
static int is_mf(char k){return k=='M'||k=='F'||k=='X';}
static int is_fp(char k){return is_mf(k)||k=='A'||k=='N';}

// expanded-sequence machine sim (window W; W=0 => infinite)
static int sim(const int *op, const char *ok, const int (*opr)[8], const int *nopr, int M, int W){
  static int fin[MAXE]; static char done[MAXE];
  memset(done,0,M);
  int issued=0, head=0, t=0, mk=0;
  while(issued<M){
    int mfu=0,fpu=0,ldu=0,stu=0;
    while(head<M && done[head]) head++;
    int lim=(W<=0||head+W>M)?M:head+W;
    for(int s=head;s<lim;s++){
      if(done[s])continue;
      int rdy=1; for(int j=0;j<nopr[s];j++){int p=opr[s][j]; if(!done[p]||fin[p]>t){rdy=0;break;}}
      if(!rdy)continue;
      char k=ok[s];
      if(k=='C'){done[s]=1;fin[s]=t;issued++;continue;}
      if(k=='L'){if(ldu>=3)continue;ldu++;}
      else if(k=='S'){if(stu>=2)continue;stu++;}
      else{int um=(k=='X')?2:is_mf(k)?1:0, uf=(k=='X')?2:is_fp(k)?1:0;
           if(mfu+um>2||fpu+uf>3)continue; mfu+=um; fpu+=uf;}
      done[s]=1; fin[s]=t+lat_of(k); issued++; if(fin[s]>mk)mk=fin[s];
    }
    t++; if(t>=HOR){fprintf(stderr,"hor\n");exit(1);}
  }
  return mk;
}

// MAXLIVE of an order (C excluded: rematerializable broadcasts)
static int maxlive(const int *ord){
  static int lastuse[MAXN], pos[MAXN];
  for(int i=0;i<N;i++)pos[ord[i]]=i;
  for(int i=0;i<N;i++){int lu=-1;
    for(int j=0;j<nsucc[i];j++){int p=pos[succ[i][j]];if(p>lu)lu=p;}
    lastuse[i]=lu;}
  int live=0,ml=0;
  static int dieat[MAXN+1]; memset(dieat,0,sizeof(int)*(N+1));
  for(int s=0;s<N;s++){
    int v=ord[s];
    live-=dieat[s];
    if(kind[v]!='C'&&lastuse[v]>=0){live++; dieat[lastuse[v]+1]++;}
    if(live>ml)ml=live;
  }
  return ml;
}

// Belady expansion under R regs -> counts + expanded sequence for sim
static int bel_stores,bel_reloads;
static int expand(const int *ord,int R,int *op,char *ok,int (*opr)[8],int *nopr){
  static int pos[MAXN],nu[MAXN][20],nnu[MAXN],cur[MAXN];
  static char inreg[MAXN],stored[MAXN]; static int lastdef[MAXN];
  for(int i=0;i<N;i++){pos[ord[i]]=i;nnu[i]=0;cur[i]=0;inreg[i]=0;stored[i]=0;lastdef[i]=-1;}
  for(int i=0;i<N;i++)for(int j=0;j<nsucc[i];j++){int u=succ[i][j];nu[i][nnu[i]++]=pos[u];}
  for(int i=0;i<N;i++){ // sort next-uses
    for(int a=0;a<nnu[i];a++)for(int b=a+1;b<nnu[i];b++)
      if(nu[i][b]<nu[i][a]){int t=nu[i][a];nu[i][a]=nu[i][b];nu[i][b]=t;}}
  int regs[64],nr=0,M=0; bel_stores=bel_reloads=0;
  #define NEXTUSE(v,s) (cur[v]<nnu[v]?nu[v][cur[v]]:1<<29)
  for(int s=0;s<N;s++){
    int v=ord[s];
    // ensure operands in regs
    for(int j=0;j<npred[v];j++){int p=pred[v][j];
      if(kind[p]=='C')continue;
      if(!inreg[p]){ // reload
        if(nr>=R){ // evict furthest (not an operand of v)
          int ev=-1,fu=-1;
          for(int q=0;q<nr;q++){int w=regs[q];int isop=0;
            for(int jj=0;jj<npred[v];jj++)if(pred[v][jj]==w)isop=1;
            if(isop)continue;
            int f=NEXTUSE(w,s); if(f>fu){fu=f;ev=q;}}
          int w=regs[ev];
          if(!stored[w]&&NEXTUSE(w,s)<(1<<29)){ // needs store
            op[M]=w;ok[M]='S';nopr[M]=1;opr[M][0]=lastdef[w];M++;bel_stores++;stored[w]=1;}
          inreg[w]=0; regs[ev]=regs[--nr];
        }
        op[M]=p;ok[M]='L';nopr[M]=0;lastdef[p]=M;M++;bel_reloads++;
        inreg[p]=1;regs[nr++]=p;
      }
    }
    // advance operand next-use cursors
    for(int j=0;j<npred[v];j++){int p=pred[v][j];
      while(cur[p]<nnu[p]&&nu[p][cur[p]]<=s)cur[p]++;}
    // result register
    if(kind[v]!='C'&&nsucc[v]>0){
      if(nr>=R){int ev=-1,fu=-1;
        for(int q=0;q<nr;q++){int w=regs[q];int isop=0;
          for(int jj=0;jj<npred[v];jj++)if(pred[v][jj]==w)isop=1;
          if(isop)continue;
          int f=NEXTUSE(w,s);if(f>fu){fu=f;ev=q;}}
        int w=regs[ev];
        if(!stored[w]&&NEXTUSE(w,s)<(1<<29)){
          op[M]=w;ok[M]='S';nopr[M]=1;opr[M][0]=lastdef[w];M++;bel_stores++;stored[w]=1;}
        inreg[w]=0;regs[ev]=regs[--nr];}
      inreg[v]=1;regs[nr++]=v;
    }
    // the op itself; deps = latest def index of each operand
    op[M]=v;ok[M]=kind[v];nopr[M]=0;
    for(int j=0;j<npred[v];j++){int p=pred[v][j];
      if(kind[p]=='C')continue; opr[M][nopr[M]++]=lastdef[p];}
    lastdef[v]=M;M++;
  }
  return M;
}

static void metrics(const int *ord,int R){
  static int op[MAXE],nopr[MAXE]; static char ok[MAXE]; static int opr[MAXE][8];
  int ml=maxlive(ord);
  // no-spill machine cycles (R=inf)
  int M0=0; for(int s=0;s<N;s++){int v=ord[s];op[M0]=v;ok[M0]=kind[v];nopr[M0]=0;
    for(int j=0;j<npred[v];j++){int p=pred[v][j];if(kind[p]!='C'){
      for(int q=M0-1;q>=0;q--)if(op[q]==p){opr[M0][nopr[M0]++]=q;break;}}}
    M0++;}
  int c1=sim(op,ok,opr,nopr,M0,1),c32=sim(op,ok,opr,nopr,M0,32),ci=sim(op,ok,opr,nopr,M0,0);
  int M=expand(ord,R,op,ok,opr,nopr);
  int e1=sim(op,ok,opr,nopr,M,1),e32=sim(op,ok,opr,nopr,M,32),ei=sim(op,ok,opr,nopr,M,0);
  printf("MAXLIVE=%d  belady@R%d: stores=%d reloads=%d spillops=%d\n",
         ml,R,bel_stores,bel_reloads,bel_stores+bel_reloads);
  printf("cycles no-spill:  W1=%d W32=%d Winf=%d\n",c1,c32,ci);
  printf("cycles w/ spills: W1=%d W32=%d Winf=%d   (expanded M=%d)\n",e1,e32,ei,M);
}

int main(int argc,char**argv){
  if(argc<4){fprintf(stderr,"usage: %s dump kinds map|cloud|frontier arg\n",argv[0]);return 1;}
  {FILE*f=fopen(argv[2],"r");int t;char k;memset(tag2i,-1,sizeof tag2i);
   while(fscanf(f,"%d %c",&t,&k)==2){tag2i[t]=N;tags[N]=t;kind[N]=k;lat[N]=lat_of(k);N++;}fclose(f);}
  {FILE*f=fopen(argv[1],"r");char line[512];
   while(fgets(line,sizeof line,f)){if(line[0]=='#'||line[0]=='\n')continue;
     int t=atoi(line);int i=tag2i[t];char*p=strchr(line,':');if(!p)continue;p++;
     while(*p){while(*p==' ')p++;if(*p<'0'||*p>'9')break;
       int q=atoi(p);pred[i][npred[i]++]=tag2i[q];succ[tag2i[q]][nsucc[tag2i[q]]++]=i;
       while(*p>='0'&&*p<='9')p++;}}fclose(f);}
  static int ord[MAXN];
  if(!strcmp(argv[3],"map")){
    if(argc>4&&strcmp(argv[4],"SU")){
      FILE*f=fopen(argv[4],"r");char ln[256];int m=0;
      while(fgets(ln,sizeof ln,f)){if(ln[0]=='#'||ln[0]=='\n')continue;
        int t=atoi(ln);if(tag2i[t]>=0)ord[m++]=tag2i[t];}
      fclose(f);if(m!=N){fprintf(stderr,"bad order\n");return 2;}
    } else for(int i=0;i<N;i++)ord[i]=i;
    metrics(ord,16); return 0;
  }
  if(!strcmp(argv[3],"cloud")){
    int T=atoi(argv[4]); srand(7);
    static int indeg[MAXN],rdy[MAXN];
    // Pareto over (W1 no-spill cycles, MAXLIVE)
    static int bestml[4096]; for(int i=0;i<4096;i++)bestml[i]=1<<29;
    for(int it=0;it<T;it++){
      memset(indeg,0,sizeof indeg);
      for(int i=0;i<N;i++)for(int j=0;j<npred[i];j++)indeg[i]++;
      int nr=0;for(int i=0;i<N;i++)if(!indeg[i])rdy[nr++]=i;
      for(int s=0;s<N;s++){int pk=rand()%nr,v=rdy[pk];rdy[pk]=rdy[--nr];ord[s]=v;
        for(int j=0;j<nsucc[v];j++)if(--indeg[succ[v][j]]==0)rdy[nr++]=succ[v][j];}
      static int op[MAXE],nopr2[MAXE];static char ok[MAXE];static int opr2[MAXE][8];
      int M0=0;for(int s=0;s<N;s++){int v=ord[s];op[M0]=v;ok[M0]=kind[v];nopr2[M0]=0;
        for(int j=0;j<npred[v];j++){int p=pred[v][j];if(kind[p]!='C'){
          for(int q=M0-1;q>=0;q--)if(op[q]==p){opr2[M0][nopr2[M0]++]=q;break;}}}M0++;}
      int c=sim(op,ok,opr2,nopr2,M0,1), ml=maxlive(ord);
      if(ml<bestml[c])bestml[c]=ml;
    }
    // print Pareto staircase
    int cur=1<<29;
    for(int c=0;c<4096;c++) if(bestml[c]<cur){printf("%d %d\n",c,bestml[c]);cur=bestml[c];}
    return 0;
  }
  if(!strcmp(argv[3],"minsp")){
    int T=atoi(argv[4]); srand(11);
    static int indeg[MAXN],rdy[MAXN];
    static int op[MAXE],nopr2[MAXE];static char ok[MAXE];static int opr2[MAXE][8];
    int b1=1<<29,b32=1<<29,bi=1<<29;
    for(int it=0;it<T;it++){
      memset(indeg,0,sizeof indeg);
      for(int i=0;i<N;i++)for(int j=0;j<npred[i];j++)indeg[i]++;
      int nr=0;for(int i=0;i<N;i++)if(!indeg[i])rdy[nr++]=i;
      for(int s2=0;s2<N;s2++){int pk=rand()%nr,v=rdy[pk];rdy[pk]=rdy[--nr];ord[s2]=v;
        for(int j=0;j<nsucc[v];j++)if(--indeg[succ[v][j]]==0)rdy[nr++]=succ[v][j];}
      int M=expand(ord,16,op,ok,opr2,nopr2);
      int e1=sim(op,ok,opr2,nopr2,M,1);
      int e32=sim(op,ok,opr2,nopr2,M,32);
      int ei=sim(op,ok,opr2,nopr2,M,0);
      if(e1<b1)b1=e1; if(e32<b32)b32=e32; if(ei<bi)bi=ei;
    }
    printf("min over %d random orders, cycles-with-belady-spills@R16: W1=%d W32=%d Winf=%d\n",T,b1,b32,bi);
    return 0;
  }
  fprintf(stderr,"frontier mode: use ilp-style B&B externally\n");
  return 1;
}
