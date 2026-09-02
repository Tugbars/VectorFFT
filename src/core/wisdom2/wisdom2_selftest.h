/* wisdom2_selftest.h — the wave-0 unit gates for the wisdom2 module
 * (README §7, campaign item 0.8), OWNED BY THE MODULE per the thin-driver
 * law: bench files only make calls; every scenario, assertion, and helper
 * lives here beside the code it gates. Zero timing anywhere.
 *
 * Entry point: vw2_g0_selftest(scratch_dir) -> number of failures (0 = ALL
 * PASS). The scratch dir is created if missing and wiped of wisdom2_*.txt.
 */
#ifndef VFFT_WISDOM2_SELFTEST_H
#define VFFT_WISDOM2_SELFTEST_H

#include <stdarg.h>
#include "wisdom2.h"

#if defined(_WIN32)
#  define VW2__MKDIR(d) _mkdir(d)
#else
#  include <sys/stat.h>
#  define VW2__MKDIR(d) mkdir(d, 0755)
#endif

static int vw2__st_fail;
#define VW2_ST_CHECK(cond, name) do { \
    if (cond) printf("  ok   %s\n", name); \
    else { printf("  FAIL %s (line %d)\n", name, __LINE__); vw2__st_fail++; } \
} while (0)

static inline void vw2__st_wipe(const char *dir)
{
    char p[700];
    int i;
    for (i = 0; i < VW2_NSHARDS; i++) {
        snprintf(p, sizeof p, "%s/%s", dir, vw2_shard_name[i]);
        remove(p);
    }
    snprintf(p, sizeof p, "%s/wisdom2_quarantine.txt", dir);
    remove(p);
    vw2__sweep_tmps(dir);
}

static inline long vw2__st_slurp(const char *path, char *buf, long cap)
{
    FILE *f = fopen(path, "rb");
    long n;
    if (!f) return -1;
    n = (long)fread(buf, 1, (size_t)cap - 1, f);
    fclose(f);
    buf[n] = 0;
    return n;
}

static inline vw2_key_t vw2__st_key(int t, int n0, long long q, int ord, int pl)
{
    vw2_key_t k;
    memset(&k, 0, sizeof k);
    k.t = (uint8_t)t; k.rank = 1; k.n[0] = n0;
    k.q = q; k.ord = (int8_t)ord; k.pl = (int8_t)pl; k.dir = VW2_DIR_NONE;
    return k;
}

/* pointer-returning key helper (static ring) so builder call sites can pass
 * the key inline. The builder takes the key BY POINTER on purpose: va_start
 * after a by-value struct parameter is miscompiled by mingw15.2 at
 * -O3 -mavx2 -mfma -march=native (the repo's race flags) — the vararg
 * overflow area is misread once the token count grows. Never put an
 * aggregate parameter directly before "..." in this codebase. */
static inline const vw2_key_t *vw2__st_keyp(int t, int n0, long long q, int ord, int pl)
{
    static vw2_key_t ring[4];
    static int slot;
    vw2_key_t *k = &ring[slot++ & 3];
    *k = vw2__st_key(t, n0, q, ord, pl);
    return k;
}

/* record builder; flat (name, sect, value) triples terminated by NULL name */
static inline vw2_rec_t vw2__st_rec(const vw2_key_t *kp, ...)
{
    va_list ap;
    vw2_rec_t r;
    const char *name;
    memset(&r, 0, sizeof r);
    r.key = *kp;
    va_start(ap, kp);
    while ((name = va_arg(ap, const char *)) != NULL) {
        int sect = va_arg(ap, int);
        const char *val = va_arg(ap, const char *);
        vw2_rec_set(&r, sect, name, val);
    }
    va_end(ap);
    return r;
}

static inline int vw2_g0_selftest(const char *dir)
{
    char path[700];
    static char buf[131072], buf2[131072];
    vw2_store_t st, st2, sa, sb;
    vw2_rec_t r;
    int rc;

    vw2__st_fail = 0;
    setvbuf(stdout, NULL, _IONBF, 0);   /* crash-site visibility */
    VW2__MKDIR(dir);
    vw2__st_wipe(dir);
    printf("[wisdom2_g0] scratch = %s\n", dir);

    /* ---- T1: bank + save + reload round-trip, three shards ------------- */
    printf("T1 round-trip:\n");
    vw2_open(&st, dir, 1);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc", "ref", 1, "cell(t=c2c,n=4096,q=1,ord=scr,place=oop)",
              "ran", 2, "1", "ns", 2, "8891.0", "metric", 2, "fwd1", "units", 2, "ns",
              "arms", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank oop-homed nat record");
    /* an ord=scr/ip row so the STRIDE shard file also exists in T1 (the
     * nat/ip record above homes in the OOP shard since the 2026-09-02
     * re-route: order verdicts live with the engine wisdom) */
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 4096, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "8.8.8.8",
              "ran", 2, "8", "ns", 2, "9000.0", "metric", 2, "fwd1", "units", 2, "ns",
              "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank stride-family record");
    /* its ref target (so lookups on it stay valid across the suite) */
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 4096, 1, VW2_ORD_SCR, VW2_PL_OOP),
              "eng", 1, "zsplit", "chain", 1, "8.8.8.8", "t2q", 1, "3",
              "ran", 2, "1", "ns", 2, "4400.0", "metric", 2, "joint2", "units", 2, "ns",
              "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank the ref target");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_R2C, 1024, 1, VW2_ORD_NAT, VW2_PL_OOP),
              "eng", 1, "zr2c", "zr_kv", 1, "3",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank real-family record");
    {
        vw2_key_t k2; memset(&k2, 0, sizeof k2);
        k2.t = VW2_T_C2C; k2.rank = 2; k2.n[0] = 64; k2.n[1] = 64;
        k2.q = 1; k2.ord = VW2_ORD_NAT; k2.pl = VW2_PL_OOP;
        r = vw2__st_rec(&k2, "rowplan", 1, "4.16", "colplan", 1, "2.16.2", "b", 1, "8",
                  "ran", 2, "1", "ns", 2, "7485.2", "metric", 2, "fwd1", "units", 2, "ns",
                  "src", 2, "migrated", "from", 2, "fft2d_c2c_wisdom.txt:19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank 2d-family record");
    }
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    vw2_close(&st);

    vw2_open(&st2, dir, 1);
    VW2_ST_CHECK(st2.nrec == 5, "reload count == 5");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st2, &q);
        VW2_ST_CHECK(hit != NULL, "exact lookup hits");
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "mode"), "zcasc"), "payload survives");
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "ns"), "8891.0"), "measure survives");
    }
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    VW2_ST_CHECK(vw2__st_slurp(path, buf, sizeof buf) > 0, "stride shard file exists");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_REAL]);
    VW2_ST_CHECK(vw2__st_slurp(path, buf, sizeof buf) > 0, "real shard file exists");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_2D]);
    VW2_ST_CHECK(vw2__st_slurp(path, buf, sizeof buf) > 0, "2d shard file exists");

    /* ---- T2: idempotent re-save (equal record replaces, bytes stable) -- */
    printf("T2 idempotency:\n");
    /* the nat/ip record homes in the OOP shard (2026-09-02 re-route) */
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_OOP]);
    vw2__st_slurp(path, buf, sizeof buf);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc", "ref", 1, "cell(t=c2c,n=4096,q=1,ord=scr,place=oop)",
              "ran", 2, "1", "ns", 2, "8891.0", "metric", 2, "fwd1", "units", 2, "ns",
              "arms", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st2, &r) == VW2_OK, "identical re-bank accepted (equal rank+date)");
    VW2_ST_CHECK(vw2_save(&st2) == VW2_OK, "re-save");
    vw2__st_slurp(path, buf2, sizeof buf2);
    VW2_ST_CHECK(!strcmp(buf, buf2), "file bytes identical after identical re-bank");

    /* ---- T3: legend/header byte-identity ------------------------------- */
    printf("T3 header:\n");
    {
        char want[4096]; size_t off = 0; int i; int okhdr;
        off += (size_t)snprintf(want + off, sizeof want - off, VW2_MAGIC " %d.%d\n", VW2_MAJOR, VW2_MINOR);
        for (i = 0; i < VW2_NLEGEND; i++)
            off += (size_t)snprintf(want + off, sizeof want - off, "@legend %s\n", vw2_legend[i]);
        okhdr = !strncmp(buf2, want, off);
        VW2_ST_CHECK(okhdr, "writer-emitted header matches the module rule table byte-for-byte");
    }

    /* ---- T4: carry-unknown-forward (tokens, records, directives) ------- */
    printf("T4 carry-unknown:\n");
    {
        FILE *f = fopen(path, "ab");
        VW2_ST_CHECK(f != NULL, "scratch appendable");
        if (f) {
            fprintf(f, "@cell t=c2c n=32 q=1 ord=scr place=ip | eng=stride zz_future=7 | ran=1 src=race date=2026-08-19\n");
            fprintf(f, "@cell t=zzz n=99 q=1 ord=scr place=ip | eng=alien | ran=1\n");
            fprintf(f, "@futuredirective payload=opaque\n");
            fclose(f);
        }
    }
    vw2_close(&st2);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 32, 1, VW2_ORD_SCR, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "zz_future"), "7"), "unknown TOKEN parsed + kept");
    }
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 64, 2, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank another record to force resave");
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "resave with unknowns present");
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "zz_future=7") != NULL, "unknown token survives resave");
    VW2_ST_CHECK(strstr(buf, "t=zzz") != NULL, "unknown-transform record carried opaque");
    VW2_ST_CHECK(strstr(buf, "@futuredirective payload=opaque") != NULL, "unknown directive carried opaque");
    /* second save cycle: opaque lines survive again, exactly once */
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 96, 2, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    vw2_bank(&st, &r);
    vw2_save(&st);
    vw2__st_slurp(path, buf, sizeof buf);
    {
        const char *one = strstr(buf, "@futuredirective");
        VW2_ST_CHECK(one && !strstr(one + 1, "@futuredirective"), "opaque line not duplicated across cycles");
    }
    vw2_close(&st);

    /* ---- T5: version refuse + poison (file never clobbered) ------------ */
    printf("T5 version-refuse:\n");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_OOP]);
    {
        FILE *f = fopen(path, "wb");
        fprintf(f, "@vw2 2.0\n@cell t=c2c n=128 q=1 ord=scr place=oop | eng=future | ran=1\n");
        fclose(f);
    }
    rc = vw2_open(&st, dir, 1);
    VW2_ST_CHECK(rc == VW2_EVERSION, "open reports version refusal");
    VW2_ST_CHECK(st.poisoned[VW2_SHARD_OOP] == 1, "shard poisoned");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 128, 1, VW2_ORD_SCR, VW2_PL_OOP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "poisoned file's cells miss (never silently served)");
    }
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 256, 1, VW2_ORD_SCR, VW2_PL_OOP),
              "eng", 1, "classic", "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_EPOISON, "bank into poisoned shard refused");
    vw2_rec_free(&r);
    vw2__st_slurp(path, buf, sizeof buf);
    vw2_save(&st);
    vw2__st_slurp(path, buf2, sizeof buf2);
    VW2_ST_CHECK(!strcmp(buf, buf2), "poisoned file bytes untouched by save");
    vw2_close(&st);
    remove(path);

    /* ---- T6: stale .tmp sweep at open ----------------------------------- */
    printf("T6 tmp-sweep:\n");
    snprintf(path, sizeof path, "%s/%s.tmp.12345", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    { FILE *f = fopen(path, "wb"); if (f) { fprintf(f, "junk from a crashed writer\n"); fclose(f); } }
    vw2_open(&st, dir, 1);
    { FILE *f = fopen(path, "rb"); VW2_ST_CHECK(f == NULL, "stale pid-suffixed .tmp swept at open"); if (f) fclose(f); }
    vw2_close(&st);

    /* ---- T7: two-store merge-on-save (concurrent sessions) ------------- */
    printf("T7 merge-on-save:\n");
    vw2_open(&sa, dir, 1);
    vw2_open(&sb, dir, 1);            /* B loads BEFORE A saves */
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "8.8.8",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&sa, &r) == VW2_OK, "A banks cellA");
    VW2_ST_CHECK(vw2_save(&sa) == VW2_OK, "A saves");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 768, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "8.96",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&sb, &r) == VW2_OK, "B banks cellB");
    VW2_ST_CHECK(vw2_save(&sb) == VW2_OK, "B saves (merge-on-save)");
    vw2_close(&sa); vw2_close(&sb);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t qa = vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP);
        vw2_key_t qb = vw2__st_key(VW2_T_C2C, 768, 8, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &qa) != NULL, "A's cell survives B's later save");
        VW2_ST_CHECK(vw2_lookup(&st, &qb) != NULL, "B's cell present");
    }

    /* ---- T8: merge rank + date tie-break -------------------------------- */
    printf("T8 merge-rank:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "4.128",
              "ran", 2, "8", "src", 2, "env:VFFT_TCUT", "arms", 2, "1", "date", 2, "2026-08-20", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "env-shaped verdict cannot replace a raced one");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "4.128",
              "ran", 2, "8", "src", 2, "seed", "date", 2, "2026-08-20", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "seed cannot replace a raced verdict");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "4.128",
              "ran", 2, "8", "src", 2, "hologram", "date", 2, "2026-08-20", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "UNKNOWN future src ranks lowest, cannot displace");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "16.32",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-18", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "older race cannot replace newer");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "16.32", "ran", 2, "8", "src", 2, "race", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "dateless challenger cannot replace a dated incumbent");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "16.32",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-21", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "newer race replaces");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "chain"), "16.32"), "replacement served");
    }

    /* ---- T9: cross-metric refusal ---------------------------------------- */
    printf("T9 cross-metric:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc",
              "ran", 2, "1", "ns", 2, "4400.0", "metric", 2, "joint2", "units", 2, "ns",
              "src", 2, "race", "date", 2, "2026-08-22", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_EMETRIC, "joint2 cannot replace fwd1 (never compare across metrics)");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc", "ran", 2, "1",
              "src", 2, "race", "date", 2, "2026-08-22", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_EMETRIC, "metric-less challenger cannot replace a measured incumbent");
    vw2_rec_free(&r);

    /* ---- T10: wildcard law ------------------------------------------------ */
    printf("T10 wildcards:\n");
    {
        vw2_key_t wk = vw2__st_key(VW2_T_C2C, 8192, -1, VW2_ORD_ANY, VW2_PL_ANY);
        r = vw2__st_rec(&wk, "eng", 1, "split_oop", "chain", 1, "8.32.32",
                  "ran", 2, "4", "ns", 2, "19289.3", "metric", 2, "fwd1", "units", 2, "ns",
                  "src", 2, "race", "date", 2, "2026-08-19", NULL);
        rc = vw2_bank(&st, &r);
        VW2_ST_CHECK(rc == VW2_EWILDCARD, "fresh wildcard bank refused (no from=)");
        vw2_rec_free(&r);
        r = vw2__st_rec(&wk, "eng", 1, "split_oop", "chain", 1, "8.32.32",
                  "ran", 2, "4", "ns", 2, "19289.3", "metric", 2, "fwd1", "units", 2, "ns",
                  "src", 2, "migrated", "from", 2, "oop_wisdom.txt:119", "date", 2, "2026-08-19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "migrated wildcard bank accepted");
    }
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 8192, 1, VW2_ORD_NAT, VW2_PL_OOP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "chain"), "8.32.32"), "wildcard serves any axis value");
    }
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 8192, 1, VW2_ORD_NAT, VW2_PL_OOP),
              "eng", 1, "split_oop", "chain", 1, "64.128",
              "ran", 2, "4", "ns", 2, "18000.0", "metric", 2, "fwd1", "units", 2, "ns",
              "src", 2, "race", "date", 2, "2026-08-22", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "concrete cell banks beside the wildcard");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 8192, 1, VW2_ORD_NAT, VW2_PL_OOP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "chain"), "64.128"), "exact hit beats wildcard");
    }
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save wildcard state");
    vw2_close(&st);

    /* ---- T11: seeds never served (wildcard AND concrete keys) ------------ */
    printf("T11 seeds:\n");
    vw2_open(&st, dir, 1);
    {
        vw2_key_t wk = vw2__st_key(VW2_T_C2C, 999, -1, VW2_ORD_ANY, VW2_PL_ANY);
        r = vw2__st_rec(&wk, "eng", 1, "stride",
                  "src", 2, "seed", "from", 2, "legacy:x", "date", 2, "2026-08-19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "wildcard seed banks fine");
    }
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 999, 4, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "wildcard seed is never served as a verdict");
    }
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 997, 4, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "src", 2, "seed", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "concrete-key seed banks fine");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 997, 4, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "concrete-key seed is never served either");
    }

    /* ---- T12: read-only guard --------------------------------------------- */
    printf("T12 read-only:\n");
    vw2_close(&st);
    vw2_open(&st, dir, 0);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 2048, 1, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_EREADONLY, "bank refused on read-only store");
    vw2_rec_free(&r);
    VW2_ST_CHECK(vw2_save(&st) == VW2_EREADONLY, "save refused on read-only store");
    VW2_ST_CHECK(vw2_quarantine_append(&st, "x", "y", "z") == VW2_EREADONLY,
          "quarantine refused on read-only store");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_update_field(&st, &q, "t2q", "1") == VW2_EREADONLY,
              "update_field refused on read-only store");
    }
    vw2_close(&st);

    /* ---- T13: ref= parse --------------------------------------------------- */
    printf("T13 ref:\n");
    {
        vw2_key_t k;
        VW2_ST_CHECK(vw2_ref_parse("cell(t=c2c,n=4096,q=1,ord=scr,place=oop)", &k) == 1, "ref parses");
        VW2_ST_CHECK(k.t == VW2_T_C2C && k.n[0] == 4096 && k.q == 1 &&
              k.ord == VW2_ORD_SCR && k.pl == VW2_PL_OOP, "ref key fields correct");
        VW2_ST_CHECK(vw2_ref_parse("cell(t=c2c,n=4096)", &k) == 0, "partial ref refused (full key required)");
    }

    /* ---- T14: quarantine append (guarded, raw to EOL) ---------------------- */
    printf("T14 quarantine:\n");
    vw2_open(&st, dir, 1);
    VW2_ST_CHECK(vw2_quarantine_append(&st, "garbage-variant-token", "oop_wisdom.txt:45",
                                "4096 1 2 3 32 16 8 690 -1435841587 32761 81219.0") == VW2_OK,
          "quarantine append");
    VW2_ST_CHECK(vw2_quarantine_append(&st, "two words", "f", "x") == VW2_EVALUE,
          "reason with a space refused (lexical law)");
    snprintf(path, sizeof path, "%s/wisdom2_quarantine.txt", dir);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "raw=4096 1 2 3 32 16 8 690 -1435841587 32761 81219.0") != NULL,
          "raw= runs to end of line, escape-free");
    VW2_ST_CHECK(!strncmp(buf, VW2_MAGIC " ", 5), "quarantine file carries the version header");

    /* ---- T15: empty-payload round-trip -------------------------------------- */
    printf("T15 empty-payload:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 40, 2, VW2_ORD_SCR, VW2_PL_IP),
              "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank record with zero payload tokens");
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    vw2_close(&st);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 40, 2, VW2_ORD_SCR, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        int i, meas_ok = 1;
        VW2_ST_CHECK(hit != NULL, "payload-less record served after reload");
        if (hit)
            for (i = 0; i < hit->ntok; i++)
                if (hit->tok[i].sect != 2) meas_ok = 0;
        VW2_ST_CHECK(meas_ok, "measure tokens kept their section (no payload migration)");
    }
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "place=ip | - | ran=2") != NULL, "empty section carries the '-' marker");

    /* ---- T16: shard routing (prime / trig) ----------------------------------- */
    printf("T16 routing:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 1009, 1, VW2_ORD_NAT, VW2_PL_OOP),
              "eng", 1, "bluestein", "m", 1, "2048", "b", 1, "1",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank bluestein record");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_DCT1, 257, 4, VW2_ORD_NAT, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "16.16",
              "ran", 2, "4", "src", 2, "migrated", "from", 2, "spike_wisdom.txt:10", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank re-keyed trig record");
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_PRIME]);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "n=1009") != NULL, "bluestein routed to the prime shard");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "t=dct1 n=257") != NULL, "trig routed to the stride shard");

    /* ---- T17: unknown KEY token => invisible + opaque carry ------------------- */
    printf("T17 unknown-key-token:\n");
    {
        FILE *f = fopen(path, "ab");
        if (f) { fprintf(f, "@cell t=c2c n=48 q=1 ord=scr place=ip nthreads=8 | eng=x | ran=1\n"); fclose(f); }
    }
    vw2_close(&st);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 48, 1, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "record with unknown key token is invisible to lookup");
    }
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 56, 2, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    vw2_bank(&st, &r);
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "resave");
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "nthreads=8") != NULL, "future-key-axis record survives resave verbatim");

    /* ---- T18: minor-version acceptance ----------------------------------------- */
    printf("T18 minor-version:\n");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_3D]);
    {
        FILE *f = fopen(path, "wb");
        if (f) {
            fprintf(f, "@vw2 1.7\n@cell t=c2c n=8x8x8 q=1 ord=nat place=oop | rowplan=x | ran=1 src=race date=2026-08-19\n");
            fclose(f);
        }
    }
    vw2_close(&st);
    rc = vw2_open(&st, dir, 1);
    VW2_ST_CHECK(rc == VW2_OK, "minor 1.7 file accepted");
    {
        vw2_key_t k3; memset(&k3, 0, sizeof k3);
        k3.t = VW2_T_C2C; k3.rank = 3; k3.n[0] = k3.n[1] = k3.n[2] = 8;
        k3.q = 1; k3.ord = VW2_ORD_NAT; k3.pl = VW2_PL_OOP;
        VW2_ST_CHECK(vw2_lookup(&st, &k3) != NULL, "record from a newer-minor file served");
    }
    vw2_close(&st);
    remove(path);

    /* ---- T19: CRLF tolerance ------------------------------------------------------ */
    printf("T19 crlf:\n");
    {
        FILE *f = fopen(path, "wb");
        if (f) {
            fprintf(f, "@vw2 1.0\r\n@cell t=c2c n=4x4x4 q=1 ord=nat place=oop | rowplan=y | ran=1 src=race date=2026-08-19\r\n");
            fclose(f);
        }
    }
    vw2_open(&st, dir, 1);
    {
        vw2_key_t k3; memset(&k3, 0, sizeof k3);
        k3.t = VW2_T_C2C; k3.rank = 3; k3.n[0] = k3.n[1] = k3.n[2] = 4;
        k3.q = 1; k3.ord = VW2_ORD_NAT; k3.pl = VW2_PL_OOP;
        VW2_ST_CHECK(vw2_lookup(&st, &k3) != NULL, "CRLF-terminated file parses");
    }
    vw2_close(&st);
    remove(path);

    /* ---- T20: field-scoped promotion ------------------------------------------------ */
    printf("T20 update-field:\n");
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_update_field(&st, &q, "t2q", "1") == VW2_OK, "update_field sets");
        VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    }
    vw2_close(&st);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && vw2_rec_get(hit, "t2q") && !strcmp(vw2_rec_get(hit, "t2q"), "1"),
              "promoted field survives reload");
    }

    /* ---- T21: re-route migration scrubs the old shard --------------------------------- */
    printf("T21 re-route:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 337, 1, VW2_ORD_NAT, VW2_PL_OOP),
              "eng", 1, "classic", "chain", 1, "337",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank eng=classic (oop shard)");
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    vw2_close(&st);
    vw2_open(&st, dir, 1);
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 337, 1, VW2_ORD_NAT, VW2_PL_OOP),
              "eng", 1, "bluestein", "m", 1, "1024", "b", 1, "1",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-21", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "re-bank eng=bluestein (re-routes to prime)");
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    vw2_close(&st);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 337, 1, VW2_ORD_NAT, VW2_PL_OOP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "eng"), "bluestein"), "re-routed verdict served");
    }
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_OOP]);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "n=337") == NULL, "stale copy scrubbed from the old shard");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_PRIME]);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "n=337") != NULL, "record lives in the new shard");

    /* ---- T22: dangling ref => MISS ------------------------------------------------------ */
    printf("T22 dangling-ref:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 6144, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc", "ref", 1, "cell(t=c2c,n=6144,q=1,ord=scr,place=oop)",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank signpost with absent target");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 6144, 1, VW2_ORD_NAT, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "dangling ref treated as MISS");
    }
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 6144, 1, VW2_ORD_SCR, VW2_PL_OOP),
              "eng", 1, "zsplit", "chain", 1, "8.768",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank the target");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 6144, 1, VW2_ORD_NAT, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) != NULL, "signpost serves once the target exists");
    }

    /* ---- T23: wildcard precedence (q=*-only beats ord/place wildcards) ------------------- */
    printf("T23 wildcard-precedence:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 3072, 7, VW2_ORD_ANY, VW2_PL_ANY),
              "eng", 1, "engB", "ran", 2, "1",
              "src", 2, "migrated", "from", 2, "x:1", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank ord/place-wildcard record FIRST");
    {
        vw2_key_t wq = vw2__st_key(VW2_T_C2C, 3072, -1, VW2_ORD_NAT, VW2_PL_OOP);
        r = vw2__st_rec(&wq, "eng", 1, "engA", "ran", 2, "1",
                  "src", 2, "migrated", "from", 2, "x:2", "date", 2, "2026-08-19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank q=*-only record SECOND");
    }
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 3072, 7, VW2_ORD_NAT, VW2_PL_OOP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "eng"), "engA"),
              "q=*-only record beats ord/place wildcard regardless of order");
    }

    /* ---- T24: request keys never carry wildcards ------------------------------------------ */
    printf("T24 request-wildcard-guard:\n");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 3072, -1, VW2_ORD_NAT, VW2_PL_OOP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "wildcard request refused (NULL)");
    }

    /* ---- T25: lexical law at the write entry ------------------------------------------------ */
    printf("T25 lexical:\n");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 88, 1, VW2_ORD_SCR, VW2_PL_IP), NULL);
    VW2_ST_CHECK(vw2_rec_set(&r, 1, "chain", "8 | 32") == VW2_EVALUE, "value with pipe/space refused");
    VW2_ST_CHECK(vw2_rec_set(&r, 1, "bad name", "x") == VW2_EVALUE, "name with space refused");
    vw2_rec_free(&r);

    /* ---- T26: long-line round-trip (no length limit) ----------------------------------------- */
    printf("T26 long-line:\n");
    {
        static char big[6000];
        int i;
        for (i = 0; i < 5990; i += 2) { big[i] = '8'; big[i + 1] = '.'; }
        big[5989] = 0;
        r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 176, 2, VW2_ORD_SCR, VW2_PL_IP),
                  "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
        VW2_ST_CHECK(vw2_rec_set(&r, 1, "chain", big) == VW2_OK, "6KB value accepted");
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank long record");
        VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save long record");
        vw2_close(&st);
        vw2_open(&st, dir, 1);
        {
            vw2_key_t q = vw2__st_key(VW2_T_C2C, 176, 2, VW2_ORD_SCR, VW2_PL_IP);
            const vw2_rec_t *hit = vw2_lookup(&st, &q);
            VW2_ST_CHECK(hit && vw2_rec_get(hit, "chain") &&
                  strlen(vw2_rec_get(hit, "chain")) == strlen(big),
                  "6KB value survives reload intact (growing reader)");
        }
    }

    /* ---- T27: @meta round-trip ------------------------------------------------------------------ */
    printf("T27 meta:\n");
    vw2_set_meta(&st, "host=i9-14900KF isa=avx2 l1d=49152");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 208, 2, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    vw2_bank(&st, &r);
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save with meta set");
    vw2_close(&st);
    vw2_open(&st, dir, 1);
    VW2_ST_CHECK(!strcmp(st.meta, "host=i9-14900KF isa=avx2 l1d=49152"), "@meta captured at load");
    r = vw2__st_rec(vw2__st_keyp(VW2_T_C2C, 224, 2, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    vw2_bank(&st, &r);
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "resave");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "@meta host=i9-14900KF isa=avx2 l1d=49152") != NULL,
          "@meta survives the resave");
    vw2_close(&st);

    /* ---- T28: zero-byte / headerless file => poison ------------------------------------------------ */
    printf("T28 headerless:\n");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_3D]);
    { FILE *f = fopen(path, "wb"); if (f) fclose(f); }
    rc = vw2_open(&st, dir, 1);
    VW2_ST_CHECK(rc == VW2_EVERSION, "zero-byte file refused (never silently empty)");
    VW2_ST_CHECK(st.poisoned[VW2_SHARD_3D] == 1, "zero-byte shard poisoned");
    vw2_save(&st);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(buf[0] == 0, "zero-byte file untouched by save");
    vw2_close(&st);
    remove(path);

    printf("\n[wisdom2_g0] %s — %d failure(s)\n", vw2__st_fail ? "FAIL" : "ALL PASS", vw2__st_fail);
    return vw2__st_fail;
}

#endif /* VFFT_WISDOM2_SELFTEST_H */
