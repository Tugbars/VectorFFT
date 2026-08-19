/* wisdom2_selftest.h — the wave-0 unit gates for the wisdom2 module
 * (README §7, campaign item 0.8), OWNED BY THE MODULE per the thin-driver
 * law: bench files only make calls; every scenario, assertion, and helper
 * lives here beside the code it gates. Zero timing anywhere.
 *
 * Covers: round-trip + idempotency, writer-emitted header byte-identity,
 * carry-unknown-forward (tokens, records, directives), version-refuse +
 * per-file poison, stale-.tmp sweep, two-store merge-on-save, merge rank +
 * date tie-break, cross-metric refusal, the wildcard law, seeds never
 * served, the read-only guard, ref= parsing, quarantine append.
 *
 * Entry point: vw2_g0_selftest(scratch_dir) -> number of failures (0 = ALL
 * PASS). The scratch dir is created if missing and wiped of wisdom2_*.txt.
 */
#ifndef VFFT_WISDOM2_SELFTEST_H
#define VFFT_WISDOM2_SELFTEST_H

#include <stdarg.h>
#include "wisdom2.h"

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
        snprintf(p, sizeof p, "%s/%s.tmp", dir, vw2_shard_name[i]);
        remove(p);
    }
    snprintf(p, sizeof p, "%s/wisdom2_quarantine.txt", dir);
    remove(p);
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

/* record builder; flat (name, sect, value) triples terminated by NULL name */
static inline vw2_rec_t vw2__st_rec(vw2_key_t k, ...)
{
    va_list ap;
    vw2_rec_t r;
    const char *name;
    memset(&r, 0, sizeof r);
    r.key = k;
    va_start(ap, k);
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
    char path[700], tmpp[760];
    static char buf[65536], buf2[65536];
    vw2_store_t st, st2, sa, sb;
    vw2_rec_t r;
    int rc;

    vw2__st_fail = 0;
#if defined(_WIN32)
    { char cmd[800]; snprintf(cmd, sizeof cmd, "mkdir \"%s\" 2>NUL", dir); system(cmd); }
#else
    { char cmd[800]; snprintf(cmd, sizeof cmd, "mkdir -p \"%s\"", dir); system(cmd); }
#endif
    vw2__st_wipe(dir);
    printf("[wisdom2_g0] scratch = %s\n", dir);

    /* ---- T1: bank + save + reload round-trip, three shards ------------- */
    printf("T1 round-trip:\n");
    vw2_open(&st, dir, 1);
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc", "ref", 1, "cell(t=c2c,n=4096,q=1,ord=scr,place=oop)",
              "ran", 2, "1", "ns", 2, "8891.0", "metric", 2, "fwd1", "units", 2, "ns",
              "arms", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank stride-family record");
    r = vw2__st_rec(vw2__st_key(VW2_T_R2C, 1024, 1, VW2_ORD_NAT, VW2_PL_OOP),
              "eng", 1, "zr2c", "zr_kv", 1, "3",
              "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank real-family record");
    {
        vw2_key_t k2; memset(&k2, 0, sizeof k2);
        k2.t = VW2_T_C2C; k2.rank = 2; k2.n[0] = 64; k2.n[1] = 64;
        k2.q = 1; k2.ord = VW2_ORD_NAT; k2.pl = VW2_PL_OOP;
        r = vw2__st_rec(k2, "rowplan", 1, "4.16", "colplan", 1, "2.16.2", "b", 1, "8",
                  "ran", 2, "1", "ns", 2, "7485.2", "metric", 2, "fwd1", "units", 2, "ns",
                  "src", 2, "migrated", "from", 2, "fft2d_c2c_wisdom.txt:19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank 2d-family record");
    }
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "save");
    vw2_close(&st);

    vw2_open(&st2, dir, 1);
    VW2_ST_CHECK(st2.nrec == 3, "reload count == 3");
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
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    vw2__st_slurp(path, buf, sizeof buf);
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
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
        fprintf(f, "@cell t=c2c n=32 q=1 ord=scr place=ip | eng=stride zz_future=7 | ran=1 src=race date=2026-08-19 |\n");
        fprintf(f, "@cell t=zzz n=99 q=1 ord=scr place=ip | eng=alien | ran=1 |\n");
        fprintf(f, "@futuredirective payload=opaque\n");
        fclose(f);
    }
    vw2_close(&st2);
    vw2_open(&st, dir, 1);
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 32, 1, VW2_ORD_SCR, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "zz_future"), "7"), "unknown TOKEN parsed + kept");
    }
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 64, 2, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "2", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "bank another record to force resave");
    VW2_ST_CHECK(vw2_save(&st) == VW2_OK, "resave with unknowns present");
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "zz_future=7") != NULL, "unknown token survives resave");
    VW2_ST_CHECK(strstr(buf, "t=zzz") != NULL, "unknown-transform record carried opaque");
    VW2_ST_CHECK(strstr(buf, "@futuredirective payload=opaque") != NULL, "unknown directive carried opaque");
    vw2_close(&st);

    /* ---- T5: version refuse + poison (file never clobbered) ------------ */
    printf("T5 version-refuse:\n");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_OOP]);
    {
        FILE *f = fopen(path, "wb");
        fprintf(f, "@vw2 2.0\n@cell t=c2c n=128 q=1 ord=scr place=oop | eng=future | ran=1 |\n");
        fclose(f);
    }
    rc = vw2_open(&st, dir, 1);
    VW2_ST_CHECK(rc == VW2_EVERSION, "open reports version refusal");
    VW2_ST_CHECK(st.poisoned[VW2_SHARD_OOP] == 1, "shard poisoned");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 128, 1, VW2_ORD_SCR, VW2_PL_OOP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "poisoned file's cells miss (never silently served)");
    }
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 256, 1, VW2_ORD_SCR, VW2_PL_OOP),
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

    /* ---- T6: stale .tmp sweep ------------------------------------------ */
    printf("T6 tmp-sweep:\n");
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_STRIDE]);
    snprintf(tmpp, sizeof tmpp, "%s.tmp", path);
    { FILE *f = fopen(tmpp, "wb"); fprintf(f, "junk from a crashed writer\n"); fclose(f); }
    vw2_open(&st, dir, 1);
    { FILE *f = fopen(tmpp, "rb"); VW2_ST_CHECK(f == NULL, "stale .tmp swept at open"); if (f) fclose(f); }
    vw2_close(&st);

    /* ---- T7: two-store merge-on-save (concurrent sessions) ------------- */
    printf("T7 merge-on-save:\n");
    vw2_open(&sa, dir, 1);
    vw2_open(&sb, dir, 1);            /* B loads BEFORE A saves */
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "8.8.8",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    VW2_ST_CHECK(vw2_bank(&sa, &r) == VW2_OK, "A banks cellA");
    VW2_ST_CHECK(vw2_save(&sa) == VW2_OK, "A saves");
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 768, 8, VW2_ORD_SCR, VW2_PL_IP),
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

    /* ---- T8: merge rank + date tie-break ------------------------------- */
    printf("T8 merge-rank:\n");
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "4.128",
              "ran", 2, "8", "src", 2, "env:VFFT_TCUT", "arms", 2, "1", "date", 2, "2026-08-20", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "env-shaped verdict cannot replace a raced one");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "4.128",
              "ran", 2, "8", "src", 2, "seed", "date", 2, "2026-08-20", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "seed cannot replace a raced verdict");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "16.32",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-18", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_ERANK, "older race cannot replace newer");
    vw2_rec_free(&r);
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "chain", 1, "16.32",
              "ran", 2, "8", "src", 2, "race", "date", 2, "2026-08-21", NULL);
    VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "newer race replaces");
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 512, 8, VW2_ORD_SCR, VW2_PL_IP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "chain"), "16.32"), "replacement served");
    }

    /* ---- T9: cross-metric refusal --------------------------------------- */
    printf("T9 cross-metric:\n");
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 4096, 1, VW2_ORD_NAT, VW2_PL_IP),
              "mode", 1, "zcasc",
              "ran", 2, "1", "ns", 2, "4400.0", "metric", 2, "joint2", "units", 2, "ns",
              "src", 2, "race", "date", 2, "2026-08-22", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_EMETRIC, "joint2 cannot replace fwd1 (never compare across metrics)");
    vw2_rec_free(&r);

    /* ---- T10: wildcard law ---------------------------------------------- */
    printf("T10 wildcards:\n");
    {
        vw2_key_t wk = vw2__st_key(VW2_T_C2C, 8192, -1, VW2_ORD_ANY, VW2_PL_ANY);
        r = vw2__st_rec(wk, "eng", 1, "split_oop", "chain", 1, "8.32.32",
                  "ran", 2, "4", "ns", 2, "19289.3", "metric", 2, "fwd1", "units", 2, "ns",
                  "src", 2, "race", "date", 2, "2026-08-19", NULL);
        rc = vw2_bank(&st, &r);
        VW2_ST_CHECK(rc == VW2_EWILDCARD, "fresh wildcard bank refused (no from=)");
        vw2_rec_free(&r);
        r = vw2__st_rec(wk, "eng", 1, "split_oop", "chain", 1, "8.32.32",
                  "ran", 2, "4", "ns", 2, "19289.3", "metric", 2, "fwd1", "units", 2, "ns",
                  "src", 2, "migrated", "from", 2, "oop_wisdom.txt:119", "date", 2, "2026-08-19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "migrated wildcard bank accepted");
    }
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 8192, 1, VW2_ORD_NAT, VW2_PL_OOP);
        const vw2_rec_t *hit = vw2_lookup(&st, &q);
        VW2_ST_CHECK(hit && !strcmp(vw2_rec_get(hit, "chain"), "8.32.32"), "wildcard serves any axis value");
    }
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 8192, 1, VW2_ORD_NAT, VW2_PL_OOP),
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

    /* ---- T11: seed records never served --------------------------------- */
    printf("T11 seeds:\n");
    vw2_open(&st, dir, 1);
    {
        vw2_key_t wk = vw2__st_key(VW2_T_C2C, 999, -1, VW2_ORD_ANY, VW2_PL_ANY);
        r = vw2__st_rec(wk, "eng", 1, "stride",
                  "src", 2, "seed", "from", 2, "c2r_wisdom.txt:2", "date", 2, "2026-08-19", NULL);
        VW2_ST_CHECK(vw2_bank(&st, &r) == VW2_OK, "seed banks fine");
    }
    {
        vw2_key_t q = vw2__st_key(VW2_T_C2C, 999, 4, VW2_ORD_SCR, VW2_PL_IP);
        VW2_ST_CHECK(vw2_lookup(&st, &q) == NULL, "seed is never served as a verdict");
    }

    /* ---- T12: read-only guard ------------------------------------------- */
    printf("T12 read-only:\n");
    vw2_close(&st);
    vw2_open(&st, dir, 0);
    r = vw2__st_rec(vw2__st_key(VW2_T_C2C, 2048, 1, VW2_ORD_SCR, VW2_PL_IP),
              "eng", 1, "stride", "ran", 2, "1", "src", 2, "race", "date", 2, "2026-08-19", NULL);
    rc = vw2_bank(&st, &r);
    VW2_ST_CHECK(rc == VW2_EREADONLY, "bank refused on read-only store");
    vw2_rec_free(&r);
    VW2_ST_CHECK(vw2_save(&st) == VW2_EREADONLY, "save refused on read-only store");
    vw2_close(&st);

    /* ---- T13: ref= parse ------------------------------------------------ */
    printf("T13 ref:\n");
    {
        vw2_key_t k;
        VW2_ST_CHECK(vw2_ref_parse("cell(t=c2c,n=4096,q=1,ord=scr,place=oop)", &k) == 1, "ref parses");
        VW2_ST_CHECK(k.t == VW2_T_C2C && k.n[0] == 4096 && k.q == 1 &&
              k.ord == VW2_ORD_SCR && k.pl == VW2_PL_OOP, "ref key fields correct");
        VW2_ST_CHECK(vw2_ref_parse("cell(t=c2c,n=4096)", &k) == 0, "partial ref refused (full key required)");
    }

    /* ---- T14: quarantine append ----------------------------------------- */
    printf("T14 quarantine:\n");
    VW2_ST_CHECK(vw2_quarantine_append(dir, "garbage-variant-token", "oop_wisdom.txt:45",
                                "4096 1 2 3 32 16 8 690 -1435841587 32761 81219.0") == VW2_OK,
          "quarantine append");
    snprintf(path, sizeof path, "%s/wisdom2_quarantine.txt", dir);
    vw2__st_slurp(path, buf, sizeof buf);
    VW2_ST_CHECK(strstr(buf, "raw=4096 1 2 3 32 16 8 690 -1435841587 32761 81219.0") != NULL,
          "raw= runs to end of line, escape-free");

    printf("\n[wisdom2_g0] %s — %d failure(s)\n", vw2__st_fail ? "FAIL" : "ALL PASS", vw2__st_fail);
    return vw2__st_fail;
}

#endif /* VFFT_WISDOM2_SELFTEST_H */
