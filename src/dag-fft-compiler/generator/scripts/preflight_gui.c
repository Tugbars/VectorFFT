/* preflight_gui.c — the measurement prelaunch screen, as a real window
 * (section 40b). Quake-3-prelaunch model: an immediate-mode raylib GUI
 * refreshing probes at 1Hz while the user closes programs and watches
 * the machine go quiet; READY/WAIT verdict; launch / auto-launch with
 * countdown / abort. Same probes, thresholds, exit codes, and snapshot
 * sidecar as the terminal preflight (preflight.sh), which remains the
 * no-display fallback.
 *
 * Build: scripts/build_preflight_gui.sh  (raylib 5.0, static)
 * Usage: preflight_gui <pinned_cpu> <output_dir>
 * Exit:  0 = launch (snapshot written), 1 = abort.
 * Test hook: PFGUI_AUTOTEST=launch|abort acts after ~2s (headless CI). */

#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define W 760
#define H 470

typedef struct {
    double load1, load5, load15;
    char gov[32];
    int sibling;            /* -1 = none */
    int sib_busy;           /* percent, -1 = unknown */
    int temp_c;             /* -1 = unknown */
    char top[3][48];
    int ntop;
    int ncpu;
    /* verdict */
    int ok_load, ok_gov, ok_sib, ready;
} probes_t;

static long long stat_total[2] = {0, 0}, stat_idle[2] = {0, 0};

static void read_loadavg(probes_t *p) {
    FILE *f = fopen("/proc/loadavg", "r");
    p->load1 = p->load5 = p->load15 = -1;
    if (f) { fscanf(f, "%lf %lf %lf", &p->load1, &p->load5, &p->load15); fclose(f); }
}

static void read_governor(int cpu, probes_t *p) {
    char path[128];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu);
    FILE *f = fopen(path, "r");
    strcpy(p->gov, "n/a");
    if (f) { if (fscanf(f, "%31s", p->gov) != 1) strcpy(p->gov, "n/a"); fclose(f); }
}

static int read_sibling(int cpu) {
    char path[128], buf[64];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return -1; }
    fclose(f);
    for (char *tok = strtok(buf, ",-\n"); tok; tok = strtok(NULL, ",-\n")) {
        int v = atoi(tok);
        if (v != cpu) return v;
    }
    return -1;
}

/* sibling busy%% from /proc/stat deltas across the 1Hz tick */
static int read_sib_busy(int sib) {
    if (sib < 0) return -1;
    char want[16], line[256];
    snprintf(want, sizeof want, "cpu%d ", sib);
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return -1;
    long long v[10] = {0}, total = 0, idle = 0;
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, want, strlen(want)) == 0) {
            sscanf(line + strlen(want),
                   "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld",
                   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9]);
            for (int i = 0; i < 10; i++) total += v[i];
            idle = v[3] + v[4];
            found = 1;
            break;
        }
    }
    fclose(f);
    if (!found) return -1;
    stat_total[0] = stat_total[1]; stat_idle[0] = stat_idle[1];
    stat_total[1] = total;         stat_idle[1] = idle;
    long long dt = stat_total[1] - stat_total[0];
    if (stat_total[0] == 0 || dt <= 0) return -1;  /* first sample */
    return (int)((dt - (stat_idle[1] - stat_idle[0])) * 100 / dt);
}

static void read_top3(probes_t *p) {
    p->ntop = 0;
    FILE *f = popen("ps -eo pcpu,comm --no-headers --sort=-pcpu 2>/dev/null", "r");
    if (!f) return;
    char line[128];
    while (p->ntop < 3 && fgets(line, sizeof line, f)) {
        double pc; char comm[64];
        if (sscanf(line, "%lf %63s", &pc, comm) != 2) continue;
        if (pc <= 0.5) break;
        if (strcmp(comm, "ps") == 0 || strcmp(comm, "preflight_gui") == 0) continue;
        snprintf(p->top[p->ntop], sizeof p->top[0], "%s %.0f%%", comm, pc);
        p->ntop++;
    }
    pclose(f);
}

static int read_temp(void) {
    int best = -1;
    for (int z = 0; z < 16; z++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", z);
        FILE *f = fopen(path, "r");
        if (!f) break;
        int t = -1;
        if (fscanf(f, "%d", &t) == 1 && t / 1000 > best) best = t / 1000;
        fclose(f);
    }
    return best;
}

static void probe(int cpu, probes_t *p) {
    read_loadavg(p);
    read_governor(cpu, p);
    p->ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    p->sibling = read_sibling(cpu);
    p->sib_busy = read_sib_busy(p->sibling);
    read_top3(p);
    p->temp_c = read_temp();
    /* verdict — identical thresholds to preflight.sh */
    p->ok_load = !(p->load1 > p->ncpu * 0.25);
    p->ok_gov  = (strcmp(p->gov, "performance") == 0 || strcmp(p->gov, "n/a") == 0);
    p->ok_sib  = !(p->sib_busy > 10);
    p->ready   = p->ok_load && p->ok_gov && p->ok_sib;
}

static void snapshot(const char *outdir, int cpu, const probes_t *p) {
    char path[512], cmd[600];
    snprintf(cmd, sizeof cmd, "mkdir -p '%s'", outdir);
    if (system(cmd) != 0) { /* best effort */ }
    snprintf(path, sizeof path, "%s/preflight_snapshot.txt", outdir);
    FILE *f = fopen(path, "w");
    if (!f) return;
    time_t now = time(NULL);
    char ts[64];
    strftime(ts, sizeof ts, "%FT%T%z", localtime(&now));
    fprintf(f, "# measurement preflight snapshot (section 40b, gui)\n");
    fprintf(f, "timestamp: %s\npinned_cpu: %d\n", ts, cpu);
    fprintf(f, "load: %.2f %.2f %.2f\ngovernor: %s\n", p->load1, p->load5, p->load15, p->gov);
    fprintf(f, "smt_sibling: %d busy: %d%%\ntemp_max: %dC\n", p->sibling, p->sib_busy, p->temp_c);
    fprintf(f, "top_consumers:");
    for (int i = 0; i < p->ntop; i++) fprintf(f, " %s ", p->top[i]);
    fprintf(f, "\n");
    fclose(f);
}

static void lamp(int x, int y, int ok, const char *warntext) {
    if (ok) {
        DrawCircle(x, y, 6, (Color){80, 220, 120, 255});
        DrawText("ok", x + 14, y - 8, 18, (Color){80, 220, 120, 255});
    } else {
        DrawCircle(x, y, 6, (Color){240, 180, 60, 255});
        DrawText(warntext, x + 14, y - 8, 18, (Color){240, 180, 60, 255});
    }
}

static int button(Rectangle r, const char *label, Color base, Vector2 mouse) {
    int hot = CheckCollisionPointRec(mouse, r);
    Color c = base;
    if (hot) { c.r = (unsigned char)(c.r + 25); c.g = (unsigned char)(c.g + 25); c.b = (unsigned char)(c.b + 25); }
    DrawRectangleRec(r, c);
    DrawRectangleLinesEx(r, 1, (Color){200, 200, 200, 60});
    int tw = MeasureText(label, 20);
    DrawText(label, (int)(r.x + (r.width - tw) / 2), (int)(r.y + r.height / 2 - 10), 20, RAYWHITE);
    return hot && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: preflight_gui <cpu> <outdir>\n"); return 2; }
    int cpu = atoi(argv[1]);
    const char *outdir = argv[2];
    const char *autotest = getenv("PFGUI_AUTOTEST");

    SetTraceLogLevel(LOG_ERROR);
    InitWindow(W, H, "VFFT measurement prelaunch");
    SetTargetFPS(60);

    probes_t p;
    memset(&p, 0, sizeof p);
    probe(cpu, &p);

    double last = GetTime(), started = GetTime();
    int autolaunch = 0, countdown = -1, result = 1, done = 0;
    char buf[160];

    while (!WindowShouldClose() && !done) {
        double now = GetTime();
        if (now - last >= 1.0) {
            last = now;
            probe(cpu, &p);
            if (autolaunch && p.ready) {
                if (countdown < 0) countdown = 3;
                else if (--countdown < 0) { result = 0; done = 1; }
            }
            if (!p.ready) countdown = -1;
        }
        if (autotest && now - started > 2.0) {
            result = (strcmp(autotest, "launch") == 0) ? 0 : 1;
            done = 1;
        }

        Vector2 mouse = GetMousePosition();

        BeginDrawing();
        ClearBackground((Color){14, 17, 22, 255});

        DrawRectangle(0, 0, W, 64, (Color){22, 28, 36, 255});
        DrawText("VFFT  MEASUREMENT  PRELAUNCH", 28, 14, 28, (Color){120, 200, 255, 255});
        snprintf(buf, sizeof buf, "pinned core: cpu%d        output: build_tuned/", cpu);
        DrawText(buf, 30, 44, 14, (Color){140, 150, 160, 255});

        int y = 96;
        DrawText("load (1/5/15m)", 40, y, 20, RAYWHITE);
        snprintf(buf, sizeof buf, "%.2f  %.2f  %.2f", p.load1, p.load5, p.load15);
        DrawText(buf, 300, y, 20, (Color){200, 210, 220, 255});
        lamp(620, y + 10, p.ok_load, "busy");

        y += 44;
        DrawText("top consumers", 40, y, 20, RAYWHITE);
        buf[0] = 0;
        for (int i = 0; i < p.ntop; i++) {
            strncat(buf, p.top[i], sizeof buf - strlen(buf) - 1);
            strncat(buf, "   ", sizeof buf - strlen(buf) - 1);
        }
        if (p.ntop == 0) strcpy(buf, "(quiet)");
        DrawText(buf, 300, y, 18, (Color){200, 210, 220, 255});

        y += 44;
        DrawText("governor", 40, y, 20, RAYWHITE);
        snprintf(buf, sizeof buf, "cpu%d: %s", cpu, p.gov);
        DrawText(buf, 300, y, 20, (Color){200, 210, 220, 255});
        lamp(620, y + 10, p.ok_gov, p.gov);

        y += 44;
        DrawText("smt sibling", 40, y, 20, RAYWHITE);
        if (p.sibling >= 0 && p.sib_busy >= 0)
            snprintf(buf, sizeof buf, "cpu%d: %d%%", p.sibling, p.sib_busy);
        else
            strcpy(buf, "n/a");
        DrawText(buf, 300, y, 20, (Color){200, 210, 220, 255});
        lamp(620, y + 10, p.ok_sib, buf);

        y += 44;
        DrawText("max temp", 40, y, 20, RAYWHITE);
        if (p.temp_c >= 0) snprintf(buf, sizeof buf, "%dC", p.temp_c);
        else strcpy(buf, "n/a");
        DrawText(buf, 300, y, 20, (Color){200, 210, 220, 255});

        /* verdict bar — pulses when READY, Q3 style */
        y += 56;
        if (p.ready) {
            float pulse = 0.75f + 0.25f * (float)((1.0 + __builtin_sin(now * 3.0)) / 2.0);
            Color g = {(unsigned char)(40 * pulse), (unsigned char)(170 * pulse),
                       (unsigned char)(90 * pulse), 255};
            DrawRectangle(28, y, W - 56, 44, g);
            if (autolaunch && countdown >= 0)
                snprintf(buf, sizeof buf, "READY - launching in %d", countdown);
            else
                snprintf(buf, sizeof buf, "READY - machine is quiet");
            DrawText(buf, 44, y + 11, 22, RAYWHITE);
        } else {
            DrawRectangle(28, y, W - 56, 44, (Color){150, 110, 30, 255});
            DrawText("WAIT - close marked items, fields update live", 44, y + 11, 22, RAYWHITE);
        }

        /* buttons */
        y += 64;
        if (button((Rectangle){28, (float)y, 210, 46}, "LAUNCH NOW",
                   (Color){30, 90, 50, 255}, mouse) || IsKeyPressed(KEY_ENTER)) {
            result = 0; done = 1;
        }
        if (button((Rectangle){258, (float)y, 230, 46},
                   autolaunch ? "AUTO: ARMED" : "AUTO-LAUNCH",
                   autolaunch ? (Color){40, 80, 120, 255} : (Color){35, 50, 70, 255},
                   mouse) || IsKeyPressed(KEY_A)) {
            autolaunch = !autolaunch;
            countdown = -1;
        }
        if (button((Rectangle){508, (float)y, 224, 46}, "ABORT",
                   (Color){90, 40, 40, 255}, mouse) || IsKeyPressed(KEY_Q)) {
            result = 1; done = 1;
        }

        EndDrawing();
    }

    if (result == 0) snapshot(outdir, cpu, &p);
    CloseWindow();
    return result;
}
