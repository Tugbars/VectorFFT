/* wisdom2_folders.h — ONE CPU, ONE FOLDER: which folder of a store root is
 * this CPU's.
 *
 * The shipped store is a root of folders, one per CPU that raced rows
 * (docs/design/wisdom_system.md §2):
 *
 *     src/wisdom/
 *       new/        the shard files with headers only: the next unknown CPU's
 *       14900KF/    one CPU's rows
 *       Zen4/       another's
 *
 * A folder belongs to the CPU whose identity its files are stamped with (the
 * `@meta` line; common/support/cpu_identity.h builds the string). Folder NAMES
 * carry no meaning; a user may rename theirs. Selection, for an identity:
 *
 *   1. the folder whose stamp EQUALS the identity (the first by name, should
 *      two ever carry one stamp);
 *   2. else `new/`, while it is unstamped: the first save stamps it, and it is
 *      this CPU's folder from then on;
 *   3. else (`new/` is another CPU's by now, or the root has none) a folder
 *      named after the identity, created here.
 *
 * Nothing is ever read from another CPU's folder: a row is a measurement of
 * the machine that raced it. This file knows no CPU: the identity is a string
 * its caller passes. A directory the caller NAMES (VFFT_WISDOM_DIR,
 * vfft_wisdom_load(dir)) is a store by itself and is never scanned.
 */
#ifndef VFFT_WISDOM2_FOLDERS_H
#define VFFT_WISDOM2_FOLDERS_H

#include "common/wisdom/wisdom2.h"
#if !defined(_WIN32)
#  include <sys/stat.h>
#endif

#define VW2_FOLDER_NEW "new"

/* how a folder was chosen */
#define VW2_FOLDER_MATCHED 1   /* its stamp equals the identity */
#define VW2_FOLDER_NEW_TAKEN 2 /* the unstamped new/ */
#define VW2_FOLDER_CREATED 3   /* a folder named after the identity */

/* The stamp a folder's files carry: the `@meta` payload of the first shard
 * file that has one ("" = unstamped). Only the header lines are read. Returns
 * the number of shard files the folder holds. */
static inline int vw2_folder_stamp(const char *dir, char *out, size_t n)
{
    int shard, nfiles = 0;
    if (n) out[0] = '\0';
    for (shard = 0; shard < VW2_NSHARDS; shard++) {
        char path[768], line[4096];
        FILE *f;
        int fresh = 1;                               /* at the start of a line */
        snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[shard]);
        f = fopen(path, "rb");
        if (!f) continue;
        nfiles++;
        while ((!n || !out[0]) && fgets(line, sizeof line, f)) {
            const size_t len = strlen(line);
            const int whole = (len && line[len - 1] == '\n');
            if (fresh) {
                if (!strncmp(line, "@cell ", 6)) break;          /* the header is over */
                if (!strncmp(line, "@meta ", 6) && n) {
                    size_t k = strlen(line + 6);
                    while (k && (line[6 + k - 1] == '\n' || line[6 + k - 1] == '\r' || line[6 + k - 1] == ' ')) k--;
                    if (k >= n) k = n - 1;
                    memcpy(out, line + 6, k);
                    out[k] = '\0';
                }
            }
            fresh = whole;
        }
        fclose(f);
        if (n && out[0]) {                           /* count the rest without reading them */
            for (shard++; shard < VW2_NSHARDS; shard++) {
                snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[shard]);
                f = fopen(path, "rb");
                if (f) { nfiles++; fclose(f); }
            }
            break;
        }
    }
    return nfiles;
}

static inline int vw2__is_dir(const char *path)
{
#if defined(_WIN32)
    const DWORD a = GetFileAttributesA(path);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

/* the name of the folder under root whose stamp equals identity, the first
 * by name; 1 = found */
static inline int vw2__folder_find(const char *root, const char *identity, char *name, size_t n)
{
    int found = 0;
    char stamp[256], path[768];
#if defined(_WIN32)
    WIN32_FIND_DATAA fd;
    HANDLE h;
    snprintf(path, sizeof path, "%s/*", root);
    h = FindFirstFileA(path, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    do {
        const char *e = fd.cFileName;
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || e[0] == '.') continue;
#else
    DIR *d = opendir(root);
    struct dirent *de;
    if (!d) return 0;
    while ((de = readdir(d)) != NULL) {
        const char *e = de->d_name;
        if (e[0] == '.') continue;
        snprintf(path, sizeof path, "%s/%s", root, e);
        if (!vw2__is_dir(path)) continue;
#endif
        snprintf(path, sizeof path, "%s/%s", root, e);
        if (!vw2_folder_stamp(path, stamp, sizeof stamp) || strcmp(stamp, identity) != 0) continue;
        if (!found || strcmp(e, name) < 0) snprintf(name, n, "%s", e);
        found = 1;
#if defined(_WIN32)
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    }
    closedir(d);
#endif
    return found;
}

/* This identity's folder under root, as a path in out. id_name is the name a
 * created folder gets. Returns VW2_FOLDER_*; one line on stderr says which
 * folder was taken whenever none matched. */
static inline int vw2_folder_select(const char *root, const char *identity, const char *id_name,
                                    char *out, size_t n)
{
    char name[256], stamp[256], path[768];
    if (vw2__folder_find(root, identity, name, sizeof name)) {
        snprintf(out, n, "%s/%s", root, name);
        return VW2_FOLDER_MATCHED;
    }
    snprintf(path, sizeof path, "%s/" VW2_FOLDER_NEW, root);
    if (vw2__is_dir(path)) {
        (void)vw2_folder_stamp(path, stamp, sizeof stamp);
        if (!stamp[0]) {
            snprintf(out, n, "%s", path);
            fprintf(stderr, "[wisdom2] no folder under %s carries this CPU's identity (%s): its rows go to "
                            "%s, which the first save stamps as this CPU's\n", root, identity, path);
            return VW2_FOLDER_NEW_TAKEN;
        }
    }
    snprintf(out, n, "%s/%s", root, id_name);
#if defined(_WIN32)
    (void)CreateDirectoryA(out, NULL);
#else
    (void)mkdir(out, 0777);
#endif
    fprintf(stderr, "[wisdom2] no folder under %s carries this CPU's identity (%s): its rows go to the "
                    "new folder %s\n", root, identity, out);
    return VW2_FOLDER_CREATED;
}

#endif /* VFFT_WISDOM2_FOLDERS_H */
