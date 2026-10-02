#!/usr/bin/env python3
"""merge_to_host.py -- HACK (2026-09-28), pending a real `--merge --host`.

The gauntlet's own stage_merge copies a run's shards into src/wisdom/ itself,
the shared store. On a second calibration host that is wrong: the per-host rule
is that stores never mix, so a Zen 4 run's verdicts belong in src/wisdom/Zen4/.

The copy cannot be wholesale either. A run's store starts as a COPY of the
shipped wisdom, so it holds the calibration host's rows as well as this run's;
copying the file would import 14900KF verdicts into the Zen 4 folder and
recreate the mixing from the other direction.

So: lift only the rows this run BANKED -- `src=race` with the run's own date --
and upsert them into the host folder's shards by their key (everything before
the first " | "), replacing a row for the same cell and appending a new one.
Every touched shard is backed up first.

  python gauntlet/merge_to_host.py --name <run> --host Zen4 [--date YYYY-MM-DD]
  python gauntlet/merge_to_host.py --name <run> --host Zen4 --dry-run
"""
import argparse, datetime, io, os, re, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED = os.path.join(ROOT, "src", "wisdom")
RESULTS = os.path.join(ROOT, "gauntlet", "results")
SHARDS = ("wisdom2_oop.txt", "wisdom2_scr.txt", "wisdom2_real.txt",
          "wisdom2_prime.txt", "wisdom2_2d.txt", "wisdom2_3d.txt")


def key_of(line):
    """A record's identity: the KEY section, everything before the first ' | '."""
    return line.split(" | ", 1)[0].strip()


def read_lines(path):
    if not os.path.isfile(path):
        return []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        return [l.rstrip("\r\n") for l in f]


def write_lines(path, lines):
    # LF only: the store's text contract (wisdom2/README 2.3)
    with io.open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the run directory under gauntlet/results/")
    ap.add_argument("--host", required=True, help="the per-host folder under src/wisdom/ (e.g. Zen4)")
    ap.add_argument("--date", default=datetime.date.today().isoformat(),
                    help="bank date to lift (default: today)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    store = os.path.join(RESULTS, a.name, "store")
    dest = os.path.join(SHIPPED, a.host)
    if not os.path.isdir(store):
        sys.exit("no store at %s" % store)
    if not os.path.isdir(dest):
        if a.dry_run:
            print("would create %s" % dest)
        else:
            os.makedirs(dest)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_new = total_repl = 0
    for shard in SHARDS:
        src_lines = read_lines(os.path.join(store, shard))
        if not src_lines:
            continue
        # this run's banked rows only
        banked = [l for l in src_lines
                  if l.startswith("@cell") and "src=race" in l and ("date=" + a.date) in l]
        if not banked:
            continue

        dpath = os.path.join(dest, shard)
        dlines = read_lines(dpath)
        if not dlines:                      # a shard the host folder does not have yet
            # 🔴 NEVER take @meta from the SOURCE store (fixed 2026-10-02). A run's
            # store is seeded from the shipped wisdom, so its shard headers carry the
            # CALIBRATION host's stamp -- copying them wrote
            # "@meta host=intel-f6m183" onto 1509 Zen 4 verdicts, which is exactly
            # the host-provenance lie the @meta mechanism exists to catch. Prefer a
            # sibling shard already in this host folder; else emit no @meta and let
            # the library stamp it on first open.
            hdr = [l for l in src_lines
                   if l.startswith("@") and not l.startswith("@cell")
                   and not l.startswith("@meta")]
            meta = None
            for sib in SHARDS:
                if sib == shard:
                    continue
                for l in read_lines(os.path.join(dest, sib)):
                    if l.startswith("@meta"):
                        meta = l
                        break
                if meta:
                    break
            if meta:
                hdr.append(meta)
            dlines = hdr if hdr else ["@vw2 1.2"]

        index = {key_of(l): i for i, l in enumerate(dlines) if l.startswith("@cell")}
        new = repl = 0
        for row in banked:
            k = key_of(row)
            if k in index:
                if dlines[index[k]] != row:
                    dlines[index[k]] = row
                    repl += 1
            else:
                dlines.append(row)
                index[k] = len(dlines) - 1
                new += 1

        total_new += new
        total_repl += repl
        print("%-20s %3d banked  ->  %3d new, %3d replaced   %s" %
              (shard, len(banked), new, repl, os.path.relpath(dpath, ROOT)))
        if a.dry_run or (new == 0 and repl == 0):
            continue
        if os.path.isfile(dpath):
            shutil.copy2(dpath, dpath + ".bak_pre_hostmerge_" + stamp)
        write_lines(dpath, dlines)

    print("%s: %d new, %d replaced%s" %
          ("DRY RUN" if a.dry_run else "merged into src/wisdom/" + a.host,
           total_new, total_repl, "" if a.dry_run else "  (.bak_pre_hostmerge_%s beside each)" % stamp))


if __name__ == "__main__":
    sys.exit(main())
