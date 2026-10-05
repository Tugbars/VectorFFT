"""wisdom_folder.py -- which folder of src/wisdom/ is this CPU's.

The shipped store is a root of per-CPU folders (src/wisdom/README.md): the
library uses the one stamped with this CPU's identity, the unstamped new/ for a
CPU the root has not seen. The selection is the library's; tools ASK it
(recal_1d_probe --where, i.e. vfft_wisdom_folder()) and never re-derive it.

    from wisdom_folder import shipped_folder
    folder, identity = shipped_folder()            # binaries beside the sources
    folder, identity = shipped_folder(bin_dir)

    python gauntlet/wisdom_folder.py               # prints both
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""


def shipped_folder(bin_dir=None):
    """(folder, identity) of the library's own store on this CPU"""
    probe = os.path.join(bin_dir or HERE, "recal_1d_probe" + EXE)
    if not os.path.isfile(probe):
        raise SystemExit("missing %s -- build the gauntlet first (see gauntlet/README.md), or pass --bin-dir" % probe)
    env = os.environ.copy()
    env.pop("VFFT_WISDOM_DIR", None)        # the library's OWN store, not a directory someone named
    r = subprocess.run([probe, "--where"], capture_output=True, text=True, errors="replace", env=env)
    out = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    if r.returncode != 0 or "folder" not in out:
        raise SystemExit("%s --where failed (exit %d): %s" % (probe, r.returncode, (r.stderr or r.stdout).strip()[-400:]))
    return os.path.normpath(out["folder"]), out.get("identity", "")


if __name__ == "__main__":
    f, i = shipped_folder(sys.argv[1] if len(sys.argv) > 1 else None)
    print("folder:   %s\nidentity: %s" % (f, i))
