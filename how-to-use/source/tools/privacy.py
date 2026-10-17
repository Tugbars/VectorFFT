"""What quoted program output must never carry: an absolute path, the user's name, the
machine's name. Used by the listings harness (it refuses such output) and by the build."""
import getpass
import os
import re
import tempfile
from pathlib import Path

ABSOLUTE = [
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"), "a Windows absolute path"),
    (re.compile(r"\\\\[A-Za-z0-9._-]+\\"), "a UNC path"),
    (re.compile(r"(?<![\w.~-])/(?:home|Users|users|c|d|e|mnt|tmp|usr|opt|var|private|root|cygdrive|etc)/"),
     "a POSIX absolute path"),
    (re.compile(r"(?<![\w])~[/\\]"), "a path in a home directory"),
]


def _names():
    names = set()
    for n in (os.environ.get("USERNAME"), os.environ.get("USER"), os.environ.get("LOGNAME")):
        if n:
            names.add(n)
    try:
        names.add(getpass.getuser())
    except Exception:
        pass
    host = {n for n in (os.environ.get("COMPUTERNAME"), os.environ.get("HOSTNAME")) if n}
    return {n for n in names if len(n) >= 3}, {h for h in host if len(h) >= 3}


def _places():
    out = set()
    for p in (Path.home(), Path(tempfile.gettempdir()), Path(__file__).resolve().parents[3]):
        s = str(p)
        out.update({s, s.replace("\\", "/")})
    return {p for p in out if len(p) > 3}


def findings(text):
    """A list of what is wrong with text (empty when it may be quoted)."""
    out = []
    users, hosts = _names()
    low = text.lower()
    for u in users:
        if u.lower() in low:
            out.append("the user name")
    for h in hosts:
        if h.lower() in low:
            out.append("the machine name")
    for p in _places():
        if p.lower() in low:
            out.append("a local path")
    for rx, what in ABSOLUTE:
        m = rx.search(text)
        if m:
            line = text[:m.start()].count("\n") + 1
            out.append("%s (line %d: %r)" % (what, line, text.split("\n")[line - 1][:120]))
    return out
