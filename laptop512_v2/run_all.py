"""AVX-512 timing of the large-radix plans on the Zen 4 laptop. One command: python run_all.py
Nothing is installed and nothing is written outside this folder. The library's wisdom is not touched: the tool only reads store/ and never writes it.
If the run is interrupted, start the same command again: it continues from the last finished job.
When it ends it writes large1d_results.zip next to this file."""
import ctypes
import os
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
EXE = os.path.join(HERE, 'large1d.exe')
OUT = os.path.join(HERE, 'results')
# name, job list, rounds, extra flags: the lists that answer the main question run first
LISTS = [
    ('narrow', 'jobs/narrow.txt', 15, ['--share']),
    ('zscan', 'jobs/mids.txt', 15, ['--share']),
    ('yscan', 'jobs/lasts.txt', 15, ['--share']),
    ('zstage', 'jobs/stages.txt', 15, ['--share']),
    ('ends', 'jobs/ends.txt', 15, ['--share']),
    ('zpad', 'jobs/padded.txt', 15, ['--share']),
    ('plans', 'jobs/plans.txt', 15, ['--share']),
    ('turn', 'jobs/turned.txt', 15, ['--share']),
    ('var', 'jobs/variants.txt', 15, ['--share']),
    ('flat', 'jobs/flat.txt', 15, ['--share']),
    ('kern', 'jobs/kernels.txt', 15, ['--share']),
    ('nd', 'jobs/rows2d3d.txt', 15, ['--share']),
]
SECONDS_PER_BLOCK = 0.3


def say(s):
    print(time.strftime('%H:%M:%S ') + s, flush=True)


def stop(s):
    say('STOPPED: ' + s)
    sys.exit(1)


def preflight():
    if not os.path.exists(EXE):
        stop('large1d.exe is not in this folder.')
    p = subprocess.run([EXE, '--check', '--jobs', 'jobs/preflight.txt'], capture_output=True, text=True)
    if p.returncode == -1073741515 or p.returncode == 3221225781:
        stop('Windows could not start large1d.exe: a DLL is missing. libwinpthread-1.dll must sit next to it.')
    if p.returncode == -1073741795 or p.returncode == 3221225501:
        stop('large1d.exe hit an illegal instruction: this CPU is not Zen 4 (it needs AVX-512).')
    lines = [l for l in p.stdout.splitlines() if l.startswith('CHECK')]
    bad = [l for l in lines if 'ok=1' not in l]
    if p.returncode != 0 or not lines or bad:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        stop('the correctness check failed (exit code %d, %d checked, %d wrong). Nothing was timed.' % (p.returncode, len(lines), len(bad)))
    say('correctness check passed on %d plans' % len(lines))


def load(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith('#')]


def main():
    os.makedirs(OUT, exist_ok=True)
    # keep the laptop awake for the length of the run
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    preflight()
    todo = []
    total = 0.0
    for name, path, rounds, flags in LISTS:
        lines = load(path)
        donef = os.path.join(OUT, name + '.done')
        done = int(open(donef).read()) if os.path.exists(donef) else 0
        for i, l in enumerate(lines):
            if i >= done:
                est = (len(l.split()) + 1) * rounds * SECONDS_PER_BLOCK
                total += est
                todo.append((name, i, l, rounds, flags, est))
    say('%d jobs to run, about %.1f hours. Leave the laptop plugged in and do not use it meanwhile.' % (len(todo), total / 3600))
    left = total
    failed = 0
    for name, i, line, rounds, flags, est in todo:
        job = os.path.join(OUT, '_job.txt')
        with open(job, 'w') as f:
            f.write(line + '\n')
        t0 = time.time()
        with open(os.path.join(OUT, 't5_' + name + '.txt'), 'a') as so, open(os.path.join(OUT, 't5_' + name + '.err'), 'a') as se:
            p = subprocess.run([EXE] + flags + ['--rounds', str(rounds), '--jobs', job], stdout=so, stderr=se)
        if p.returncode != 0:
            failed += 1
            with open(os.path.join(OUT, 'failures.txt'), 'a') as f:
                f.write('%s line %d exit %d: %s\n' % (name, i, p.returncode, line[:200]))
            say('job %s %d FAILED with exit %d, carrying on' % (name, i, p.returncode))
        with open(os.path.join(OUT, name + '.done'), 'w') as f:
            f.write(str(i + 1))
        left -= est
        say('%s job %d done in %.0f s, about %.1f hours left' % (name, i + 1, time.time() - t0, left / 3600))
    z = os.path.join(HERE, 'large1d_results.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for n in sorted(os.listdir(OUT)):
            if not n.startswith('_'):
                f.write(os.path.join(OUT, n), n)
    say('DONE, %d jobs failed. Send back: %s' % (failed, z))


main()
