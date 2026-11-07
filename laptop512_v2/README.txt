LARGE-RADIX AVX-512 TIMING, ZEN 4 LAPTOP
========================================

What it is
  Times large-radix plans against the laptop's banked plans, all in the same run.
  The plans under test do not exist in the library. They were built and checked in a scratch copy on the desktop.
  This answers the one thing the desktop cannot: whether they are faster on real AVX-512 hardware.
  The first list compares each forced power-of-two plan with its twiddles stored full width and stored as one (cos, sin) pair, side by side in the same run.

What it does not do
  It installs nothing, builds nothing and writes nothing outside this folder.
  It does not race and does not touch the library's wisdom. store/ is a read-only copy of the Zen 4 wisdom.
  There is no MKL in it.

How to run it
  1. Copy this whole folder to the laptop.
  2. Plug in the charger and close other programs.
  3. Open a terminal in this folder and run:   python run_all.py
  4. It takes about 3 hours and prints the time left. The laptop stays awake by itself.
  5. When it says DONE, send back large1d_results.zip.

If it stops
  Run the same command again. It continues from the last finished job.
  If it says STOPPED, it explains why, and nothing was timed.
  A single job that fails is written to results/failures.txt and the run carries on.
