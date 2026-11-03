import os, sys
sys.stdout.flush(); sys.stderr.flush()
os.execv(sys.executable, sys.orig_argv)
