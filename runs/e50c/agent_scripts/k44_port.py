import sys, os
g = save_edit.__globals__
print("kernel PORT:", g.get("PORT"), "MODEL:", str(g.get("MODEL"))[-60:], "MAX_RUNS:", g.get("MAX_RUNS"))
print("argv:", sys.argv, "orig_argv:", getattr(sys, "orig_argv", None))
print("executable:", sys.executable, "pid", os.getpid(), "ppid", os.getppid())
