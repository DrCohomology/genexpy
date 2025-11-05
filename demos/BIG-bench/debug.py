import os
from pathlib import Path

import genexpy.managers as mu
from importlib import reload

demo_dir = Path(".") / "demos" / "BIG-bench"
main = mu.ProjectManager("config.yaml", demo_dir)
df_nstar = main.generalizability_analysis()






