"""Generate plots for any existing standardized experiment JSON records."""
import json
from pathlib import Path
from plot_results import generate_run_plots
ROOT=Path("results")
for p in ROOT.rglob("*.json"):
    if "comparison" in p.parts: continue
    try:
        r=json.loads(p.read_text(encoding="utf-8"))
    except Exception: continue
    if isinstance(r,dict) and "run_id" in r:
        generate_run_plots(r,p.parent)
        print("Plotted:",p)
