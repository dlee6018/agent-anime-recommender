"""Evaluate all baseline models on dev + eval sets; log to W&B."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wandb  # noqa: E402

from src.data import titles  # noqa: E402
from src.evaluate import evaluate, load_eval_set, print_report  # noqa: E402
from src.registry import get_model  # noqa: E402

MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "pop", "genre", "content", "als", "blend"]

dev_raw = json.load(open(ROOT / "data" / "dev_set.json"))["queries"]
dev_set = {int(q): [int(r) for r in v] for q, v in dev_raw.items()}
eval_set = load_eval_set()
tt = titles()

for name in MODELS:
    fn = get_model(name)
    dev = evaluate(fn, dev_set)
    ev = evaluate(fn, eval_set)
    run = wandb.init(project="anime-rec", name=f"baseline-{name}",
                     config={"model": name}, reinit=True)
    run.log({"dev/p5": dev["precision_at_k"], "dev/mrr": dev["mrr"],
             "eval/p5": ev["precision_at_k"], "eval/p1": ev["precision_at_1"],
             "eval/mrr": ev["mrr"]})
    run.finish(quiet=True)
    print(f"\n=== {name}:  dev P@5={dev['precision_at_k']:.3f}  "
          f"eval P@5={ev['precision_at_k']:.3f}")
    if name == MODELS[-1]:
        print_report(ev, tt, n_worst=8)
