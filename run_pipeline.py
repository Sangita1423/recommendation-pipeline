"""
Entry point: run the whole recommendation pipeline end-to-end.

Terminal:
    python run_pipeline.py                     # full run
    python run_pipeline.py --only train_models evaluate_models
    python run_pipeline.py --config configs/config.yaml --retries 2

Jupyter notebook (paste this whole file in a cell, or):
    from run_pipeline import main
    main()                                     # full run
    main(["--only", "evaluate_models"])        # subset

Schedule it (e.g. daily 02:00) with cron:
    0 2 * * * cd /path/to/recommendation-pipeline && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
"""
import argparse
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    """Folder that contains run_pipeline.py and src/ (works in scripts and notebooks)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:                      # running inside Jupyter: no __file__
        here = Path.cwd()
        for cand in [here, *here.parents, here / "recommendation-pipeline"]:
            if (cand / "src").is_dir() and (cand / "configs").is_dir():
                return cand
        raise FileNotFoundError(
            "Could not find the project folder (the one containing src/ and configs/). "
            "Use os.chdir(r'...\\recommendation-pipeline') before running this cell.")


ROOT = _project_root()
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestration.pipeline import DAG, run_pipeline  # noqa: E402
from src.utils import load_config  # noqa: E402


def _in_notebook() -> bool:
    return "ipykernel" in sys.modules


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config YAML")
    ap.add_argument("--retries", type=int, default=1, help="retries per task")
    ap.add_argument("--only", nargs="*", choices=list(DAG), help="run a subset of tasks")

    if argv is None and _in_notebook():
        argv = []                          # ignore Jupyter's own --f=kernel.json argument
    args, _unknown = ap.parse_known_args(argv)

    record = run_pipeline(load_config(args.config), retries=args.retries, only=args.only)
    print(f"\nPipeline status: {record['status'].upper()}  "
          f"({record['duration_sec']}s)  ->  reports/runs/latest_run.png")
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    code = main()
    if not _in_notebook():                 # sys.exit would raise SystemExit inside Jupyter
        sys.exit(code)
