"""
Entry point: run the experiment, then post-hoc analysis.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python run.py               # full pipeline (writes results/run_final.json)
    python run.py --analyze     # only re-run analysis on existing run_final.json
    python run.py --add-models  # run only models in config not yet in run_final.json,
                                # merge them in, then re-analyze (incremental)
"""

import argparse

import config
import experiment
import analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="Skip the experiment and re-run analysis on the existing run_final.json")
    ap.add_argument("--add-models", action="store_true",
                    help="Run only config models missing from run_final.json, merge, re-analyze")
    args = ap.parse_args()

    if args.add_models:
        experiment.add_models(config.MODELS)
        print()
    elif not args.analyze:
        experiment.run_experiment()
        print()

    analysis.main()


if __name__ == "__main__":
    main()
