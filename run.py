"""
Entry point: run the experiment, then post-hoc analysis.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python run.py            # full pipeline
    python run.py --analyze  # only re-run analysis on existing results/run.json
"""

import argparse

import experiment
import analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="Skip the experiment and re-run analysis on the existing run.json")
    args = ap.parse_args()

    if not args.analyze:
        experiment.run_experiment()
        print()

    analysis.main()


if __name__ == "__main__":
    main()
