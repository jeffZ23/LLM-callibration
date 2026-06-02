"""
Persona-count ablation: does the persona *ensemble* matter, or just having 3 votes?

The main study fixes three diverse personas (analytical / devil's advocate /
knowledge). This sub-study varies ONLY the persona set, holding everything else
identical (same 4 models, same 150 questions, same temperature, same H>0.6
trigger, same four reflection conditions):

  - nopersona : 3 agents with the SAME neutral prompt. Disagreement can only come
                from sampling noise, so this isolates the value of persona
                *diversity* vs. merely having three votes.
  - 3persona  : the existing main run (results/run_final.json) — the reference.
  - 5persona  : the original 3 + two more distinct stances (intuitive,
                probabilistic), to see whether more diverse voices sharpen the
                disagreement signal or just add noise.

We compare, per model and per persona set: baseline calibration (ECE/acc),
reflection trigger rate, and the reflection effect (ΔECE under `full`).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python persona_study.py                 # run both variants + analyze
    python persona_study.py --run nopersona # run one variant only
    python persona_study.py --analyze       # analyze existing run files, no API
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import config
import experiment
from analysis import _final_under_condition
from calibration import expected_calibration_error

RESULTS_DIR = "results"

# Capture the original three personas before any reassignment.
ORIGINAL_PERSONAS = dict(config.PERSONAS)
ORIGINAL_THRESHOLD = config.ENTROPY_THRESHOLD  # 0.6 in the main study

# With 5 agents the smallest non-zero vote entropy is the 4-vs-1 split (H=0.500);
# 3-2 is 0.673, etc. The main 0.6 threshold therefore *misses* 4-1 disagreement.
# Lowering the 5-persona threshold below 0.500 makes "any disagreement triggers,"
# matching the 3-persona design where the minimal 2-1 split (H=0.637) already
# triggers. We use 0.4 (clear of the 0.500 boundary; 5-0 agreement at H=0 still
# never triggers).
FIVE_PERSONA_THRESHOLD = 0.4

# --- nopersona: three identical neutral voices (sampling-only disagreement) ---
_NEUTRAL = (
    "You are a helpful assistant answering a multiple-choice question. "
    "Read the question and the choices carefully and pick the best answer."
)
NEUTRAL_PERSONAS = {"neutral_1": _NEUTRAL, "neutral_2": _NEUTRAL, "neutral_3": _NEUTRAL}

# --- 5persona: original three + two more distinct reasoning stances ---
EXTRA_PERSONAS = {
    "intuitive": (
        "You are an Intuitive Responder. Answer from immediate intuition and "
        "common knowledge. Go with the first answer that feels right rather than "
        "deliberating at length."
    ),
    "probabilistic": (
        "You are a Probabilistic Reasoner. Think in terms of base rates and "
        "likelihoods. Weigh how probable each option is, and let your stated "
        "confidence reflect that probability."
    ),
}
FIVE_PERSONAS = {**ORIGINAL_PERSONAS, **EXTRA_PERSONAS}

# variant name -> (persona set, run file, entropy threshold)
VARIANTS = {
    "nopersona": (NEUTRAL_PERSONAS, os.path.join(RESULTS_DIR, "run_nopersona.json"),
                  ORIGINAL_THRESHOLD),
    "5persona": (FIVE_PERSONAS, os.path.join(RESULTS_DIR, "run_5persona.json"),
                 FIVE_PERSONA_THRESHOLD),
}

# For the comparison table: label -> run file. The 3-persona column reuses the
# main study's run so we never re-spend on it.
COMPARE = [
    ("1-voice (no persona)", os.path.join(RESULTS_DIR, "run_nopersona.json")),
    ("3-persona (main)", os.path.join(RESULTS_DIR, "run_final.json")),
    ("5-persona", os.path.join(RESULTS_DIR, "run_5persona.json")),
]

MODEL_ORDER = list(config.MODELS)


# ---------- Run ----------

def run_variant(name: str):
    personas, out_path, threshold = VARIANTS[name]
    print(f"\n=== Running variant '{name}' "
          f"({len(personas)} personas: {list(personas)}; H>{threshold} trigger) "
          f"-> {out_path} ===")
    # experiment.* reads config.PERSONAS and config.ENTROPY_THRESHOLD dynamically,
    # so swapping them here is enough.
    config.PERSONAS = personas
    config.ENTROPY_THRESHOLD = threshold
    try:
        experiment.run_experiment(out_path=out_path)
    finally:
        config.PERSONAS = ORIGINAL_PERSONAS
        config.ENTROPY_THRESHOLD = ORIGINAL_THRESHOLD


# ---------- Analyze ----------

def _variant_stats(run: list[dict]) -> dict:
    """Per-model: baseline ECE/acc/conf, trigger rate, and ΔECE under `full`."""
    r1 = defaultdict(lambda: {"conf": [], "correct": []})
    trig = defaultdict(int)
    tot = defaultdict(int)
    reflecting = defaultdict(list)
    for q in run:
        for m, t in q.get("per_model_reflection_triggered", {}).items():
            tot[m] += 1
            if t:
                trig[m] += 1
        for a in q["agents"]:
            r1[a["model_key"]]["conf"].append(a["round1_confidence"])
            r1[a["model_key"]]["correct"].append(a["round1_correct"])
            if a.get("reflected"):
                reflecting[a["model_key"]].append(a)

    stats = {}
    for m, s in r1.items():
        ece = expected_calibration_error(s["conf"], s["correct"], config.ECE_N_BINS)
        acc = float(np.mean(s["correct"]))
        conf = float(np.mean(s["conf"]))
        trate = 100.0 * trig[m] / tot[m] if tot[m] else float("nan")

        agents = reflecting[m]
        if agents:
            r1c = [a["round1_confidence"] for a in agents]
            r1k = [a["round1_correct"] for a in agents]
            r1e = expected_calibration_error(r1c, r1k, config.ECE_N_BINS)
            fc, fk = [], []
            for a in agents:
                cc, kk = _final_under_condition(a, "full")
                fc.append(cc)
                fk.append(kk)
            post_e = expected_calibration_error(fc, fk, config.ECE_N_BINS)
            d_full = post_e - r1e
        else:
            r1e = post_e = d_full = float("nan")

        stats[m] = {
            "n_agents": len(s["conf"]),
            "acc": acc,
            "conf": conf,
            "ece": ece,
            "trate": trate,
            "n_trig": len(agents),
            "pre_ece": r1e,
            "post_ece": post_e,
            "d_full": d_full,
        }
    return stats


def _fmt(x, pct=False, signed=False):
    if x != x:  # NaN
        return "—"
    if pct:
        return f"{x:.1f}%"
    if signed:
        return f"{x:+.3f}"
    return f"{x:.3f}"


def analyze():
    loaded = []
    for label, path in COMPARE:
        if os.path.exists(path):
            loaded.append((label, _variant_stats(json.load(open(path)))))
        else:
            print(f"  (missing {path}; skipping '{label}')")

    if not loaded:
        raise SystemExit("No run files found. Run the variants first.")

    lines = [
        "# Persona-count ablation: does the persona ensemble matter?",
        "",
        "Everything fixed except the persona set: same 4 models, 150 TruthfulQA "
        "questions, temperature, and four reflection conditions. "
        "**1-voice** = three identical neutral prompts (disagreement from sampling "
        "only); **3-persona** = the main study; **5-persona** = main three + "
        "intuitive + probabilistic.",
        "",
        "Trigger: 1-voice and 3-persona use H>0.6 (their minimal 2-1 split is "
        "H=0.637). With 5 agents the minimal 4-1 split is only H=0.500, so the "
        "5-persona set uses H>0.4 — i.e. *any* disagreement triggers, matching the "
        "3-persona design.",
        "",
        "ΔECE(full) is the reflection effect on the triggered subset under the "
        "`full` peer signal (negative = reflection improved calibration).",
        "",
        "| Model | Persona set | #agents/q | Baseline ECE | Acc | Mean conf | "
        "Trigger rate | n triggered | ΔECE(full) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_q = {"1-voice (no persona)": 3, "3-persona (main)": 3, "5-persona": 5}
    for m in MODEL_ORDER:
        for label, stats in loaded:
            if m not in stats:
                continue
            s = stats[m]
            lines.append(
                f"| {m} | {label} | {per_q.get(label, '')} | {_fmt(s['ece'])} | "
                f"{_fmt(100 * s['acc'], pct=True)} | {s['conf']:.1f}% | "
                f"{_fmt(s['trate'], pct=True)} | {s['n_trig']} | "
                f"{_fmt(s['d_full'], signed=True)} |"
            )
        lines.append("| | | | | | | | | |")

    table = "\n".join(lines)
    out_md = os.path.join(RESULTS_DIR, "persona_comparison.md")
    with open(out_md, "w") as f:
        f.write(table + "\n")
    print(f"\nWrote {out_md}\n")
    print(table)

    _plot(loaded)


def _plot(loaded):
    labels = [lab for lab, _ in loaded]
    x = np.arange(len(MODEL_ORDER))
    width = 0.8 / max(1, len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    # (1) baseline ECE by model across persona sets
    ax = axes[0]
    for i, (lab, stats) in enumerate(loaded):
        ys = [stats.get(m, {}).get("ece", np.nan) for m in MODEL_ORDER]
        ax.bar(x + i * width, ys, width=width * 0.95, label=lab,
               edgecolor="black", alpha=0.85)
    ax.set_xticks(x + (len(labels) - 1) * width / 2)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Baseline ECE (lower = better)")
    ax.set_title("Calibration vs. persona set")
    ax.legend(fontsize=8)

    # (2) trigger rate by model across persona sets
    ax = axes[1]
    for i, (lab, stats) in enumerate(loaded):
        ys = [stats.get(m, {}).get("trate", np.nan) for m in MODEL_ORDER]
        ax.bar(x + i * width, ys, width=width * 0.95, label=lab,
               edgecolor="black", alpha=0.85)
    ax.set_xticks(x + (len(labels) - 1) * width / 2)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Reflection trigger rate (%)")
    ax.set_title("Disagreement vs. persona set")
    ax.legend(fontsize=8)

    # (3) before/after ECE on the triggered subset (condition=full), per persona
    #     set: paired pre (hatched) and post (solid) bars within each set's slot.
    ax = axes[2]
    for i, (lab, stats) in enumerate(loaded):
        pre = [stats.get(m, {}).get("pre_ece", np.nan) for m in MODEL_ORDER]
        post = [stats.get(m, {}).get("post_ece", np.nan) for m in MODEL_ORDER]
        base = x + i * width
        color = f"C{i}"
        ax.bar(base - width * 0.22, pre, width=width * 0.42, color=color,
               alpha=0.45, edgecolor="black", hatch="//",
               label=f"{lab} pre" if i == 0 else None)
        ax.bar(base + width * 0.22, post, width=width * 0.42, color=color,
               alpha=0.9, edgecolor="black",
               label=f"{lab} post" if i == 0 else None)
    ax.set_xticks(x + (len(labels) - 1) * width / 2)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("ECE on triggered subset (full)")
    ax.set_title("Reflection: before (hatched) → after (solid)")
    # legend keyed by persona-set color, plus the pre/post hatch convention
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=f"C{i}", edgecolor="black", alpha=0.85, label=lab)
               for i, (lab, _) in enumerate(loaded)]
    handles += [Patch(facecolor="white", edgecolor="black", hatch="//", label="pre"),
                Patch(facecolor="grey", edgecolor="black", label="post")]
    ax.legend(handles=handles, fontsize=7, ncol=2)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "persona_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["nopersona", "5persona", "both"],
                    default="both", help="which variant(s) to run (calls the API)")
    ap.add_argument("--analyze", action="store_true",
                    help="only analyze existing run files; no API calls")
    args = ap.parse_args()

    if not args.analyze:
        names = ["nopersona", "5persona"] if args.run == "both" else [args.run]
        for name in names:
            run_variant(name)
    analyze()


if __name__ == "__main__":
    main()
