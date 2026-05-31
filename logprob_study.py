"""
Confidence-elicitation sub-study (GPT-4o only): verbalized vs. logprob confidence.

Motivation (see report 4.1): "confidence" in the main study is the model's
verbalized self-report. Here we add a second, independent measure on the SAME
questions — the model's token-probability mass on the answer letter (Kadavath
et al., 2022) — and ask whether the two even agree, and which is better
calibrated (cf. Tian et al., 2023, who find verbalized can beat logprobs for
RLHF'd models).

Scope: GPT-4o only (it exposes logprobs cleanly through OpenRouter; the 7-8B
open models generally do not). We reuse the existing 150 questions and pair each
logprob measurement with the verbalized answer already stored in run_final.json.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python logprob_study.py            # run + analyze
    python logprob_study.py --analyze  # analyze existing results/logprob_gpt4o.json
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

import config
from calibration import (
    expected_calibration_error,
    brier_score,
    bootstrap_metric,
    reliability_bins,
)
from client import call_letter_logprobs
from experiment import load_questions

MODEL_KEY = "gpt-4o"
MODEL_ID = config.MODELS[MODEL_KEY]
RESULTS_DIR = "results"
RUN_PATH = os.path.join(RESULTS_DIR, "run_final.json")
OUT_PATH = os.path.join(RESULTS_DIR, "logprob_gpt4o.json")


# ---------- Prompt ----------

def letter_prompt(persona_desc: str, q) -> tuple[str, str]:
    system = (
        f"{persona_desc}\n\n"
        "Answer the multiple-choice question with ONLY the single capital letter "
        "of the best choice (e.g. B). Output just the letter — no punctuation, no "
        "explanation."
    )
    user = f"Question: {q.text}\n\nChoices:\n{q.formatted_choices()}\n\nAnswer:"
    return system, user


# ---------- Run ----------

def _verbalized_index(run: list[dict]) -> dict[tuple[int, str], dict]:
    """Map (qid, persona) -> gpt-4o Round 1 verbalized record."""
    idx = {}
    for q in run:
        for a in q["agents"]:
            if a["model_key"] == MODEL_KEY:
                idx[(q["qid"], a["persona"])] = a
    return idx


def _one(q, persona_key, persona_desc, correct_letter, verb):
    sys_p, usr_p = letter_prompt(persona_desc, q)
    try:
        lp = call_letter_logprobs(MODEL_ID, sys_p, usr_p, q.letters)
    except Exception as e:
        print(f"  ! logprob ({q.qid}/{persona_key}) failed: {e}")
        return None
    v_ans = verb["round1_answer"]
    return {
        "qid": q.qid,
        "persona": persona_key,
        "correct_letter": correct_letter,
        # verbalized (from main run)
        "verbalized_answer": v_ans,
        "verbalized_confidence": verb["round1_confidence"],
        "verbalized_correct": bool(verb["round1_correct"]),
        # logprob (this study)
        "logprob_answer": lp.answer,
        "logprob_confidence": 100.0 * lp.prob,          # prob of logprob's own pick
        "logprob_correct": bool(lp.answer == correct_letter),
        # internal prob the model placed on the answer it VERBALLY gave
        "p_verbalized_answer": 100.0 * lp.dist.get(v_ans, 0.0),
        "dist": lp.dist,
    }


def run_study():
    if not os.path.exists(RUN_PATH):
        raise SystemExit(f"{RUN_PATH} not found — run the main experiment first.")
    run = json.load(open(RUN_PATH))
    verb_idx = _verbalized_index(run)
    if not verb_idx:
        raise SystemExit(f"No '{MODEL_KEY}' records in {RUN_PATH}.")

    questions = load_questions()
    rows = []
    executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    futures = []
    for q in questions:
        for persona_key, persona_desc in config.PERSONAS.items():
            verb = verb_idx.get((q.qid, persona_key))
            if verb is None:
                continue
            futures.append(executor.submit(
                _one, q, persona_key, persona_desc, q.correct_letter, verb))

    done = 0
    for fut in as_completed(futures):
        r = fut.result()
        if r is not None:
            rows.append(r)
        done += 1
        if done % 50 == 0:
            print(f"  ... {done}/{len(futures)}")
    executor.shutdown(wait=True)

    rows.sort(key=lambda r: (r["qid"], r["persona"]))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(rows, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}  ({len(rows)} paired measurements)")
    return rows


# ---------- Analyze ----------

def analyze(rows: list[dict]):
    v_conf = [r["verbalized_confidence"] for r in rows]
    v_corr = [r["verbalized_correct"] for r in rows]
    l_conf = [r["logprob_confidence"] for r in rows]
    l_corr = [r["logprob_correct"] for r in rows]
    p_verb = [r["p_verbalized_answer"] for r in rows]

    v_ece = bootstrap_metric(v_conf, v_corr, "ece", config.ECE_N_BINS,
                             n_boot=config.BOOTSTRAP_N)
    l_ece = bootstrap_metric(l_conf, l_corr, "ece", config.ECE_N_BINS,
                             n_boot=config.BOOTSTRAP_N)
    v_bri = brier_score(v_conf, v_corr)
    l_bri = brier_score(l_conf, l_corr)
    v_acc = float(np.mean(v_corr))
    l_acc = float(np.mean(l_corr))

    agree = float(np.mean([r["logprob_answer"] == r["verbalized_answer"] for r in rows]))
    # correlation: stated confidence vs internal prob of the stated answer
    corr = float(np.corrcoef(v_conf, p_verb)[0, 1])

    lines = [
        "# Confidence elicitation: verbalized vs. logprob (GPT-4o)",
        "",
        f"N = {len(rows)} paired (question, persona) measurements. "
        "Verbalized = self-reported 0-100; logprob = softmax mass on the answer "
        "letter (first-token logprobs). ECE shown with bootstrap 95% CI.",
        "",
        "| Confidence source | Mean conf | Accuracy | ECE | Brier |",
        "|---|---:|---:|---:|---:|",
        f"| Verbalized (self-report) | {np.mean(v_conf):.1f}% | {v_acc:.1%} | "
        f"{v_ece[0]:.3f} [{v_ece[1]:.3f}, {v_ece[2]:.3f}] | {v_bri:.3f} |",
        f"| Logprob (token prob) | {np.mean(l_conf):.1f}% | {l_acc:.1%} | "
        f"{l_ece[0]:.3f} [{l_ece[1]:.3f}, {l_ece[2]:.3f}] | {l_bri:.3f} |",
        "",
        f"- **Answer agreement** (logprob argmax == verbalized answer): {agree:.1%}",
        f"- **Correlation** between stated confidence and internal probability of "
        f"the *stated* answer: r = {corr:+.3f}",
        f"- **Mean internal prob on the stated answer**: {np.mean(p_verb):.1f}% "
        f"(vs {np.mean(v_conf):.1f}% stated)",
    ]
    table = "\n".join(lines)
    out_md = os.path.join(RESULTS_DIR, "logprob_comparison.md")
    with open(out_md, "w") as f:
        f.write(table + "\n")
    print(f"\nWrote {out_md}\n")
    print(table)

    _plots(rows, v_conf, v_corr, l_conf, l_corr, p_verb)


def _plots(rows, v_conf, v_corr, l_conf, l_corr, p_verb):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (1) two reliability curves
    ax = axes[0]
    for conf, corr, label, color in [
        (v_conf, v_corr, "verbalized", "tab:blue"),
        (l_conf, l_corr, "logprob", "tab:orange"),
    ]:
        centers, accs, counts = reliability_bins(conf, corr, config.ECE_N_BINS)
        xs = [c for c, a, n in zip(centers, accs, counts) if n > 0 and not np.isnan(a)]
        ys = [a for a, n in zip(accs, counts) if n > 0 and not np.isnan(a)]
        ax.plot(xs, ys, "o-", color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy in bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("GPT-4o reliability: verbalized vs. logprob")
    ax.legend()

    # (2) stated confidence vs internal prob of the stated answer
    ax = axes[1]
    x = np.array(v_conf) / 100.0
    y = np.array(p_verb) / 100.0
    jit = (np.random.default_rng(0).random(len(x)) - 0.5) * 0.01
    ax.scatter(x + jit, y, alpha=0.4, s=18)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="stated = internal")
    ax.set_xlabel("Stated (verbalized) confidence")
    ax.set_ylabel("Internal prob on the stated answer")
    ax.set_xlim(0, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Does GPT-4o believe what it says?")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "logprob.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true",
                    help="Analyze existing results/logprob_gpt4o.json without calling the API")
    args = ap.parse_args()

    if args.analyze:
        rows = json.load(open(OUT_PATH))
    else:
        rows = run_study()
    analyze(rows)


if __name__ == "__main__":
    main()
