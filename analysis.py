"""
Post-hoc analysis: produce slide-ready outputs from results/run_final.json.

Final study additions over the pilot:
  - Three model families (Llama, Qwen, Mistral).
  - Reflection ablation: four conditions (full / votes_only / confidence_only /
    reasoning_only) isolating which peer signal drives correction vs herding.
  - Brier score alongside ECE (a strictly proper scoring rule).
  - Bootstrap 95% confidence intervals on every headline metric.

Generates:
  results/summary_final.md        — baseline acc / ECE / Brier per model, with CIs
  results/ablation_table.md       — per-model x per-condition reflection effect
  results/ablation.png            — ECE by reflection condition, grouped by model
  results/reliability_final.png   — reliability diagram per model (Round 1)
  results/disagreement_final.png  — vote entropy vs calibration gap (Round 1)
  results/case_correction_final.md — a question where reflection fixed a wrong answer
  results/case_herding_final.md    — a question where agents eroded a correct answer
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import config
from calibration import (
    expected_calibration_error,
    brier_score,
    bootstrap_metric,
    reliability_bins,
    vote_entropy,
)


RESULTS_DIR = "results"
RUN_PATH = os.path.join(RESULTS_DIR, "run_final.json")


def load_run() -> list[dict]:
    with open(RUN_PATH) as f:
        return json.load(f)


# ---------- Helpers ----------

def _final_under_condition(agent: dict, condition: str) -> tuple[float, bool]:
    """Final (confidence, correct) for an agent under a reflection condition.

    Uses the agent's revised answer if it reflected and produced output for this
    condition; otherwise falls back to its Round 1 answer (untriggered agents
    never change)."""
    if agent.get("reflected") and condition in agent.get("reflections", {}):
        r = agent["reflections"][condition]
        return r["confidence"], r["correct"]
    return agent["round1_confidence"], agent["round1_correct"]


def _ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------- Baseline summary (RQ1) ----------

def summary_table(run: list[dict]) -> str:
    by_model = defaultdict(lambda: {"conf": [], "correct": []})
    for q in run:
        for a in q["agents"]:
            by_model[a["model_key"]]["conf"].append(a["round1_confidence"])
            by_model[a["model_key"]]["correct"].append(a["round1_correct"])

    lines = [
        "# Baseline calibration (Round 1, no reflection)",
        "",
        "Point estimate with bootstrap 95% CI in brackets "
        f"({config.BOOTSTRAP_N} resamples).",
        "",
        "| Model | N | Accuracy | Mean conf | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, s in by_model.items():
        n = len(s["conf"])
        acc_p, acc_lo, acc_hi = bootstrap_metric(
            s["conf"], s["correct"], "acc",
            n_boot=config.BOOTSTRAP_N, alpha=config.BOOTSTRAP_ALPHA)
        ece_p, ece_lo, ece_hi = bootstrap_metric(
            s["conf"], s["correct"], "ece", config.ECE_N_BINS,
            n_boot=config.BOOTSTRAP_N, alpha=config.BOOTSTRAP_ALPHA)
        bri_p, bri_lo, bri_hi = bootstrap_metric(
            s["conf"], s["correct"], "brier",
            n_boot=config.BOOTSTRAP_N, alpha=config.BOOTSTRAP_ALPHA)
        mean_conf = float(np.mean(s["conf"]))
        lines.append(
            f"| {m} | {n} | {_ci(acc_p, acc_lo, acc_hi)} | {mean_conf:.1f}% | "
            f"{_ci(ece_p, ece_lo, ece_hi)} | {_ci(bri_p, bri_lo, bri_hi)} |"
        )
    table = "\n".join(lines)
    out = os.path.join(RESULTS_DIR, "summary_final.md")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"Wrote {out}\n")
    print(table)
    return table


# ---------- Reflection ablation (RQ2 mechanism) ----------

def ablation_table(run: list[dict]) -> str:
    """For each model, restricted to the agents that actually reflected, compare
    Round 1 to each ablation condition. Identical subset across conditions => a
    clean within-subject ablation of the peer signal."""
    # reflecting agents per model
    by_model = defaultdict(list)
    for q in run:
        for a in q["agents"]:
            if a.get("reflected"):
                by_model[a["model_key"]].append(a)

    lines = [
        "# Reflection ablation (agents where disagreement triggered reflection)",
        "",
        "Each triggered agent was re-prompted once per condition, each revealing a",
        "different slice of peer information. Subset is identical across conditions.",
        "",
        "- **full** = peer votes + mean peer confidence",
        "- **votes_only** = peer vote distribution only",
        "- **confidence_only** = mean peer confidence only",
        "- **reasoning_only** = anonymized peer one-sentence reasonings only",
        "",
        "| Model | Condition | n | R1 ECE | ECE | ΔECE | Brier | acc | w→r | r→w | net | Δconf |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for m, agents in by_model.items():
        n = len(agents)
        r1_conf = [a["round1_confidence"] for a in agents]
        r1_correct = [a["round1_correct"] for a in agents]
        r1_ece = expected_calibration_error(r1_conf, r1_correct, config.ECE_N_BINS)
        r1_meanconf = float(np.mean(r1_conf))

        for c in config.REFLECTION_CONDITIONS:
            f_conf, f_correct, wr, rw = [], [], 0, 0
            for a in agents:
                conf, correct = _final_under_condition(a, c)
                f_conf.append(conf)
                f_correct.append(correct)
                if (not a["round1_correct"]) and correct:
                    wr += 1
                elif a["round1_correct"] and (not correct):
                    rw += 1
            ece = expected_calibration_error(f_conf, f_correct, config.ECE_N_BINS)
            bri = brier_score(f_conf, f_correct)
            acc = float(np.mean(f_correct))
            d_ece = ece - r1_ece
            d_conf = float(np.mean(f_conf)) - r1_meanconf
            lines.append(
                f"| {m} | {c} | {n} | {r1_ece:.3f} | {ece:.3f} | {d_ece:+.3f} | "
                f"{bri:.3f} | {acc:.1%} | {wr} | {rw} | {wr - rw:+d} | {d_conf:+.1f}pp |"
            )
        lines.append("| | | | | | | | | | | | |")  # spacer row between models

    table = "\n".join(lines)
    out = os.path.join(RESULTS_DIR, "ablation_table.md")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"\nWrote {out}\n")
    print(table)
    return table


def ablation_plot(run: list[dict]):
    """Grouped bars: post-reflection ECE on the reflecting subset, by condition,
    grouped by model, with the Round 1 baseline drawn as a marker per group."""
    by_model = defaultdict(list)
    for q in run:
        for a in q["agents"]:
            if a.get("reflected"):
                by_model[a["model_key"]].append(a)

    models = list(by_model)
    conditions = config.REFLECTION_CONDITIONS
    if not models:
        print("  (no reflecting agents; skipping ablation plot)")
        return

    cond_ece = {c: [] for c in conditions}
    baseline = []
    for m in models:
        agents = by_model[m]
        baseline.append(expected_calibration_error(
            [a["round1_confidence"] for a in agents],
            [a["round1_correct"] for a in agents], config.ECE_N_BINS))
        for c in conditions:
            f_conf = [_final_under_condition(a, c)[0] for a in agents]
            f_correct = [_final_under_condition(a, c)[1] for a in agents]
            cond_ece[c].append(
                expected_calibration_error(f_conf, f_correct, config.ECE_N_BINS))

    x = np.arange(len(models))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(2.6 * len(models) + 2, 5))
    for i, c in enumerate(conditions):
        ax.bar(x + i * width, cond_ece[c], width=width * 0.95, label=c,
               edgecolor="black", alpha=0.85)
    # baseline R1 ECE marker spanning each model group
    for j, b in enumerate(baseline):
        x0 = x[j] - width / 2
        x1 = x[j] + (len(conditions) - 0.5) * width
        ax.plot([x0, x1], [b, b], "k--", lw=1.5,
                label="Round 1 baseline" if j == 0 else None)
    ax.set_xticks(x + (len(conditions) - 1) * width / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel("ECE on reflecting subset (lower = better)")
    ax.set_title("Reflection ablation: which peer signal recalibrates?")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------- Reliability diagram (Round 1) ----------

def reliability_diagram(run: list[dict]):
    by_model = defaultdict(lambda: {"conf": [], "correct": []})
    for q in run:
        for a in q["agents"]:
            by_model[a["model_key"]]["conf"].append(a["round1_confidence"])
            by_model[a["model_key"]]["correct"].append(a["round1_correct"])

    fig, axes = plt.subplots(1, len(by_model), figsize=(5 * len(by_model), 4.5),
                             squeeze=False)
    for ax, (m, s) in zip(axes[0], by_model.items()):
        centers, accs, counts = reliability_bins(s["conf"], s["correct"],
                                                 config.ECE_N_BINS)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        width = 1.0 / config.ECE_N_BINS
        for c, a, n in zip(centers, accs, counts):
            if np.isnan(a) or n == 0:
                continue
            ax.bar(c, a, width=width * 0.9, alpha=0.6, edgecolor="black")
            ax.text(c, a + 0.02, str(n), ha="center", fontsize=7)
        ece = expected_calibration_error(s["conf"], s["correct"], config.ECE_N_BINS)
        ax.set_title(f"{m}\nECE = {ece:.3f}  (n={len(s['conf'])})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy in bin")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Baseline reliability diagram (Round 1, no reflection)")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "reliability_final.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------- Disagreement vs error (Round 1) ----------

def disagreement_plot(run: list[dict]):
    xs, ys, models = [], [], []
    for q in run:
        by_model = defaultdict(list)
        for a in q["agents"]:
            by_model[a["model_key"]].append(a)
        for m, group in by_model.items():
            answers = [a["round1_answer"] for a in group if a["round1_answer"] != "?"]
            if not answers:
                continue
            H = vote_entropy(answers)
            mean_conf = np.mean([a["round1_confidence"] for a in group]) / 100.0
            acc = np.mean([a["round1_correct"] for a in group])
            xs.append(H); ys.append(mean_conf - acc); models.append(m)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    uniq = list(dict.fromkeys(models))
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(uniq)}
    for m in uniq:
        mx = [x for x, mm in zip(xs, models) if mm == m]
        my = [y for y, mm in zip(ys, models) if mm == m]
        ax.scatter(mx, my, label=m, alpha=0.7, color=cmap[m])
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(config.ENTROPY_THRESHOLD, color="red", linestyle="--", lw=1,
               label=f"reflection trigger (H={config.ENTROPY_THRESHOLD})")
    ax.set_xlabel("Vote entropy (within model, across personas)")
    ax.set_ylabel("Mean confidence − accuracy   (>0 = overconfident)")
    ax.set_title("Disagreement vs calibration gap (Round 1)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "disagreement_final.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------- Case studies (using the 'full' condition) ----------

CASE_CONDITION = "full"


def _format_question(q: dict) -> str:
    lines = [f"**Question:** {q['text']}", "", "**Choices:**"]
    for L, c in zip(q["letters"], q["choices"]):
        marker = " ✅" if L == q["correct_letter"] else ""
        lines.append(f"- {L}. {c}{marker}")
    return "\n".join(lines)


def _format_agents(q: dict) -> str:
    rows = ["| Model | Persona | R1 ans | R1 conf | R1 ✓ | R2 ans | R2 conf | R2 ✓ |",
            "|---|---|:-:|---:|:-:|:-:|---:|:-:|"]
    for a in q["agents"]:
        r1_ok = "✓" if a["round1_correct"] else "✗"
        refl = a.get("reflections", {}).get(CASE_CONDITION)
        if a.get("reflected") and refl:
            r2_ok = "✓" if refl["correct"] else "✗"
            rows.append(
                f"| {a['model_key']} | {a['persona']} | {a['round1_answer']} | "
                f"{a['round1_confidence']:.0f}% | {r1_ok} | "
                f"{refl['answer']} | {refl['confidence']:.0f}% | {r2_ok} |"
            )
        else:
            rows.append(
                f"| {a['model_key']} | {a['persona']} | {a['round1_answer']} | "
                f"{a['round1_confidence']:.0f}% | {r1_ok} | — | — | — |"
            )
    return "\n".join(rows)


def _format_reasoning(q: dict) -> str:
    parts = []
    for a in q["agents"]:
        head = f"**{a['model_key']} / {a['persona']}**"
        parts.append(f"{head} (R1, {a['round1_answer']}@{a['round1_confidence']:.0f}%): "
                     f"_{a['round1_reasoning']}_")
        refl = a.get("reflections", {}).get(CASE_CONDITION)
        if a.get("reflected") and refl:
            parts.append(f"{head} (R2/{CASE_CONDITION}, {refl['answer']}@"
                         f"{refl['confidence']:.0f}%): _{refl['reasoning']}_")
    return "\n\n".join(parts)


def _score_correction(q: dict) -> int:
    s = 0
    for a in q["agents"]:
        refl = a.get("reflections", {}).get(CASE_CONDITION)
        if refl and not a["round1_correct"] and refl["correct"]:
            s += 1
    return s


def _score_herding(q: dict) -> int:
    s = 0
    for a in q["agents"]:
        refl = a.get("reflections", {}).get(CASE_CONDITION)
        if refl and a["round1_correct"] and not refl["correct"]:
            s += 1
    return s


def case_studies(run: list[dict]):
    correction = max(run, key=_score_correction, default=None)
    herding = max(run, key=_score_herding, default=None)

    for label, q, score_fn, title in [
        ("case_correction_final", correction, _score_correction,
         "Reflection corrected an error"),
        ("case_herding_final", herding, _score_herding,
         "Agents abandoned a correct answer under disagreement"),
    ]:
        if q is None or score_fn(q) == 0:
            note = ("_No instance of this pattern in the current run "
                    f"(condition = {CASE_CONDITION}). Try more questions or a "
                    "higher-difficulty dataset (GPQA) to surface it._")
        else:
            note = ""
        out = os.path.join(RESULTS_DIR, f"{label}.md")
        body = f"# Case study: {title}\n\n_(reflection condition shown: {CASE_CONDITION})_\n\n"
        if q is not None:
            body += _format_question(q) + "\n\n"
            body += "## Round 1 entropy per model\n\n"
            for m, H in q["per_model_entropy_round1"].items():
                trig = q["per_model_reflection_triggered"].get(m, False)
                body += f"- **{m}**: H = {H:.3f}  {'→ reflection triggered' if trig else ''}\n"
            body += "\n## Agent answers\n\n" + _format_agents(q) + "\n\n"
            body += "## Reasoning trace\n\n" + _format_reasoning(q) + "\n"
        if note:
            body += "\n" + note + "\n"
        with open(out, "w") as f:
            f.write(body)
        print(f"Wrote {out}")


# ---------- Entry ----------

def main():
    run = load_run()
    print(f"Loaded {len(run)} questions from {RUN_PATH}\n")
    summary_table(run)
    ablation_table(run)
    print()
    ablation_plot(run)
    reliability_diagram(run)
    disagreement_plot(run)
    case_studies(run)
    print("\nAll outputs in", RESULTS_DIR)


if __name__ == "__main__":
    main()
