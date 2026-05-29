"""
Post-hoc analysis: produce slide-ready outputs from results/run.json.

Generates:
  results/summary_table.md     — accuracy + ECE per model, baseline vs post-reflection
  results/reliability.png      — reliability diagram per model
  results/disagreement.png     — vote-entropy vs |confidence - accuracy|
  results/case_correction.md   — a question where reflection corrected a wrong answer
  results/case_herding.md      — a question where agents herded toward a wrong majority
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import config
from calibration import (
    expected_calibration_error,
    reliability_bins,
    vote_entropy,
)


RESULTS_DIR = "results"
RUN_PATH = os.path.join(RESULTS_DIR, "run.json")


def load_run() -> list[dict]:
    with open(RUN_PATH) as f:
        return json.load(f)


# ---------- Summary table ----------

def summary_table(run: list[dict]) -> str:
    """Per-model accuracy + ECE, baseline vs post-reflection.
    Post-reflection metrics use round-2 answer where reflected, else round-1."""
    by_model: dict[str, dict[str, list]] = defaultdict(lambda: {
        "r1_conf": [], "r1_correct": [],
        "final_conf": [], "final_correct": [],
        "n_reflected": 0, "n_total": 0,
    })

    for q in run:
        for a in q["agents"]:
            m = a["model_key"]
            slot = by_model[m]
            slot["n_total"] += 1
            slot["r1_conf"].append(a["round1_confidence"])
            slot["r1_correct"].append(a["round1_correct"])
            if a["reflected"]:
                slot["n_reflected"] += 1
                slot["final_conf"].append(a["round2_confidence"])
                slot["final_correct"].append(a["round2_correct"])
            else:
                slot["final_conf"].append(a["round1_confidence"])
                slot["final_correct"].append(a["round1_correct"])

    lines = [
        "| Model | N | Acc (baseline) | ECE (baseline) | Acc (post-ref) | ECE (post-ref) | % reflected |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, s in by_model.items():
        n = s["n_total"]
        acc1 = float(np.mean(s["r1_correct"]))
        ece1 = expected_calibration_error(s["r1_conf"], s["r1_correct"], config.ECE_N_BINS)
        accF = float(np.mean(s["final_correct"]))
        eceF = expected_calibration_error(s["final_conf"], s["final_correct"], config.ECE_N_BINS)
        pct_r = 100.0 * s["n_reflected"] / max(n, 1)
        lines.append(
            f"| {m} | {n} | {acc1:.2%} | {ece1:.3f} | {accF:.2%} | {eceF:.3f} | {pct_r:.1f}% |"
        )
    table = "\n".join(lines)

    out = os.path.join(RESULTS_DIR, "summary_table.md")
    with open(out, "w") as f:
        f.write("# Calibration summary\n\n" + table + "\n")
    print(f"Wrote {out}")
    print()
    print(table)
    return table


# ---------- Reliability diagram ----------

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
        # Bars: bin accuracy. Width = 1/n_bins.
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
    out = os.path.join(RESULTS_DIR, "reliability.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------- Disagreement vs error ----------

def disagreement_plot(run: list[dict]):
    """Per (question, model) point: H vs (mean confidence - accuracy)."""
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
    cmap = {m: c for m, c in zip(set(models), ["tab:blue", "tab:orange", "tab:green"])}
    for m in set(models):
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
    out = os.path.join(RESULTS_DIR, "disagreement.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------- Case studies ----------

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
        if a["reflected"]:
            r2_ok = "✓" if a["round2_correct"] else "✗"
            rows.append(
                f"| {a['model_key']} | {a['persona']} | {a['round1_answer']} | "
                f"{a['round1_confidence']:.0f}% | {r1_ok} | "
                f"{a['round2_answer']} | {a['round2_confidence']:.0f}% | {r2_ok} |"
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
        if a["reflected"]:
            parts.append(f"{head} (R2, {a['round2_answer']}@{a['round2_confidence']:.0f}%): "
                         f"_{a['round2_reasoning']}_")
    return "\n\n".join(parts)


def _score_correction(q: dict) -> int:
    """How many agents flipped from wrong to correct in reflection?"""
    s = 0
    for a in q["agents"]:
        if a["reflected"] and not a["round1_correct"] and a["round2_correct"]:
            s += 1
    return s


def _score_herding(q: dict) -> int:
    """How many agents flipped from correct to wrong in reflection?"""
    s = 0
    for a in q["agents"]:
        if a["reflected"] and a["round1_correct"] and not a["round2_correct"]:
            s += 1
    return s


def case_studies(run: list[dict]):
    # Best correction case
    correction = max(run, key=_score_correction, default=None)
    herding = max(run, key=_score_herding, default=None)

    for label, q, score_fn in [
        ("case_correction", correction, _score_correction),
        ("case_herding", herding, _score_herding),
    ]:
        if q is None or score_fn(q) == 0:
            note = ("_No instance of this pattern in the current run. "
                    "This is itself a finding worth flagging — try more questions "
                    "or a higher-difficulty dataset (GPQA) to surface it._")
        else:
            note = ""
        out = os.path.join(RESULTS_DIR, f"{label}.md")
        title = "Reflection corrected an error" if label == "case_correction" \
                else "Agents herded toward a wrong majority"
        body = f"# Case study: {title}\n\n"
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
    print()
    reliability_diagram(run)
    disagreement_plot(run)
    case_studies(run)
    print("\nAll outputs in", RESULTS_DIR)


if __name__ == "__main__":
    main()
