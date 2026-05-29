# Preliminary Experiment: Disagreement-Based Reflection for LLM Calibration

**Group 3 — Calibration Across Models**
**Status:** Preliminary results from a 50-question pilot run, intended to validate the experimental pipeline ahead of the full study.

## 1. Summary

We implemented and ran a working prototype of the proposed two-stage experiment (calibration → reflection) on 50 questions from TruthfulQA, using two underlying LLMs and three persona-conditioned agents per LLM. Both research questions are addressed in a directional sense by the data, with appropriate caveats about sample size. The pilot also surfaces a non-trivial negative result that strengthens the motivation for the final study.

## 2. Experimental setup

**Dataset.** TruthfulQA mc1 (multiple-choice, single correct option), `validation` split. Fifty questions sampled with a fixed random seed. Answer-choice positions are shuffled per question to remove the dataset's known position bias (correct option is otherwise always first).

**Models.** Two open-source LLMs queried via the OpenRouter API:
- `meta-llama/llama-3.1-8b-instruct`
- `qwen/qwen-2.5-7b-instruct`

These were chosen as similarly-sized models from different families, allowing a clean cross-architecture comparison without confounding by model scale.

**Agents.** Each model is instantiated as three persona-conditioned agents, following Jiang (2026):
- *Analytical Reasoner* — stepwise logical analysis
- *Devil's Advocate* — challenges the obvious answer; flags misconceptions
- *Knowledge-Focused* — anchors on verifiable facts; instructed to express uncertainty as lower confidence rather than guessing

This yields six agents per question (2 models × 3 personas), 300 baseline answers in total.

**Round 1 (calibration).** Each agent independently answers the question and reports a self-reported confidence in [0, 100]. No agent sees any other agent's response.

**Round 2 (reflection).** For each model independently, we compute Shannon vote entropy across its three personas. If `H > 0.6` (corresponding roughly to a 2-vs-1 disagreement), all three agents from that model are re-prompted with:
- their own initial answer and confidence
- the anonymized peer answer distribution (leave-one-out)
- the mean peer confidence

Each agent then commits to a final answer and revised confidence. Reflection within a model is independent of the other model's results — this isolates the underlying-LLM variable for RQ2.

**Metrics.**
- *Expected Calibration Error (ECE)* with M = 10 bins: weighted average gap between bin-mean confidence and bin-mean accuracy (Guo et al., 2017).
- *Vote entropy* (Shannon, in nats) over within-model answer distributions, used as the disagreement signal.
- *Reward* (post-hoc, not used for training): r = 1{correct} − 0.5 · |c − 1{correct}|.

## 3. Results

### 3.1 Baseline calibration (RQ1)

Both models are systematically overconfident, with similar magnitude but distinct distributional patterns.

| Model        | N   | Accuracy | Mean confidence | Gap (conf − acc) | ECE   |
|--------------|----:|---------:|----------------:|-----------------:|------:|
| llama-3.1-8b | 150 |    68.7% |           88.1% |          +19.4pp | 0.199 |
| qwen-2.5-7b  | 150 |    66.0% |           84.9% |          +18.9pp | 0.219 |

Reliability diagrams (in `results/reliability.png`) show that Llama concentrates predictions in the 0.7–1.0 confidence range with a narrow miscalibration band, whereas Qwen produces a wider confidence distribution with several poorly-calibrated bins. ECE values are close, but the *shapes* of miscalibration differ.

A more substantive finding is that **within-model vote entropy predicts overconfidence**. Per-question correlations between vote entropy and (mean confidence − accuracy):

| Model        | Pearson r | Mean gap when H ≤ 0.6 | Mean gap when H > 0.6 |
|--------------|----------:|----------------------:|----------------------:|
| llama-3.1-8b |    +0.547 |                +0.058 |                +0.626 |
| qwen-2.5-7b  |    +0.262 |                +0.125 |                +0.356 |

When Llama's three personas agree, its confidence is within 6 percentage points of accuracy — i.e., **Llama is well-calibrated when it knows the answer**. When the personas disagree, the gap explodes to +63 points. Qwen exhibits the same pattern with weaker effect size. Disagreement is therefore a usable signal of overconfidence, and the strength of this signal is itself model-dependent.

### 3.2 Reflection effect (RQ2)

Reflection was triggered on 24% of questions for Llama and 28% for Qwen. Naive reflection slightly *worsened* calibration in both models:

| Model        | ECE (R1) | ECE (post-reflection) | Δ ECE  |
|--------------|---------:|----------------------:|-------:|
| llama-3.1-8b |    0.199 |                 0.207 | +0.008 |
| qwen-2.5-7b  |    0.219 |                 0.234 | +0.015 |

Restricting the analysis to the agent-answers where reflection actually fired makes the picture more pointed:

| Model        | n  | R1 accuracy | R1 ECE | R2 ECE | Δ ECE  |
|--------------|---:|------------:|-------:|-------:|-------:|
| llama-3.1-8b | 36 |       19.4% |  0.626 |  0.678 | +0.052 |
| qwen-2.5-7b  | 42 |       42.9% |  0.461 |  0.498 | +0.037 |

The questions on which reflection triggered were objectively hard — Llama scored below random on its triggered subset — yet agents reported confidences near 95%. Reflection failed to recalibrate them; if anything, it amplified the calibration gap.

The two models also behaved oppositely along a second axis: **mean confidence change** after reflection was −0.4 pp for Llama and **+9.5 pp for Qwen**. That is, Llama largely held its prior confidence, while Qwen raised confidence after seeing disagreement. The same reflection prompt produced opposite confidence dynamics across models — direct evidence that the reflection effect is model-dependent in *both* sign and magnitude. This asymmetry widened from the 30-question pilot (−3.4 pp vs +5.6 pp), increasing our confidence that it is not an artifact of small samples.

Net flip counts among reflecting agents:

| Model        | wrong → right | right → wrong | net |
|--------------|--------------:|--------------:|----:|
| llama-3.1-8b |             4 |             6 |  −2 |
| qwen-2.5-7b  |            10 |            12 |  −2 |

Both models lose more correct answers to herding than they gain in correction.

### 3.3 Mechanism — case studies

Two qualitative examples (in `results/case_correction.md` and `results/case_herding.md`) illustrate the dynamics.

**Correction (success factors).** Llama's three personas independently agreed on the correct answer (no reflection). Qwen split 2/3 wrong (D, "hard work") and 1/3 right (E, "many factors"). Reflection successfully converted both wrong agents to E. *Reflection works when the dissenter is correct and peer reasoning supports them.*

**Herding (prevalence of homosexuality).** All three Llama personas were confidently wrong (H = 0, no reflection triggered — a separate failure mode of within-model entropy as a signal). Qwen split 2/3 right (E, "less than 6%") and 1/3 wrong (B, "≥10%"). After reflection, both correct Qwen agents flipped to B; the wrong agent flipped to E. The Qwen knowledge persona's revised reasoning explicitly invoked imagined supporting studies ("more recent studies suggest the percentage is higher"), suggesting a confabulation in service of the herd. *Reflection fails when peer popularity is misaligned with truth, and the failure produces post-hoc rationalization rather than honest revision.*

A third worked example (`How many neurons are in the human brain?`, in conversation logs) shows both models triggering reflection at the threshold, with Qwen converging correctly on the right answer while Llama partially converged but lost a previously-correct agent — illustrating the model-dependent reflection effect at the level of a single question.

## 4. Discussion

The pilot supports a directional answer to both research questions:

- **RQ1** (yes): Both 7–8B open models are systematically overconfident on TruthfulQA, with similar overall ECE but distinct miscalibration shapes. Vote entropy among same-model personas correlates with overconfidence, with the strength of the correlation itself differing across models.
- **RQ2** (qualified no, with novel finding): Naive disagreement-triggered reflection does not improve calibration; in this prototype it slightly degrades it. The reflection effect is model-dependent in both magnitude and *direction* of confidence updates.

The herding case suggests that the failure mode is driven by *vote popularity* rather than *peer reasoning quality* — agents collapse toward the majority even when the minority is correct. This is the central mechanism that the proposed final study will probe via prompt-component ablations (votes vs reasoning vs confidences).

## 5. Limitations

1. **Sample size.** N = 150 per model. ECE estimates fluctuate by ~0.05 across the 30- and 50-question runs; we report directions, not significance. The final study will use N ≥ 300 with bootstrapped confidence intervals.
2. **Two model families.** RQ2's interaction-effect claim requires more than two LLMs to generalize; we plan to add a third family (Mistral or Gemma) and a frontier model (Claude or GPT-4o-mini) in the final study.
3. **Self-reported confidence.** Open 7B-class models report confidence in coarse multiples of 5 or 10, which limits the granularity of the reliability diagram. We plan to compare against logprob-derived confidence where available.
4. **Single dataset.** TruthfulQA is designed to elicit confident misconceptions; we do not yet know whether the herding pattern persists on harder, lower-prior datasets such as GPQA. The final study will stratify by dataset to test this.

## 6. Proposed next steps

The following experimental pipeline follows directly from the pilot's findings:

1. **Phase 2 (mechanism).** Ablate the reflection prompt into four variants — full information, votes-only, confidence-only, and reasoning-only — to isolate which signal drives correction versus herding.
2. **Phase 3 (intervention).** Use the best-performing variant from Phase 2 to construct a calibration-preserving reflection prompt and test it head-to-head against the naive prompt.
3. **Phase 4 (generalization).** Add a third open model and one frontier model; add GPQA alongside TruthfulQA. Test whether the model-dependent interaction effect holds and whether reflection's success/failure stratifies cleanly by dataset difficulty.
4. **Phase 5 (optional).** Replace the fixed entropy threshold with a learned trigger policy (bandit-style update on the reward function), addressing the agentic-RL component of the course.

## 7. Code and reproducibility

All code is in `~/Desktop/calibration_prototype/`:

- `config.py` — experiment parameters
- `client.py` — OpenRouter API wrapper with structured output parsing
- `experiment.py` — Round 1 + Round 2 pipeline; per-question checkpointing
- `calibration.py` — ECE, reliability bins, vote entropy, reward function
- `analysis.py` — summary table, reliability diagram, disagreement scatter, case-study generators
- `run.py` — entry point; full run takes ~20 minutes with rate-limited free-tier API access

Raw per-agent results are saved to `results/run.json`.

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *arXiv:1706.04599.*
- Jiang, B. (2026). DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles. *arXiv:2603.20975.*
- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *arXiv:2109.07958.*
- Ozer, O., et al. (2025). MAR: Multi-Agent Reflexion Improves Reasoning Abilities in LLMs. *arXiv:2512.20845.*
- Yang, R., et al. (2024). Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation. *ICLR 2024 Workshop on Reliable and Responsible Foundation Models.*
