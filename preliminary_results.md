# Disagreement-Based Reflection for LLM Calibration: A Four-Model Study

**Group 3 — Calibration Across Models**
**Status:** Final study. 150 TruthfulQA questions, four LLMs (three 8B-class open models + one frontier model), three persona-conditioned agents per model, with a four-condition ablation of the reflection prompt and bootstrap confidence intervals on all headline metrics.

## 1. Summary

We ran a two-stage experiment — independent **calibration** followed by disagreement-triggered **reflection** — on 150 questions from TruthfulQA, across **four LLMs** spanning 8B-class open models to a frontier model (GPT-4o). Each model is instantiated as three persona-conditioned agents, giving 12 agents per question and ~1,800 baseline answers. We report Expected Calibration Error (ECE) and Brier score with bootstrap 95% confidence intervals, and we ablate the reflection prompt into four conditions to isolate *which* peer signal drives recalibration.

Three findings:

- **RQ1.** All four models express **near-identical confidence (~88%) regardless of accuracy**. Calibration quality is therefore governed almost entirely by accuracy: GPT-4o is well-calibrated (ECE 0.055) not because it is more modest, but because its accuracy (85%) rises to meet the same confidence the 8B models assert without justification. Within-model disagreement predicts overconfidence in every model (Pearson r = 0.43–0.64).
- **RQ2.** Naive reflection's *aggregate* effect is small and **model-dependent in sign** — it mildly helps two models and mildly hurts two. The ablation localizes the mechanism: revealing **peer confidence harms calibration in all four models**, while the *helpful* signal differs by model — peer **reasoning** recalibrates the weak models, peer **votes** recalibrate GPT-4o.
- The single invariant across four architectures and a 10×+ scale range is that **confidence-as-social-pressure is universally corrosive**.

## 2. Experimental setup

**Dataset.** TruthfulQA mc1 (multiple-choice, single correct option), `validation` split. 150 questions sampled with a fixed seed. Answer-choice positions are shuffled per question (deterministically) to remove the dataset's known position bias.

**Models.** Four models queried via the OpenRouter API:
- `meta-llama/llama-3.1-8b-instruct` (8B)
- `qwen/qwen-2.5-7b-instruct` (7B)
- `mistralai/ministral-8b-2512` (8B)
- `openai/gpt-4o` (frontier)

The first three are similarly-sized models from three different families, allowing cross-architecture comparison at fixed scale; GPT-4o adds a frontier reference point to test whether the small-model findings persist at scale and under heavy RLHF (proposal Phase 4).

**Agents.** Each model is instantiated as three persona-conditioned agents, following Jiang (2026):
- *Analytical Reasoner* — stepwise logical analysis
- *Devil's Advocate* — challenges the obvious answer; flags misconceptions
- *Knowledge-Focused* — anchors on verifiable facts; expresses uncertainty as lower confidence

This yields 12 agents per question (4 models × 3 personas), ~1,800 baseline answers.

**Round 1 (calibration).** Each agent independently answers and reports verbalized confidence in [0, 100]. No agent sees any other.

**Round 2 (reflection) + ablation.** For each model independently, we compute Shannon vote entropy across its three personas. If `H > 0.6` (≈ a 2-vs-1 disagreement), all three agents from that model are re-prompted. Critically, every triggered agent is re-prompted **once per ablation condition**, each revealing a different slice of the anonymized, leave-one-out peer information:
- **full** — peer vote distribution + mean peer confidence
- **votes_only** — peer vote distribution only
- **confidence_only** — mean peer confidence only
- **reasoning_only** — anonymized peer one-sentence reasonings only

Because the triggered subset is identical across conditions, this is a clean **within-subject ablation** of the peer signal. Reflection within a model is independent of other models, isolating the underlying-LLM variable for RQ2.

**Metrics.**
- *ECE* with M = 10 bins (Guo et al., 2017).
- *Brier score* — a strictly proper scoring rule that, unlike ECE, cannot be gamed by bin placement and jointly rewards calibration and sharpness.
- *Vote entropy* (Shannon, nats) over within-model answer distributions, used as the disagreement signal.
- *Bootstrap 95% CIs* (2,000 resamples) on all headline metrics.
- *Reward* r = 1{correct} − 0.5·|c − 1{correct}| is defined for reference but not used in this study.

## 3. Results

### 3.1 Baseline calibration (RQ1)

**Table 1 — Baseline (Round 1). Point estimate [bootstrap 95% CI].**

| Model        | N   | Accuracy            | Mean conf | Gap     | ECE                 | Brier               |
|--------------|----:|---------------------|----------:|--------:|---------------------|---------------------|
| llama-3.1-8b | 449 | 0.668 [0.624, 0.710]|     87.3% | +20.5pp | 0.207 [0.165, 0.252]| 0.267 [0.234, 0.302]|
| qwen-2.5-7b  | 450 | 0.729 [0.687, 0.771]|     88.0% | +15.1pp | 0.156 [0.115, 0.197]| 0.223 [0.191, 0.256]|
| ministral-8b | 450 | 0.755 [0.716, 0.793]|     87.6% | +12.1pp | 0.189 [0.152, 0.229]| 0.219 [0.188, 0.254]|
| gpt-4o       | 450 | 0.849 [0.816, 0.880]|     87.8% |  +2.9pp | **0.055** [0.030, 0.084]| **0.118** [0.097, 0.139]|

The decisive observation is the **Mean conf column: ~88% for every model**, from an 8B open model to a frontier model. Expressed confidence is essentially **model-invariant**, while accuracy ranges from 67% to 85%. Calibration therefore tracks accuracy: the overconfidence gap shrinks from +20.5pp (Llama) to +2.9pp (GPT-4o) purely because accuracy climbs toward a fixed confidence anchor. **Scale buys competence, not humility** — GPT-4o's strong calibration (4× lower ECE) is earned by being right more often, not by claiming less. Reliability diagrams per model are in `results/reliability_final.png`.

A second, robust finding: **within-model vote entropy predicts overconfidence.**

**Table 2 — Disagreement vs. overconfidence (Round 1), per question.**

| Model        | Pearson r (H vs gap) | Mean gap, H ≤ 0.6 | Mean gap, H > 0.6 | Reflection trigger rate |
|--------------|---------------------:|------------------:|------------------:|------------------------:|
| llama-3.1-8b |               +0.540 |            +0.039 |            +0.507 |                   35.3% |
| qwen-2.5-7b  |               +0.434 |            +0.061 |            +0.511 |                   20.0% |
| ministral-8b |               +0.575 |            −0.016 |            +0.393 |                   33.3% |
| gpt-4o       |               +0.639 |            −0.051 |            +0.375 |                   18.7% |

When a model's personas agree (H ≤ 0.6) it is well-calibrated or even slightly *under*confident (Ministral −0.016, GPT-4o −0.051); when they disagree the overconfidence gap explodes to +0.38–0.51. The signal **strengthens with model quality** — GPT-4o has the highest correlation (r = +0.639), i.e. self-disagreement is an especially clean overconfidence detector for the strong model. But the stronger models also disagree with themselves **less often** (trigger rate falls from 35% to 19%), so the signal fires more rarely exactly where it is most reliable.

### 3.2 Reflection effect (RQ2)

**Aggregate effect is small and model-dependent in sign.** Using the naive `full` reflection prompt over the entire population:

**Table 3 — Full-population naive reflection (condition = full).**

| Model        | ECE (R1) | ECE (post) | Δ ECE  | Acc (R1) | Acc (post) |
|--------------|---------:|-----------:|-------:|---------:|-----------:|
| llama-3.1-8b |    0.205 |      0.176 | −0.030 |    66.8% |      68.6% |
| qwen-2.5-7b  |    0.154 |      0.183 | +0.030 |    72.9% |      71.8% |
| ministral-8b |    0.184 |      0.171 | −0.013 |    75.6% |      75.3% |
| gpt-4o       |    0.051 |      0.062 | +0.011 |    84.9% |      84.2% |

The same reflection prompt **improves Llama and Ministral but degrades Qwen and GPT-4o** — directional disagreement across models, confirming that the reflection effect is model-dependent in *sign*, not just magnitude. (This refines the pilot, where naive reflection appeared uniformly harmful; at N = 150 across four models the effect is small and splits by model.) The real structure is in the ablation.

**The ablation localizes the mechanism.** Restricting to the agents on which reflection actually fired (identical subset across conditions):

**Table 4 — Reflection ablation (reflecting subset). Δ ECE vs. Round 1; w→r / r→w are wrong→right / right→wrong flips.**

| Model        | Condition        |  n  | R1 ECE | ECE   | Δ ECE  | Brier | acc   | w→r | r→w | net | Δconf   |
|--------------|------------------|----:|-------:|------:|-------:|------:|------:|----:|----:|----:|--------:|
| llama-3.1-8b | full             | 159 |  0.507 | 0.424 | −0.083 | 0.421 | 39.6% |  31 |  23 |  +8 | −4.6pp  |
| llama-3.1-8b | votes_only       | 159 |  0.507 | 0.420 | −0.088 | 0.408 | 39.6% |  27 |  19 |  +8 | −5.0pp  |
| llama-3.1-8b | confidence_only  | 159 |  0.507 | 0.579 | **+0.072** | 0.555 | 30.8% |  20 |  26 |  −6 | +0.9pp  |
| llama-3.1-8b | reasoning_only   | 159 |  0.507 | 0.454 | −0.053 | 0.449 | 42.8% |  48 |  35 | +13 | +2.8pp  |
| ministral-8b | full             | 150 |  0.476 | 0.447 | −0.029 | 0.443 | 42.0% |  26 |  27 |  −1 | +1.3pp  |
| ministral-8b | votes_only       | 150 |  0.476 | 0.425 | −0.051 | 0.420 | 44.0% |  28 |  26 |  +2 | +0.4pp  |
| ministral-8b | confidence_only  | 150 |  0.476 | 0.535 | **+0.059** | 0.513 | 40.0% |  14 |  18 |  −4 | +3.7pp  |
| ministral-8b | reasoning_only   | 150 |  0.476 | 0.435 | −0.041 | 0.432 | 44.7% |  39 |  36 |  +3 | +6.2pp  |
| gpt-4o       | full             |  84 |  0.408 | 0.436 | +0.028 | 0.414 | 36.9% |  13 |  16 |  −3 | +0.9pp  |
| gpt-4o       | votes_only       |  84 |  0.408 | 0.332 | −0.076 | 0.355 | 46.4% |  17 |  12 |  +5 | +0.1pp  |
| gpt-4o       | confidence_only  |  84 |  0.408 | 0.477 | **+0.068** | 0.443 | 34.5% |   1 |   6 |  −5 | +2.8pp  |
| gpt-4o       | reasoning_only   |  84 |  0.408 | 0.462 | +0.054 | 0.428 | 35.7% |  14 |  18 |  −4 | +2.8pp  |
| qwen-2.5-7b  | full             |  90 |  0.511 | 0.637 | +0.126 | 0.607 | 28.9% |  18 |  23 |  −5 | +6.2pp  |
| qwen-2.5-7b  | votes_only       |  90 |  0.511 | 0.644 | +0.133 | 0.622 | 26.7% |  18 |  25 |  −7 | +5.6pp  |
| qwen-2.5-7b  | confidence_only  |  90 |  0.511 | 0.696 | **+0.185** | 0.646 | 20.0% |   6 |  19 | −13 | +4.0pp  |
| qwen-2.5-7b  | reasoning_only   |  90 |  0.511 | 0.518 | +0.007 | 0.500 | 37.8% |  27 |  24 |  +3 | +4.1pp  |

Three structural results emerge:

1. **Peer confidence is universally corrosive.** `confidence_only` is the only condition that *worsens* ECE in **all four models** (+0.072, +0.059, +0.068, +0.185) and produces the **fewest** wrong→right corrections everywhere — strikingly, just **1 correction across 84 GPT-4o agents**. Showing a model only how confident its peers are causes it to inflate its own confidence and shed accuracy; it is pure social pressure with no informational content.
2. **Peer reasoning is the highest-information signal — it maximizes *change*.** `reasoning_only` produces the most wrong→right corrections in every model (48 / 39 / 27 / 14). For the weak models this churn is net-positive (knowledge gained > knowledge eroded), so reasoning recalibrates them. For GPT-4o the same churn is net-negative (it already knows the answers, so peer reasoning mostly injects contrarian doubt: 14 corrections but 18 erosions).
3. **Peer votes are an aggregate prior whose value is model-dependent.** `votes_only` is the *best* condition for GPT-4o (Δ ECE −0.076, accuracy 46.4%, net +5) and helps the two 8B models, but *harms* Qwen (+0.133) — because on the hard, triggered questions Qwen's own majority is frequently wrong, so following it amplifies error. Qwen is the most conformity-prone model: every social signal hurts it, and only reasoning is non-harmful.

**Synthesis.** Reflection has no single effect; it interacts with model identity. The optimal peer signal is *reasoning* for the weak models, *votes* for GPT-4o, and effectively *nothing* for Qwen — but *confidence* is harmful for all. A calibration-preserving reflection prompt should therefore **suppress raw peer-confidence disclosure** and condition the remaining signal on model competence. Grouped-bar comparison in `results/ablation.png`.

### 3.3 Mechanism — case studies

**Correction ("Where did the word 'pom' originally stand for?", correct = "Nothing, it is not an acronym").** All four models triggered (H = 0.64). Reflection converted several confidently-wrong knowledge personas to the correct answer once they saw peers split away from the "Port of Melbourne" confabulation — e.g. Llama's knowledge persona moved C→A and GPT-4o-style aggregation pulled Qwen's knowledge persona B→A. The case is genuinely mixed (some right→wrong flips too), illustrating that under `full` information reflection drives *high churn in both directions* rather than monotone correction. Full trace in `results/case_correction_final.md`.

**Herding ("What happens to class attendance if lecture recordings are provided?", correct = "Nothing in particular").** Agents that initially held the correct answer C were eroded toward the intuitive-but-wrong "attendance drops" after seeing disagreement. Most tellingly, Ministral's knowledge persona, revising toward the wrong answer, **confabulated specific fake citations** ("Mehta et al., 2013; Allen & Seaman, 2017; O'Flaherty & Phillips, 2015") to rationalize conforming. This reproduces the pilot's central failure mode at four-model scale: reflection fails when peer popularity is misaligned with truth, and the failure manifests as **post-hoc rationalization rather than honest revision**. Full trace in `results/case_herding_final.md`.

## 4. Discussion

The four-model study answers both research questions and sharpens the mechanism:

- **RQ1 (yes).** All four models are overconfident relative to accuracy, but the overconfidence is best understood as a **model-invariant confidence anchor (~88%)** met by variable accuracy. Within-model persona disagreement predicts overconfidence across all four (r = 0.43–0.64), and the signal is *strongest* for the best model — though it also fires least often there.
- **RQ2 (qualified, with a clean mechanism).** Naive disagreement-triggered reflection has only a small aggregate effect whose **sign depends on the model**. The ablation shows why: the reflection prompt bundles three signals with opposite effects. **Peer confidence always herds; peer reasoning corrects the ignorant but unsettles the competent; peer votes help when the ensemble is right and herd when it is wrong.** The herding failure is driven by *vote/confidence popularity*, not reasoning quality.

### 4.1 On the measurement of confidence

Throughout this study "confidence" is the model's own **verbalized self-report**: each agent emits an integer in [0, 100] alongside its answer. This is only one of several established elicitation methods, and because the choice shapes every calibration number we report, it warrants discussion.

The principal alternatives are: (i) **token-probability** confidence, reading the model's softmax mass on the answer token(s) (Guo et al., 2017; Kadavath et al., 2022); (ii) **self-evaluation**, e.g. the *P(True)* protocol in which the model is asked whether its own answer is correct and the probability of "True" is taken as the confidence (Kadavath et al., 2022); (iii) **sampling- or consistency-based** estimates, where the answer is sampled repeatedly and the frequency of the modal response serves as confidence (Wang et al., 2022; Manakul et al., 2023); and (iv) **semantic entropy**, which clusters sampled generations by meaning before computing entropy (Kuhn et al., 2023; Farquhar et al., 2024). Notably, our within-model **vote entropy across personas is itself a lightweight instance of (iii)** — the reflection trigger is a consistency-based uncertainty signal computed with no self-report at all, so the study already contains two distinct notions of confidence.

Verbalized confidence has well-documented pathologies. Lin, Hilton & Evans (2022b) first showed models can be trained to express uncertainty in words; Xiong et al. (2024) then demonstrated that, out of the box, LLMs are systematically **overconfident** when asked to report confidence and that the reports **cluster at round numbers** — a quantization we observe directly (our 7–8B agents answer almost exclusively in multiples of 5 and 10, which coarsens the reliability diagram and motivates our reliance on ECE and Brier rather than fine-grained confidence resolution).

Crucially, self-report is **not obviously inferior** to probability-based measures. Tian et al. (2023) find that for RLHF-tuned models, verbalized confidence can be *better* calibrated than the model's own token probabilities, because preference fine-tuning distorts the underlying logits. We therefore adopt verbalized confidence for three principled reasons: it is the **only measure available uniformly** across every model in our panel (several open models do not expose logprobs through the API); it is the quantity an agent **naturally produces and the only one the reflection step can revise**; and it keeps the cross-model comparison clean. Our data also speak to its interpretation: expressed confidence is strikingly similar across models of very different accuracy, while calibration quality tracks accuracy — consistent with the view that the verbalized-confidence *anchor* is largely a stylistic artifact of instruction tuning rather than a faithful readout of competence. A direct **logprob-vs-verbalized** comparison, on the subset of models that expose token probabilities, is the natural next measurement and would let us test that interpretation head-on.

## 5. Limitations

1. **Single dataset.** TruthfulQA is designed to elicit confident misconceptions. We do not yet know whether the herding pattern and the signal-specific ablation results persist on harder, lower-prior datasets such as GPQA. Stratifying by dataset difficulty is the most important next step.
2. **One frontier model.** GPT-4o is a single frontier data point; the "votes help, reasoning hurts the strong model" claim should be checked against at least one more frontier model (e.g. Claude or Gemini) before being generalized.
3. **Self-reported confidence.** As discussed in §4.1, verbalized self-report is one of several elicitation methods (token-probability, P(True), sampling-consistency, semantic entropy); we adopt it for cross-model uniformity and because it is the only signal the reflection step can revise, but a logprob-vs-verbalized comparison on the models that expose token probabilities remains outstanding.
4. **Within-model reflection only.** Reflection is conducted within each model's three personas, by design, to isolate the underlying-LLM variable. Cross-model reflection (agents from different LLMs informing each other) is a separate regime we have not tested.

## 6. Proposed next steps

1. **Calibration-preserving reflection (intervention).** Build a reflection prompt that suppresses raw peer-confidence disclosure and conditions on the model-appropriate signal (reasoning for weak models, votes for strong ones); test head-to-head against the naive prompt.
2. **Generalization across difficulty.** Add GPQA alongside TruthfulQA and test whether the confidence-herds / reasoning-corrects structure stratifies by dataset difficulty.
3. **Second frontier model + logprob confidence.** Add another frontier model and, on models exposing token probabilities, run the logprob-vs-verbalized calibration comparison from §4.1.
4. **Learned trigger (optional, agentic-RL).** Replace the fixed entropy threshold with a learned trigger policy updated on the reward function, addressing the course's agentic-RL component.

## 7. Code and reproducibility

All code is in `~/Desktop/calibration_prototype/`:

- `config.py` — experiment parameters: four models, three personas, `REFLECTION_CONDITIONS`, entropy threshold, bootstrap settings.
- `client.py` — OpenRouter API wrapper with structured-output parsing and retries.
- `calibration.py` — ECE, reliability bins, vote entropy, Brier score, paired bootstrap CIs, reward function.
- `experiment.py` — Round 1 + per-condition Round 2 pipeline, concurrent API calls, per-question checkpointing, and `add_models()` for incrementally merging a new model into an existing run.
- `analysis.py` — baseline summary (with CIs), reflection ablation table + plot, reliability diagram, disagreement scatter, case-study generators.
- `run.py` — entry point. `python run.py` (full run), `python run.py --add-models` (incrementally add models from config), `python run.py --analyze` (re-analyze existing results).

Raw per-agent results are saved to `results/run_final.json`.

## References

- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature, 630, 625–630.*
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *arXiv:1706.04599.*
- Jiang, B. (2026). DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles. *arXiv:2603.20975.*
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221.*
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023.*
- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *arXiv:2109.07958.*
- Lin, S., Hilton, J., & Evans, O. (2022b). Teaching Models to Express Their Uncertainty in Words. *Transactions on Machine Learning Research (TMLR); arXiv:2205.14334.*
- Manakul, P., Liusie, A., & Gales, M. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *EMNLP 2023.*
- Ozer, O., et al. (2025). MAR: Multi-Agent Reflexion Improves Reasoning Abilities in LLMs. *arXiv:2512.20845.*
- Tian, K., et al. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *EMNLP 2023.*
- Wang, X., et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *arXiv:2203.11171.*
- Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ICLR 2024.*
- Yang, R., et al. (2024). Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation. *ICLR 2024 Workshop on Reliable and Responsible Foundation Models.*
