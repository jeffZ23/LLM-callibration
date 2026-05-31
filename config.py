"""
Experiment configuration for the disagreement-based reflection prototype.

Two underlying LLMs x three personas = six agents per question.
Round 1: independent answers + self-reported confidence (baseline calibration).
Round 2: triggered when same-model agents disagree (vote entropy > THRESHOLD);
each agent sees the anonymized peer answer distribution and revises.
"""

# OpenRouter model IDs. All have free/cheap tiers as of 2025-2026.
# Swap if needed: any OpenRouter model ID works.
# Third family (Mistral) added for the final study so RQ2's cross-model
# interaction claim rests on more than two architectures.
MODELS = {
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "ministral-8b": "mistralai/ministral-8b-2512",
    # Large frontier model (Phase 4 generalization): tests whether the small-model
    # overconfidence + herding findings persist at frontier scale / heavy RLHF.
    "gpt-4o": "openai/gpt-4o",
}

# Three personas (cut from the proposal's five for prototype-scale cost/time).
# Bring the other two back when scaling to the full study.
PERSONAS = {
    "analytical": (
        "You are an Analytical Reasoner. Approach every problem with rigorous "
        "step-by-step logical analysis. Decompose the problem, examine each "
        "piece, and only then commit to an answer."
    ),
    "devils_advocate": (
        "You are a Devil's Advocate. Actively challenge the obvious answer. "
        "Look for what most people would miss, common misconceptions, or "
        "edge cases where the intuitive answer fails."
    ),
    "knowledge": (
        "You are Knowledge-Focused. Rely on established domain knowledge and "
        "verifiable facts. If you are uncertain, say so via lower confidence "
        "rather than guessing confidently."
    ),
}

# Dataset: TruthfulQA mc1 (single correct option per question).
# Designed to elicit confidently-wrong answers via common misconceptions —
# the regime where disagreement-based reflection should help most.
DATASET_NAME = "truthfulqa/truthful_qa"
DATASET_CONFIG = "multiple_choice"
DATASET_SPLIT = "validation"
N_QUESTIONS = 150  # final study scale (was 50 for the pilot)

# Reflection ablation conditions. When disagreement triggers reflection, every
# triggered agent is re-prompted once PER condition, each revealing a different
# slice of peer information. This isolates which signal drives correction vs
# herding (RQ2 mechanism).
#   full            -> peer vote distribution + mean peer confidence
#   votes_only      -> peer vote distribution only
#   confidence_only -> mean peer confidence only (no vote breakdown)
#   reasoning_only  -> anonymized peer one-sentence reasonings only (no votes)
REFLECTION_CONDITIONS = ["full", "votes_only", "confidence_only", "reasoning_only"]

# Concurrency for API calls (ThreadPoolExecutor). OpenRouter tolerates modest
# parallelism on these models; lower this if you hit rate limits.
MAX_WORKERS = 8

# Bootstrap settings for confidence intervals on metrics.
BOOTSTRAP_N = 2000
BOOTSTRAP_ALPHA = 0.05  # -> 95% CI

# Reflection trigger: vote entropy threshold (in nats).
# With 3 personas, max entropy ~= ln(3) = 1.10. Threshold of 0.6 ~= "moderate
# disagreement" (e.g., 2 agents agree, 1 differs => H ~= 0.64).
ENTROPY_THRESHOLD = 0.6

# Reward function lambda from the proposal: r = 1{correct} - lambda * |c - 1{correct}|
LAMBDA = 0.5

# ECE binning
ECE_N_BINS = 10

# Reproducibility
RANDOM_SEED = 42
