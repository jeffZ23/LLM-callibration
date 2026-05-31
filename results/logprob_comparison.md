# Confidence elicitation: verbalized vs. logprob (GPT-4o)

N = 450 paired (question, persona) measurements. Verbalized = self-reported 0-100; logprob = softmax mass on the answer letter (first-token logprobs). ECE shown with bootstrap 95% CI.

| Confidence source | Mean conf | Accuracy | ECE | Brier |
|---|---:|---:|---:|---:|
| Verbalized (self-report) | 87.8% | 84.9% | 0.055 [0.030, 0.084] | 0.118 |
| Logprob (token prob) | 94.4% | 83.6% | 0.115 [0.087, 0.144] | 0.123 |

- **Answer agreement** (logprob argmax == verbalized answer): 93.3%
- **Correlation** between stated confidence and internal probability of the *stated* answer: r = +0.265
- **Mean internal prob on the stated answer**: 91.3% (vs 87.8% stated)
