# Reflection ablation (agents where disagreement triggered reflection)

Each triggered agent was re-prompted once per condition, each revealing a
different slice of peer information. Subset is identical across conditions.

- **full** = peer votes + mean peer confidence
- **votes_only** = peer vote distribution only
- **confidence_only** = mean peer confidence only
- **reasoning_only** = anonymized peer one-sentence reasonings only

| Model | Condition | n | R1 ECE | ECE | ΔECE | Brier | acc | w→r | r→w | net | Δconf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b | full | 159 | 0.507 | 0.424 | -0.083 | 0.421 | 39.6% | 31 | 23 | +8 | -4.6pp |
| llama-3.1-8b | votes_only | 159 | 0.507 | 0.420 | -0.088 | 0.408 | 39.6% | 27 | 19 | +8 | -5.0pp |
| llama-3.1-8b | confidence_only | 159 | 0.507 | 0.579 | +0.072 | 0.555 | 30.8% | 20 | 26 | -6 | +0.9pp |
| llama-3.1-8b | reasoning_only | 159 | 0.507 | 0.454 | -0.053 | 0.449 | 42.8% | 48 | 35 | +13 | +2.8pp |
| | | | | | | | | | | | |
| ministral-8b | full | 150 | 0.476 | 0.447 | -0.029 | 0.443 | 42.0% | 26 | 27 | -1 | +1.3pp |
| ministral-8b | votes_only | 150 | 0.476 | 0.425 | -0.051 | 0.420 | 44.0% | 28 | 26 | +2 | +0.4pp |
| ministral-8b | confidence_only | 150 | 0.476 | 0.535 | +0.059 | 0.513 | 40.0% | 14 | 18 | -4 | +3.7pp |
| ministral-8b | reasoning_only | 150 | 0.476 | 0.435 | -0.041 | 0.432 | 44.7% | 39 | 36 | +3 | +6.2pp |
| | | | | | | | | | | | |
| gpt-4o | full | 84 | 0.408 | 0.436 | +0.028 | 0.414 | 36.9% | 13 | 16 | -3 | +0.9pp |
| gpt-4o | votes_only | 84 | 0.408 | 0.332 | -0.076 | 0.355 | 46.4% | 17 | 12 | +5 | +0.1pp |
| gpt-4o | confidence_only | 84 | 0.408 | 0.477 | +0.068 | 0.443 | 34.5% | 1 | 6 | -5 | +2.8pp |
| gpt-4o | reasoning_only | 84 | 0.408 | 0.462 | +0.054 | 0.428 | 35.7% | 14 | 18 | -4 | +2.8pp |
| | | | | | | | | | | | |
| qwen-2.5-7b | full | 90 | 0.511 | 0.637 | +0.126 | 0.607 | 28.9% | 18 | 23 | -5 | +6.2pp |
| qwen-2.5-7b | votes_only | 90 | 0.511 | 0.644 | +0.133 | 0.622 | 26.7% | 18 | 25 | -7 | +5.6pp |
| qwen-2.5-7b | confidence_only | 90 | 0.511 | 0.696 | +0.185 | 0.646 | 20.0% | 6 | 19 | -13 | +4.0pp |
| qwen-2.5-7b | reasoning_only | 90 | 0.511 | 0.518 | +0.007 | 0.500 | 37.8% | 27 | 24 | +3 | +4.1pp |
| | | | | | | | | | | | |
