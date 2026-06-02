# Persona-count ablation: does the persona ensemble matter?

Everything fixed except the persona set: same 4 models, 150 TruthfulQA questions, temperature, and four reflection conditions. **1-voice** = three identical neutral prompts (disagreement from sampling only); **3-persona** = the main study; **5-persona** = main three + intuitive + probabilistic.

Trigger: 1-voice and 3-persona use H>0.6 (their minimal 2-1 split is H=0.637). With 5 agents the minimal 4-1 split is only H=0.500, so the 5-persona set uses H>0.4 — i.e. *any* disagreement triggers, matching the 3-persona design.

ΔECE(full) is the reflection effect on the triggered subset under the `full` peer signal (negative = reflection improved calibration).

| Model | Persona set | #agents/q | Baseline ECE | Acc | Mean conf | Trigger rate | n triggered | ΔECE(full) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b | 1-voice (no persona) | 3 | 0.279 | 60.7% | 87.8% | 21.3% | 96 | -0.032 |
| llama-3.1-8b | 3-persona (main) | 3 | 0.205 | 66.8% | 87.3% | 35.3% | 159 | -0.083 |
| llama-3.1-8b | 5-persona | 5 | 0.231 | 63.7% | 86.2% | 46.0% | 345 | -0.058 |
| | | | | | | | | |
| qwen-2.5-7b | 1-voice (no persona) | 3 | 0.206 | 66.0% | 85.1% | 13.3% | 60 | +0.073 |
| qwen-2.5-7b | 3-persona (main) | 3 | 0.154 | 72.9% | 88.0% | 20.0% | 90 | +0.126 |
| qwen-2.5-7b | 5-persona | 5 | 0.174 | 70.7% | 86.5% | 30.7% | 230 | +0.119 |
| | | | | | | | | |
| ministral-8b | 1-voice (no persona) | 3 | 0.245 | 69.8% | 94.0% | 8.0% | 36 | +0.077 |
| ministral-8b | 3-persona (main) | 3 | 0.184 | 75.6% | 87.6% | 33.3% | 150 | -0.029 |
| ministral-8b | 5-persona | 5 | 0.190 | 72.9% | 88.9% | 42.7% | 320 | +0.014 |
| | | | | | | | | |
| gpt-4o | 1-voice (no persona) | 3 | 0.135 | 79.3% | 92.4% | 4.0% | 18 | -0.192 |
| gpt-4o | 3-persona (main) | 3 | 0.051 | 84.9% | 87.8% | 18.7% | 84 | +0.028 |
| gpt-4o | 5-persona | 5 | 0.064 | 82.9% | 88.3% | 22.7% | 170 | -0.049 |
| | | | | | | | | |
