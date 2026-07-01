# Metis — Structured Reasoning Discipline

**Experimental.** This project explores whether a structured 5-stage reasoning pipeline (Initial Hypothesis → Assumption Audit → Self-Critique → Revised Reasoning → Grounded Conclusion) improves LLM answer accuracy compared to naive single-pass prompting.

## Pipeline

| Script | Purpose |
|--------|---------|
| `generate.py` | Creates synthetic 5-stage reasoning examples across 6 domains, quality-scored and filtered |
| `transform.py` | Converts external reasoning traces (AlibabaApsara/Superior-Reasoning-SFT) into Metis format |
| `evaluate.py` | Compares naive vs Metis answers by extracting `\boxed{}` output and comparing against ground truth |
| `inspect_dataset.py` | Quick inspection of the transformed dataset |
| `test_api.py` | Tests API connectivity and judge model latency |

## Models

All prompts use `nvidia/nemotron-3-nano-30b-a3b` (same model for both conditions).

## Results

Preliminary evaluation on math problems with extractable boxed answers shows **no significant accuracy difference** between naive and Metis prompting — both achieve ~50% on a 14-example subset. The structured pipeline introduces more opportunities for answer-format drift without a clear correctness benefit.

An earlier LLM-judge-based evaluation reported a 100% Metis win rate, but this was driven by verbosity/formatting bias rather than actual accuracy.

## Setup

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
cp .env.example .env   # add your NVIDIA_API_KEY
```
