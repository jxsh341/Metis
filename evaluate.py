import os
import json
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Model config ───────────────────────────────────────────────────────────────
# Both naive and Metis use the same model; difference is prompting strategy
MODEL  = "nvidia/nemotron-3-nano-30b-a3b"
JUDGE_MODEL  = "nvidia/nemotron-3-ultra-550b-a55b"  # kept for potential fallback

client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)


# ── Answer extraction helpers ──────────────────────────────────────────────────
def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \boxed{...} in the text."""
    if not text:
        return None
    # Find all \boxed{...} patterns, handle nested braces
    matches = list(re.finditer(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_ground_truth(ex: dict) -> Optional[str]:
    """Extract ground truth answer from dataset entry."""
    # Try grounded_conclusion first (has the final answer in boxed format)
    gc = ex.get("grounded_conclusion", "")
    ans = extract_boxed(gc)
    if ans:
        return ans
    # Fallback: try original_thinking
    ot = ex.get("original_thinking", "")
    ans = extract_boxed(ot)
    if ans:
        return ans
    return None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize answer for comparison: strip whitespace, lowercase, remove $."""
    if ans is None:
        return None
    ans = ans.strip()
    ans = ans.replace('$', '')
    
    # Remove LaTeX spacing commands and \ (backslash-space)
    ans = re.sub(r'\\(?:displaystyle|;|,|:|!|quad|qquad)\s*', '', ans)
    ans = ans.replace('\\ ', ' ')
    
    # Normalize \dfrac, \tfrac -> \frac first
    ans = re.sub(r'\\(?:dfrac|tfrac)', '\\frac', ans)
    
    # Normalize \frac{a}{b} -> a/b
    def frac_repl(m):
        num = m.group(1).strip()
        den = m.group(2).strip()
        if re.match(r'^[^+\-*/(){}[\]|,;]+$', num) and re.match(r'^[^+\-*/(){}[\]|,;]+$', den):
            return f"{num}/{den}"
        return f"({num})/({den})"
    ans = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', frac_repl, ans)
    while re.search(r'\\frac\{[^{}]*\{[^{}]*\}[^{]*\}\{[^{}]*\}', ans):
        ans = re.sub(r'\\frac\{([^{}]*\{[^{}]*\}[^{}]*)\}\{([^{}]+)\}', frac_repl, ans)
        ans = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]*\{[^{}]*\}[^{}]*)\}', frac_repl, ans)
    
    # Normalize \binom{n}{k} -> C(n,k)
    ans = re.sub(r'\\binom\{([^{}]+)\}\{([^{}]+)\}', r'C(\1,\2)', ans)
    
    # Normalize \mid -> | (divides)
    ans = ans.replace('\\mid', '|')
    ans = ans.replace('\\text{divides}', '|')
    ans = ans.replace('divides', '|')
    
    # Remove \left, \right and \bigl, \bigr, etc.
    ans = re.sub(r'\\(?:left|right|big[lr]?|Big[lr]?|bigm|Bigm|biggm|Biggm)', '', ans)
    
    # Clean up backslashes left from \ (backslash-space) after spacing removal
    ans = re.sub(r'\\(\s|$)', '', ans)
    
    # Normalize \text{...} -> ...
    ans = re.sub(r'\\text\{([^{}]+)\}', r'\1', ans)
    
    # Normalize subscripts: _{0} -> _0, {0} -> 0
    ans = re.sub(r'_\{([^}]+)\}', r'_\1', ans)
    ans = re.sub(r'\{([0-9]+)\}', r'\1', ans)
    
    # Normalize parentheses and tuples: (x,y,z) = (a,b,c) -> a,b,c
    # Extract tuple content
    ans = re.sub(r'\([^)]*\)\s*=\s*', '', ans)
    
    # Remove all whitespace
    ans = re.sub(r'\s+', '', ans)
    
    # Remove trailing periods
    ans = ans.rstrip('.')
    
    return ans.lower()


def safe_print(*args, **kwargs):
    """Print that handles encoding issues on Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace problematic chars
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                arg = arg.encode('ascii', 'replace').decode('ascii')
            new_args.append(arg)
        print(*new_args, **kwargs)


def answers_match(pred: Optional[str], truth: Optional[str]) -> bool:
    """Check if predicted answer matches ground truth."""
    if pred is None or truth is None:
        return False
    return normalize_answer(pred) == normalize_answer(truth)


# ── Data structure ─────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    domain:             str
    subfield:           str
    difficulty:         str
    problem:            str
    naive_response:     str
    metis_response:     str
    ground_truth:       str
    naive_answer:       Optional[str] = None
    metis_answer:       Optional[str] = None
    naive_correct:      bool = False
    metis_correct:      bool = False


# ── API call helper ────────────────────────────────────────────────────────────
def call(model: str, prompt: str, system: str, temperature: float = 0.7) -> Optional[str]:
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2000,
                timeout=120,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            wait = 5 * (2 ** attempt)
            print(f"  Empty response, waiting {wait}s ({attempt+1}/3)...")
            time.sleep(wait)
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"  API error: {e}, waiting {wait}s ({attempt+1}/3)...")
            time.sleep(wait)
    return None


# ── Condition 1: Naive single-pass ─────────────────────────────────────────────
def naive_response(problem: str) -> Optional[str]:
    system = "You are a helpful assistant. Answer the problem as best you can. Output your final answer within \\boxed{}."
    prompt = f"Problem:\n{problem}\n\nProvide your best answer. Put the final answer in \\boxed{{}}."
    return call(MODEL, prompt, system, temperature=0.7)


# ── Condition 2: Metis five-stage reasoning ────────────────────────────────────
def metis_response(problem: str) -> Optional[str]:
    """Run the full five-stage Metis reasoning pattern."""

    # Stage 1 — Initial Hypothesis
    s1 = call(MODEL,
        f"Problem:\n{problem}\n\nGenerate your INITIAL HYPOTHESIS. Commit to a first answer. Be direct. Do not hedge.",
        "You are a reasoning model in Stage 1 of a structured reasoning process.",
        temperature=0.8)
    if not s1: return None
    time.sleep(2)

    # Stage 2 — Assumption Audit
    s2 = call(MODEL,
        f"Problem:\n{problem}\n\nYour Initial Hypothesis:\n{s1}\n\nPerform an ASSUMPTION AUDIT. List every assumption. Flag each as VERIFIED, UNCERTAIN, or POTENTIALLY WRONG. Be ruthless.",
        "You are a reasoning model in Stage 2 of a structured reasoning process.",
        temperature=0.7)
    if not s2: return None
    time.sleep(2)

    # Stage 3 — Self-Critique
    s3 = call(MODEL,
        f"Problem:\n{problem}\n\nInitial Hypothesis:\n{s1}\n\nAssumption Audit:\n{s2}\n\nPerform a SELF-CRITIQUE. Challenge your hypothesis using the flagged assumptions. What did you get wrong? Be specific.",
        "You are a reasoning model in Stage 3 of a structured reasoning process.",
        temperature=0.7)
    if not s3: return None
    time.sleep(2)

    # Stage 4 — Revised Reasoning
    s4 = call(MODEL,
        f"Problem:\n{problem}\n\nInitial Hypothesis:\n{s1}\n\nAssumption Audit:\n{s2}\n\nSelf-Critique:\n{s3}\n\nGenerate your GROUNDED CONCLUSION. Complete revised reasoning chain. End with CONFIDENCE: [0.0-1.0] and one sentence on how self-critique improved your answer.",
        "You are a reasoning model in Stage 4 of a structured reasoning process.",
        temperature=0.6)
    if not s4: return None

    # Stage 5 — Final Answer with boxed format
    s5 = call(MODEL,
        f"Problem:\n{problem}\n\nFull reasoning chain:\n=== STAGE 1: INITIAL HYPOTHESIS ===\n{s1}\n\n=== STAGE 2: ASSUMPTION AUDIT ===\n{s2}\n\n=== STAGE 3: SELF-CRITIQUE ===\n{s3}\n\n=== STAGE 4: GROUNDED CONCLUSION ===\n{s4}\n\nNow provide your FINAL ANSWER. Put the final answer in \\boxed{{}}.",
        "You are a reasoning model producing the final answer after structured reasoning.",
        temperature=0.3)
    if not s5: return None

    # Return full chain as structured output
    return f"""=== STAGE 1: INITIAL HYPOTHESIS ===
{s1}

=== STAGE 2: ASSUMPTION AUDIT ===
{s2}

=== STAGE 3: SELF-CRITIQUE ===
{s3}

=== STAGE 4: GROUNDED CONCLUSION ===
{s4}

=== STAGE 5: FINAL ANSWER ===
{s5}"""


# ── Programmatic answer comparison ─────────────────────────────────────────────
def evaluate_answers(problem: str, naive: str, metis: str, ground_truth: str) -> dict:
    """Extract boxed answers and compare against ground truth."""
    naive_ans = extract_boxed(naive)
    metis_ans = extract_boxed(metis)
    
    naive_correct = answers_match(naive_ans, ground_truth)
    metis_correct = answers_match(metis_ans, ground_truth)
    
    return {
        "naive_answer": naive_ans,
        "metis_answer": metis_ans,
        "ground_truth": ground_truth,
        "naive_correct": naive_correct,
        "metis_correct": metis_correct,
    }


# ── Visualization ──────────────────────────────────────────────────────────────
def visualize_results(results: list[EvalResult]):
    os.makedirs("data/eval", exist_ok=True)

    domains      = [r.domain      for r in results]
    difficulties = [r.difficulty for r in results]
    naive_correct = [1 if r.naive_correct else 0 for r in results]
    metis_correct = [1 if r.metis_correct else 0 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Project Metis — Reasoning Discipline Empirical Evaluation",
                 fontsize=14, fontweight="bold", y=1.02)

    # Plot 1 — Accuracy comparison per example
    ax = axes[0]
    x = np.arange(len(results))
    ax.bar(x - 0.2, naive_correct, 0.4, label="Naive",  color="#ef4444", alpha=0.8)
    ax.bar(x + 0.2, metis_correct, 0.4, label="Metis",  color="#3b82f6", alpha=0.8)
    ax.set_title("Accuracy per Example: Naive vs Metis")
    ax.set_xlabel("Example")
    ax.set_ylabel("Correct (0/1)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=np.mean(naive_correct), color="#ef4444", linestyle="--", alpha=0.5, label="Naive mean")
    ax.axhline(y=np.mean(metis_correct), color="#3b82f6", linestyle="--", alpha=0.5, label="Metis mean")

    # Plot 2 — Accuracy delta by difficulty
    ax = axes[1]
    diff_order  = ["mid", "advanced", "expert"]
    diff_deltas = []
    for d in diff_order:
        idxs = [i for i, r in enumerate(results) if r.difficulty == d]
        if idxs:
            delta = np.mean([metis_correct[i] - naive_correct[i] for i in idxs])
            diff_deltas.append(delta)
        else:
            diff_deltas.append(0)
    colors = ["#22c55e" if d > 0 else "#ef4444" for d in diff_deltas]
    ax.bar(diff_order, diff_deltas, color=colors, alpha=0.8)
    ax.set_title("Metis Accuracy Delta by Difficulty")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Accuracy Delta (Metis - Naive)")
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Plot 3 — Accuracy delta by domain
    ax = axes[2]
    unique_domains = list(set(domains))
    domain_deltas  = []
    for dom in unique_domains:
        idxs = [i for i, r in enumerate(results) if r.domain == dom]
        if idxs:
            delta = np.mean([metis_correct[i] - naive_correct[i] for i in idxs])
            domain_deltas.append(delta)
        else:
            domain_deltas.append(0)
    colors = ["#22c55e" if d > 0 else "#ef4444" for d in domain_deltas]
    ax.barh(unique_domains, domain_deltas, color=colors, alpha=0.8)
    ax.set_title("Metis Accuracy Delta by Domain")
    ax.set_xlabel("Accuracy Delta (Metis - Naive)")
    ax.axvline(x=0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("data/eval/metis_evaluation.png", dpi=150, bbox_inches="tight")
    safe_print("\n  Saved: data/eval/metis_evaluation.png")

    # Summary stats
    safe_print(f"\n{'='*50}")
    safe_print(f"METIS EVALUATION SUMMARY")
    safe_print(f"{'='*50}")
    safe_print(f"Examples evaluated:     {len(results)}")
    safe_print(f"Naive accuracy:         {np.mean(naive_correct):.1%}")
    safe_print(f"Metis accuracy:         {np.mean(metis_correct):.1%}")
    safe_print(f"Mean accuracy delta:    {np.mean(np.array(metis_correct) - np.array(naive_correct)):+.1%}")
    wins = sum(1 for n, m in zip(naive_correct, metis_correct) if m > n)
    ties = sum(1 for n, m in zip(naive_correct, metis_correct) if m == n)
    losses = sum(1 for n, m in zip(naive_correct, metis_correct) if m < n)
    safe_print(f"Metis wins/ties/losses: {wins}/{ties}/{losses}")


# ── Main eval loop ─────────────────────────────────────────────────────────────
def run_evaluation(n_examples: int = 20):
    filtered_path = "data/metis/metis_dataset.json"

    if not os.path.exists(filtered_path):
        safe_print("No filtered dataset found. Run generate.py first.")
        return

    with open(filtered_path) as f:
        dataset = json.load(f)

    if len(dataset) < n_examples:
        print(f"Only {len(dataset)} filtered examples available. Using all of them.")
        n_examples = len(dataset)

    # Sample evenly across domains and difficulties
    selected = dataset[:n_examples]
    results  = []

    for i, ex in enumerate(selected):
        domain = ex.get("domain") or "unknown"
        subfield = ex.get("subfield") or "unknown"
        difficulty = ex.get("difficulty") or "mid"
        problem = ex.get("problem") or ""

# Extract ground truth from dataset
        ground_truth = extract_ground_truth(ex)
        if not ground_truth:
            safe_print(f"  Skipping \u2014 no ground truth found")
            continue

        safe_print(f"\n[{i+1}/{n_examples}] {domain} | {subfield} | {difficulty}")
        safe_print(f"  Ground truth: {ground_truth}")

        naive = naive_response(problem)
        time.sleep(3)
        metis = metis_response(problem)
        time.sleep(3)

        if not naive or not metis:
            safe_print("  Skipping \u2014 incomplete responses")
            continue

        eval_result = evaluate_answers(problem, naive, metis, ground_truth)

        naive_answer = eval_result["naive_answer"]
        metis_answer = eval_result["metis_answer"]
        naive_correct = eval_result["naive_correct"]
        metis_correct = eval_result["metis_correct"]

        result = EvalResult(
            domain=domain, subfield=subfield,
            difficulty=difficulty, problem=problem,
            naive_response=naive, metis_response=metis,
            ground_truth=ground_truth,
            naive_answer=naive_answer, metis_answer=metis_answer,
            naive_correct=naive_correct, metis_correct=metis_correct,
        )
        results.append(result)

        safe_print(f"  Naive answer: {naive_answer} | Metis answer: {metis_answer}")
        safe_print(f"  Naive correct: {naive_correct} | Metis correct: {metis_correct}")

        # Save progress
        os.makedirs("data/eval", exist_ok=True)
        with open("data/eval/eval_results.json", "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)

    visualize_results(results)


if __name__ == "__main__":
    run_evaluation(n_examples=100)