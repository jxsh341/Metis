import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

# ── Model routing ──────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"

NANO  = "nvidia/nemotron-3-nano-30b-a3b"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"

DOMAIN_MODEL_MAP = {
    "Physics":                        NANO,
    "Code Debugging":                 NANO,
    "Mathematics":                    ULTRA,
    "Logical Reasoning":              ULTRA,
    "Scientific Hypothesis Formation": ULTRA,
    "Causal Reasoning":               ULTRA,
}

SUBFIELDS = {
    "Physics":          ["Quantum Mechanics", "Thermodynamics", "Relativity",
                         "Electrodynamics", "Statistical Mechanics", "Condensed Matter"],
    "Mathematics":      ["Number Theory", "Topology", "Combinatorics",
                         "Real Analysis", "Abstract Algebra", "Differential Geometry"],
    "Logical Reasoning":["Constraint Satisfaction", "Modal Logic", "Counterfactual Reasoning",
                         "Paradox Resolution", "Formal Argumentation", "Epistemic Logic"],
    "Code Debugging":   ["Memory Management", "Concurrency Bugs", "Algorithmic Complexity",
                         "Type System Errors", "Race Conditions", "Undefined Behavior"],
    "Scientific Hypothesis Formation": ["Experimental Design", "Falsifiability Analysis",
                         "Confound Identification", "Replication Failure Analysis",
                         "Measurement Error", "Causal Inference"],
    "Causal Reasoning": ["Counterfactual Analysis", "Simpson's Paradox",
                         "Confounding Variables", "Feedback Loops",
                         "Intervention vs Observation", "Temporal Causality"],
}

DIFFICULTIES = ["mid", "advanced", "expert"]

client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)


# ── Data structure ─────────────────────────────────────────────────────────────
@dataclass
class MetisExample:
    domain:               str
    subfield:             str
    difficulty:           str
    problem:              str
    initial_hypothesis:   str
    assumption_audit:     str
    self_critique:        str
    grounded_conclusion:  str
    critique_depth_score: Optional[float] = None
    improvement_delta_score: Optional[float] = None


# ── Core generation helpers ────────────────────────────────────────────────────
def call(model: str, prompt: str, system: str, temperature: float = 0.7) -> str:
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=1500,
                timeout=120,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            print(f"  Empty response, retrying ({attempt+1}/3)...")
            time.sleep(5)
        except Exception as e:
            print(f"  API error: {e}, retrying ({attempt+1}/3)...")
            time.sleep(5)
    return None


def generate_problem(domain: str, subfield: str, difficulty: str, model: str) -> str:
    system = "You are an expert problem designer. Output only the problem statement, nothing else."
    difficulty_guidance = {
        "mid":      "Requires multi-step reasoning with at least one non-obvious assumption.",
        "advanced": "The initial hypothesis will almost certainly be wrong in at least one dimension.",
        "expert":   "Embed a subtle trap in the framing of the problem itself.",
    }
    prompt = f"""Design a {difficulty}-level problem in {domain} — {subfield}.

Difficulty guidance: {difficulty_guidance[difficulty]}

The problem should genuinely stress-test structured reasoning.
Output only the problem statement."""
    return call(model, prompt, system, temperature=0.9)


def generate_initial_hypothesis(problem: str, model: str) -> str:
    system = "You are a reasoning model in Stage 1 of a structured reasoning process."
    prompt = f"""Problem:
{problem}

Generate your INITIAL HYPOTHESIS. 
Rules:
- Commit to a first answer. Be direct.
- Do NOT hedge or say 'it depends'.
- This is deliberately rough — it is the starting point, not the final answer.
- For expert-level problems, your hypothesis may contain flawed assumptions. That is expected."""
    return call(model, prompt, system, temperature=0.8)


def generate_assumption_audit(problem: str, hypothesis: str, model: str) -> str:
    system = "You are a reasoning model in Stage 2 of a structured reasoning process."
    prompt = f"""Problem:
{problem}

Your Initial Hypothesis:
{hypothesis}

Perform an ASSUMPTION AUDIT.
- List every assumption you made in your initial hypothesis.
- Flag each one as: VERIFIED, UNCERTAIN, or POTENTIALLY WRONG.
- Be ruthless. Surface assumptions you made implicitly, not just explicitly."""
    return call(model, prompt, system, temperature=0.7)


def generate_self_critique(problem: str, hypothesis: str, audit: str, model: str) -> str:
    system = "You are a reasoning model in Stage 3 of a structured reasoning process."
    prompt = f"""Problem:
{problem}

Your Initial Hypothesis:
{hypothesis}

Your Assumption Audit:
{audit}

Perform a SELF-CRITIQUE.
- Challenge your initial hypothesis using the flagged assumptions.
- What did you get wrong? What did you miss?
- For expert-level problems: identify the trap hidden in the problem framing.
- Be specific. Vague critique is worthless."""
    return call(model, prompt, system, temperature=0.7)


def generate_grounded_conclusion(problem: str, hypothesis: str,
                                  audit: str, critique: str, model: str) -> str:
    system = "You are a reasoning model in Stage 4 of a structured reasoning process."
    prompt = f"""Problem:
{problem}

Initial Hypothesis:
{hypothesis}

Assumption Audit:
{audit}

Self-Critique:
{critique}

Generate your GROUNDED CONCLUSION.
- Produce a complete revised reasoning chain informed by all prior stages.
- End with: CONFIDENCE: [0.0-1.0] and one sentence explaining how your self-critique improved your initial hypothesis."""
    return call(model, prompt, system, temperature=0.6)


# ── Quality scoring ────────────────────────────────────────────────────────────
def score_example(example: MetisExample, model: str) -> MetisExample:
    system = "You are a quality evaluator. Respond only with valid JSON."
    prompt = f"""Evaluate this reasoning chain:

PROBLEM: {example.problem}
INITIAL HYPOTHESIS: {example.initial_hypothesis}
ASSUMPTION AUDIT: {example.assumption_audit}
SELF CRITIQUE: {example.self_critique}
GROUNDED CONCLUSION: {example.grounded_conclusion}

Score on two dimensions from 0.0 to 1.0:

1. critique_depth_score: Did the assumption audit catch REAL flaws, or was it superficial?
   - 0.0-0.3: Superficial, obvious assumptions only
   - 0.4-0.6: Caught some real flaws but missed important ones
   - 0.7-1.0: Deep, caught non-obvious assumptions and implicit flaws

2. improvement_delta_score: Is the grounded conclusion MEANINGFULLY better than the initial hypothesis?
   - 0.0-0.3: Little to no improvement
   - 0.4-0.6: Moderate improvement
   - 0.7-1.0: Substantially better — the reasoning discipline clearly added value

Respond with only this JSON:
{{"critique_depth_score": 0.0, "improvement_delta_score": 0.0}}"""

    try:
        result = call(ULTRA, prompt, system, temperature=0.1)
        if result:
            clean = result.strip().replace("```json", "").replace("```", "").strip()
            scores = json.loads(clean)
            example.critique_depth_score    = scores["critique_depth_score"]
            example.improvement_delta_score = scores["improvement_delta_score"]
    except Exception as e:
        print(f"  Scoring failed: {e}")
        example.critique_depth_score    = 0.0
        example.improvement_delta_score = 0.0
    return example


# ── Main generation loop ───────────────────────────────────────────────────────
def generate_example(domain: str, subfield: str, difficulty: str) -> Optional[MetisExample]:
    model = DOMAIN_MODEL_MAP[domain]
    print(f"  Generating: {domain} | {subfield} | {difficulty} | model: {model.split('/')[-1]}")

    try:
        problem    = generate_problem(domain, subfield, difficulty, model)
        hypothesis = generate_initial_hypothesis(problem, model)
        audit      = generate_assumption_audit(problem, hypothesis, model)
        critique   = generate_self_critique(problem, hypothesis, audit, model)
        conclusion = generate_grounded_conclusion(problem, hypothesis, audit, critique, model)

        example = MetisExample(
            domain=domain, subfield=subfield, difficulty=difficulty,
            problem=problem, initial_hypothesis=hypothesis,
            assumption_audit=audit, self_critique=critique,
            grounded_conclusion=conclusion,
        )

        example = score_example(example, model)
        return example

    except Exception as e:
        print(f"  Failed: {e}")
        return None


def run_generation(target: int = 108, quality_threshold: float = 0.7):
    """Generate one example per seed combination, filter by quality."""
    os.makedirs("data/raw",      exist_ok=True)
    os.makedirs("data/filtered", exist_ok=True)

    raw, filtered = [], []

    for domain, subfields in SUBFIELDS.items():
        for subfield in subfields:
            for difficulty in DIFFICULTIES:
                print(f"\n[{len(raw)+1}/{target}]")
                example = generate_example(domain, subfield, difficulty)

                if example:
                    raw.append(example.__dict__)

                    c_score = example.critique_depth_score or 0.0
                    d_score = example.improvement_delta_score or 0.0

                    passes = (c_score >= quality_threshold and d_score >= quality_threshold)

                    if passes:
                        filtered.append(example.__dict__)
                        print(f"  ✓ PASSED quality gate (critique={c_score:.2f}, delta={d_score:.2f})")
                    else:
                        print(f"  ✗ FAILED quality gate (critique={c_score:.2f}, delta={d_score:.2f})")

                # Save progress after every example
                with open("data/raw/metis_raw.json", "w") as f:
                    json.dump(raw, f, indent=2)
                with open("data/filtered/metis_filtered.json", "w") as f:
                    json.dump(filtered, f, indent=2)

                time.sleep(3)  # respect free tier rate limits

    print(f"\n{'='*50}")
    print(f"Generation complete.")
    print(f"Raw examples:      {len(raw)}")
    print(f"Filtered examples: {len(filtered)}")
    print(f"Pass rate:         {len(filtered)/max(len(raw),1)*100:.1f}%")


if __name__ == "__main__":
    run_generation(target=108, quality_threshold=0.7)