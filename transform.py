import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from typing import Optional

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
NANO  = "nvidia/nemotron-3-nano-30b-a3b"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
TRANSFORM_MODEL = NANO  # Fast and reliable for restructuring tasks

# Route by domain — same logic as before
DOMAIN_MODEL_MAP = {
    "math":    ULTRA,
    "code":    NANO,
    "science": ULTRA,
    "other":   NANO,
}

client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)


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
            print(f"  Empty response, waiting {wait}s ({attempt+1}/5)...")
            time.sleep(wait)
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"  API error: {e}, waiting {wait}s ({attempt+1}/5)...")
            time.sleep(wait)
    return None


# ── Extract thinking trace from output ────────────────────────────────────────
def extract_thinking(output: str) -> str:
    """Pull content from inside <think> tags if present."""
    if "<think>" in output and "</think>" in output:
        start = output.index("<think>") + len("<think>")
        end   = output.index("</think>")
        return output[start:end].strip()
    return output.strip()


# ── Transform one example into Metis five-stage format ────────────────────────
def transform_to_metis(problem: str, thinking: str, model: str) -> Optional[dict]:
    system = "You are an expert reasoning analyst. You restructure existing reasoning traces into a precise five-stage format. Be faithful to the original reasoning — do not invent new content."

    prompt = f"""You are given a problem and an existing reasoning trace.
Restructure the reasoning trace into exactly five stages.

PROBLEM:
{problem}

EXISTING REASONING TRACE:
{thinking[:3000]}

Restructure into these five stages. Be faithful to what the original reasoning actually did:

STAGE 1 — INITIAL HYPOTHESIS:
The first committed answer or approach the reasoning attempted. One clear statement.

STAGE 2 — ASSUMPTION AUDIT:
List every assumption made in Stage 1. Flag each as VERIFIED, UNCERTAIN, or POTENTIALLY WRONG.

STAGE 3 — SELF CRITIQUE:
Where did the initial approach go wrong or need correction? What did it miss or overcomplicate?

STAGE 4 — REVISED REASONING:
The corrected reasoning chain, informed by the critique.

STAGE 5 — GROUNDED CONCLUSION:
The final answer with confidence score [0.0-1.0] and one sentence explaining how the critique improved the initial hypothesis.

Format your response exactly like this:
===STAGE1===
[content]
===STAGE2===
[content]
===STAGE3===
[content]
===STAGE4===
[content]
===STAGE5===
[content]"""

    result = call(model, prompt, system, temperature=0.4)
    if not result:
        return None

    # Flexible stage extraction — handles multiple delimiter styles
    import re
    stage_patterns = [
        r"===STAGE(\d)===\s*(.*?)(?====STAGE\d===|$)",
        r"STAGE\s*(\d)[:\-–]\s*(.*?)(?=STAGE\s*\d[:\-–]|$)",
        r"\*\*Stage\s*(\d)[:\-–][^*]*\*\*\s*(.*?)(?=\*\*Stage\s*\d|$)",
        r"#{1,3}\s*Stage\s*(\d)[:\-–][^\n]*\n(.*?)(?=#{1,3}\s*Stage\s*\d|$)",
    ]

    stages = {}
    for pattern in stage_patterns:
        matches = re.findall(pattern, result, re.DOTALL | re.IGNORECASE)
        if len(matches) >= 3:
            for num, content in matches:
                stages[f"STAGE{num}"] = content.strip()
            break

    # Last resort — split on any numbered stage marker
    if len(stages) < 3:
        parts = re.split(r'\n(?=(?:stage|step)\s*[1-5])', result, flags=re.IGNORECASE)
        if len(parts) >= 3:
            for i, part in enumerate(parts[:5], 1):
                stages[f"STAGE{i}"] = part.strip()

    if len(stages) < 3:
        print(f"  Incomplete stage extraction: {list(stages.keys())}")
        print(f"  Raw response preview: {result[:300]}")
        return None

    return {
        "problem":             problem,
        "original_thinking":   thinking[:3000],
        "initial_hypothesis":  stages.get("STAGE1", ""),
        "assumption_audit":    stages.get("STAGE2", ""),
        "self_critique":       stages.get("STAGE3", ""),
        "revised_reasoning":   stages.get("STAGE4", ""),
        "grounded_conclusion": stages.get("STAGE5", ""),
        "domain":              "",
    }


# ── Main transformation loop ───────────────────────────────────────────────────
def run_transformation(target: int = 50):
    os.makedirs("data/metis", exist_ok=True)
    output_path = "data/metis/metis_dataset.json"

    # Load existing progress
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing examples.")
    else:
        results = []

    already_done = len(results)

    # Stream the dataset
    ds = load_dataset(
        "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        "stage1",
        split="train",
        streaming=True,
    )

    processed = 0
    for example in ds:
        if len(results) >= target:
            break

        # Skip already processed examples
        if processed < already_done:
            processed += 1
            continue

        domain  = example.get("domain", "other")
        problem = example.get("input",  "")
        output  = example.get("output", "")

        if not problem or not output:
            processed += 1
            continue

        thinking = extract_thinking(output)
        model = TRANSFORM_MODEL

        print(f"\n[{len(results)+1}/{target}] domain={domain} model={model.split('/')[-1]}")
        print(f"  Problem: {problem[:80]}...")

        transformed = transform_to_metis(problem, thinking, model)

        if transformed:
            transformed["domain"] = domain
            results.append(transformed)
            print(f"  ✓ Transformed successfully")
        else:
            print(f"  ✗ Transformation failed")

        # Save progress after every example
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        processed += 1
        time.sleep(5)

    print(f"\n{'='*50}")
    print(f"Transformation complete.")
    print(f"Total examples: {len(results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_transformation(target=50)