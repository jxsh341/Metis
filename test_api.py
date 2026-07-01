import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)

candidate_judges = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-4-340b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.3-70b-instruct"
]

prompt = """You are evaluating two responses to the same problem.
PROBLEM:
What is 2+2?
RESPONSE A:
4
RESPONSE B:
Four
Score each response from 0.0 to 1.0.
Respond with only this JSON:
{
  "naive_correctness": 1.0,
  "naive_reasoning_depth": 1.0,
  "naive_self_awareness": 1.0,
  "metis_correctness": 1.0,
  "metis_reasoning_depth": 1.0,
  "metis_self_awareness": 1.0,
  "reasoning": "Both are correct."
}"""

for model in candidate_judges:
    print(f"Testing model as judge: {model}")
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an impartial expert evaluator. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200,
            timeout=20
        )
        elapsed = time.time() - start
        print(f"  Success in {elapsed:.2f}s!")
        print(f"  Output: {response.choices[0].message.content.strip()}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  Failed after {elapsed:.2f}s! Error: {e}")
    print("-" * 50)
