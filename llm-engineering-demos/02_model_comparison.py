"""Demo 2 — Run the same prompt across several models and compare.

Evaluating and comparing Foundation Models for a given use case is a core
GenAI engineering task: quality, latency and cost all vary by model.
Edit the MODELS list to match the models available on your provider.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.getenv("BASE_URL", "https://api.groq.com/openai/v1"),
)

# Check your provider's console for currently available model names.
MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

PROMPT = (
    "A truck fleet manager asks: what are the top 3 benefits of "
    "predictive maintenance? Answer in 3 short bullet points."
)


def run(model: str) -> None:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.3,
    )
    elapsed = time.perf_counter() - start
    usage = response.usage
    print(f"\n{'=' * 60}")
    print(f"Model: {model}  |  {elapsed:.2f}s  |  {usage.total_tokens} tokens")
    print(f"{'=' * 60}")
    print(response.choices[0].message.content.strip())


if __name__ == "__main__":
    for model in MODELS:
        run(model)
