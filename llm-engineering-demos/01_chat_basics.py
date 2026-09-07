"""Demo 1 — Connect to an LLM API and stream a response.

Uses an OpenAI-compatible client: works with Groq, Mistral, OpenAI,
or any local server (Ollama, vLLM) by changing BASE_URL / MODEL_NAME.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.getenv("BASE_URL", "https://api.groq.com/openai/v1"),
)
MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")


def chat(prompt: str, system: str = "You are a concise, helpful assistant.") -> str:
    """Send a prompt and stream the answer token by token."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        stream=True,
    )
    full_answer = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_answer.append(delta)
    print()
    return "".join(full_answer)


if __name__ == "__main__":
    print(f"Model: {MODEL}\n")
    chat("Explain in 3 sentences what Retrieval-Augmented Generation is.")
