"""Demo 3 — Minimal RAG pipeline: chunk, vectorize, retrieve, generate.

TF-IDF and cosine similarity are implemented from scratch (standard library
only) to make the retrieval mechanics explicit — and to keep the demo free of
heavy compiled dependencies.

The pipeline shape is the same as a production RAG system (like my project
MIA: Azure OpenAI + CosmosDB vector search):
  1. Split documents into chunks
  2. Turn chunks into vectors (here: TF-IDF; in prod: neural embeddings)
  3. Vectorize the user question and retrieve the top-k similar chunks
  4. Build a grounded prompt and let the LLM answer FROM the documents
"""

import math
import os
import re
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.getenv("BASE_URL", "https://api.groq.com/openai/v1"),
)
MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# --- 1. A tiny internal knowledge base (imagine maintenance manuals) ------
DOCUMENTS = [
    "Engine oil on the FH16 truck must be replaced every 60,000 km. "
    "Use grade VDS-5 oil only. Failure to do so voids the warranty.",
    "Brake pads should be inspected every 20,000 km. Replace them when "
    "thickness is below 5 mm. Always replace pads on both wheels of an axle.",
    "The telematics unit reports fault code P0420 when the catalytic "
    "converter efficiency drops. First check the oxygen sensors before "
    "replacing the converter.",
    "Battery health checks are mandatory before winter. A resting voltage "
    "below 12.4 V indicates the battery should be recharged or replaced.",
]


def chunk(documents: list[str]) -> list[str]:
    """Real systems split large docs into overlapping chunks.
    Our docs are already small, so each doc is one chunk."""
    return documents


def tokenize(text: str) -> list[str]:
    """Lowercase and keep alphanumeric words only."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Term Frequency x Inverse Document Frequency, as a sparse dict.

    TF   = how often a word appears in THIS text (relative)
    IDF  = how rare the word is across ALL texts -> rare words weigh more,
           so 'the' contributes almost nothing while 'brake' is decisive.
    """
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {
        word: (count / total) * idf.get(word, 0.0)
        for word, count in counts.items()
    }


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity: the angle between two vectors, in [0, 1] here.
    1.0 = same direction (same topic), 0.0 = no shared terms."""
    shared = set(a) & set(b)
    dot = sum(a[word] * b[word] for word in shared)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(question: str, chunks: list[str], k: int = 2) -> list[str]:
    """Vectorize chunks + question, return the k most similar chunks."""
    tokenized = [tokenize(text) for text in chunks]

    # IDF is computed on the corpus only — the question must not change it.
    n_docs = len(tokenized)
    document_frequency = Counter(
        word for tokens in tokenized for word in set(tokens)
    )
    idf = {
        word: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for word, freq in document_frequency.items()
    }

    chunk_vectors = [tfidf_vector(tokens, idf) for tokens in tokenized]
    question_vector = tfidf_vector(tokenize(question), idf)

    scored = [
        (cosine(question_vector, vector), text)
        for vector, text in zip(chunk_vectors, chunks)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [text for score, text in scored[:k] if score > 0]


def answer(question: str) -> str:
    context = retrieve(question, chunk(DOCUMENTS))
    print("Retrieved context:")
    for i, passage in enumerate(context, 1):
        print(f"  [{i}] {passage[:70]}...")
    grounded_prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n" + "\n".join(f"- {p}" for p in context) +
        f"\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": grounded_prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    question = "When should I change the oil on an FH16 and which grade?"
    print(f"Question: {question}\n")
    print("\nAnswer:\n" + answer(question))
