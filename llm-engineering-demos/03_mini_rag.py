"""Demo 3 — Minimal RAG pipeline: chunk, vectorize, retrieve, generate.

The pipeline shape is the same as a production RAG system (like my project
MIA: Azure OpenAI + CosmosDB vector search):
  1. Split documents into chunks
  2. Turn chunks into vectors (here: TF-IDF, offline; in prod: embeddings)
  3. Vectorize the user question and retrieve the top-k similar chunks
  4. Build a grounded prompt and let the LLM answer FROM the documents
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def retrieve(question: str, chunks: list[str], k: int = 2) -> list[str]:
    """Vectorize chunks + question, return the k most similar chunks."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks + [question])
    chunk_vectors, question_vector = matrix[:-1], matrix[-1]
    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [text for score, text in ranked[:k] if score > 0]


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
