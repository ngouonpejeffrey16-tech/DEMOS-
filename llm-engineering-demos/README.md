# LLM Engineering Demos

Hands-on demos of core GenAI engineering patterns in Python: calling LLM APIs, comparing models, Retrieval-Augmented Generation (RAG), and agent orchestration with LangGraph.

> Built as a companion to my main project **MIA** — an enterprise RAG chatbot (Azure OpenAI + CosmosDB + FastAPI, integrated into Microsoft Teams) that cut document search time by ~70%.

## What's inside

| Script | Concept |
|---|---|
| `01_chat_basics.py` | Connect to an LLM API (OpenAI-compatible), send a prompt, stream the answer |
| `02_model_comparison.py` | Run the same prompt across several models and compare outputs side by side |
| `03_mini_rag.py` | Minimal RAG pipeline: chunking, vectorization, similarity search, grounded generation |
| `04_langgraph_agent.py` | A LangGraph agent: StateGraph, tool node, conditional edges, loop until done |
| `05_data_cleaning.py` | Data cleaning with pandas: duplicates, mixed date formats, missing values, outliers |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key (free key at https://console.groq.com)
cp .env.example .env
# then edit .env and paste your key

# 3. Run any demo
python 01_chat_basics.py
python 02_model_comparison.py
python 03_mini_rag.py
python 04_langgraph_agent.py
python 05_data_cleaning.py   # no API key needed
```

The demos use an **OpenAI-compatible client**, so the same code works with Groq, Mistral, OpenAI, or a local server (Ollama, vLLM) — just change `BASE_URL` and `MODEL_NAME` in `.env`.

## Design notes

- **`05_data_cleaning.py`** runs fully offline. Key principle demonstrated: order matters — normalize text before deduplicating, parse dates explicitly (never guess day/month), and flag outliers *before* imputing missing values so they don't pollute the statistics.

- **`03_mini_rag.py`** uses TF-IDF vectors so it runs offline with zero API cost for the retrieval step. In production (as in MIA), you would swap this for a proper embedding model and a vector database (e.g. Azure OpenAI embeddings + CosmosDB vector search). The pipeline shape — chunk, vectorize, retrieve top-k by similarity, ground the prompt — is identical.
- **`04_langgraph_agent.py`** shows the canonical agent loop: an LLM node decides whether to call a tool, a conditional edge routes to the tool node or to END, and the tool result loops back to the LLM.

## AI-assisted development

This repository was built with AI assistance (Claude) as part of my daily workflow — I use LLM assistants for scaffolding, code review and documentation, then verify, test and refine everything myself. Prompting and reviewing AI output effectively is a skill I actively practice.

## Author

**Jeffrey Gandhi Ngouonpe** — Master's student in Data & AI at EFREI Paris.
Certified AZ-900 · AI-900 · SC-900.
