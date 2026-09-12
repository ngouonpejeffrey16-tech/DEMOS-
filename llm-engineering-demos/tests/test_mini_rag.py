"""Tests for the RAG retrieval step (03_mini_rag.py).

The retrieval step is deterministic and testable offline — we never call the
LLM here. Testing retrieval separately from generation is good practice:
most RAG failures come from bad retrieval, not from the model.

Run with:  pytest -v
"""

import importlib.util
import os
from pathlib import Path

import pytest

# The demo builds an API client at import time, so give it a dummy key.
os.environ.setdefault("LLM_API_KEY", "test-key-not-used")

SPEC = importlib.util.spec_from_file_location(
    "mini_rag", Path(__file__).parent.parent / "03_mini_rag.py"
)
mini_rag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mini_rag)


def test_retrieve_returns_requested_number_of_chunks():
    results = mini_rag.retrieve("oil change interval", mini_rag.DOCUMENTS, k=2)
    assert len(results) <= 2


def test_retrieve_finds_the_relevant_document():
    """A question about oil must surface the oil-change document first."""
    results = mini_rag.retrieve(
        "When should engine oil be replaced?", mini_rag.DOCUMENTS, k=1
    )
    assert "oil" in results[0].lower()


def test_retrieve_ranks_brake_question_to_brake_document():
    results = mini_rag.retrieve(
        "What is the minimum brake pad thickness?", mini_rag.DOCUMENTS, k=1
    )
    assert "brake" in results[0].lower()


def test_retrieve_drops_completely_unrelated_questions():
    """Zero-similarity chunks are filtered out rather than returned as noise."""
    results = mini_rag.retrieve("xyzzy quuxfoo", mini_rag.DOCUMENTS, k=2)
    assert results == []


@pytest.mark.parametrize("k", [1, 2, 3])
def test_retrieve_respects_k(k):
    results = mini_rag.retrieve("battery voltage winter", mini_rag.DOCUMENTS, k=k)
    assert len(results) <= k


# --- TF-IDF internals ------------------------------------------------------
def test_cosine_of_identical_vectors_is_one():
    vector = {"brake": 0.5, "pad": 0.5}
    assert mini_rag.cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_of_disjoint_vectors_is_zero():
    assert mini_rag.cosine({"brake": 1.0}, {"battery": 1.0}) == 0.0


def test_tokenize_lowercases_and_strips_punctuation():
    assert mini_rag.tokenize("Oil, 60,000 km!") == ["oil", "60", "000", "km"]
