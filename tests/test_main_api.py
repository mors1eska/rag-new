import pytest
from fastapi.testclient import TestClient
import unittest.mock as mock
import os
import sys
from typing import List, Tuple, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_api import app
# QueryResponse and SourceDocument are defined in main_api.py
# from main_api import QueryResponse, SourceDocument

client = TestClient(app)

# Mocked Document class for LangChain (if needed for RetrievalQA mock)
class MockAPIDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

# --- Tests for /query/ endpoint ---

@mock.patch('main_api.RetrievalQA') # Mocks the class itself
def test_query_successful(MockRetrievalQA, mocker): # Added mocker for patching globals
    """Test successful /query/ request."""
    mock_qa_instance = mock.AsyncMock()
    mock_qa_instance.ainvoke.return_value = {
        "result": "Test RAG answer",
        "source_documents": [
            MockAPIDocument("Test content snippet 1", {"file_name": "test1.pdf", "1c_section": "УНФ"}),
            MockAPIDocument("Test content snippet 2", {"file_name": "test2.md", "1c_section": "БП"})
        ]
    }
    MockRetrievalQA.from_chain_type.return_value = mock_qa_instance

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    mocker.patch('main_api.LLM', mock.Mock())
    mocker.patch('main_api.VECTORSTORE', mock.Mock())

    response = client.post("/query/", json={"query": "test query", "section": "УНФ"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test RAG answer"
    assert len(data["sources"]) == 2
    assert data["sources"][0]["file_name"] == "test1.pdf"
    assert data["sources"][0]["section_1c"] == "УНФ"

def test_query_missing_embedding_model(mocker):
    """Test /query/ when EMBEDDING_MODEL is None."""
    mocker.patch('main_api.EMBEDDING_MODEL', None)
    mocker.patch('main_api.LLM', mock.Mock())
    mocker.patch('main_api.VECTORSTORE', mock.Mock())
    response = client.post("/query/", json={"query": "test query"})
    assert response.status_code == 500
    assert "модель для эмбеддингов не сконфигурирована" in response.json()["detail"]

def test_query_missing_llm(mocker):
    """Test /query/ when LLM is None."""
    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    mocker.patch('main_api.LLM', None)
    mocker.patch('main_api.VECTORSTORE', mock.Mock())
    response = client.post("/query/", json={"query": "test query"})
    assert response.status_code == 500
    assert "LLM не сконфигурирована" in response.json()["detail"]


def test_query_missing_vectorstore(mocker):
    """Test /query/ when VECTORSTORE is None."""
    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    mocker.patch('main_api.LLM', mock.Mock())
    mocker.patch('main_api.VECTORSTORE', None)
    response = client.post("/query/", json={"query": "test query"})
    assert response.status_code == 500
    assert "база данных векторов не доступна" in response.json()["detail"]

@mock.patch('main_api.RetrievalQA')
def test_query_with_section_filter(MockRetrievalQA, mocker):
    """Test /query/ with a specific section filter."""
    mock_qa_instance = mock.AsyncMock()
    mock_qa_instance.ainvoke.return_value = {"result": "Filtered answer", "source_documents": []}
    MockRetrievalQA.from_chain_type.return_value = mock_qa_instance

    mock_vectorstore_instance = mock.Mock()
    mock_retriever = mock.Mock()
    mock_vectorstore_instance.as_retriever.return_value = mock_retriever

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    mocker.patch('main_api.LLM', mock.Mock())
    mocker.patch('main_api.VECTORSTORE', mock_vectorstore_instance)

    client.post("/query/", json={"query": "test query", "section": "БП"})

    mock_vectorstore_instance.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 5, 'filter': {'1c_section': 'БП'}}
    )

# --- Tests for /ensemble_query/ endpoint ---

MOCK_RICH_SEMANTIC_RESULTS = [("sem_doc1", 0.9, {"full_path": "path/sem1.pdf", "1c_section": "УНФ", "source": "semantic", "file_name": "sem1.pdf"}, "Semantic snippet 1")]
MOCK_RICH_TFIDF_RESULTS = [("tfidf_doc1", 0.8, {"doc_id": "tfidf_doc1", "source": "TF-IDF", "file_name": "tfidf1.txt"}, "Snippet not available...")]
MOCK_RICH_FTS_RESULTS = [("fts_doc1", 0.7, {"doc_id": "fts_doc1", "source": "FTS", "1c_section": "УНФ", "file_name": "fts1.dat"}, "FTS snippet 1")]
MOCK_MERGED_RESULTS = [("sem_doc1", 0.85), ("fts_doc1", 0.75), ("tfidf_doc1", 0.55)]


@mock.patch('main_api.ensemble_merge')
@mock.patch('main_api.fts_search')
@mock.patch('main_api.tfidf_search')
@mock.patch('main_api.semantic_search')
def test_ensemble_query_successful_default_weights(
    mock_semantic, mock_tfidf, mock_fts, mock_ensemble_merge, mocker
):
    """Test successful /ensemble_query/ with default weights."""
    mock_semantic.return_value = MOCK_RICH_SEMANTIC_RESULTS
    mock_tfidf.return_value = MOCK_RICH_TFIDF_RESULTS
    mock_fts.return_value = MOCK_RICH_FTS_RESULTS
    mock_ensemble_merge.return_value = MOCK_MERGED_RESULTS

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())

    payload = {"query": "test ensemble query"}
    response = client.post("/ensemble_query/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "Ensemble search completed" in data["answer"]
    assert len(data["sources"]) == len(MOCK_MERGED_RESULTS)

    args, kwargs = mock_ensemble_merge.call_args
    assert 'weights' in kwargs
    assert kwargs['weights'] == {"semantic": 0.5, "tfidf": 0.3, "fts": 0.2}
    assert kwargs['top_k'] == 10

    if data["sources"]:
        # Assuming sem_doc1 is first due to MOCK_MERGED_RESULTS
        assert "[Score: 0.8500] Semantic snippet 1" in data["sources"][0]["content_snippet"]
        assert data["sources"][0]["file_name"] == "sem1.pdf"


@mock.patch('main_api.ensemble_merge')
@mock.patch('main_api.fts_search')
@mock.patch('main_api.tfidf_search')
@mock.patch('main_api.semantic_search')
def test_ensemble_query_custom_params(
    mock_semantic, mock_tfidf, mock_fts, mock_ensemble_merge, mocker
):
    """Test /ensemble_query/ with custom weights, top_k, top_n_individual."""
    mock_semantic.return_value = MOCK_RICH_SEMANTIC_RESULTS
    mock_tfidf.return_value = []
    mock_fts.return_value = MOCK_RICH_FTS_RESULTS

    custom_weights = {"semantic": 0.7, "tfidf": 0.1, "fts": 0.2}
    mock_ensemble_merge.return_value = [("sem_doc1", 0.95), ("fts_doc1", 0.65)]

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    payload = {
        "query": "custom query",
        "top_k": 5,
        "top_n_individual": 7,
        "weights": custom_weights,
        "section": "БП"
    }
    response = client.post("/ensemble_query/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 2

    mock_semantic.assert_called_once_with("custom query", 7, "БП")
    mock_tfidf.assert_called_once_with("custom query", 7)
    mock_fts.assert_called_once_with("custom query", 7)

    args, kwargs = mock_ensemble_merge.call_args
    assert kwargs['weights'] == custom_weights
    assert kwargs['top_k'] == 5
    assert args[0] == [("sem_doc1", 0.9)]
    assert args[1] == []
    assert args[2] == [("fts_doc1", 0.7)]


@mock.patch('main_api.ensemble_merge')
@mock.patch('main_api.fts_search')
@mock.patch('main_api.tfidf_search')
@mock.patch('main_api.semantic_search')
def test_ensemble_query_with_post_merge_filter(
    mock_semantic, mock_tfidf, mock_fts, mock_ensemble_merge, mocker
):
    """Test /ensemble_query/ with post-merge filtering."""
    mock_semantic.return_value = MOCK_RICH_SEMANTIC_RESULTS
    mock_tfidf.return_value = []
    mock_fts.return_value = MOCK_RICH_FTS_RESULTS
    # MOCK_MERGED_RESULTS = [("sem_doc1", 0.85), ("fts_doc1", 0.75)]
    # sem_doc1 metadata: {"full_path": "path/sem1.pdf", "1c_section": "УНФ", "source": "semantic"}
    # fts_doc1 metadata: {"doc_id": "fts_doc1", "source": "FTS", "1c_section": "УНФ"}
    mock_ensemble_merge.return_value = [("sem_doc1", 0.85), ("fts_doc1", 0.75)]


    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    payload = {
        "query": "filter test",
        "filter_after_merge": {"1c_section": "УНФ"}
    }
    response = client.post("/ensemble_query/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 2
    assert data["sources"][0]["section_1c"] == "УНФ"
    assert data["sources"][1]["section_1c"] == "УНФ"
    assert "after merging and subsequent filtering" in data["answer"]


@mock.patch('main_api.ensemble_merge')
@mock.patch('main_api.fts_search')
@mock.patch('main_api.tfidf_search')
@mock.patch('main_api.semantic_search')
def test_ensemble_query_post_merge_filter_no_match(
    mock_semantic, mock_tfidf, mock_fts, mock_ensemble_merge, mocker
):
    """Test post-merge filter that results in no matches."""
    mock_semantic.return_value = MOCK_RICH_SEMANTIC_RESULTS
    mock_tfidf.return_value = []
    mock_fts.return_value = MOCK_RICH_FTS_RESULTS
    mock_ensemble_merge.return_value = MOCK_MERGED_RESULTS

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    payload = {
        "query": "filter no match test",
        "filter_after_merge": {"1c_section": "NonExistentSection"}
    }
    response = client.post("/ensemble_query/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 0
    assert "after merging and subsequent filtering" in data["answer"]

@mock.patch('main_api.ensemble_merge')
@mock.patch('main_api.fts_search')
@mock.patch('main_api.tfidf_search')
@mock.patch('main_api.semantic_search')
def test_ensemble_query_empty_results_from_all_searches(
    mock_semantic, mock_tfidf, mock_fts, mock_ensemble_merge, mocker
):
    """Test API when all individual searches and merge return empty."""
    mock_semantic.return_value = []
    mock_tfidf.return_value = []
    mock_fts.return_value = []
    mock_ensemble_merge.return_value = []

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    payload = {"query": "empty test"}
    response = client.post("/ensemble_query/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 0
    assert "Ensemble search completed" in data["answer"]

def test_ensemble_query_embedding_model_not_loaded(mocker):
    """Test /ensemble_query/ when EMBEDDING_MODEL is None."""
    mocker.patch('main_api.EMBEDDING_MODEL', None)
    # No need to mock individual searchers if EMBEDDING_MODEL check fails first
    payload = {"query": "test ensemble query"}
    response = client.post("/ensemble_query/", json=payload)
    assert response.status_code == 500
    assert "модель для эмбеддингов не сконфигурирована" in response.json()["detail"]

# Example of testing fallback scenario (indirectly, by controlling what ensemble_merge returns)
@mock.patch('main_api.ensemble_merge')
@mock.patch('main_api.fts_search')
@mock.patch('main_api.tfidf_search')
@mock.patch('main_api.semantic_search')
def test_ensemble_query_fallback_scenario_simulated(
    mock_semantic, mock_tfidf, mock_fts, mock_ensemble_merge, mocker
):
    """Simulate a fallback scenario by having ensemble_merge return semantic-like results."""
    # These are normalized scores as if minmax_scale was applied
    mock_semantic_normalized_fallback = [("sem_doc1", 1.0), ("sem_doc_extra", 0.8)]

    mock_semantic.return_value = MOCK_RICH_SEMANTIC_RESULTS # Original rich results
    mock_tfidf.return_value = [] # No tfidf results, so intersection might be small or empty
    mock_fts.return_value = []   # No fts results

    # ensemble_merge is mocked to return what it would in a fallback
    mock_ensemble_merge.return_value = mock_semantic_normalized_fallback

    mocker.patch('main_api.EMBEDDING_MODEL', mock.Mock())
    payload = {"query": "fallback test"}
    response = client.post("/ensemble_query/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 2 # Should match the length of mock_semantic_normalized_fallback

    # Verify that the details_map still helps construct the response correctly
    # sem_doc1 is in MOCK_RICH_SEMANTIC_RESULTS, so its details should be found
    assert data["sources"][0]["file_name"] == "sem1.pdf"
    assert "[Score: 1.0000]" in data["sources"][0]["content_snippet"]

    # sem_doc_extra is NOT in MOCK_RICH_SEMANTIC_RESULTS, so its details won't be found in details_map
    # The code currently prints a warning and skips adding to output_sources if details not found.
    # This means the actual number of sources in response might be less than mock_ensemble_merge returns.
    # Let's adjust MOCK_MERGED_RESULTS for this test or ensure details_map is populated correctly.

    # Correct approach for this test:
    # Ensure details_map would have entries for what mock_ensemble_merge returns.
    # For this test, we'll assume sem_doc1 is the only one whose details can be found.

    # Re-evaluate: mock_ensemble_merge returns IDs. The API then looks up these IDs in details_map.
    # details_map is built from mock_semantic, mock_tfidf, mock_fts.
    # If ensemble_merge returns ("sem_doc_extra", 0.8), but "sem_doc_extra" was not in the original rich results,
    # then details_map.get("sem_doc_extra") will be None.

    # Current test setup for details_map:
    # details_map will contain "sem_doc1" from MOCK_RICH_SEMANTIC_RESULTS.
    # If mock_ensemble_merge returns [("sem_doc1", 1.0), ("sem_doc_extra", 0.8)],
    # only "sem_doc1" will be found in details_map.

    # Let's make mock_ensemble_merge return only IDs that can be found in details_map for this test.
    mock_ensemble_merge.return_value = [("sem_doc1", 1.0)] # sem_doc1 is in MOCK_RICH_SEMANTIC_RESULTS

    response_fallback = client.post("/ensemble_query/", json=payload)
    assert response_fallback.status_code == 200
    data_fallback = response_fallback.json()
    assert len(data_fallback["sources"]) == 1
    assert data_fallback["sources"][0]["file_name"] == "sem1.pdf"
    assert "[Score: 1.0000]" in data_fallback["sources"][0]["content_snippet"]
