"""
Unit tests for query_data.py search functions.
"""
import pytest
import unittest.mock as mock
import os
import sys
from typing import List, Tuple, Dict, Any
import numpy as np # Added missing import
import sqlite3 # Added missing import for FTS tests

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query_data import semantic_search, tfidf_search, fts_search
# Mocked Document class for LangChain
class MockDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

# --- Constants for tests ---
CHROMA_PERSIST_DIR_TEST = "test_db_chroma_qd" # Используем отдельную директорию для тестов Chroma
SQLITE_DB_PATH_TEST = ":memory:" # Для FTS тестов в памяти
SQLITE_FTS_TABLE_TEST = "documents_fts_test"

TFIDF_MODEL_PATH_TEST = os.path.join(CHROMA_PERSIST_DIR_TEST, "tfidf_model.joblib")
TFIDF_VECTORS_PATH_TEST = os.path.join(CHROMA_PERSIST_DIR_TEST, "tfidf_vectors_and_ids.joblib")


# --- Fixtures and Mocks for semantic_search ---

@pytest.fixture
def mock_semantic_dependencies(mocker):
    """Mocks dependencies for semantic_search."""
    mocker.patch('query_data.get_embedding_model', return_value=mock.Mock())

    mock_chroma_instance = mock.Mock()
    # Пример возвращаемых данных для similarity_search_with_score
    mock_docs_with_scores = [
        (MockDocument("Semantic content 1", {"full_path": "semantic/path1.pdf", "1c_section": "УНФ", "page_number": 1}), 0.2),
        (MockDocument("Semantic content 2", {"full_path": "semantic/path2.md", "1c_section": "БП", "page_number": 1}), 0.5),
    ]
    mock_chroma_instance.similarity_search_with_score.return_value = mock_docs_with_scores

    mocker.patch('langchain_community.vectorstores.Chroma', return_value=mock_chroma_instance)
    # mocker.patch('query_data.CHROMA_PERSIST_DIR', CHROMA_PERSIST_DIR_TEST) # Уже используется в query_data

    # Mock os.path.exists for CHROMA_PERSIST_DIR
    mock_os_path_exists = mocker.patch('os.path.exists')

    return mock_chroma_instance, mock_os_path_exists

# --- Tests for semantic_search ---

def test_semantic_search_successful(mock_semantic_dependencies):
    """Test successful semantic_search execution."""
    _, mock_os_path_exists = mock_semantic_dependencies
    mock_os_path_exists.return_value = True # CHROMA_PERSIST_DIR exists

    results = semantic_search("test query", top_n=2)

    assert len(results) == 2
    doc_id, score, metadata, snippet = results[0]
    assert "semantic/path1.pdf" in doc_id
    assert score == pytest.approx(1 / (1 + 0.2))
    assert metadata["1c_section"] == "УНФ"
    assert snippet == "Semantic content 1"

    doc_id_2, score_2, metadata_2, snippet_2 = results[1]
    assert "semantic/path2.md" in doc_id_2
    assert score_2 == pytest.approx(1 / (1 + 0.5))
    assert metadata_2["1c_section"] == "БП"
    assert snippet_2 == "Semantic content 2"

def test_semantic_search_with_filter(mock_semantic_dependencies):
    """Test semantic_search with a section filter."""
    mock_chroma, mock_os_path_exists = mock_semantic_dependencies
    mock_os_path_exists.return_value = True

    semantic_search("query with filter", top_n=2, section_filter="УНФ")
    # Проверяем, что filter был передан в similarity_search_with_score
    mock_chroma.similarity_search_with_score.assert_called_once_with(
        "query with filter",
        k=2,
        filter={'1c_section': 'УНФ'}
    )

def test_semantic_search_chroma_dir_not_exists(mock_semantic_dependencies):
    """Test semantic_search when CHROMA_PERSIST_DIR does not exist."""
    _, mock_os_path_exists = mock_semantic_dependencies
    mock_os_path_exists.return_value = False # CHROMA_PERSIST_DIR does NOT exist

    results = semantic_search("test query", top_n=2)
    assert results == []

def test_semantic_search_empty_results_from_chroma(mock_semantic_dependencies):
    """Test semantic_search when Chroma returns an empty list."""
    mock_chroma, mock_os_path_exists = mock_semantic_dependencies
    mock_os_path_exists.return_value = True
    mock_chroma.similarity_search_with_score.return_value = [] # Chroma returns no results

    results = semantic_search("test query", top_n=2)
    assert results == []

def test_semantic_search_score_conversion(mock_semantic_dependencies):
    """Verify correct distance to similarity score conversion."""
    mock_chroma, mock_os_path_exists = mock_semantic_dependencies
    mock_os_path_exists.return_value = True

    # distance = 0 -> similarity = 1
    # distance = 1 -> similarity = 0.5
    # distance = 3 -> similarity = 0.25
    mock_chroma.similarity_search_with_score.return_value = [
        (MockDocument("Content A", {"full_path": "pathA"}), 0.0),
        (MockDocument("Content B", {"full_path": "pathB"}), 1.0),
        (MockDocument("Content C", {"full_path": "pathC"}), 3.0),
    ]
    results = semantic_search("score test", top_n=3)
    assert len(results) == 3
    assert results[0][1] == pytest.approx(1.0)  # 1 / (1 + 0)
    assert results[1][1] == pytest.approx(0.5)  # 1 / (1 + 1)
    assert results[2][1] == pytest.approx(0.25) # 1 / (1 + 3)


# --- Fixtures and Mocks for tfidf_search ---

@pytest.fixture
def mock_tfidf_dependencies(mocker):
    """Mocks dependencies for tfidf_search."""
    mock_vectorizer_instance = mock.Mock()
    mock_vectorizer_instance.transform.return_value = np.array([[0.1, 0.2, 0.3]]) # Example transformed query

    # Mock joblib.load
    # It's called twice: once for the vectorizer, once for vectors_data
    mock_joblib_load = mocker.patch('joblib.load')

    # Test data for tfidf_vectors_and_ids.joblib
    test_doc_ids = ["tfidf/path1.txt", "tfidf/path2.txt"]
    # Ensure this matrix matches dimensions with query_vector and expected cosine_similarity behavior
    test_doc_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Configure side_effect for joblib.load
    mock_joblib_load.side_effect = [
        mock_vectorizer_instance, # First call loads vectorizer
        {'ids': test_doc_ids, 'vectors': test_doc_vectors} # Second call loads vectors data
    ]

    # Mock os.path.exists
    mock_os_path_exists_tfidf = mocker.patch('os.path.exists')

    # Mock cosine_similarity
    # This is a bit more direct if we want to control similarity scores precisely
    mock_cosine_similarity = mocker.patch('sklearn.metrics.pairwise.cosine_similarity')
    # Example: query vector is [0.1, 0.2, 0.3]
    # doc_vectors are [[0.1,0.2,0.3], [0.4,0.5,0.6]]
    # Expected similarity: perfect match with first, some match with second
    mock_cosine_similarity.return_value = np.array([[1.0, 0.5]])

    # Patch constants used within tfidf_search for file paths
    mocker.patch('query_data.CHROMA_PERSIST_DIR', CHROMA_PERSIST_DIR_TEST)


    return mock_joblib_load, mock_os_path_exists_tfidf, mock_cosine_similarity, mock_vectorizer_instance

# --- Tests for tfidf_search ---

def test_tfidf_search_successful(mock_tfidf_dependencies):
    """Test successful tfidf_search execution."""
    _, mock_os_path_exists, _, _ = mock_tfidf_dependencies
    # Simulate that both files exist
    mock_os_path_exists.side_effect = lambda path: True

    results = tfidf_search("test query", top_n=2)

    assert len(results) == 2

    doc_id_1, score_1, metadata_1, snippet_1 = results[0]
    assert doc_id_1 == "tfidf/path1.txt"
    assert score_1 == pytest.approx(1.0)
    assert metadata_1['source'] == 'TF-IDF'
    assert "Snippet not available" in snippet_1

    doc_id_2, score_2, metadata_2, snippet_2 = results[1]
    assert doc_id_2 == "tfidf/path2.txt"
    assert score_2 == pytest.approx(0.5)
    assert metadata_2['source'] == 'TF-IDF'

def test_tfidf_search_files_not_exist(mock_tfidf_dependencies):
    """Test tfidf_search when model/vector files do not exist."""
    _, mock_os_path_exists, _, _ = mock_tfidf_dependencies
    mock_os_path_exists.return_value = False # Files do not exist

    results = tfidf_search("test query", top_n=2)
    assert results == []

def test_tfidf_search_one_file_not_exist(mock_tfidf_dependencies):
    """Test tfidf_search when one of the two files does not exist."""
    _, mock_os_path_exists, _, _ = mock_tfidf_dependencies
    # Simulate only vectorizer exists, but vectors_data does not
    mock_os_path_exists.side_effect = lambda path: TFIDF_MODEL_PATH_TEST in path

    results = tfidf_search("test query", top_n=2)
    assert results == []

def test_tfidf_search_empty_results_from_similarity(mock_tfidf_dependencies):
    """Test tfidf_search when cosine_similarity returns no good matches (e.g. all zeros)."""
    mock_joblib_load, mock_os_path_exists, mock_cosine_similarity, _ = mock_tfidf_dependencies
    mock_os_path_exists.return_value = True
    mock_cosine_similarity.return_value = np.array([[0.0, 0.0]]) # No similarity

    results = tfidf_search("test query", top_n=2)
    assert len(results) == 0 # No results because scores are not > 0


# --- Fixtures and Mocks for fts_search ---

@pytest.fixture
def setup_test_fts_db(mocker):
    """Sets up an in-memory SQLite FTS5 database with test data."""
    # Patch SQLITE_DB_PATH and SQLITE_FTS_TABLE in query_data for the duration of the test
    mocker.patch('query_data.SQLITE_DB_PATH', SQLITE_DB_PATH_TEST) # Use ":memory:"
    mocker.patch('query_data.SQLITE_FTS_TABLE', SQLITE_FTS_TABLE_TEST)

    conn = sqlite3.connect(SQLITE_DB_PATH_TEST) # This will be an in-memory DB for this connection
    cursor = conn.cursor()

    # Create FTS table
    cursor.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {SQLITE_FTS_TABLE_TEST} (
        doc_id TEXT UNIQUE,
        content TEXT,
        tokenize = 'porter unicode61'
    );
    """)

    # Insert test data
    test_data = [
        ("fts_doc1", "This is a test document about fts queries."),
        ("fts_doc2", "Another test document, focusing on sqlite features."),
        ("fts_doc3", "A document specifically about a query for fts search."),
        ("fts_doc4", "Completely unrelated content for testing no match."),
    ]
    cursor.executemany(f"INSERT INTO {SQLITE_FTS_TABLE_TEST} (doc_id, content) VALUES (?, ?)", test_data)
    conn.commit()

    # Mock sqlite3.connect to return this specific connection when called from query_data
    mock_connect = mocker.patch('sqlite3.connect')
    mock_connect.return_value = conn

    yield conn # Provide the connection to the test if needed, though mock_connect handles it

    conn.close()


# --- Tests for fts_search ---

def test_fts_search_successful(setup_test_fts_db):
    """Test successful fts_search execution."""
    mocker.patch('os.path.exists', return_value=True) # Assume DB file "exists" (for in-memory this isn't strictly needed but good for consistency)

    results = fts_search("fts query", top_n=2) # "fts" and "query" are in doc1 and doc3

    assert len(results) == 2

    # Results are ordered by rank (BM25). Closer to 0 is better.
    # "fts query" - doc3 is a very good match, doc1 is also good.
    # We can't easily predict exact BM25 scores, so check doc_ids and relative order if possible

    doc_ids_found = [r[0] for r in results]
    assert "fts_doc3" in doc_ids_found
    assert "fts_doc1" in doc_ids_found

    # Check that the best match (likely fts_doc3) comes first or has a better (less negative) score
    score_doc3 = 0.0
    score_doc1 = 0.0
    for r_id, r_score, _, _ in results:
        if r_id == "fts_doc3":
            score_doc3 = r_score
        if r_id == "fts_doc1":
            score_doc1 = r_score

    # FTS rank is negative, closer to zero is better. Our processed score is -rank. So higher is better.
    # "A document specifically about a query for fts search." (fts_doc3)
    # "This is a test document about fts queries." (fts_doc1)
    # fts_doc3 should be more relevant to "fts query"
    assert score_doc3 > score_doc1


    # Check structure of a result item
    doc_id, score, metadata, snippet = results[0]
    assert isinstance(doc_id, str)
    assert isinstance(score, float)
    assert isinstance(metadata, dict)
    assert isinstance(snippet, str)
    assert metadata['source'] == 'FTS'
    assert snippet.startswith(setup_test_fts_db.execute(f"SELECT content FROM {SQLITE_FTS_TABLE_TEST} WHERE doc_id = '{doc_id}'").fetchone()[0][:50])


def test_fts_search_no_match(setup_test_fts_db):
    """Test fts_search when the query finds no matches."""
    mocker.patch('os.path.exists', return_value=True)
    results = fts_search("nonexistentXYZ", top_n=2)
    assert len(results) == 0

def test_fts_search_db_path_not_exist(setup_test_fts_db, mocker): # setup_test_fts_db still runs for consistency but os.path.exists is key here
    """Test fts_search when the SQLite DB path does not exist."""
    mock_os_exist = mocker.patch('os.path.exists')
    mock_os_exist.return_value = False # DB file does not exist

    # Unpatch sqlite3.connect if it was patched by setup_test_fts_db to ensure it's not called
    # This is tricky because setup_test_fts_db already patches it.
    # A cleaner way might be to have a separate fixture for this specific case,
    # or to ensure that the os.path.exists check in fts_search happens before sqlite3.connect.
    # For now, we rely on fts_search's internal os.path.exists check.

    results = fts_search("any query", top_n=2)
    assert results == []
    mock_os_exist.assert_called_with(SQLITE_DB_PATH_TEST) # Check that the patched path was used
