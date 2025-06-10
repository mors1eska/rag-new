"""
Unit tests for ensemble_logic.py functions.
"""
import pytest # Import pytest for potential use of fixtures or specific features later
from typing import List, Tuple, Dict
import sys
import os

# Add the project root to the Python path to allow importing from ensemble_logic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ensemble_logic import minmax_scale, ensemble_merge

# --- Tests for minmax_scale ---

def test_minmax_scale_empty_list():
    """Test minmax_scale with an empty list."""
    assert minmax_scale([]) == []

def test_minmax_scale_single_item():
    """Test minmax_scale with a single item list."""
    assert minmax_scale([("doc1", 10.0)]) == [("doc1", 1.0)]

def test_minmax_scale_all_same_scores():
    """Test minmax_scale when all scores are the same."""
    results = [("doc1", 5.0), ("doc2", 5.0), ("doc3", 5.0)]
    expected = [("doc1", 1.0), ("doc2", 1.0), ("doc3", 1.0)]
    assert minmax_scale(results) == expected

def test_minmax_scale_positive_scores_ascending():
    """Test minmax_scale with positive scores in ascending order."""
    results = [("doc1", 10.0), ("doc2", 20.0), ("doc3", 30.0)]
    expected = [("doc1", 0.0), ("doc2", 0.5), ("doc3", 1.0)]
    scaled_results = minmax_scale(results)
    for (res_id, res_score), (exp_id, exp_score) in zip(scaled_results, expected):
        assert res_id == exp_id
        assert res_score == pytest.approx(exp_score)

def test_minmax_scale_positive_scores_descending():
    """Test minmax_scale with positive scores in descending order."""
    results = [("doc1", 30.0), ("doc2", 20.0), ("doc3", 10.0)]
    expected = [("doc1", 1.0), ("doc2", 0.5), ("doc3", 0.0)]
    scaled_results = minmax_scale(results)
    for (res_id, res_score), (exp_id, exp_score) in zip(scaled_results, expected):
        assert res_id == exp_id
        assert res_score == pytest.approx(exp_score)

def test_minmax_scale_scores_include_zero():
    """Test minmax_scale with scores including zero."""
    results = [("docA", 0.0), ("docB", 5.0), ("docC", 10.0)]
    expected = [("docA", 0.0), ("docB", 0.5), ("docC", 1.0)]
    scaled_results = minmax_scale(results)
    for (res_id, res_score), (exp_id, exp_score) in zip(scaled_results, expected):
        assert res_id == exp_id
        assert res_score == pytest.approx(exp_score)

def test_minmax_scale_mixed_positive_and_negative_scores():
    """Test minmax_scale with a mix of positive and negative scores."""
    results = [("neg2", -10.0), ("neg1", -5.0), ("zero", 0.0), ("pos1", 5.0), ("pos2", 10.0)]
    expected = [("neg2", 0.0), ("neg1", 0.25), ("zero", 0.5), ("pos1", 0.75), ("pos2", 1.0)]
    scaled_results = minmax_scale(results)
    for (res_id, res_score), (exp_id, exp_score) in zip(scaled_results, expected):
        assert res_id == exp_id
        assert res_score == pytest.approx(exp_score)

def test_minmax_scale_negative_scores_only():
    """Test minmax_scale with only negative scores."""
    results = [("doc1", -5.0), ("doc2", -10.0), ("doc3", -2.0)]
    expected = {"doc1": 0.625, "doc2": 0.0, "doc3": 1.0} # Using dict for order-insensitive comparison
    scaled_results = minmax_scale(results)
    assert dict(scaled_results) == pytest.approx(expected)

# --- Tests for ensemble_merge ---

# Sorted semantic results for predictable fallback in tests
SEM_RESULTS_SORTED = [("docA", 0.9), ("docB", 0.8), ("docC", 0.6)]
# Corresponding normalized: docA=1, docB=2/3, docC=0 (approx 0.666)

TFIDF_RESULTS_1 = [("docA", 20.0), ("docB", 10.0), ("docC", 5.0)]
# Normalized: docA=1, docB=1/3, docC=0 (approx 0.333)

FTS_RESULTS_1 = [("docA", -1.0), ("docB", -5.0), ("docC", -2.0)] # Original scores
# Normalized (min=-5, max=-1, range=4): docA=1, docB=0, docC=0.75

WEIGHTS_1 = {'semantic': 0.5, 'tfidf': 0.3, 'fts': 0.2}
# Expected scores for common docs (A, B, C) with WEIGHTS_1:
# docA: (1.0 * 0.5) + (1.0 * 0.3) + (1.0 * 0.2) = 1.0
# docB: (0.6666 * 0.5) + (0.3333 * 0.3) + (0.0 * 0.2) = 0.3333 + 0.09999 + 0 = 0.43329
# docC: (0.0 * 0.5) + (0.0 * 0.3) + (0.75 * 0.2) = 0 + 0 + 0.15 = 0.15
# Expected order: docA, docB, docC

def test_ensemble_merge_basic_intersection():
    """Test basic intersection and weighting."""
    merged = ensemble_merge(SEM_RESULTS_SORTED, TFIDF_RESULTS_1, FTS_RESULTS_1, WEIGHTS_1, top_k=3)
    assert len(merged) == 3
    assert merged[0][0] == "docA"
    assert merged[0][1] == pytest.approx(1.0)
    assert merged[1][0] == "docB"
    assert merged[1][1] == pytest.approx(0.43333, abs=1e-3)
    assert merged[2][0] == "docC"
    assert merged[2][1] == pytest.approx(0.15)

def test_ensemble_merge_no_common_documents_fallback():
    """Test fallback when no common documents exist (strict intersection fails)."""
    sem_results = [("docX", 0.8)] # norm: [("docX", 1.0)]
    tfidf_results = [("docY", 20.0)]
    fts_results = [("docZ", -1.0)]
    merged = ensemble_merge(sem_results, tfidf_results, fts_results, WEIGHTS_1, top_k=1, min_intersection_results=1)
    assert len(merged) == 1
    assert merged[0] == ("docX", 1.0) # Falls back to normalized semantic

def test_ensemble_merge_one_list_empty_fallback():
    """Test fallback when one input list is empty (e.g., tfidf)."""
    # Uses SEM_RESULTS_SORTED, norm_semantic: [("docA",1.0), ("docB",0.666...), ("docC",0.0)]
    merged = ensemble_merge(SEM_RESULTS_SORTED, [], FTS_RESULTS_1, WEIGHTS_1, top_k=2, min_intersection_results=1)
    assert len(merged) == 2
    assert merged[0][0] == "docA"
    assert merged[0][1] == pytest.approx(1.0)
    assert merged[1][0] == "docB"
    assert merged[1][1] == pytest.approx(2/3, abs=1e-3)

def test_ensemble_merge_min_intersection_trigger_fallback():
    """Test fallback when common docs < min_intersection_results."""
    sem_results = [("common1", 0.9), ("common2", 0.8), ("sem_only", 0.7)] # norm: c1=1, c2=0.5, so=0 (sorted by score)
    tfidf_results = [("common1", 15.0), ("common2", 25.0), ("tfidf_only", 10.0)]
    fts_results = [("common1", -2.0), ("common2", -3.0), ("fts_only", -1.0)]
    # Common docs: common1, common2. Length is 2.
    # If min_intersection_results is 3, should fallback to (normalized) semantic.
    merged = ensemble_merge(sem_results, tfidf_results, fts_results, WEIGHTS_1, top_k=2, min_intersection_results=3)
    assert len(merged) == 2
    assert merged[0][0] == "common1"
    assert merged[0][1] == pytest.approx(1.0)
    assert merged[1][0] == "common2"
    assert merged[1][1] == pytest.approx(0.5) # (0.8-0.7)/(0.9-0.7) = 0.5

def test_ensemble_merge_all_lists_empty():
    """Test behavior when all input lists are empty."""
    merged = ensemble_merge([], [], [], WEIGHTS_1, top_k=3)
    assert merged == []

def test_ensemble_merge_fallback_with_empty_semantic():
    """Test fallback when semantic results (the fallback source) are also empty."""
    merged = ensemble_merge([], TFIDF_RESULTS_1, FTS_RESULTS_1, WEIGHTS_1, top_k=3, min_intersection_results=1)
    assert merged == []

def test_ensemble_merge_sufficient_intersection():
    """Test normal merge when common docs >= min_intersection_results."""
    merged = ensemble_merge(SEM_RESULTS_SORTED, TFIDF_RESULTS_1, FTS_RESULTS_1, WEIGHTS_1, top_k=3, min_intersection_results=2)
    assert len(merged) == 3
    assert merged[0][0] == "docA"
    assert merged[0][1] == pytest.approx(1.0)

def test_ensemble_merge_top_k_respected_merge():
    """Test that top_k parameter is respected in normal merge mode."""
    merged = ensemble_merge(SEM_RESULTS_SORTED, TFIDF_RESULTS_1, FTS_RESULTS_1, WEIGHTS_1, top_k=1, min_intersection_results=1)
    assert len(merged) == 1
    assert merged[0][0] == "docA"

def test_ensemble_merge_top_k_respected_fallback():
    """Test that top_k parameter is respected in fallback mode."""
    merged = ensemble_merge(SEM_RESULTS_SORTED, [], [], WEIGHTS_1, top_k=2, min_intersection_results=1)
    assert len(merged) == 2
    assert merged[0][0] == "docA"
    assert merged[1][0] == "docB"


def test_ensemble_merge_weights_effect():
    """Test that different weights change scores and potentially order."""
    weights_sem_heavy = {'semantic': 0.8, 'tfidf': 0.1, 'fts': 0.1}
    # norm_semantic: docA=1, docB=0.666, docC=0
    # norm_tfidf: docA=1, docB=0.333, docC=0
    # norm_fts: docA=1, docB=0, docC=0.75
    # docA_new_score = (1.0 * 0.8) + (1.0 * 0.1) + (1.0 * 0.1) = 1.0
    # docB_new_score = (0.6666 * 0.8) + (0.3333 * 0.1) + (0.0 * 0.1) = 0.53328 + 0.03333 + 0 = 0.56661
    # docC_new_score = (0.0 * 0.8) + (0.0 * 0.1) + (0.75 * 0.1) = 0 + 0 + 0.075 = 0.075

    merged_sem_heavy = ensemble_merge(SEM_RESULTS_SORTED, TFIDF_RESULTS_1, FTS_RESULTS_1, weights_sem_heavy, top_k=3)
    assert len(merged_sem_heavy) == 3
    assert merged_sem_heavy[0][0] == "docA"
    assert merged_sem_heavy[0][1] == pytest.approx(1.0)
    assert merged_sem_heavy[1][0] == "docB"
    assert merged_sem_heavy[1][1] == pytest.approx(0.56666, abs=1e-3)
    assert merged_sem_heavy[2][0] == "docC"
    assert merged_sem_heavy[2][1] == pytest.approx(0.075)

    merged_orig_weights = ensemble_merge(SEM_RESULTS_SORTED, TFIDF_RESULTS_1, FTS_RESULTS_1, WEIGHTS_1, top_k=3)
    # Scores for docA should be the same as it gets max normalized score from all sources
    assert merged_sem_heavy[0][1] == pytest.approx(merged_orig_weights[0][1])
    # Scores for docB and docC should differ due to different weighting
    assert merged_sem_heavy[1][1] != pytest.approx(merged_orig_weights[1][1])
    assert merged_sem_heavy[2][1] != pytest.approx(merged_orig_weights[2][1])
