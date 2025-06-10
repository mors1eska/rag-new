"""
This module provides functions for ensembling search results from different retrieval methods.
"""

from typing import List, Tuple, Dict, Set

def minmax_scale(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Normalizes a list of scores to a 0-1 scale using Min-Max scaling.

    Args:
        results: A list of (doc_id, score) tuples, where 'score' is a float
                 and higher scores are considered better.

    Returns:
        A list of (doc_id, normalized_score) tuples, where normalized_score
        is between 0 and 1.
        Returns an empty list if the input is empty.
        If all scores are the same, all normalized scores will be 1.0.
    """
    if not results:
        return []

    scores = [score for _, score in results]

    if not scores: # Should be caught by 'if not results' but as a safeguard
        return []

    min_score = min(scores)
    max_score = max(scores)

    # If all scores are the same (or only one item)
    if min_score == max_score:
        return [(doc_id, 1.0) for doc_id, _ in results]

    normalized_results: List[Tuple[str, float]] = []
    for doc_id, score in results:
        try:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_results.append((doc_id, normalized_score))
        except ZeroDivisionError:
            # This case should be covered by min_score == max_score check,
            # but as an absolute safeguard, assign 1.0.
            normalized_results.append((doc_id, 1.0))

    return normalized_results

if __name__ == "__main__":
    print("Testing minmax_scale function...")

    # Test case 1: Basic scaling
    sample_results_1 = [("doc1", 10.0), ("doc2", 20.0), ("doc3", 30.0)]
    print(f"\nOriginal 1: {sample_results_1}")
    scaled_results_1 = minmax_scale(sample_results_1)
    print(f"Scaled 1: {scaled_results_1}")
    # Expected: [('doc1', 0.0), ('doc2', 0.5), ('doc3', 1.0)]

    # Test case 2: Scores include 0
    sample_results_2 = [("docA", 0.0), ("docB", 5.0), ("docC", 10.0)]
    print(f"\nOriginal 2: {sample_results_2}")
    scaled_results_2 = minmax_scale(sample_results_2)
    print(f"Scaled 2: {scaled_results_2}")
    # Expected: [('docA', 0.0), ('docB', 0.5), ('docC', 1.0)]

    # Test case 3: All scores are the same
    sample_results_3 = [("docX", 5.0), ("docY", 5.0), ("docZ", 5.0)]
    print(f"\nOriginal 3: {sample_results_3}")
    scaled_results_3 = minmax_scale(sample_results_3)
    print(f"Scaled 3: {scaled_results_3}")
    # Expected: [('docX', 1.0), ('docY', 1.0), ('docZ', 1.0)]

    # Test case 4: Single item
    sample_results_4 = [("single_doc", 100.0)]
    print(f"\nOriginal 4: {sample_results_4}")
    scaled_results_4 = minmax_scale(sample_results_4)
    print(f"Scaled 4: {scaled_results_4}")
    # Expected: [('single_doc', 1.0)]

    # Test case 5: Empty list
    sample_results_5: List[Tuple[str, float]] = []
    print(f"\nOriginal 5: {sample_results_5}")
    scaled_results_5 = minmax_scale(sample_results_5)
    print(f"Scaled 5: {scaled_results_5}")
    # Expected: []

    # Test case 6: Scores in descending order
    sample_results_6 = [("doc1", 30.0), ("doc2", 20.0), ("doc3", 10.0)]
    print(f"\nOriginal 6: {sample_results_6}")
    scaled_results_6 = minmax_scale(sample_results_6)
    print(f"Scaled 6: {scaled_results_6}")
    # Expected: [('doc1', 1.0), ('doc2', 0.5), ('doc3', 0.0)]

    # Test case 7: Negative scores (assuming higher is still better)
    sample_results_7 = [("neg1", -5.0), ("neg2", -10.0), ("neg3", 0.0)]
    # min_score = -10, max_score = 0
    # neg1: (-5 - (-10)) / (0 - (-10)) = 5 / 10 = 0.5
    # neg2: (-10 - (-10)) / (0 - (-10)) = 0 / 10 = 0.0
    # neg3: (0 - (-10)) / (0 - (-10)) = 10 / 10 = 1.0
    print(f"\nOriginal 7: {sample_results_7}")
    scaled_results_7 = minmax_scale(sample_results_7)
    print(f"Scaled 7: {scaled_results_7}")
    # Expected: [('neg1', 0.5), ('neg2', 0.0), ('neg3', 1.0)]

    print("\nTesting minmax_scale complete.")


def ensemble_merge(
    results_semantic: List[Tuple[str, float]],
    results_tfidf: List[Tuple[str, float]],
    results_fts: List[Tuple[str, float]],
    weights: Dict[str, float],
    top_k: int = 10,
    min_intersection_results: int = 1
) -> List[Tuple[str, float]]:
    """
    Merges results from semantic, TF-IDF, and FTS searches using weighted scoring
    after Min-Max scaling. If the number of documents common to all three result
    sets is below 'min_intersection_results', it falls back to returning the
    top_k results from the normalized semantic search.

    Args:
        results_semantic: List of (doc_id, score) from semantic search.
                          Assumed to be sorted by relevance (higher score is better).
        results_tfidf: List of (doc_id, score) from TF-IDF search.
        results_fts: List of (doc_id, score) from FTS search.
        weights: Dictionary with weights for each search type,
                 e.g., {'semantic': 0.6, 'tfidf': 0.2, 'fts': 0.2}.
        top_k: The number of top results to return.
        min_intersection_results: Minimum number of common documents required
                                  to proceed with intersection logic. If below
                                  this, fallback to semantic search results.

    Returns:
        A list of (doc_id, score) tuples, sorted by score in descending order,
        limited to top_k. Score is either aggregated or normalized semantic.
        Returns an empty list if all input lists are empty, or if semantic
        results are empty during fallback.
    """
    if not results_semantic and not results_tfidf and not results_fts:
        print("Ensemble merge: All input result lists are empty. Returning empty list.")
        return []

    # Normalize scores for each result set
    # Note: minmax_scale preserves the order of items if their original scores are distinct,
    # or if all scores are the same (all become 1.0).
    # If results_semantic is already sorted by relevance, norm_semantic will also be.
    norm_semantic = minmax_scale(results_semantic)
    norm_tfidf = minmax_scale(results_tfidf)
    norm_fts = minmax_scale(results_fts)

    # Handle cases where some lists might be empty after normalization (if they were initially empty)
    # This check is more robust for the fallback logic.
    if not norm_semantic and not norm_tfidf and not norm_fts:
         print("Ensemble merge: All input lists were empty initially or after normalization. Returning empty list.")
         return []


    # Convert normalized results to dictionaries for easy lookup
    norm_semantic_map = {doc_id: score for doc_id, score in norm_semantic}
    norm_tfidf_map = {doc_id: score for doc_id, score in norm_tfidf}
    norm_fts_map = {doc_id: score for doc_id, score in norm_fts}

    # Get sets of doc_ids from each original (or normalized) result list
    semantic_ids = set(norm_semantic_map.keys()) if norm_semantic else set()
    tfidf_ids = set(norm_tfidf_map.keys()) if norm_tfidf else set()
    fts_ids = set(norm_fts_map.keys()) if norm_fts else set()

    common_doc_ids = set()
    # Intersection logic only makes sense if all sources provided results
    if norm_semantic and norm_tfidf and norm_fts:
        common_doc_ids = semantic_ids.intersection(tfidf_ids).intersection(fts_ids)
        print(f"Ensemble merge: Found {len(common_doc_ids)} common document IDs across all three sources.")
    elif norm_semantic: # If some lists are empty, no three-way intersection is possible
        print("Ensemble merge: Not all three search methods provided results. Cannot compute three-way intersection.")
        # Fallback will be triggered if len(common_doc_ids) is less than min_intersection_results
    else: # Semantic search itself is empty, and it's our primary fallback
        print("Ensemble merge: Semantic search results are empty. Cannot proceed with merge or fallback to semantic.")
        return []


    if len(common_doc_ids) < min_intersection_results:
        print(f"Ensemble merge: Number of common documents ({len(common_doc_ids)}) is less than min_intersection_results ({min_intersection_results}).")
        if not norm_semantic:
            print("Ensemble merge: Fallback triggered, but semantic search results are also empty. Returning empty list.")
            return []
        print(f"Ensemble merge: Falling back to top {top_k} normalized semantic search results.")
        # norm_semantic is already sorted by original semantic relevance due to minmax_scale behavior on sorted input.
        # If semantic scores were identical, they all become 1.0, order is preserved.
        # If scores were different, normalization preserves order.
        return norm_semantic[:top_k]
    else:
        print(f"Ensemble merge: Proceeding with aggregation for {len(common_doc_ids)} common document IDs.")
        aggregated_scores_list: List[Tuple[str, float]] = []
        for doc_id in common_doc_ids:
            aggregated_score = 0.0
            aggregated_score += norm_semantic_map.get(doc_id, 0.0) * weights.get('semantic', 0.0)
            aggregated_score += norm_tfidf_map.get(doc_id, 0.0) * weights.get('tfidf', 0.0)
            aggregated_score += norm_fts_map.get(doc_id, 0.0) * weights.get('fts', 0.0)
            aggregated_scores_list.append((doc_id, aggregated_score))

        # Sort by aggregated score in descending order
        aggregated_scores_list.sort(key=lambda x: x[1], reverse=True)
        return aggregated_scores_list[:top_k]


if __name__ == "__main__":
    # ... (previous minmax_scale tests remain here) ...
    print("\nRunning minmax_scale tests again for context before ensemble_merge tests:")
    # Test case 1: Basic scaling
    sample_results_1 = [("doc1", 10.0), ("doc2", 20.0), ("doc3", 30.0)]
    print(f"\nOriginal 1: {sample_results_1}")
    scaled_results_1 = minmax_scale(sample_results_1)
    print(f"Scaled 1: {scaled_results_1}")
    # Expected: [('doc1', 0.0), ('doc2', 0.5), ('doc3', 1.0)]

    # Test case 2: Scores include 0
    sample_results_2 = [("docA", 0.0), ("docB", 5.0), ("docC", 10.0)]
    print(f"\nOriginal 2: {sample_results_2}")
    scaled_results_2 = minmax_scale(sample_results_2)
    print(f"Scaled 2: {scaled_results_2}")
    # Expected: [('docA', 0.0), ('docB', 0.5), ('docC', 1.0)]

    # Test case 3: All scores are the same
    sample_results_3 = [("docX", 5.0), ("docY", 5.0), ("docZ", 5.0)]
    print(f"\nOriginal 3: {sample_results_3}")
    scaled_results_3 = minmax_scale(sample_results_3)
    print(f"Scaled 3: {scaled_results_3}")
    # Expected: [('docX', 1.0), ('docY', 1.0), ('docZ', 1.0)]

    # Test case 4: Single item
    sample_results_4 = [("single_doc", 100.0)]
    print(f"\nOriginal 4: {sample_results_4}")
    scaled_results_4 = minmax_scale(sample_results_4)
    print(f"Scaled 4: {scaled_results_4}")
    # Expected: [('single_doc', 1.0)]

    # Test case 5: Empty list
    sample_results_5: List[Tuple[str, float]] = []
    print(f"\nOriginal 5: {sample_results_5}")
    scaled_results_5 = minmax_scale(sample_results_5)
    print(f"Scaled 5: {scaled_results_5}")
    # Expected: []

    # Test case 6: Scores in descending order
    sample_results_6 = [("doc1", 30.0), ("doc2", 20.0), ("doc3", 10.0)]
    print(f"\nOriginal 6: {sample_results_6}")
    scaled_results_6 = minmax_scale(sample_results_6)
    print(f"Scaled 6: {scaled_results_6}")
    # Expected: [('doc1', 1.0), ('doc2', 0.5), ('doc3', 0.0)]

    # Test case 7: Negative scores (assuming higher is still better)
    sample_results_7 = [("neg1", -5.0), ("neg2", -10.0), ("neg3", 0.0)]
    print(f"\nOriginal 7: {sample_results_7}")
    scaled_results_7 = minmax_scale(sample_results_7)
    print(f"Scaled 7: {scaled_results_7}")
    # Expected: [('neg1', 0.5), ('neg2', 0.0), ('neg3', 1.0)]
    print("\nTesting minmax_scale complete.")


    print("\n\nTesting ensemble_merge function...")

    # Test Case 1: Basic intersection and weighting
    sem_results1 = [("docA", 0.8), ("docB", 0.6), ("docC", 0.9)] # norm: docA 0.66, docB 0, docC 1
    tfidf_results1 = [("docA", 20.0), ("docB", 10.0), ("docC", 5.0)]# norm: docA 1, docB 0.33, docC 0
    fts_results1 = [("docA", -1.0), ("docB", -5.0), ("docC", -2.0)] # norm: docA 1, docB 0, docC 0.75 (using -rank, so higher is better)
                                                                # min=-5, max=-1. docA: (-1 - -5)/(4) = 1. docB: (-5 - -5)/(4)=0. docC: (-2 - -5)/(4)=0.75

    weights1 = {'semantic': 0.5, 'tfidf': 0.3, 'fts': 0.2}
    # docA: (0.66*0.5) + (1*0.3) + (1*0.2) = 0.33 + 0.3 + 0.2 = 0.83
    # docB: (0*0.5) + (0.33*0.3) + (0*0.2) = 0 + 0.099 + 0 = 0.099
    # docC: (1*0.5) + (0*0.3) + (0.75*0.2) = 0.5 + 0 + 0.15 = 0.65
    # Expected order: docA, docC, docB

    print(f"\nEnsemble Test 1:")
    merged1 = ensemble_merge(sem_results1, tfidf_results1, fts_results1, weights1, top_k=3)
    print(f"Merged 1: {merged1}")

    # Test Case 2: No common documents
    sem_results2 = [("docX", 0.8)]
    tfidf_results2 = [("docY", 20.0)]
    fts_results2 = [("docZ", -1.0)]
    print(f"\nEnsemble Test 2 (no common):")
    merged2 = ensemble_merge(sem_results2, tfidf_results2, fts_results2, weights1, top_k=3)
    print(f"Merged 2: {merged2}") # Expected: []

    # Test Case 3: One result list is empty - now triggers fallback if semantic is present
    print(f"\nEnsemble Test 3 (one list empty, e.g., tfidf):")
    # Fallback to semantic: norm_semantic1 = [('docC', 1.0), ('docA', 0.66...), ('docB', 0.0)]
    # Expected: top 3 from norm_semantic1
    merged3 = ensemble_merge(sem_results1, [], fts_results1, weights1, top_k=3, min_intersection_results=1)
    print(f"Merged 3: {merged3}")
    # Expected: (docC, 1.0), (docA, 0.66...), (docB, 0.0) if sem_results1 is sorted by score desc before norm.
    # Let's ensure sem_results1 is sorted for predictable fallback for testing:
    sem_results1_sorted = sorted(sem_results1, key=lambda x:x[1], reverse=True)
    # sem_results1_sorted = [('docC', 0.9), ('docA', 0.8), ('docB', 0.6)]
    # norm_semantic_sorted = [('docC', 1.0), ('docA', 0.66...), ('docB', 0.0)]
    merged3_sorted_sem_input = ensemble_merge(sem_results1_sorted, [], fts_results1, weights1, top_k=3, min_intersection_results=1)
    print(f"Merged 3 (sorted semantic input for fallback): {merged3_sorted_sem_input}")


    # Test Case 4: Different doc_ids, some overlap, but intersection < min_intersection_results
    # common1, common2 are in all. len(common_doc_ids) = 2
    # If min_intersection_results = 3, it should fallback.
    print(f"\nEnsemble Test 4 (partial overlap, intersection < min_intersection_results):")
    sem_results4 = [("common1", 0.9), ("common2", 0.8), ("sem_only", 0.7)] # norm: c1=1, c2=0.5, sem_only=0
    sem_results4_sorted = sorted(sem_results4, key=lambda x:x[1], reverse=True)
    tfidf_results4 = [("common1", 15.0), ("common2", 25.0), ("tfidf_only", 10.0)]
    fts_results4 = [("common1", -2.0), ("common2", -3.0), ("fts_only", -1.0)]
    merged4_fallback = ensemble_merge(sem_results4_sorted, tfidf_results4, fts_results4, weights1, top_k=2, min_intersection_results=3)
    print(f"Merged 4 (fallback expected): {merged4_fallback}")
    # Expected: top 2 from sem_results4_sorted (normalized) -> [('common1', 1.0), ('common2', 0.5)]

    # Test Case 5: All lists empty - should return empty
    print(f"\nEnsemble Test 5 (all empty):")
    merged5 = ensemble_merge([], [], [], weights1, top_k=3, min_intersection_results=1)
    print(f"Merged 5: {merged5}") # Expected: []

    # Test Case 6: Sufficient intersection, normal merge
    print(f"\nEnsemble Test 6 (sufficient intersection):")
    # Using sem_results1, tfidf_results1, fts_results1 from Test Case 1. Intersection size is 3.
    # min_intersection_results = 2. Should proceed with normal merge.
    merged6 = ensemble_merge(sem_results1, tfidf_results1, fts_results1, weights1, top_k=3, min_intersection_results=2)
    print(f"Merged 6: {merged6}")
    # Expected: same as Merged 1: docA, docC, docB (scores might differ slightly based on norm map if any list was empty)

    # Test Case 7: Fallback when semantic is empty (should return empty)
    print(f"\nEnsemble Test 7 (fallback with empty semantic):")
    merged7 = ensemble_merge([], tfidf_results1, fts_results1, weights1, top_k=3, min_intersection_results=1)
    print(f"Merged 7: {merged7}") # Expected: []


    print("\nEnsemble testing complete.")
