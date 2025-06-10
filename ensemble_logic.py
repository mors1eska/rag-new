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

    # NOTE on doc_id consistency:
    # The effectiveness of the intersection and subsequent merging depends heavily on
    # the consistency of doc_ids generated by different search methods.
    # For example, semantic search and FTS might produce chunk-specific IDs (e.g., "path/doc.pdf_page_1_chunk_0"),
    # while TF-IDF search might currently use document-level IDs (e.g., "path/doc.pdf").
    # If doc_id granularity differs, the intersection might be smaller than expected
    # or not capture true overlaps at the chunk level for all methods.
    # This implementation assumes doc_ids are directly comparable for intersection.
    # Future improvements might involve a doc_id canonicalization step or ensuring
    # all search methods produce IDs of similar granularity if chunk-level merging is desired.

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
