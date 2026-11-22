"""Reciprocal rank fusion: pure logic, no data needed."""

from papertrail.fusion import RRF_K, rrf


def test_item_in_both_lists_beats_item_in_one():
    fused = [item for item, _ in rrf([1, 2, 3], [3, 4, 5])]
    assert fused[0] == 3 or fused[0] == 1  # 3 is 3rd + 1st, 1 is 1st only: 3 must win
    assert fused.index(3) < fused.index(1)


def test_score_is_sum_of_reciprocal_ranks():
    scores = dict(rrf([10, 20], [20]))
    assert scores[10] == 1 / (RRF_K + 1)
    assert scores[20] == 1 / (RRF_K + 2) + 1 / (RRF_K + 1)


def test_every_item_appears_once_best_first():
    fused = rrf([1, 2, 3], [3, 4])
    items = [i for i, _ in fused]
    assert sorted(items) == [1, 2, 3, 4] and len(items) == len(set(items))
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_ties_keep_the_first_list_first():
    # 1 is 1st in list A only; 9 is 1st in list B only: equal scores, A's item stays first
    assert [i for i, _ in rrf([1], [9])] == [1, 9]


def test_empty_inputs():
    assert rrf() == [] and rrf([], []) == []
    assert [i for i, _ in rrf([5, 6], [])] == [5, 6]
