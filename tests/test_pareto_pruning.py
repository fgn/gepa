from gepa.gepa_utils import is_dominated, remove_dominated_programs


def test_is_dominated_requires_single_common_dominator():
    # y appears in three fronts; their intersection (excluding y) is empty.
    fronts = [
        {0, 1, 2, 3, 4},
        {1, 3, 4},
        {0, 3},
    ]
    y = 3
    others = {0, 1, 2, 4}
    assert is_dominated(y, others, fronts) is False


def test_is_dominated_true_when_single_common_exists():
    fronts = [
        {1, 4},  # both 1 and 4; remove y=1 candidate dominator is 4
        {1, 2, 4},  # 4 still present
        {0, 1, 4},  # 4 still present -> intersection={4}
    ]
    y = 1
    others = {0, 2, 4}
    assert is_dominated(y, others, fronts) is True


def test_remove_dominated_programs_does_not_remove_without_single_dominator():
    fronts = [
        {0, 1, 2, 3, 4},
        {1, 3, 4},
        {0, 3},
    ]
    # Scores aren't critical; they just order iteration.
    scores = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    new_fronts = remove_dominated_programs(fronts, scores=scores)
    # Program 3 should still appear in all fronts where it used to appear.
    assert {3}.issubset(new_fronts[0]) and {3}.issubset(new_fronts[1]) and {3}.issubset(new_fronts[2])
