import pytest
from tag_research.class_ap import average_precision

def test_noninterpolated_reference_example():
    # Standard mathematical metric fixture; not model evaluation data.
    assert average_precision([0, 0, 1, 1], [.1, .4, .35, .8]) == pytest.approx(5 / 6)

def test_tied_thresholds_do_not_depend_on_input_order():
    assert average_precision([1, 0], [.5, .5]) == .5
    assert average_precision([0, 1], [.5, .5]) == .5

def test_missing_positive_class_is_explicitly_undefined():
    assert average_precision([0, 0], [.1, .2]) is None

def test_perfect_ranking():
    assert average_precision([0, 1, 1], [.1, .9, .8]) == 1
