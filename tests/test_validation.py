import pytest
from tag_research.validation import paired_interval

def test_identical_paired_results_have_zero_interval():
    # Mathematical regression fixture, not generated model observations.
    result = paired_interval([0, 0, 0], resamples=100)
    assert result['bootstrap_95_percentile_interval'] == [0, 0]
    assert result['tied_images'] == 3

def test_constant_difference_and_pair_counts():
    result = paired_interval([.25, .25, .25], resamples=100)
    assert result['mean_difference'] == .25
    assert result['bootstrap_95_percentile_interval'] == [.25, .25]
    assert result['improved_images'] == 3

@pytest.mark.parametrize('values', [[], [1], [float('nan'), 1]])
def test_invalid_observations_rejected(values):
    with pytest.raises(ValueError):
        paired_interval(values)
