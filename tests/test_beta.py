"""Beta best-belief math (paper Section 4)."""

from rqgm.beta import (
    best_belief,
    beta_ppf,
    posterior_mean,
    regularized_incomplete_beta,
)


def test_best_belief_uninformed_prior_is_epsilon():
    # Beta(1, 1) is uniform, so its 5% quantile is 0.05.
    assert abs(best_belief(0, 0) - 0.05) < 1e-6


def test_best_belief_monotone_in_evidence():
    assert best_belief(9, 1) > best_belief(7, 3) > best_belief(3, 7)


def test_best_belief_tightens_with_sample_size():
    # Same 0.9 success rate, 10x the evidence -> a strictly higher lower bound.
    assert best_belief(90, 10) > best_belief(9, 1)


def test_ppf_cdf_roundtrip():
    for a, b, q in [(2.0, 5.0, 0.3), (10.0, 2.0, 0.05), (4.0, 8.0, 0.7)]:
        x = beta_ppf(q, a, b)
        assert abs(regularized_incomplete_beta(x, a, b) - q) < 1e-6


def test_incomplete_beta_bounds_and_symmetry():
    assert regularized_incomplete_beta(0.0, 2.0, 3.0) == 0.0
    assert regularized_incomplete_beta(1.0, 2.0, 3.0) == 1.0
    # I_x(a, b) == 1 - I_{1-x}(b, a)
    left = regularized_incomplete_beta(0.3, 2.0, 3.0)
    right = 1.0 - regularized_incomplete_beta(0.7, 3.0, 2.0)
    assert abs(left - right) < 1e-9


def test_posterior_mean():
    assert posterior_mean(0, 0) == 0.5
    assert abs(posterior_mean(9, 1) - 10.0 / 12.0) < 1e-12
