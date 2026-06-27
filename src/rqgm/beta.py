"""Beta-posterior best-belief scoring (paper Section 4, Eq. for ``BB_epsilon``).

Pure standard-library implementation of the regularized incomplete Beta
function and its inverse, used to compute the conservative best-belief score

    BB_epsilon(a) = I^{-1}_epsilon(1 + S, 1 + F)

i.e. the ``epsilon``-quantile of the ``Beta(1 + S, 1 + F)`` posterior over the
success probability of a candidate with ``S`` successes and ``F`` failures.

No third-party dependencies: this keeps the core library portable and trivial
to vendor.
"""

from __future__ import annotations

import math

__all__ = [
    "regularized_incomplete_beta",
    "beta_ppf",
    "best_belief",
    "posterior_mean",
]

_TINY = 1e-30


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued fraction for the incomplete Beta function (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return ``I_x(a, b)``, the regularized incomplete Beta function.

    This is the CDF of a ``Beta(a, b)`` distribution evaluated at ``x``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    # Use the continued fraction in whichever region converges fastest.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Inverse CDF (quantile function) of ``Beta(a, b)`` via bisection.

    ``regularized_incomplete_beta`` is monotone in ``x`` so a simple bisection
    on ``[0, 1]`` is robust and dependency-free.
    """
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def best_belief(successes: int, failures: int, epsilon: float = 0.05) -> float:
    """Paper ``BB_epsilon``: the ``epsilon``-quantile of ``Beta(1 + S, 1 + F)``.

    A conservative lower bound on the candidate's success probability that
    rewards evidence: ``best_belief(0, 0) == epsilon`` and the score rises with
    successes, falls with failures, and tightens with sample size.
    """
    return beta_ppf(epsilon, 1.0 + successes, 1.0 + failures)


def posterior_mean(successes: int, failures: int) -> float:
    """Posterior mean of ``Beta(1 + S, 1 + F)`` (Laplace-smoothed rate)."""
    return (1.0 + successes) / (2.0 + successes + failures)
