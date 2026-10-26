"""The Clopper-Pearson exact lower bound on a success rate (ruling D6).

Implemented here on the standard library rather than imported from SciPy. The
core has no runtime dependencies, and one function is not worth adding a
compiled stack to the image and the lock. The bound is defined by a binomial
tail, so it is computed from that tail directly, which the tests can check in
exact rational arithmetic.
"""

from __future__ import annotations

import math

#: Relative amount the bound is shaded down by, so float error errs toward failing.
#: One part in a billion: far above the arithmetic's error, far below any
#: difference a threshold could mean.
_MARGIN = 1e-9


def clopper_pearson_lower(successes: int, episodes: int, *, confidence: float = 0.95) -> float:
    """Lower endpoint of the two-sided Clopper-Pearson interval at `confidence`.

    For k successes in n episodes this is the p_L that solves
    P(X >= k | n, p_L) = (1 - confidence) / 2, and 0 when k = 0. At the default
    it is the lower end of the exact 95% interval, so the probability that the
    true rate is below it is at most 2.5%.

    The root is found by bisection down to adjacent floats, then shaded down by
    `_MARGIN`. Float error in the target and the tail (about 1e-15 relative at
    n = 1, growing with n) could otherwise overstate the bound in its last
    digits; shading by a margin far above that error means rounding can only
    understate the bound, so a gate never passes on a rounding error.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if episodes < 0:
        raise ValueError(f"episodes must be non-negative, got {episodes}")
    if not 0 <= successes <= episodes:
        raise ValueError(f"successes must be in [0, {episodes}], got {successes}")
    if successes == 0:
        return 0.0

    log_target = math.log((1.0 - confidence) / 2.0)
    # log C(n, i) for i = k..n, from exact integers. Computed once, outside the search.
    coefficient = math.comb(episodes, successes)
    log_coefficients = []
    for i in range(successes, episodes + 1):
        log_coefficients.append(math.log(coefficient))
        coefficient = coefficient * (episodes - i) // (i + 1)

    def log_upper_tail(p: float) -> float:
        """log P(X >= k | n, p), summed in log space so a large n cannot overflow."""
        log_p, log_q = math.log(p), math.log1p(-p)
        terms = [
            log_c + i * log_p + (episodes - i) * log_q
            for i, log_c in enumerate(log_coefficients, start=successes)
        ]
        peak = max(terms)
        return peak + math.log(math.fsum(math.exp(term - peak) for term in terms))

    # The tail rises with p from 0 at p = 0 to 1 at p = 1. Invariant: the tail at
    # `low` is below the target and the tail at `high` is not.
    low, high = 0.0, 1.0
    while True:
        middle = (low + high) / 2.0
        if middle in (low, high):
            return low * (1.0 - _MARGIN)
        if log_upper_tail(middle) < log_target:
            low = middle
        else:
            high = middle
