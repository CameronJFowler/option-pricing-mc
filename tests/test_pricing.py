import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing import black_scholes_price, monte_carlo_price  # noqa: E402

S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


def test_known_call_value():
    # Standard textbook case, 100/100/5%/20%/1y.
    assert black_scholes_price(S, K, r, sigma, T, "call") == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    c = black_scholes_price(S, K, r, sigma, T, "call")
    p = black_scholes_price(S, K, r, sigma, T, "put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


def test_zero_vol_is_discounted_intrinsic():
    assert black_scholes_price(S, 90.0, r, 0.0, T, "call") == pytest.approx(
        S - 90.0 * np.exp(-r * T), abs=1e-10)


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes_price(S, K, r, sigma, T, "straddle")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_matches_closed_form(option_type):
    """The estimate should land within 4 standard errors of the true value.
    Fixed seed, so this either passes always or is a genuine bug.
    """
    rng = np.random.default_rng(12345)
    mc = monte_carlo_price(S, K, r, sigma, T, 200_000, option_type, rng=rng)
    bs = black_scholes_price(S, K, r, sigma, T, option_type)
    assert abs(mc.price - bs) < 4 * mc.stderr


def test_standard_error_shrinks_as_sqrt_n():
    rng = np.random.default_rng(7)
    small = monte_carlo_price(S, K, r, sigma, T, 10_000, rng=rng)
    large = monte_carlo_price(S, K, r, sigma, T, 160_000, rng=rng)
    ratio = small.stderr / large.stderr
    # 16x the samples should be 4x tighter. Allow 20% either way.
    assert 3.2 < ratio < 4.8


def test_antithetic_reduces_variance():
    plain = monte_carlo_price(S, K, r, sigma, T, 100_000,
                              rng=np.random.default_rng(1))
    anti = monte_carlo_price(S, K, r, sigma, T, 100_000, antithetic=True,
                             rng=np.random.default_rng(1))
    assert anti.stderr < plain.stderr
