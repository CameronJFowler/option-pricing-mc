"""European option pricing: closed form Black-Scholes and Monte Carlo.

Both functions take the same five parameters:

    S      spot price
    K      strike
    r      risk-free rate, annualised, continuously compounded
    sigma  volatility, annualised
    T      time to expiry in years
"""
from collections import namedtuple

import numpy as np
from scipy.stats import norm

MCResult = namedtuple("MCResult", "price stderr")


def black_scholes_price(S, K, r, sigma, T, option_type="call"):
    """Closed-form Black-Scholes price for a European option."""
    if T <= 0:
        intrinsic = S - K if option_type == "call" else K - S
        return max(intrinsic, 0.0)
    if sigma <= 0:
        fwd = S - K * np.exp(-r * T)
        return max(fwd, 0.0) if option_type == "call" else max(-fwd, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def monte_carlo_price(S, K, r, sigma, T, n_simulations=100_000,
                      option_type="call", antithetic=False, rng=None):
    """Price a European option by simulating terminal spot under the risk-neutral
    measure. Returns the discounted mean payoff and its standard error.

    S_T = S * exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1)

    With antithetic=True each draw Z is paired with -Z. The two payoffs are
    negatively correlated, so averaging the pair before taking the sample mean
    cuts the variance for the same number of draws.
    """
    rng = np.random.default_rng() if rng is None else rng
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T)

    def payoff(Z):
        ST = S * np.exp(drift + diffusion * Z)
        return np.maximum(ST - K, 0.0) if option_type == "call" \
            else np.maximum(K - ST, 0.0)

    if antithetic:
        half = n_simulations // 2
        Z = rng.standard_normal(half)
        samples = 0.5 * (payoff(Z) + payoff(-Z))
    else:
        Z = rng.standard_normal(n_simulations)
        samples = payoff(Z)

    discount = np.exp(-r * T)
    price = discount * samples.mean()
    stderr = discount * samples.std(ddof=1) / np.sqrt(samples.size)
    return MCResult(price, stderr)


def convergence(S, K, r, sigma, T, max_sims=100_000, step=1_000, **kwargs):
    """Run an independent Monte Carlo at each sample size from step to max_sims.

    Independent runs rather than one cumulative average, so the spread of the
    points at a given n is the actual sampling distribution of the estimator
    rather than one correlated path through it.
    """
    sizes = list(range(step, max_sims + step, step))
    results = [monte_carlo_price(S, K, r, sigma, T, n, **kwargs) for n in sizes]
    return sizes, [x.price for x in results], [x.stderr for x in results]
