"""Plot Monte Carlo prices against the Black-Scholes value as n grows."""
import matplotlib.pyplot as plt
import numpy as np

from pricing import black_scholes_price, convergence, monte_carlo_price

S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

if __name__ == "__main__":
    bs = black_scholes_price(S, K, r, sigma, T)
    sizes, prices, errs = convergence(S, K, r, sigma, T)

    prices = np.array(prices)
    errs = np.array(errs)

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, prices - 2 * errs, prices + 2 * errs, alpha=0.2,
                     label="Monte Carlo, 2 standard errors")
    plt.plot(sizes, prices, linewidth=1, label="Monte Carlo price")
    plt.axhline(bs, linestyle="--", color="black", label="Black-Scholes price")
    plt.xlabel("Number of simulations")
    plt.ylabel("Option price")
    plt.title("Monte Carlo convergence to Black-Scholes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plain = monte_carlo_price(S, K, r, sigma, T, 100_000)
    anti = monte_carlo_price(S, K, r, sigma, T, 100_000, antithetic=True)

    print(f"Black-Scholes      {bs:.4f}")
    print(f"Monte Carlo        {plain.price:.4f}  (se {plain.stderr:.4f})")
    print(f"  with antithetic  {anti.price:.4f}  (se {anti.stderr:.4f})")
    print(f"  variance cut     {(1 - anti.stderr / plain.stderr) * 100:.0f}%")
