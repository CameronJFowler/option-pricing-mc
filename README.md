# mc-option-pricing

Pricing European options two ways and checking they agree: the Black-Scholes formula,
and Monte Carlo simulation of the terminal spot price. The point of the repo is the
comparison, not either method on its own. Black-Scholes gives the exact answer, so it
works as ground truth for the simulation, and the simulation is the thing you can
actually extend to payoffs that have no closed form.

## The maths

Under the risk-neutral measure the spot at expiry is

```
S_T = S * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z),   Z ~ N(0, 1)
```

Draw a few hundred thousand `Z`, work out the payoff for each, average them, discount
back at `exp(-rT)`. That average is the price. Because it is a sample mean its error
shrinks with the square root of the number of draws, which is slow: 100 times the
compute buys 10 times the accuracy. `monte_carlo_price` returns the standard error
alongside the estimate for exactly that reason. A price with no error bar on it is not
telling you much.

There is a variance reduction option too. With `antithetic=True` each draw `Z` is paired
with `-Z` and the two payoffs are averaged before the sample mean is taken. The pair is
negatively correlated, so the variance of the average drops without needing more draws.
On the base case it cuts the standard error by about 30%.

## Running it

```bash
pip install -r requirements.txt
python convergence_plot.py
```

That prints the closed-form price, the Monte Carlo price with and without antithetic
sampling, and plots the estimates against sample size with a two-standard-error band.
On the default case, spot 100, strike 100, 5% rate, 20% vol, one year:

```
Black-Scholes      10.4506
Monte Carlo        10.4281  (se 0.0465)
  with antithetic  10.4810  (se 0.0329)
  variance cut     29%
```

The convergence curve runs an independent simulation at each sample size rather than one
cumulative running average. It is more work, but the vertical spread of the points at a
given `n` is then the real sampling distribution of the estimator, which is the thing
worth looking at. A cumulative average looks smoother and is misleading about it.

## Tests

```bash
pytest tests -q
```

Eight checks. The interesting ones:

- The closed form matches the textbook value of 10.4506 for the standard case.
- Put-call parity holds exactly, which catches sign errors in `d1` and `d2`.
- The Monte Carlo estimate is within four standard errors of the closed form, for both
  calls and puts. Seeded, so a failure is a real bug rather than bad luck.
- Sixteen times the samples gives roughly four times the precision, confirming the
  square-root behaviour rather than assuming it.
- Antithetic sampling produces a smaller standard error than plain sampling on the same
  seed.

## Files

```
pricing.py            black_scholes_price, monte_carlo_price, convergence
convergence_plot.py   the comparison plot and the printed summary
tests/test_pricing.py
```

## What this does not do

Only European calls and puts on a non-dividend-paying underlying, priced off a single
terminal draw. There is no path simulation, so anything path-dependent (barriers, Asians,
American exercise) is out of reach as written. Volatility is a constant, which is the
assumption that makes Black-Scholes wrong in practice and is why real desks quote a
surface instead of one number.

The natural next steps are stepping the path with an Euler or exact scheme so barrier
options become possible, adding control variates on top of the antithetic pairs, and
Longstaff-Schwartz for early exercise.
