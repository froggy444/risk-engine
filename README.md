# risk-engine

A vectorized portfolio risk engine in Python. It prices a book of European options with Black-Scholes-Merton, computes Greeks, explains daily PnL by Greek, and runs historical-simulation VaR with full revaluation. Positions and results are stored as Parquet via Apache Arrow.

The focus is on how a production risk system is built: columnar data layout, a single vectorized pricing kernel shared by every calculation, incremental recomputation when one market input moves, bounded memory for scenario analysis, and tests that verify the math rather than just the code paths.

## Features

| Area | What it does |
|---|---|
| Pricing | Black-Scholes-Merton with continuous dividend yield; price, delta, gamma, vega and theta in one pass |
| Risk engine | Full book revaluation, plus incremental updates that reprice only positions on the underlying that moved |
| PnL attribution | Splits actual PnL into delta, gamma, vega and theta, and reports the unexplained residual |
| VaR | Historical simulation with full revaluation, 99% VaR and Expected Shortfall, block-processed scenarios |
| Storage | Portfolio and position-level risk written to and read from Parquet (zstd) through Arrow |

## Performance

Measured with `python benchmarks/bench.py` on a standard cloud VM (single thread). Numbers vary by machine.

| Stage | 10,000 positions | 100,000 positions |
|---|---|---|
| Full revaluation + all Greeks | ~1.5 ms | ~12 ms |
| Incremental update, one spot move | ~0.1 ms | ~0.3 ms |
| PnL attribution | ~3 ms | ~28 ms |
| Historical VaR, 250 full-reval scenarios | ~0.2 s | ~3–4 s |

## Quick start

```bash
git clone https://github.com/<your-username>/risk-engine.git
cd risk-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

risk-engine                    # demo on a synthetic 10K-position book
pytest                         # 28 tests
python benchmarks/bench.py     # timings
```

Library usage:

```python
from risk_engine import RiskEngine, attribute_pnl, historical_var
from risk_engine.data import make_history, make_market, make_portfolio

market = make_market(n_underlyings=50)
book = make_portfolio(market, n_positions=10_000)

engine = RiskEngine(book, market)
print(engine.result.totals)  # value, delta, gamma, vega, theta

engine.update_spot(underlying=7, spot=101.25)  # reprices only positions on underlying 7

spot_ret, vol_chg = make_history(market, n_days=250)
var = historical_var(book, market, spot_ret, vol_chg, confidence=0.99)
print(var.var, var.expected_shortfall)
```

## Design

**Columnar portfolio.** The book is a struct of NumPy arrays (strike, expiry, quantity and so on) instead of a list of position objects. Every calculation is one vectorized expression over contiguous memory, and the layout maps one-to-one onto an Arrow table, so Parquet reads and writes are column copies with no per-row object construction.

**One pricing kernel.** `black_scholes()` broadcasts over any shape. The same function prices one option, a 10K book, or a (scenarios × positions) matrix for VaR. Shared intermediates (d1, d2, discount factors, N(d1), n(d1)) are computed once and reused for price and all Greeks. Calls and puts are handled in the same pass with a sign trick, since a put's terms are N(-d1) and N(-d2).

**Incremental recomputation.** At construction the portfolio precomputes position indices per underlying. When one spot or vol moves, the engine reprices only those positions and patches cached totals by the difference, so cost scales with the positions affected (about 2% of the book here), not with book size. A test checks that a long sequence of incremental updates matches a from-scratch revaluation to floating-point tolerance.

**PnL attribution.** Actual PnL is a full revaluation between two market snapshots, including one day of time decay. Explained PnL is the second-order expansion around start-of-day Greeks:

```
dV ≈ delta·dS + ½·gamma·dS² + vega·dσ + theta·dt
```

The unexplained residual holds cross and higher-order terms (vanna, volga, speed). Across 250 simulated days the median residual is under 1% of actual PnL, rising on large-move days as you'd expect, and a test confirms it shrinks faster than linearly as moves get smaller, which is what a correct second-order expansion should do.

**Historical VaR.** Each historical day provides spot log-returns and vol changes for every underlying. The book is fully repriced in each scenario rather than approximated with delta-gamma, so it captures the nonlinearity of options. Scenarios are processed in blocks, so peak memory is `block_size × n_positions` regardless of history length. The synthetic history uses a fat-tailed (Student-t) market factor with vol moving opposite to spot, so down days come with vol spikes, as in real equity markets.

## Testing

`pytest` runs 28 tests, and `mypy --strict` and `ruff` run clean. The tests check correctness of the math, not just that the code runs:

- Price matches the textbook reference value (Hull: ATM one-year call, 20% vol, 5% rate = 10.4506).
- Put-call parity holds across 500 random options.
- Delta, gamma, vega and theta match central finite-difference estimates.
- Incremental updates match full revaluation exactly, and only affected positions change.
- Attribution components sum to explained PnL, and the residual behaves as a higher-order term.
- Vectorized VaR matches a naive per-scenario loop, results don't depend on block size, and ES ≥ VaR.
- Parquet round-trips preserve every column exactly.

## Project layout

```
src/risk_engine/
  pricing.py       Black-Scholes kernel and analytic Greeks
  portfolio.py     Columnar book, per-underlying index for incremental updates
  market.py        Immutable market snapshot (spot, vol, dividend yield, rate)
  engine.py        RiskEngine: full and incremental revaluation, aggregation
  attribution.py   PnL explain by Greek
  var.py           Historical VaR and Expected Shortfall
  io.py            Arrow/Parquet persistence
  data.py          Reproducible synthetic books and market history
  cli.py           Demo entry point
tests/             Pricing, engine, attribution, VaR and IO tests
benchmarks/        Timing harness
```

## Limitations and next steps

- Flat vol per underlying. A volatility surface would replace `vol[underlying]` with interpolation over strike and expiry.
- European options only. American exercise would need a lattice or least-squares Monte Carlo pricer.
- Single-threaded NumPy. VaR scenario blocks are independent, so they parallelize naturally across processes or with Numba.
- Synthetic data. A loader for real positions and market history would sit alongside `io.py`.

## License

MIT
