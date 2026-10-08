"""Reproducible synthetic books and market history for demos, tests and benchmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from risk_engine.market import MarketData
from risk_engine.portfolio import Portfolio

FloatArray = NDArray[np.float64]


def make_market(n_underlyings: int = 50, seed: int = 7) -> MarketData:
    rng = np.random.default_rng(seed)
    return MarketData(
        spot=rng.uniform(20.0, 500.0, n_underlyings),
        vol=rng.uniform(0.15, 0.60, n_underlyings),
        div_yield=rng.uniform(0.0, 0.03, n_underlyings),
        rate=0.04,
    )


def make_portfolio(market: MarketData, n_positions: int = 10_000, seed: int = 11) -> Portfolio:
    rng = np.random.default_rng(seed)
    und = rng.integers(0, market.n_underlyings, n_positions).astype(np.int64)
    moneyness = rng.lognormal(mean=0.0, sigma=0.15, size=n_positions)
    return Portfolio(
        position_id=np.arange(n_positions, dtype=np.int64),
        underlying=und,
        strike=np.round(market.spot[und] * moneyness, 2),
        expiry=rng.uniform(7 / 365, 2.0, n_positions),
        is_call=rng.random(n_positions) < 0.5,
        quantity=rng.integers(-50, 51, n_positions).astype(np.float64),
        multiplier=np.full(n_positions, 100.0),
    )


def make_history(
    market: MarketData,
    n_days: int = 250,
    seed: int = 23,
) -> tuple[FloatArray, FloatArray]:
    """Correlated daily spot log-returns and vol changes via a one-factor model.

    Vol moves are negatively correlated with the market factor (the leverage effect),
    so down days come with vol spikes, as in real equity markets.
    """
    rng = np.random.default_rng(seed)
    n = market.n_underlyings
    daily_vol = market.vol / np.sqrt(252.0)
    beta = rng.uniform(0.6, 1.4, n)
    market_factor = rng.standard_t(df=4, size=(n_days, 1)) * 0.01
    idio = rng.standard_normal((n_days, n)) * daily_vol * 0.6
    spot_returns = beta * market_factor + idio
    vol_changes = -0.8 * market_factor + rng.standard_normal((n_days, n)) * 0.005
    return spot_returns, vol_changes
