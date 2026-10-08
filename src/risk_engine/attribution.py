"""Daily PnL attribution ("PnL explain") by Greek.

Actual PnL is a full revaluation: V(t1, market_1) - V(t0, market_0).
Explained PnL is a second-order Taylor expansion around the start-of-day Greeks:

    dV ~= delta * dS + 0.5 * gamma * dS^2 + vega * d_sigma + theta * dt

The unexplained residual captures higher-order and cross terms (vanna, volga,
speed) and is the number a risk team watches: a large residual means the Greeks
no longer describe the book well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from risk_engine.market import MarketData
from risk_engine.portfolio import Portfolio
from risk_engine.pricing import black_scholes

FloatArray = NDArray[np.float64]

COMPONENTS = ("delta", "gamma", "vega", "theta")


@dataclass(frozen=True)
class AttributionResult:
    """Per-position PnL split into Greek components, with totals."""

    actual: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray

    @property
    def explained(self) -> FloatArray:
        return self.delta + self.gamma + self.vega + self.theta

    @property
    def unexplained(self) -> FloatArray:
        return self.actual - self.explained

    def totals(self) -> dict[str, float]:
        out = {c: float(getattr(self, c).sum()) for c in COMPONENTS}
        out["explained"] = float(self.explained.sum())
        out["unexplained"] = float(self.unexplained.sum())
        out["actual"] = float(self.actual.sum())
        return out


def attribute_pnl(
    portfolio: Portfolio,
    market_t0: MarketData,
    market_t1: MarketData,
    dt: float,
) -> AttributionResult:
    """Attribute PnL between two market snapshots separated by ``dt`` years."""
    und = portfolio.underlying
    scale = portfolio.scale

    g0 = black_scholes(
        market_t0.spot[und],
        portfolio.strike,
        portfolio.expiry,
        market_t0.vol[und],
        market_t0.rate,
        market_t0.div_yield[und],
        portfolio.is_call,
    )
    rolled = portfolio.roll(dt)
    v1 = black_scholes(
        market_t1.spot[und],
        rolled.strike,
        rolled.expiry,
        market_t1.vol[und],
        market_t1.rate,
        market_t1.div_yield[und],
        rolled.is_call,
    ).price

    d_spot = (market_t1.spot - market_t0.spot)[und]
    d_vol = (market_t1.vol - market_t0.vol)[und]

    return AttributionResult(
        actual=(v1 - g0.price) * scale,
        delta=g0.delta * d_spot * scale,
        gamma=0.5 * g0.gamma * d_spot * d_spot * scale,
        vega=g0.vega * d_vol * scale,
        theta=g0.theta * dt * scale,
    )
