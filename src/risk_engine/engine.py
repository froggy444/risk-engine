"""Stateful risk engine with full and incremental revaluation.

A full ``compute()`` revalues the whole book in one vectorized pass. When a single
market input moves (one spot or one vol), ``update_spot`` / ``update_vol`` reprice
only the positions on that underlying and patch the cached totals with the delta,
so the cost scales with the positions affected, not the size of the book.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from risk_engine.market import MarketData
from risk_engine.portfolio import Portfolio
from risk_engine.pricing import black_scholes

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_FIELDS = ("value", "delta", "gamma", "vega", "theta")


@dataclass(frozen=True)
class RiskResult:
    """Position-level risk (already scaled by quantity x multiplier) plus book totals.

    ``delta`` and ``gamma`` are in underlying-unit terms (shares-equivalent); multiply
    delta by spot for dollar delta.
    """

    value: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    totals: dict[str, float]

    def by_underlying(self, underlying: IntArray, n_underlyings: int) -> dict[str, FloatArray]:
        """Aggregate each measure per underlying with a single bincount per field."""
        return {
            f: np.bincount(underlying, weights=getattr(self, f), minlength=n_underlyings).astype(
                np.float64
            )
            for f in _FIELDS
        }


class RiskEngine:
    """Caches position-level risk and keeps it consistent as market inputs change."""

    def __init__(self, portfolio: Portfolio, market: MarketData) -> None:
        if len(portfolio) and int(portfolio.underlying.max()) >= market.n_underlyings:
            raise ValueError("portfolio references an underlying missing from market data")
        self.portfolio = portfolio
        self.market = market
        self._scale = portfolio.scale
        self._cols: dict[str, FloatArray] = {}
        self._totals: dict[str, float] = {}
        self.compute()

    # ------------------------------------------------------------------ full pass
    def compute(self) -> RiskResult:
        """Revalue every position against the current market snapshot."""
        cols = self._price(np.arange(len(self.portfolio), dtype=np.int64))
        self._cols = cols
        self._totals = {f: float(cols[f].sum()) for f in _FIELDS}
        return self.result

    # ----------------------------------------------------------- incremental pass
    def update_spot(self, underlying: int, spot: float) -> RiskResult:
        """Move one spot and reprice only the positions on that underlying."""
        self.market = self.market.with_spot(underlying, spot)
        return self._reprice_subset(self.portfolio.indices_for(underlying))

    def update_vol(self, underlying: int, vol: float) -> RiskResult:
        """Move one vol and reprice only the positions on that underlying."""
        self.market = self.market.with_vol(underlying, vol)
        return self._reprice_subset(self.portfolio.indices_for(underlying))

    def _reprice_subset(self, idx: IntArray) -> RiskResult:
        if idx.size == 0:
            return self.result
        fresh = self._price(idx)
        for f in _FIELDS:
            old = self._cols[f][idx]
            self._totals[f] += float(fresh[f].sum() - old.sum())
            self._cols[f][idx] = fresh[f]
        return self.result

    # ------------------------------------------------------------------ helpers
    def _price(self, idx: IntArray) -> dict[str, FloatArray]:
        p, m = self.portfolio, self.market
        und = p.underlying[idx]
        g = black_scholes(
            S=m.spot[und],
            K=p.strike[idx],
            T=p.expiry[idx],
            sigma=m.vol[und],
            r=m.rate,
            q=m.div_yield[und],
            is_call=p.is_call[idx],
        )
        s = self._scale[idx]
        return {
            "value": g.price * s,
            "delta": g.delta * s,
            "gamma": g.gamma * s,
            "vega": g.vega * s,
            "theta": g.theta * s,
        }

    @property
    def result(self) -> RiskResult:
        return RiskResult(
            value=self._cols["value"],
            delta=self._cols["delta"],
            gamma=self._cols["gamma"],
            vega=self._cols["vega"],
            theta=self._cols["theta"],
            totals=dict(self._totals),
        )
