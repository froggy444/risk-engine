"""Historical-simulation VaR and Expected Shortfall with full revaluation.

Each historical day supplies a vector of spot log-returns and vol changes for every
underlying. The book is fully repriced under each scenario (no delta-gamma
approximation), giving a PnL distribution from which VaR and ES are read off.

Scenarios are processed in blocks so peak memory stays bounded at
``block_size x n_positions`` floats regardless of history length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from risk_engine.market import MarketData
from risk_engine.portfolio import Portfolio
from risk_engine.pricing import black_scholes

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VaRResult:
    confidence: float
    var: float
    expected_shortfall: float
    scenario_pnl: FloatArray

    @property
    def worst_scenario(self) -> int:
        return int(np.argmin(self.scenario_pnl))


def scenario_pnl(
    portfolio: Portfolio,
    market: MarketData,
    spot_returns: FloatArray,
    vol_changes: FloatArray | None = None,
    horizon: float = 1.0 / 252.0,
    block_size: int = 64,
) -> FloatArray:
    """Book PnL under each scenario.

    Parameters
    ----------
    spot_returns : (n_scenarios, n_underlyings) log-returns.
    vol_changes  : (n_scenarios, n_underlyings) absolute vol shifts, or None.
    horizon      : time decay applied in every scenario (default one trading day).
    """
    n_scen, n_und = spot_returns.shape
    if n_und != market.n_underlyings:
        raise ValueError("spot_returns must have one column per underlying")
    if vol_changes is not None and vol_changes.shape != spot_returns.shape:
        raise ValueError("vol_changes must match spot_returns shape")

    und = portfolio.underlying
    scale = portfolio.scale
    base_value = float(
        (
            black_scholes(
                market.spot[und],
                portfolio.strike,
                portfolio.expiry,
                market.vol[und],
                market.rate,
                market.div_yield[und],
                portfolio.is_call,
            ).price
            * scale
        ).sum()
    )
    t_shifted = np.maximum(portfolio.expiry - horizon, 1e-6)

    out = np.empty(n_scen, dtype=np.float64)
    for start in range(0, n_scen, block_size):
        stop = min(start + block_size, n_scen)
        # (block, n_underlyings) -> gather to (block, n_positions)
        spots = (market.spot * np.exp(spot_returns[start:stop]))[:, und]
        if vol_changes is None:
            vols = np.broadcast_to(market.vol[und], spots.shape)
        else:
            vols = np.maximum(market.vol + vol_changes[start:stop], 1e-4)[:, und]
        prices = black_scholes(
            spots,
            portfolio.strike,
            t_shifted,
            vols,
            market.rate,
            market.div_yield[und],
            portfolio.is_call,
        ).price
        out[start:stop] = prices @ scale - base_value
    return out


def historical_var(
    portfolio: Portfolio,
    market: MarketData,
    spot_returns: FloatArray,
    vol_changes: FloatArray | None = None,
    confidence: float = 0.99,
    horizon: float = 1.0 / 252.0,
) -> VaRResult:
    """VaR and ES at ``confidence``, reported as positive loss numbers."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    pnl = scenario_pnl(portfolio, market, spot_returns, vol_changes, horizon)
    cutoff = np.quantile(pnl, 1.0 - confidence, method="lower")
    tail = pnl[pnl <= cutoff]
    return VaRResult(
        confidence=confidence,
        var=float(-cutoff),
        expected_shortfall=float(-tail.mean()),
        scenario_pnl=pnl,
    )
