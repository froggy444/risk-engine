"""Vectorized portfolio risk engine: Black-Scholes Greeks, PnL attribution and historical VaR."""

from risk_engine.attribution import AttributionResult, attribute_pnl
from risk_engine.engine import RiskEngine, RiskResult
from risk_engine.market import MarketData
from risk_engine.portfolio import Portfolio
from risk_engine.pricing import Greeks, black_scholes
from risk_engine.var import VaRResult, historical_var

__all__ = [
    "AttributionResult",
    "Greeks",
    "MarketData",
    "Portfolio",
    "RiskEngine",
    "RiskResult",
    "VaRResult",
    "attribute_pnl",
    "black_scholes",
    "historical_var",
]

__version__ = "0.1.0"
