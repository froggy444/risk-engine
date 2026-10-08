"""Timing benchmark: median of repeated runs for each stage of the engine."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from risk_engine.attribution import attribute_pnl
from risk_engine.data import make_history, make_market, make_portfolio
from risk_engine.engine import RiskEngine
from risk_engine.market import MarketData
from risk_engine.var import historical_var


def bench(label: str, fn: Callable[[], object], repeat: int = 20) -> None:
    fn()  # warm-up
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1e3)
    print(f"{label:<44} median {statistics.median(runs):8.2f} ms   min {min(runs):8.2f} ms")


def run(market: MarketData, n_positions: int) -> None:
    book = make_portfolio(market, n_positions)
    engine = RiskEngine(book, market)
    spot_ret, vol_chg = make_history(market, 250)
    t1 = market.bump(spot_ret[-1], vol_chg[-1])
    print(f"\n--- {n_positions:,} positions, {market.n_underlyings} underlyings ---")
    bench("Full revaluation + Greeks", engine.compute)
    bench(
        "Incremental update (one spot)",
        lambda: engine.update_spot(0, float(engine.market.spot[0]) * 1.0001),
    )
    bench("PnL attribution", lambda: attribute_pnl(book, market, t1, 1 / 252))
    bench(
        "Historical VaR, 250 full-reval scenarios",
        lambda: historical_var(book, market, spot_ret, vol_chg),
        repeat=5,
    )


def main() -> None:
    market = make_market(50)
    for n in (10_000, 100_000):
        run(market, n)


if __name__ == "__main__":
    main()
