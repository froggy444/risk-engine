"""Command-line demo: build a book, compute risk, explain PnL, run VaR, write Parquet."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from risk_engine.attribution import attribute_pnl
from risk_engine.data import make_history, make_market, make_portfolio
from risk_engine.engine import RiskEngine
from risk_engine.io import write_portfolio, write_risk
from risk_engine.var import historical_var


def _ms(t0: float) -> str:
    return f"{(time.perf_counter() - t0) * 1e3:8.2f} ms"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="risk-engine", description=__doc__)
    ap.add_argument("--positions", type=int, default=10_000)
    ap.add_argument("--underlyings", type=int, default=50)
    ap.add_argument("--days", type=int, default=250, help="historical scenarios for VaR")
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args(argv)

    market = make_market(args.underlyings)
    book = make_portfolio(market, args.positions)
    print(f"Book: {len(book):,} positions on {market.n_underlyings} underlyings\n")

    t0 = time.perf_counter()
    engine = RiskEngine(book, market)
    print(f"Full revaluation + Greeks          {_ms(t0)}")

    t0 = time.perf_counter()
    engine.update_spot(0, float(market.spot[0]) * 1.01)
    print(
        f"Incremental update (1 spot move)   {_ms(t0)}  "
        f"({book.indices_for(0).size} positions repriced)"
    )

    totals = engine.result.totals
    print("\nBook totals")
    for k, v in totals.items():
        print(f"  {k:<6} {v:>18,.2f}")

    spot_ret, vol_chg = make_history(market, args.days)
    t1 = engine.market.bump(spot_ret[-1], vol_chg[-1])
    t0 = time.perf_counter()
    attr = attribute_pnl(book, engine.market, t1, dt=1 / 252)
    print(f"\nPnL attribution (1 day)            {_ms(t0)}")
    for k, v in attr.totals().items():
        print(f"  {k:<12} {v:>18,.2f}")

    t0 = time.perf_counter()
    var = historical_var(book, engine.market, spot_ret, vol_chg, confidence=0.99)
    print(f"\nHistorical VaR ({args.days} scenarios)      {_ms(t0)}")
    print(f"  99% 1-day VaR  {var.var:>18,.2f}")
    print(f"  99% 1-day ES   {var.expected_shortfall:>18,.2f}")
    print(f"  worst day      #{var.worst_scenario} ({var.scenario_pnl.min():,.2f})")

    args.out.mkdir(parents=True, exist_ok=True)
    write_portfolio(book, args.out / "portfolio.parquet")
    write_risk(book, engine.result, args.out / "risk.parquet")
    np.save(args.out / "var_scenario_pnl.npy", var.scenario_pnl)
    print(f"\nWrote Parquet outputs to {args.out}/")


if __name__ == "__main__":
    main()
