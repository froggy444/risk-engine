"""Arrow / Parquet persistence for portfolios and risk results.

The columnar Portfolio maps directly onto an Arrow table, so reads and writes are
column copies with no per-row object construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from risk_engine.engine import RiskResult
from risk_engine.portfolio import Portfolio

PORTFOLIO_SCHEMA = pa.schema(
    [
        ("position_id", pa.int64()),
        ("underlying", pa.int64()),
        ("strike", pa.float64()),
        ("expiry", pa.float64()),
        ("is_call", pa.bool_()),
        ("quantity", pa.float64()),
        ("multiplier", pa.float64()),
    ]
)


def portfolio_to_arrow(portfolio: Portfolio) -> pa.Table:
    return pa.table(
        {name: getattr(portfolio, name) for name in PORTFOLIO_SCHEMA.names},
        schema=PORTFOLIO_SCHEMA,
    )


def portfolio_from_arrow(table: pa.Table) -> Portfolio:
    table = table.select(PORTFOLIO_SCHEMA.names).cast(PORTFOLIO_SCHEMA)
    cols = {name: table.column(name).to_numpy() for name in PORTFOLIO_SCHEMA.names}
    cols["is_call"] = cols["is_call"].astype(bool)
    return Portfolio(**cols)


def write_portfolio(portfolio: Portfolio, path: str | Path) -> None:
    pq.write_table(portfolio_to_arrow(portfolio), path, compression="zstd")


def read_portfolio(path: str | Path) -> Portfolio:
    return portfolio_from_arrow(pq.read_table(path))


def write_risk(portfolio: Portfolio, result: RiskResult, path: str | Path) -> None:
    """Persist position-level risk alongside identifiers for downstream dashboards."""
    table = pa.table(
        {
            "position_id": portfolio.position_id,
            "underlying": portfolio.underlying,
            "value": result.value,
            "delta": result.delta,
            "gamma": result.gamma,
            "vega": result.vega,
            "theta": result.theta,
        }
    )
    pq.write_table(table, path, compression="zstd")


def read_risk(path: str | Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy() for name in table.column_names}
