"""Parquet round-trips preserve every column exactly."""

import numpy as np

from risk_engine.engine import RiskEngine
from risk_engine.io import read_portfolio, read_risk, write_portfolio, write_risk


def test_portfolio_roundtrip(tmp_path, book):
    path = tmp_path / "book.parquet"
    write_portfolio(book, path)
    loaded = read_portfolio(path)
    for col in (
        "position_id",
        "underlying",
        "strike",
        "expiry",
        "is_call",
        "quantity",
        "multiplier",
    ):
        np.testing.assert_array_equal(getattr(loaded, col), getattr(book, col))


def test_risk_roundtrip(tmp_path, book, market):
    res = RiskEngine(book, market).result
    path = tmp_path / "risk.parquet"
    write_risk(book, res, path)
    loaded = read_risk(path)
    np.testing.assert_array_equal(loaded["delta"], res.delta)
    np.testing.assert_array_equal(loaded["position_id"], book.position_id)
