"""PnL explain: components sum correctly and the residual shrinks with move size."""

import numpy as np
import pytest

from risk_engine.attribution import attribute_pnl

DT = 1 / 252


def _residual_ratio(book, market, scale, dt=DT):
    rng = np.random.default_rng(5)
    t1 = market.bump(
        rng.standard_normal(market.n_underlyings) * 0.01 * scale,
        rng.standard_normal(market.n_underlyings) * 0.005 * scale,
    )
    tot = attribute_pnl(book, market, t1, dt).totals()
    return abs(tot["unexplained"]) / (abs(tot["actual"]) + 1e-12), tot


def test_components_sum_to_explained(book, market):
    _, tot = _residual_ratio(book, market, 1.0)
    parts = tot["delta"] + tot["gamma"] + tot["vega"] + tot["theta"]
    assert parts == pytest.approx(tot["explained"])
    assert tot["explained"] + tot["unexplained"] == pytest.approx(tot["actual"])


def test_typical_day_is_well_explained(book, market):
    ratio, _ = _residual_ratio(book, market, 1.0)
    assert ratio < 0.05


def test_residual_is_higher_order(book, market):
    # With time frozen, the residual comes from cross and higher-order terms (vanna,
    # volga, speed), so shrinking the move 10x should shrink it far more than 10x.
    _, small = _residual_ratio(book, market, 0.1, dt=0.0)
    _, large = _residual_ratio(book, market, 1.0, dt=0.0)
    assert abs(small["unexplained"]) < 0.05 * abs(large["unexplained"])


def test_pure_time_decay_matches_theta(book, market):
    tot = attribute_pnl(book, market, market, 1e-5).totals()
    assert tot["theta"] == pytest.approx(tot["actual"], rel=1e-3)
