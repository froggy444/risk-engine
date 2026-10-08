"""Incremental updates must match a from-scratch revaluation exactly (to float tolerance)."""

import numpy as np
import pytest

from risk_engine.engine import RiskEngine


def test_totals_equal_sum_of_positions(book, market):
    res = RiskEngine(book, market).result
    for f in ("value", "delta", "gamma", "vega", "theta"):
        assert res.totals[f] == pytest.approx(getattr(res, f).sum())


@pytest.mark.parametrize("kind", ["spot", "vol"])
def test_incremental_matches_full(book, market, kind):
    eng = RiskEngine(book, market)
    rng = np.random.default_rng(0)
    for _ in range(25):
        u = int(rng.integers(market.n_underlyings))
        if kind == "spot":
            eng.update_spot(u, float(eng.market.spot[u] * rng.uniform(0.95, 1.05)))
        else:
            eng.update_vol(u, float(eng.market.vol[u] + rng.uniform(-0.02, 0.02)))

    fresh = RiskEngine(book, eng.market).result
    inc = eng.result
    for f in ("value", "delta", "gamma", "vega", "theta"):
        np.testing.assert_allclose(getattr(inc, f), getattr(fresh, f), rtol=1e-12, atol=1e-9)
        assert inc.totals[f] == pytest.approx(fresh.totals[f], rel=1e-10, abs=1e-6)


def test_update_touches_only_affected_positions(book, market):
    eng = RiskEngine(book, market)
    before = eng.result.value.copy()
    eng.update_spot(3, float(market.spot[3] * 1.02))
    changed = np.flatnonzero(eng.result.value != before)
    assert set(changed) <= set(book.indices_for(3).tolist())


def test_by_underlying_sums_to_total(book, market):
    res = RiskEngine(book, market).result
    agg = res.by_underlying(book.underlying, market.n_underlyings)
    assert agg["delta"].sum() == pytest.approx(res.totals["delta"])


def test_rejects_unknown_underlying(book, market):
    small = type(market)(market.spot[:2], market.vol[:2], market.div_yield[:2], market.rate)
    with pytest.raises(ValueError):
        RiskEngine(book, small)
