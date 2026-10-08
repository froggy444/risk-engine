"""Historical VaR: matches a naive loop, ES >= VaR, and block size does not change results."""

import numpy as np
import pytest

from risk_engine.pricing import black_scholes
from risk_engine.var import historical_var, scenario_pnl


def _naive_pnl(book, market, spot_ret, vol_chg, horizon):
    und, scale = book.underlying, book.scale
    base = (
        black_scholes(
            market.spot[und],
            book.strike,
            book.expiry,
            market.vol[und],
            market.rate,
            market.div_yield[und],
            book.is_call,
        ).price
        * scale
    ).sum()
    out = []
    for i in range(spot_ret.shape[0]):
        m = market.bump(spot_ret[i], vol_chg[i])
        v = black_scholes(
            m.spot[und],
            book.strike,
            np.maximum(book.expiry - horizon, 1e-6),
            m.vol[und],
            m.rate,
            m.div_yield[und],
            book.is_call,
        ).price
        out.append((v * scale).sum() - base)
    return np.array(out)


def test_matches_naive_loop(book, market, history):
    spot_ret, vol_chg = history
    fast = scenario_pnl(book, market, spot_ret, vol_chg)
    slow = _naive_pnl(book, market, spot_ret, vol_chg, 1 / 252)
    np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("block", [1, 17, 64, 1000])
def test_block_size_invariant(book, market, history, block):
    spot_ret, vol_chg = history
    ref = scenario_pnl(book, market, spot_ret, vol_chg, block_size=64)
    np.testing.assert_allclose(scenario_pnl(book, market, spot_ret, vol_chg, block_size=block), ref)


def test_es_at_least_var(book, market, history):
    res = historical_var(book, market, *history, confidence=0.99)
    assert res.expected_shortfall >= res.var
    assert res.scenario_pnl.shape == (250,)


def test_higher_confidence_higher_var(book, market, history):
    v95 = historical_var(book, market, *history, confidence=0.95).var
    v99 = historical_var(book, market, *history, confidence=0.99).var
    assert v99 >= v95


def test_invalid_confidence(book, market, history):
    with pytest.raises(ValueError):
        historical_var(book, market, *history, confidence=1.0)
