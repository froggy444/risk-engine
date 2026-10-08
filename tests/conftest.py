import pytest

from risk_engine.data import make_history, make_market, make_portfolio


@pytest.fixture(scope="session")
def market():
    return make_market(n_underlyings=20, seed=1)


@pytest.fixture(scope="session")
def book(market):
    return make_portfolio(market, n_positions=2_000, seed=2)


@pytest.fixture(scope="session")
def history(market):
    return make_history(market, n_days=250, seed=3)
