"""Pricing correctness: reference values, parity, and Greeks vs finite differences."""

import numpy as np
import pytest

from risk_engine.pricing import black_scholes

RNG = np.random.default_rng(42)
N = 500
S = RNG.uniform(50, 150, N)
K = S * RNG.uniform(0.7, 1.3, N)
T = RNG.uniform(0.05, 2.0, N)
VOL = RNG.uniform(0.1, 0.6, N)
R, Q = 0.03, RNG.uniform(0.0, 0.03, N)
CALL = RNG.random(N) < 0.5


def greeks():
    return black_scholes(S, K, T, VOL, R, Q, CALL)


def price(s=S, k=K, t=T, v=VOL, r=R, q=Q, c=CALL):
    return black_scholes(s, k, t, v, r, q, c).price


def test_reference_value():
    # Hull, ATM 1y call, sigma 20%, r 5%: 10.4506
    g = black_scholes(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    assert g.price == pytest.approx(10.450583572185565, abs=1e-10)


def test_put_call_parity():
    c = black_scholes(S, K, T, VOL, R, Q, True).price
    p = black_scholes(S, K, T, VOL, R, Q, False).price
    np.testing.assert_allclose(c - p, S * np.exp(-Q * T) - K * np.exp(-R * T), atol=1e-10)


def test_delta_matches_finite_difference():
    h = 1e-4 * S
    fd = (price(s=S + h) - price(s=S - h)) / (2 * h)
    np.testing.assert_allclose(greeks().delta, fd, atol=1e-6)


def test_gamma_matches_finite_difference():
    h = 1e-3 * S
    fd = (price(s=S + h) - 2 * price() + price(s=S - h)) / (h * h)
    np.testing.assert_allclose(
        black_scholes(S, K, T, VOL, R, Q, CALL).gamma, fd, rtol=1e-3, atol=1e-6
    )


def test_vega_matches_finite_difference():
    h = 1e-5
    fd = (price(v=VOL + h) - price(v=VOL - h)) / (2 * h)
    np.testing.assert_allclose(
        black_scholes(S, K, T, VOL, R, Q, CALL).vega, fd, rtol=1e-6, atol=1e-6
    )


def test_theta_matches_finite_difference():
    h = 1e-6
    fd = -(price(t=T + h) - price(t=T - h)) / (2 * h)  # theta = -dV/dT
    np.testing.assert_allclose(
        black_scholes(S, K, T, VOL, R, Q, CALL).theta, fd, rtol=1e-5, atol=1e-5
    )


def test_bounds_and_signs():
    g = black_scholes(S, K, T, VOL, R, Q, CALL)
    assert np.all(g.price >= 0)
    assert np.all(g.gamma > 0) and np.all(g.vega > 0)
    assert np.all(g.delta[CALL] >= 0) and np.all(g.delta[~CALL] <= 0)


def test_broadcasts_over_scenarios():
    spots = S[None, :] * np.array([[0.9], [1.0], [1.1]])
    g = black_scholes(spots, K, T, VOL, R, Q, CALL)
    assert g.price.shape == (3, N)
    np.testing.assert_allclose(g.price[1], price())
