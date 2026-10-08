"""Vectorized Black-Scholes-Merton pricing and analytic Greeks.

All inputs are NumPy arrays (or scalars) that broadcast together, so the same
function prices a single option, a 10K-position book, or a (scenarios x positions)
matrix without Python loops.

Conventions
-----------
* ``T`` is time to expiry in years; ``r`` and ``q`` are continuously compounded.
* ``vega`` is dV/dsigma per 1.00 change in volatility (multiply by 0.01 for per vol point).
* ``theta`` is dV/dt in calendar time per year (i.e. -dV/dT); negative for a typical long option.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_MIN_T = 1e-8
_MIN_VOL = 1e-8


@dataclass(frozen=True, slots=True)
class Greeks:
    """Per-unit option value and sensitivities, one entry per option."""

    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray


def _norm_pdf(x: FloatArray) -> FloatArray:
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * x * x), dtype=np.float64)


def black_scholes(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> Greeks:
    """Price European options and compute analytic Greeks in one vectorized pass.

    Shared intermediates (d1, d2, discount factors, N(d1), n(d1)) are computed once
    and reused across price and every Greek, which keeps a full book revaluation in
    the low milliseconds.
    """
    S_ = np.asarray(S, dtype=np.float64)
    K_ = np.asarray(K, dtype=np.float64)
    T_ = np.maximum(np.asarray(T, dtype=np.float64), _MIN_T)
    vol = np.maximum(np.asarray(sigma, dtype=np.float64), _MIN_VOL)
    r_ = np.asarray(r, dtype=np.float64)
    q_ = np.asarray(q, dtype=np.float64)
    call = np.asarray(is_call, dtype=bool)

    sqrt_t = np.sqrt(T_)
    vol_sqrt_t = vol * sqrt_t
    d1 = (np.log(S_ / K_) + (r_ - q_ + 0.5 * vol * vol) * T_) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    df_r = np.exp(-r_ * T_)
    df_q = np.exp(-q_ * T_)
    pdf_d1 = _norm_pdf(d1)

    # Sign trick: a put uses N(-d1), N(-d2), so evaluate N(sign * d) once for both types.
    sign = np.where(call, 1.0, -1.0)
    n_d1 = ndtr(sign * d1)
    n_d2 = ndtr(sign * d2)

    s_dq = S_ * df_q
    k_dr = K_ * df_r

    price = sign * (s_dq * n_d1 - k_dr * n_d2)
    delta = sign * df_q * n_d1
    gamma = df_q * pdf_d1 / (S_ * vol_sqrt_t)
    vega = s_dq * pdf_d1 * sqrt_t
    theta = -s_dq * pdf_d1 * vol / (2.0 * sqrt_t) + sign * (q_ * s_dq * n_d1 - r_ * k_dr * n_d2)

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)
