"""Market data snapshot: one spot, vol and dividend yield per underlying."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarketData:
    """Immutable market snapshot indexed by underlying id (0..n_underlyings-1).

    Volatility is flat per underlying (no smile). That keeps the engine focused on
    the risk mechanics; a surface lookup would replace ``vol[underlying]`` with an
    interpolation over (strike, expiry).
    """

    spot: FloatArray
    vol: FloatArray
    div_yield: FloatArray
    rate: float

    def __post_init__(self) -> None:
        n = self.spot.shape[0]
        if self.vol.shape != (n,) or self.div_yield.shape != (n,):
            raise ValueError("spot, vol and div_yield must all have shape (n_underlyings,)")
        if np.any(self.spot <= 0) or np.any(self.vol <= 0):
            raise ValueError("spot and vol must be strictly positive")

    @property
    def n_underlyings(self) -> int:
        return int(self.spot.shape[0])

    def bump(
        self,
        spot_return: FloatArray | None = None,
        vol_change: FloatArray | None = None,
    ) -> MarketData:
        """Return a new snapshot with log-returns applied to spot and absolute shifts to vol."""
        spot = self.spot if spot_return is None else self.spot * np.exp(spot_return)
        vol = self.vol if vol_change is None else np.maximum(self.vol + vol_change, 1e-4)
        return replace(self, spot=spot, vol=vol)

    def with_spot(self, underlying: int, value: float) -> MarketData:
        spot = self.spot.copy()
        spot[underlying] = value
        return replace(self, spot=spot)

    def with_vol(self, underlying: int, value: float) -> MarketData:
        vol = self.vol.copy()
        vol[underlying] = value
        return replace(self, vol=vol)
