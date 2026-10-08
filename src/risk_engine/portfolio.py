"""Columnar option portfolio stored as NumPy arrays (struct-of-arrays layout)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Portfolio:
    """A book of European options.

    Stored column-wise rather than as a list of objects, so every risk calculation
    is a single vectorized expression over contiguous arrays and the book maps
    one-to-one onto an Arrow table or Parquet file.
    """

    position_id: IntArray
    underlying: IntArray
    strike: FloatArray
    expiry: FloatArray
    is_call: BoolArray
    quantity: FloatArray
    multiplier: FloatArray
    _groups: dict[int, IntArray] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = self.position_id.shape[0]
        for name in ("underlying", "strike", "expiry", "is_call", "quantity", "multiplier"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"column '{name}' must have shape ({n},)")
        if np.any(self.strike <= 0):
            raise ValueError("strikes must be strictly positive")
        if np.any(self.expiry <= 0):
            raise ValueError("expiries must be strictly positive (in years)")
        object.__setattr__(self, "_groups", self._build_groups())

    def _build_groups(self) -> dict[int, IntArray]:
        """Precompute position indices per underlying so an update touches only k positions."""
        order = np.argsort(self.underlying, kind="stable")
        uniq, starts = np.unique(self.underlying[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        return {int(u): order[s:e] for u, s, e in zip(uniq, starts, ends, strict=True)}

    def __len__(self) -> int:
        return int(self.position_id.shape[0])

    @property
    def scale(self) -> FloatArray:
        """Quantity x contract multiplier: converts per-unit values into position values."""
        return self.quantity * self.multiplier

    def indices_for(self, underlying: int) -> IntArray:
        """Indices of positions on a given underlying (empty if none)."""
        return self._groups.get(underlying, np.empty(0, dtype=np.int64))

    def roll(self, dt: float) -> Portfolio:
        """Same book with every expiry shortened by ``dt`` years (floored just above zero)."""
        return Portfolio(
            position_id=self.position_id,
            underlying=self.underlying,
            strike=self.strike,
            expiry=np.maximum(self.expiry - dt, 1e-6),
            is_call=self.is_call,
            quantity=self.quantity,
            multiplier=self.multiplier,
        )
