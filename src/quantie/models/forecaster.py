"""Forecaster interface for model output."""
from __future__ import annotations

from typing import Protocol

from quantie.models.ensemble import ModelOutput


class Forecaster(Protocol):
    def predict(
        self,
        feature_row: dict,
        S0: float,
        S_t: float,
        sigma_per_sec: float,
        tau_sec: float,
        drift_per_sec: float = 0.0,
    ) -> ModelOutput:
        ...

