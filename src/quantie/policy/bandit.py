"""Linear UCB contextual bandit for execution actions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantie.policy.decision import Action


_ARMS: tuple[Action, ...] = (
    Action.HOLD,
    Action.TAKE_UP,
    Action.TAKE_DOWN,
    Action.MAKE_UP,
    Action.MAKE_DOWN,
)


@dataclass(frozen=True)
class BanditConfig:
    alpha_paper: float = 0.35
    alpha_live: float = 0.0
    l2_lambda: float = 1.0
    min_samples_per_arm: int = 3


class LinUCBBandit:
    def __init__(self, dim: int, cfg: BanditConfig | None = None) -> None:
        self._cfg = cfg or BanditConfig()
        self._dim = dim
        self._A: dict[Action, np.ndarray] = {
            arm: np.eye(dim) * self._cfg.l2_lambda for arm in _ARMS
        }
        self._b: dict[Action, np.ndarray] = {
            arm: np.zeros(dim, dtype=float) for arm in _ARMS
        }
        self._pulls: dict[Action, int] = {arm: 0 for arm in _ARMS}

    @property
    def arms(self) -> tuple[Action, ...]:
        return _ARMS

    def is_warm(self) -> bool:
        return all(self._pulls[arm] >= self._cfg.min_samples_per_arm for arm in _ARMS)

    def select(self, x: np.ndarray, *, is_live: bool) -> Action:
        alpha = self._cfg.alpha_live if is_live else self._cfg.alpha_paper
        best_arm = Action.HOLD
        best_score = -1e18
        for arm in _ARMS:
            score = self.score(arm, x, alpha)
            if score > best_score:
                best_score = score
                best_arm = arm
        return best_arm

    def score(self, arm: Action, x: np.ndarray, alpha: float) -> float:
        A = self._A[arm]
        b = self._b[arm]
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b
        exploit = float(theta @ x)
        explore = float(alpha * np.sqrt(x @ A_inv @ x))
        return exploit + explore

    def update(self, arm: Action, x: np.ndarray, reward: float) -> None:
        self._A[arm] += np.outer(x, x)
        self._b[arm] += reward * x
        self._pulls[arm] += 1

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for arm in _ARMS:
            key = arm.value.lower()
            payload[f"A_{key}"] = self._A[arm]
            payload[f"b_{key}"] = self._b[arm]
            payload[f"n_{key}"] = np.array(self._pulls[arm], dtype=int)
        payload["dim"] = np.array(self._dim, dtype=int)
        np.savez(path, **payload)

    def load(self, path: Path) -> None:
        data = np.load(Path(path))
        dim = int(data["dim"])
        if dim != self._dim:
            raise ValueError(f"bandit dim mismatch: {dim} != {self._dim}")
        for arm in _ARMS:
            key = arm.value.lower()
            self._A[arm] = data[f"A_{key}"]
            self._b[arm] = data[f"b_{key}"]
            self._pulls[arm] = int(data[f"n_{key}"])

