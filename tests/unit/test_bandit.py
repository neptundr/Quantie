from __future__ import annotations

import numpy as np

from quantie.policy.bandit import LinUCBBandit
from quantie.policy.decision import Action


def test_bandit_select_and_update() -> None:
    b = LinUCBBandit(dim=3)
    x = np.array([1.0, 0.2, -0.1])
    arm = b.select(x, is_live=False)
    assert arm in b.arms
    b.update(Action.TAKE_UP, x, reward=1.0)
    score_after = b.score(Action.TAKE_UP, x, alpha=0.0)
    assert score_after > -10


def test_bandit_save_and_load(tmp_path) -> None:
    p = tmp_path / "bandit_state.npz"
    b1 = LinUCBBandit(dim=3)
    x = np.array([1.0, 0.0, 0.0])
    b1.update(Action.TAKE_DOWN, x, reward=0.5)
    b1.save(p)

    b2 = LinUCBBandit(dim=3)
    b2.load(p)
    assert b2.score(Action.TAKE_DOWN, x, alpha=0.0) == b1.score(Action.TAKE_DOWN, x, alpha=0.0)

