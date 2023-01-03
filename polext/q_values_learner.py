from typing import (
    Optional,
    Tuple,
)

import numpy as np


class QValuesLearner:
    def __init__(self) -> None:
        self.visits = {}
        self.Qtable = {}
        self.nactions = -1
        self._total_visits = 0

    def add_one_visit(self, state: Tuple[bool, ...], Q_values: np.ndarray):
        if self.nactions < 0:
            self.nactions = Q_values.shape[0]
        self._total_visits += 1
        if state not in self.visits:
            # Add new state
            self.visits[state] = 0
            self.Qtable[state] = np.zeros((self.nactions), dtype=float)
        self.visits[state] += 1
        self.Qtable[state] += (Q_values - self.Qtable[state]) / self.visits[state]

    def state_probability(self, state: Tuple[bool, ...]) -> float:
        return self.visits.get(state, 0) / max(1, self._total_visits)

    def state_visits(self, state: Tuple[bool, ...]) -> int:
        return self.visits.get(state, 0)

    def __getitem__(self, state: Tuple[bool, ...]) -> Optional[np.ndarray]:
        return self.Qtable.get(state, None)

    def reset_Q(self):
        self.Qtable = {s: Q * 0 for s, Q in self.Qtable.items()}

    def learn_qvalues(
        self,
        state: Tuple[int, ...],
        action: int,
        r: float,
        next_state: Tuple[int, ...],
        done: bool,
        alpha: float = 0.01,
        gamma: float = 0.99,
    ):
        if not done:
            nQ = self.Qtable.get(next_state, None)
            nval = np.max(nQ) if nQ is not None else 0
        else:
            nval = 0
        if state not in self.Qtable:
            self.Qtable[state] = np.zeros((self.nactions), dtype=float)
        self.Qtable[state][action] += alpha * (
            r + gamma * nval - self.Qtable[state][action]
        )

    def mix_with(self, other: "QValuesLearner", coefficient: float) -> None:
        # Merge
        for state in list(self.Qtable.keys()):
            other_q = other[state]
            if other_q is not None:
                self.Qtable[state] = (
                    self[state] * (1 - coefficient) + coefficient * other_q  # type: ignore
                )
