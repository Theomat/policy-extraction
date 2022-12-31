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
        self.normalised = False
        self._total_visits = 0

    def add_one_visit(self, state: Tuple[int, ...], Q_values: np.ndarray):
        if self.nactions < 0:
            self.nactions = Q_values.shape[0]
        self._total_visits += 1
        if state not in self.visits:
            # Add new state
            self.visits[state] = 0
            self.Qtable[state] = np.zeros((self.nactions), dtype=float)
        self.visits[state] += 1
        self.Qtable[state] += (Q_values - self.Qtable[state]) / self.visits[state]

    def state_probability(self, state: Tuple[int, ...]) -> float:
        return self.visits.get(state, 0) / max(1, self._total_visits)

    def state_visits(self, state: Tuple[int, ...]) -> int:
        return self.visits.get(state, 0)

    def state_Q(self, state: Tuple[int, ...]) -> Optional[np.ndarray]:
        Qvalues = self.Qtable.get(state, None)
        if Qvalues is None:
            return None
        if self.normalised:
            return Qvalues * self.visits[state]
        return Qvalues

    def state_normalised_Q(self, state: Tuple[int, ...]) -> Optional[np.ndarray]:
        Qvalues = self.Qtable.get(state, None)
        if Qvalues is None:
            return None
        if self.normalised:
            return Qvalues
        return Qvalues / self.visits.get(state, 0)

    def __getitem__(self, state: Tuple[int, ...]) -> Optional[np.ndarray]:
        return self.state_Q(state)

    def reset_Q(self):
        self.Qtable = {s: Q * 0 for s, Q in self.Qtable.items()}
        self.normalised = False

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
        self.normalised = True

    def mix_with(self, other: "QValuesLearner", coefficient: float) -> None:
        self._total_visits = 0
        # Merge
        for state in list(self.Qtable.keys()):
            out = self.state_normalised_Q(state)
            visits = self.state_visits(state)
            other_q = other.state_normalised_Q(state)
            if other_q is not None:
                out = out * (1 - coefficient) + coefficient * other_q
                visits = visits * (1 - coefficient) + coefficient * other.visits[state]
            self.Qtable[state] = out
            self.visits[state] = visits
            self._total_visits += visits

        # Copy missing from our dict
        for state, value in other.Qtable.items():
            if state not in self.Qtable:
                visits = other.visits[state]
                self.Qtable[state] = value / visits
                self.visits[state] = visits
                self._total_visits += visits

        self.normalised = True
