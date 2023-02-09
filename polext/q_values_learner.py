from typing import (
    Optional,
    Tuple,
    Union,
    overload,
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

    @overload
    def __getitem__(self, state: Tuple[bool, ...]) -> Optional[np.ndarray]:
        pass

    @overload
    def __getitem__(self, pair: Tuple[Tuple[bool, ...], int]) -> float:
        pass

    def __getitem__(
        self, state: Union[Tuple[bool, ...], Tuple[Tuple[bool, ...], int]]
    ) -> Union[Optional[np.ndarray], float]:
        if len(state) == 2:
            s, action = state
            if s in self.Qtable:
                return self.Qtable[s][action]
            return 0
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
        self.learn_qvalues_width_default(state, action, r, next_state, done, alpha, gamma)

    def learn_qvalues_width_default(
        self,
        state: Tuple[int, ...],
        action: int,
        r: float,
        next_state: Tuple[int, ...],
        done: bool,
        alpha: float = 0.01,
        gamma: float = 0.99,
        q_def_s: Optional[np.ndarray] = None,
        q_def_stp1: Optional[np.ndarray] = None,
    ):
        # Next state Q-value
        if not done:
            if self.visits.get(next_state, 0) == 0:
                nQ = q_def_stp1
            else:
                nQ = self.Qtable.get(next_state, None)
            nval = np.max(nQ) if nQ is not None else 0
        else:
            nval = 0
        # Init state Q-value
        if state not in self.Qtable:
            self.Qtable[state] = np.zeros((self.nactions), dtype=float)
            self.visits[state] = 0
            if q_def_s:
                self.Qtable[state] += q_def_s
        # Update
        self.visits[state] += 1
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
