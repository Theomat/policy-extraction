from typing import Generic, Iterable, List, Set, TypeVar

import numpy as np

from polext.predicate import Predicate

S = TypeVar("S")


class PredicateSpace(Generic[S]):
    def __init__(
        self, predicates: Iterable[Predicate[S]], guarantee_uniques: bool = True
    ) -> None:
        self.predicates = list(predicates)
        self.pred2int = {p: i for i, p in enumerate(self.predicates)}
        self.states_weight = {}
        self.Qtable = {}
        self.uniques = guarantee_uniques
        self.representatives = {}
        self.predicates_set = {p: set() for p in self.predicates}
        self.nactions = -1

    def _add_stats_for_representative_(self, state: S, Q_values: List[float]):
        repres = {p for p in self.predicates if p(state)}
        if repres in self.representatives:
            state = self.representatives[repres]
            self._add_stats_(state, Q_values)
        else:
            # Add new state
            self.states_weight[state] = 1
            self.Qtable[state] = np.array(Q_values)
            # Add state as a representative
            self.representatives[repres] = state
            # Update predicate set
            for p in repres:
                self.predicates_set[p].add(state)

    def _add_stats_(self, state: S, Q_values: List[float]):
        self.states_weight[state] += 1
        self.Qtable[state] += Q_values

    def add_state(self, state: S, Q_values: List[float]):
        if self.nactions < 0:
            self.nactions = len(Q_values)
        if state in self.states_weight:
            if not self.uniques:
                self._add_stats_(state, Q_values)
        else:
            self._add_stats_for_representative_(state, Q_values)

    @property
    def states(self) -> Set[S]:
        return set(self.Qtable.keys())
