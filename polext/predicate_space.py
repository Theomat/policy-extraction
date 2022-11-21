from typing import Generic, Iterable, List, Set, Tuple, TypeVar

import numpy as np

from polext.predicate import Predicate

S = TypeVar("S")


class PredicateSpace(Generic[S]):
    def __init__(
        self, predicates: Iterable[Predicate[S]], use_representatives: bool = False
    ) -> None:
        self.predicates = list(predicates)
        self.pred2int = {p: i for i, p in enumerate(self.predicates)}
        self.counts = {}
        self.Qtable = {}
        self.use_representatives = use_representatives
        self.representatives = {}
        self.predicates_set = {p: set() for p in self.predicates}
        self.nactions = -1
        self._total_visits = 0

    def _add_stats_for_representative_(self, state: S, Q_values: List[float]):
        self._total_visits += 1
        if state in self.counts:
            self._add_stats_(state, Q_values)
        else:
            # Add new state
            self.counts[state] = 1
            self.Qtable[state] = np.array(Q_values)
            if not self.use_representatives:
                for p in self.predicates:
                    if p(state):
                        self.predicates_set[p].add(state)

    def get_representative(self, state: S) -> S:
        repres = tuple(p for p in self.predicates if p(state))
        if repres in self.representatives:
            state = self.representatives[repres]
        else:
            self.representatives[repres] = state
            # Update predicate set
            for p in repres:
                self.predicates_set[p].add(state)
        return state

    def state_probability(self, state: S) -> float:
        return self.counts.get(state, 0) / max(1, self._total_visits)

    def _add_stats_(self, state: S, Q_values: List[float]):
        self.counts[state] += 1
        self.Qtable[state] += Q_values

    def visit_state(self, state: S, Q_values: List[float]):
        if self.nactions < 0:
            self.nactions = len(Q_values)
        if self.use_representatives:
            state = self.get_representative(state)
        self._add_stats_for_representative_(state, Q_values)

    def children(
        self, predicate: Predicate[S]
    ) -> "Tuple[PredicateSpace[S], PredicateSpace[S]]":
        selected_states = self.predicates_set[predicate]
        left = PredicateSpace(
            {p for p in self.predicates if p != predicate}, self.use_representatives
        )
        right = PredicateSpace(
            {p for p in self.predicates if p != predicate}, self.use_representatives
        )
        # Update nactions
        right.nactions = self.nactions
        left.nactions = right.nactions
        # Update counts and Qtable
        for s in self.states:
            target = left if s in selected_states else right
            target.counts[s] = self.counts[s]
            target.Qtable[s] = self.Qtable[s]
        # Update predicates_set
        left.predicates_set = {
            p: a.intersection(selected_states)
            for p, a in self.predicates_set.items()
            if p != predicate
        }
        right.predicates_set = {
            p: a.difference(selected_states)
            for p, a in self.predicates_set.items()
            if p != predicate
        }
        # Update representatives
        if self.use_representatives:
            for repres, s in self.representatives:
                target = left if s in selected_states else right
                new_repres = tuple(p for p in repres if p != predicate)
                target.representatives[new_repres] = s
        # Update visits
        left._total_visits = sum(self.counts[s] for s in selected_states)
        right._total_visits = self._total_visits - left._total_visits
        return left, right

    @property
    def states(self) -> Set[S]:
        return set(self.Qtable.keys())
