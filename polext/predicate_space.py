from typing import (
    Callable,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np

from polext.predicate import Predicate

S = TypeVar("S")


def __make_hashable__(x: S) -> S:
    if isinstance(x, np.ndarray):
        return tuple(__make_hashable__(y) for y in x)  # type: ignore
    return x


class PredicateSpace(Generic[S]):
    def __init__(
        self, predicates: Iterable[Predicate[S]], use_representatives: bool = False
    ) -> None:
        self.predicates = list(predicates)
        self.counts = {}
        self.Qtable = {}
        self.use_representatives = use_representatives
        self.representatives = {}
        self.predicates_set = {p: set() for p in self.predicates}
        self.nactions = -1
        self._total_visits = 0
        self.learnt_Q = {}

    def _add_stats_for_representative_(self, state: S, Q_values: np.ndarray):
        self._total_visits += 1
        if state in self.counts:
            self._add_stats_(state, Q_values)
        else:
            # Add new state
            self.counts[state] = 0
            self.Qtable[state] = np.zeros((self.nactions), dtype=float)
            self._add_stats_(state, Q_values)
            if not self.use_representatives:
                for p in self.predicates:
                    if p(state):
                        self.predicates_set[p].add(state)

    def get_representative(self, state: S) -> S:
        repres = tuple(p for p in self.predicates if p(state))
        if repres in self.representatives:
            state = self.representatives[repres]
        else:
            state = __make_hashable__(state)
            self.representatives[repres] = state
            # Update predicate set
            for p in repres:
                self.predicates_set[p].add(state)
        return state

    def state_probability(self, state: S) -> float:
        """
        State must be a representative
        """
        state = __make_hashable__(state)
        return self.counts.get(state, 0) / max(1, self._total_visits)

    def state_visits(self, state: S) -> int:
        """
        State can be any state
        """
        state = self.get_representative(state)
        return self.counts.get(state, 0)

    def state_Q(self, state: S) -> Optional[np.ndarray]:
        """
        State can be any state
        """
        state = self.get_representative(state)
        return self.Qtable.get(state, None)

    def _add_stats_(self, state: S, Q_values: np.ndarray):
        self.counts[state] += 1
        self.Qtable[state] += Q_values

    def visit_state(self, state: S, Q_values: np.ndarray):
        if self.nactions < 0:
            self.nactions = Q_values.shape[0]
        if self.use_representatives:
            state = self.get_representative(state)
        self._add_stats_for_representative_(state, Q_values)

    def reset_count(self):
        self.Qtable = {s: Q * 0 for s, Q in self.Qtable.items()}
        self._total_visits = 0
        self.counts = {s: 0 for s in self.counts.keys()}
        self.learnt_Q = {}

    def predicate_state_Q(self, pred_state: Tuple[Predicate[S], ...]) -> np.ndarray:
        return self.Qtable[self.representatives[pred_state]]

    def predicates_states(self) -> Set[Tuple[Predicate[S], ...]]:
        return set(self.representatives.keys())

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
            for repres, s in self.representatives.items():
                target = left if s in selected_states else right
                new_repres = tuple(p for p in repres if p != predicate)
                target.representatives[new_repres] = s
        # Update visits
        left._total_visits = sum(c for c in left.counts.values())
        right._total_visits = self._total_visits - left._total_visits
        return left, right

    def learn_qvalues(
        self,
        state: S,
        action: int,
        r: float,
        next_state: S,
        done: bool,
        alpha: float = 0.01,
        gamma: float = 0.99,
    ):
        s = self.get_representative(state)
        if not done:
            stp1 = self.get_representative(next_state)
            nQ = self.learnt_Q.get(stp1, None)
            nval = np.max(nQ) if nQ is not None else 0
        else:
            nval = 0
        if s not in self.learnt_Q:
            self.learnt_Q[s] = np.zeros((self.nactions), dtype=float)
        self.learnt_Q[s][action] += alpha * (
            r + gamma * nval - self.learnt_Q[s][action]
        )

    def mix_learnt(self, current: float, learnt: float):
        """
        Qvals = current * Qval_visit + learnt * Qval_learnt
        """
        for state, Qvals in self.Qtable.items():
            if self.counts[state] == 0:
                continue
            Qvals /= self.counts[state]
            if state in self.learnt_Q:
                Qvals *= current
                Qvals += learnt * self.learnt_Q[state]

    @property
    def states(self) -> Set[S]:
        return set(self.Qtable.keys())

    def random_splits(self, seed: int) -> "Generator[PredicateSpace[S], None, None]":
        states = list(self.states)
        sample_size = int(np.floor(np.sqrt(len(states))))
        rng = np.random.default_rng(seed)
        while True:
            sub_space = PredicateSpace(self.predicates, self.use_representatives)
            selected_states = {
                tuple(s) for s in rng.choice(states, size=sample_size, replace=False)  # type: ignore
            }
            # Update nactions
            sub_space.nactions = self.nactions
            # Update counts and Qtable
            for s in selected_states:
                sub_space.counts[s] = self.counts[s]
                sub_space.Qtable[s] = self.Qtable[s]
            # Update predicates_set
            sub_space.predicates_set = {
                p: a.intersection(selected_states)
                for p, a in self.predicates_set.items()
            }
            # Update representatives
            if self.use_representatives:
                for repres, s in self.representatives.items():
                    if s in selected_states:
                        sub_space.representatives[repres] = s
            # Update visits
            sub_space._total_visits = self._total_visits
            yield sub_space


def enumerated_space(
    states: List[S],
    Q: Callable[[S], np.ndarray],
    predicates: List[Predicate[S]],
    use_representatives: bool = False,
) -> PredicateSpace[S]:
    space = PredicateSpace(predicates, use_representatives)
    for s in states:
        space.visit_state(s, Q(s))
    return space
